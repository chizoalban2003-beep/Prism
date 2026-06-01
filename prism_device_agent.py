"""
prism_device_agent.py
=====================
Executes actual tasks on the user's device.

Provides:
  PrismDeviceAgent    — high-level agent; execute(), rescan(), capabilities
  DeviceTaskResult    — result dataclass
  CapabilityMap       — maps categories to available tools
  DeviceCapabilityScanner — fast (<100 ms) startup scanner
  ToolResolver        — resolution chain: stdlib → installed → synthesise →
                        suggest install → manual
"""

from __future__ import annotations

import importlib.util
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# CapabilityMap
# ---------------------------------------------------------------------------

@dataclass
class CapabilityMap:
    cli_tools:   dict[str, list[str]]
    py_packages: list[str]
    platform:    str
    has_browser: bool

    # Category → stdlib fallback flag (always available)
    _STDLIB_CATEGORIES: frozenset = field(
        default_factory=lambda: frozenset({
            "list_files", "read_file", "write_file", "move_file",
            "copy_file", "delete_file", "find_file", "search_in_files",
            "compress_zip",
        }),
        repr=False,
        compare=False,
    )

    def can_do(self, category: str) -> bool:
        """Return True if the category can be handled (stdlib or installed tool)."""
        if category in self._STDLIB_CATEGORIES:
            return True
        return bool(self.cli_tools.get(category)) or bool(self.best_tool(category))

    def best_tool(self, category: str) -> Optional[str]:
        """Return the preferred CLI tool for a category, or None."""
        tools = self.cli_tools.get(category, [])
        return tools[0] if tools else None

    def summary(self) -> str:
        lines = [f"Platform: {self.platform}", f"Browser: {self.has_browser}"]
        if self.cli_tools:
            cats = ", ".join(
                f"{cat}({','.join(tools)})"
                for cat, tools in self.cli_tools.items()
                if tools
            )
            lines.append(f"CLI: {cats}")
        if self.py_packages:
            lines.append(f"Packages: {', '.join(self.py_packages[:10])}")
        return " | ".join(lines)


# ---------------------------------------------------------------------------
# DeviceCapabilityScanner
# ---------------------------------------------------------------------------

# Map from category → candidate CLI names
_CLI_CANDIDATES: dict[str, list[str]] = {
    "compress_zip":    ["zip", "7z", "tar"],
    "compress_tar":    ["tar", "7z"],
    "image_resize":    ["convert", "ffmpeg", "magick"],
    "image_compress":  ["convert", "cjpeg", "ffmpeg"],
    "video":           ["ffmpeg", "vlc"],
    "git":             ["git"],
    "search":          ["rg", "grep", "ag"],
    "find_file":       ["find", "fd", "locate"],
    "open_app":        ["xdg-open", "open", "start"],
    "package_manager": ["pip", "pip3", "brew", "apt", "apt-get", "choco"],
}

# Python packages to check for
_PY_CANDIDATES = [
    "PIL", "Pillow", "requests", "numpy", "pandas",
    "flask", "fastapi", "django", "aiohttp",
    "psutil", "pydantic", "boto3",
]

# Browser executables
_BROWSER_CANDIDATES = [
    "google-chrome", "chromium", "chromium-browser",
    "firefox", "brave-browser", "safari", "msedge",
]


class DeviceCapabilityScanner:
    def scan(self) -> CapabilityMap:
        cli_tools: dict[str, list[str]] = {}
        for category, candidates in _CLI_CANDIDATES.items():
            found = [c for c in candidates if shutil.which(c)]
            if found:
                cli_tools[category] = found

        py_packages = [
            pkg for pkg in _PY_CANDIDATES
            if importlib.util.find_spec(pkg) is not None
        ]

        # Also check PIL under the Pillow import name
        if "PIL" not in py_packages and importlib.util.find_spec("PIL") is not None:
            py_packages.append("PIL")

        has_browser = any(shutil.which(b) for b in _BROWSER_CANDIDATES)
        if sys.platform == "darwin":
            # macOS: Safari and built-in browsers
            has_browser = has_browser or Path("/Applications/Safari.app").exists()

        return CapabilityMap(
            cli_tools=cli_tools,
            py_packages=py_packages,
            platform=sys.platform,
            has_browser=has_browser,
        )


# ---------------------------------------------------------------------------
# ToolResolution
# ---------------------------------------------------------------------------

@dataclass
class ToolResolution:
    resolved:         bool
    method:           str        # "stdlib" | "cli" | "py_package" | "suggest_install" | "manual"
    tool_name:        str = ""
    command_template: str = ""
    requires_install: bool = False
    install_hint:     str = ""
    description:      str = ""


class ToolResolver:
    """
    Resolution chain: stdlib → installed → synthesise → suggest install → manual
    """

    # task_type → (stdlib handler name, description)
    _STDLIB_MAP: dict[str, tuple[str, str]] = {
        "list_files":       ("_stdlib_list_files",     "List directory contents via os.listdir"),
        "read_file":        ("_stdlib_read_file",      "Read file via open()"),
        "write_file":       ("_stdlib_write_file",     "Write file via open()"),
        "copy_file":        ("_stdlib_copy_file",      "Copy file via shutil.copy2"),
        "move_file":        ("_stdlib_move_file",      "Move/rename via shutil.move"),
        "delete_file":      ("_stdlib_delete_file",    "Move to trash via shutil.move"),
        "find_file":        ("_stdlib_find_file",      "Find files via pathlib.glob"),
        "search_in_files":  ("_stdlib_search_files",   "Search text via pathlib + re"),
        "compress_zip":     ("_stdlib_compress_zip",   "Create zip via zipfile module"),
    }

    # task_type → (cli category, py package, install hint)
    _EXTERNAL_MAP: dict[str, tuple[str, str, str]] = {
        "image_resize":    ("image_resize",   "PIL",     "pip install Pillow"),
        "image_compress":  ("image_compress", "PIL",     "pip install Pillow"),
        "video_convert":   ("video",          "",        "Install ffmpeg"),
        "git_commit":      ("git",            "",        "Install git"),
        "git_push":        ("git",            "",        "Install git"),
        "git_pull":        ("git",            "",        "Install git"),
        "git_status":      ("git",            "",        "Install git"),
        "compress_zip":    ("compress_zip",   "zipfile", ""),
    }

    def resolve(
        self,
        task_type: str,
        task_detail: str,
        capabilities: CapabilityMap,
    ) -> ToolResolution:
        # 1. stdlib
        if task_type in self._STDLIB_MAP:
            handler, desc = self._STDLIB_MAP[task_type]
            return ToolResolution(
                resolved=True, method="stdlib",
                tool_name="stdlib", description=desc,
            )

        # 2. installed CLI
        ext = self._EXTERNAL_MAP.get(task_type)
        if ext:
            cli_cat, py_pkg, install_hint = ext
            cli_tool = capabilities.best_tool(cli_cat)
            if cli_tool:
                return ToolResolution(
                    resolved=True, method="cli",
                    tool_name=cli_tool,
                )
            # 3. installed Python package
            if py_pkg and py_pkg in capabilities.py_packages:
                return ToolResolution(
                    resolved=True, method="py_package",
                    tool_name=py_pkg,
                )
            # 4. suggest install
            if install_hint:
                return ToolResolution(
                    resolved=False, method="suggest_install",
                    requires_install=True, install_hint=install_hint,
                    description=f"Not available locally. {install_hint}",
                )

        # 5. manual
        return ToolResolution(
            resolved=False, method="manual",
            description=f"No handler found for '{task_type}'. Manual action required.",
        )


# ---------------------------------------------------------------------------
# DeviceTaskResult
# ---------------------------------------------------------------------------

@dataclass
class DeviceTaskResult:
    success:        bool
    output:         str
    files_created:  list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    tool_used:      str = ""
    command_run:    str = ""
    elapsed_ms:     float = 0.0
    error:          str = ""
    undo_command:   str = ""
    needs_approval: bool = False   # True when policy requires user confirmation


def _fail(error: str, elapsed_ms: float = 0.0) -> DeviceTaskResult:
    return DeviceTaskResult(success=False, output="", error=error, elapsed_ms=elapsed_ms)


# ---------------------------------------------------------------------------
# Danger detection
# ---------------------------------------------------------------------------

_DANGEROUS_PATTERNS = [
    r"rm\s+-[rf]+\s+/",
    r"rmdir\s+/",
    r"format\s+[a-z]:",
    r"mkfs\.",
    r"dd\s+if=.*of=/dev/",
    r">\s*/dev/(s|h)d[a-z]",
    r"chmod\s+-R\s+777\s+/",
    r"chown\s+-R\s+.*\s+/",
    r"shutdown|reboot|halt|poweroff",
]
_DANGER_RE = re.compile("|".join(_DANGEROUS_PATTERNS), re.IGNORECASE)


def _is_dangerous(command: str) -> bool:
    return bool(_DANGER_RE.search(command))


# ---------------------------------------------------------------------------
# Task type classifier
# ---------------------------------------------------------------------------

_TASK_PATTERNS: list[tuple[str, str]] = [
    (r"list\s+files?\s+(?:in\s+)?(.+)",       "list_files"),
    (r"read\s+file\s+(.+)",                    "read_file"),
    (r"write\s+(?:to\s+)?file\s+(.+)",         "write_file"),
    (r"copy\s+(.+)\s+to\s+(.+)",               "copy_file"),
    (r"move\s+(.+)\s+to\s+(.+)",               "move_file"),
    (r"rename\s+(.+)\s+to\s+(.+)",             "move_file"),
    (r"delete\s+(?:file\s+)?(.+)",             "delete_file"),
    (r"remove\s+(?:file\s+)?(.+)",             "delete_file"),
    (r"find\s+files?\s+(?:named\s+)?(.+)",     "find_file"),
    (r"search\s+(?:for\s+|in\s+)(.+)",         "search_in_files"),
    (r"grep\s+(.+)",                           "search_in_files"),
    (r"compress|zip\s+(.+)",                   "compress_zip"),
    (r"resize\s+(?:image\s+)?(.+)",            "image_resize"),
    (r"convert\s+(?:image\s+)?(.+)",           "image_resize"),
    (r"git\s+commit",                          "git_commit"),
    (r"git\s+push",                            "git_push"),
    (r"git\s+pull",                            "git_pull"),
    (r"git\s+status",                          "git_status"),
    (r"run\s+(?:command|script)\s+(.+)",       "run_command"),
    (r"execute\s+(.+)",                        "run_command"),
    (r"open\s+(?:app|application)\s+(.+)",     "open_app"),
    (r"install\s+(?:package|app)\s+(.+)",      "install_package"),
    (r"what(?:'s| is)\s+(?:on|in)\s+my\s+(.+)","list_files"),
    (r"show\s+me\s+(?:my\s+)?files",           "list_files"),
]


def _classify_task(message: str) -> tuple[str, dict]:
    """Return (task_type, extracted_params) from a natural-language task message."""
    lowered = message.lower().strip()
    for pattern, task_type in _TASK_PATTERNS:
        m = re.search(pattern, lowered)
        if m:
            return task_type, {"_match": m.groups()}
    return "unknown", {}


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

_PROTECTED_WRITE_DIRS = frozenset({
    "/etc", "/sys", "/proc", "/dev", "/boot",
    "/lib", "/lib64", "/usr", "/sbin", "/bin",
    "C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)",
})


def _validate_path(path_str: str, writable: bool = False) -> tuple[Path, str]:
    """
    Resolve and validate a user-provided path.
    Returns (resolved_path, error_string). error_string is empty if path is safe.
    """
    if not path_str or not path_str.strip():
        return Path("."), "Empty path"
    # Block path traversal before any resolution
    if ".." in path_str:
        return Path("."), "Path traversal not allowed"
    try:
        # Resolve to an absolute, canonical path
        resolved = os.path.realpath(os.path.expanduser(path_str.strip()))
    except Exception as exc:
        return Path("."), f"Invalid path: {exc}"

    # Validate the resolved path stays within allowed directory roots
    _safe_roots = (
        os.path.realpath(str(Path.home())),
        os.path.realpath(tempfile.gettempdir()),
    )
    if not any(
        resolved == root or resolved.startswith(root + os.sep)
        for root in _safe_roots
    ):
        return Path("."), f"Access denied: path outside safe directories: {resolved}"

    p = Path(resolved)

    if writable:
        p_str = str(p)
        for protected in _PROTECTED_WRITE_DIRS:
            if p_str == protected or p_str.startswith(protected + os.sep):
                return p, f"Write access denied to protected directory: {p}"

    return p, ""


# ---------------------------------------------------------------------------
# Execution helpers (stdlib)
# ---------------------------------------------------------------------------

_TRASH_DIR = Path("~/.prism/trash").expanduser()


def _ensure_trash() -> Path:
    _TRASH_DIR.mkdir(parents=True, exist_ok=True)
    return _TRASH_DIR


def _exec_list_files(path_str: str) -> DeviceTaskResult:
    p, err = _validate_path(path_str)
    if err:
        return _fail(err)
    if not p.exists():
        return _fail(f"Path not found: {p}")
    if p.is_file():
        return DeviceTaskResult(success=True, output=str(p), tool_used="stdlib")
    entries = sorted(p.iterdir())
    lines = [f"[{'D' if entry.is_dir() else 'F'}] {entry.name}" for entry in entries]
    return DeviceTaskResult(
        success=True, output="\n".join(lines) or "(empty)",
        tool_used="stdlib", command_run=f"listdir({p})",
    )


def _exec_read_file(path_str: str) -> DeviceTaskResult:
    p, err = _validate_path(path_str)
    if err:
        return _fail(err)
    if not p.exists():
        return _fail(f"File not found: {p}")
    try:
        content = p.read_text(errors="replace")
        return DeviceTaskResult(
            success=True, output=content,
            tool_used="stdlib", command_run=f"read({p})",
        )
    except Exception as exc:
        return _fail(str(exc))


def _exec_copy_file(src: str, dst: str) -> DeviceTaskResult:
    sp, err = _validate_path(src)
    if err:
        return _fail(err)
    dp, err = _validate_path(dst, writable=True)
    if err:
        return _fail(err)
    if not sp.exists():
        return _fail(f"Source not found: {sp}")
    try:
        shutil.copy2(sp, dp)
        return DeviceTaskResult(
            success=True, output=f"Copied {sp} → {dp}",
            files_created=[str(dp)],
            tool_used="stdlib", command_run=f"copy({sp},{dp})",
            undo_command=f"delete {dp}",
        )
    except Exception as exc:
        return _fail(str(exc))


def _exec_move_file(src: str, dst: str) -> DeviceTaskResult:
    sp, err = _validate_path(src, writable=True)
    if err:
        return _fail(err)
    dp, err = _validate_path(dst, writable=True)
    if err:
        return _fail(err)
    if not sp.exists():
        return _fail(f"Source not found: {sp}")
    try:
        shutil.move(str(sp), str(dp))
        return DeviceTaskResult(
            success=True, output=f"Moved {sp} → {dp}",
            files_modified=[str(dp)],
            tool_used="stdlib", command_run=f"move({sp},{dp})",
            undo_command=f"move {dp} {sp}",
        )
    except Exception as exc:
        return _fail(str(exc))


def _exec_delete_file(path_str: str) -> DeviceTaskResult:
    p, err = _validate_path(path_str, writable=True)
    if err:
        return _fail(err)
    if not p.exists():
        return _fail(f"File not found: {p}")
    trash = _ensure_trash()
    dest = trash / p.name
    # Avoid collision
    if dest.exists():
        dest = trash / f"{p.stem}_{int(time.time())}{p.suffix}"
    try:
        shutil.move(str(p), str(dest))
        return DeviceTaskResult(
            success=True, output=f"Moved {p} to trash ({dest})",
            files_modified=[str(p)],
            tool_used="stdlib", command_run=f"trash({p})",
            undo_command=f"move {dest} {p}",
        )
    except Exception as exc:
        return _fail(str(exc))


def _exec_find_file(pattern: str) -> DeviceTaskResult:
    import fnmatch

    search_dir = Path.home()
    # If pattern contains a slash, treat the leading part as dir
    parts = pattern.rsplit("/", 1)
    if len(parts) == 2:
        candidate, err = _validate_path(parts[0])
        if not err and candidate.exists():
            search_dir = candidate
            pat = parts[1]
        else:
            pat = pattern.strip()
    else:
        pat = pattern.strip()

    found = []
    try:
        for item in search_dir.rglob("*"):
            if fnmatch.fnmatch(item.name, pat):
                found.append(str(item))
                if len(found) >= 100:
                    break
    except Exception:
        pass
    output = "\n".join(found) if found else f"No files matching '{pat}' found."
    return DeviceTaskResult(
        success=True, output=output,
        tool_used="stdlib", command_run=f"rglob({search_dir},{pat})",
    )


def _exec_search_in_files(query: str) -> DeviceTaskResult:
    """Search for a string in files under a directory."""
    # Parse "for <term> in <dir>" using simple split rather than complex regex
    search_dir = Path.cwd()
    term = query.strip()

    # Check for "in <dir>" suffix
    in_idx = query.rfind(" in ")
    if in_idx != -1:
        dir_candidate = query[in_idx + 4:].strip()
        p, err = _validate_path(dir_candidate)
        if not err and p.exists():
            search_dir = p
            term = query[:in_idx].strip()

    # Strip leading "for " keyword
    if term.lower().startswith("for "):
        term = term[4:].strip()
    # Strip surrounding quotes
    if len(term) >= 2 and term[0] in ('"', "'") and term[-1] == term[0]:
        term = term[1:-1]

    if not term:
        return _fail("No search term provided.")

    matches = []
    try:
        for fpath in search_dir.rglob("*"):
            if not fpath.is_file():
                continue
            try:
                text = fpath.read_text(errors="replace")
                for i, line in enumerate(text.splitlines(), 1):
                    if term.lower() in line.lower():
                        matches.append(f"{fpath}:{i}: {line.strip()}")
                        if len(matches) >= 50:
                            break
            except Exception:
                pass
            if len(matches) >= 50:
                break
    except Exception as exc:
        return _fail(str(exc))

    output = "\n".join(matches) if matches else f"No matches for '{term}'."
    return DeviceTaskResult(
        success=True, output=output,
        tool_used="stdlib", command_run=f"search({search_dir},{term!r})",
    )


def _exec_compress_zip(path_str: str, dest_str: str = "") -> DeviceTaskResult:
    import zipfile

    src, err = _validate_path(path_str)
    if err:
        return _fail(err)
    if not src.exists():
        return _fail(f"Path not found: {src}")
    if dest_str:
        out, err = _validate_path(dest_str, writable=True)
        if err:
            return _fail(err)
    else:
        out = src.with_suffix(".zip")
    try:
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            if src.is_dir():
                for f in src.rglob("*"):
                    if f.is_file():
                        zf.write(f, f.relative_to(src))
            else:
                zf.write(src, src.name)
        return DeviceTaskResult(
            success=True, output=f"Created {out}",
            files_created=[str(out)],
            tool_used="stdlib", command_run=f"zipfile({src}→{out})",
            undo_command=f"delete {out}",
        )
    except Exception as exc:
        return _fail(str(exc))


def _exec_run_command(command: str) -> DeviceTaskResult:
    """Run a command with safety checks, avoiding shell injection."""
    if _is_dangerous(command):
        return _fail("Command refused: dangerous pattern detected.")
    try:
        args = shlex.split(command)
    except ValueError as exc:
        return _fail(f"Could not parse command: {exc}")
    if not args:
        return _fail("Empty command.")
    try:
        result = subprocess.run(
            args,
            shell=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = (result.stdout + result.stderr).strip()
        return DeviceTaskResult(
            success=result.returncode == 0,
            output=output or "(no output)",
            tool_used="subprocess",
            command_run=command,
            error="" if result.returncode == 0 else f"Exit {result.returncode}: {result.stderr[:200]}",
        )
    except subprocess.TimeoutExpired:
        return _fail("Command timed out after 30 s")
    except Exception as exc:
        return _fail(str(exc))


def _exec_git(sub: str, params: dict) -> DeviceTaskResult:
    """Run a git sub-command using a safe argument list."""
    if sub == "commit":
        msg = params.get("message", "auto commit by PRISM")
        # Two sequential calls to avoid compound shell command
        try:
            r1 = subprocess.run(
                ["git", "add", "-A"],
                capture_output=True, text=True, timeout=30,
            )
            if r1.returncode != 0:
                return _fail(f"git add failed: {r1.stderr[:200]}")
            r2 = subprocess.run(
                ["git", "commit", "-m", msg],
                capture_output=True, text=True, timeout=30,
            )
            output = (r2.stdout + r2.stderr).strip()
            return DeviceTaskResult(
                success=r2.returncode == 0,
                output=output or "(no output)",
                tool_used="git",
                command_run=f"git add -A && git commit -m {shlex.quote(msg)}",
                error="" if r2.returncode == 0 else f"Exit {r2.returncode}: {r2.stderr[:200]}",
            )
        except Exception as exc:
            return _fail(str(exc))

    allowed_subs = {"push", "pull", "status", "log", "diff", "fetch"}
    if sub not in allowed_subs:
        return _fail(f"git sub-command '{sub}' is not permitted.")
    try:
        result = subprocess.run(
            ["git", sub],
            capture_output=True, text=True, timeout=30,
        )
        output = (result.stdout + result.stderr).strip()
        return DeviceTaskResult(
            success=result.returncode == 0,
            output=output or "(no output)",
            tool_used="git",
            command_run=f"git {sub}",
            error="" if result.returncode == 0 else f"Exit {result.returncode}: {result.stderr[:200]}",
        )
    except Exception as exc:
        return _fail(str(exc))


# ---------------------------------------------------------------------------
# PrismDeviceAgent
# ---------------------------------------------------------------------------


def _split_src_dst(lowered: str, prefix_pattern: str) -> tuple[str, str]:
    """
    Extract src and dst from a string like '<prefix> <src> to <dst>'.
    Uses a simple string split on ' to ' rather than backtracking regex.
    Returns ('', '') if parsing fails.
    """
    m = re.search(prefix_pattern, lowered)
    if not m:
        return "", ""
    remainder = lowered[m.end():]
    # Split on the first literal " to "
    parts = remainder.split(" to ", 1)
    if len(parts) != 2:
        return "", ""
    src_part = parts[0].strip()
    dst_part = parts[1].strip()
    src = src_part.split()[0] if src_part else ""
    dst = dst_part.split()[0] if dst_part else ""
    return src, dst


class PrismDeviceAgent:
    """
    Executes device tasks on behalf of the PRISM agent.
    """

    def __init__(
        self,
        capabilities:  CapabilityMap,
        policy_engine=None,
        on_approval:   Optional[Callable[[str, str], bool]] = None,
        collaborator=None,
        user:          str = "default",
    ) -> None:
        self._capabilities = capabilities
        self._policy       = policy_engine
        self._on_approval  = on_approval
        self._collaborator = collaborator
        self._user         = user
        self._resolver     = ToolResolver()

    @classmethod
    def setup(
        cls,
        policy_engine=None,
        on_approval:   Optional[Callable[[str, str], bool]] = None,
        collaborator=None,
        user:          str = "default",
    ) -> "PrismDeviceAgent":
        caps = DeviceCapabilityScanner().scan()
        return cls(
            capabilities=caps,
            policy_engine=policy_engine,
            on_approval=on_approval,
            collaborator=collaborator,
            user=user,
        )

    @property
    def capabilities(self) -> CapabilityMap:
        return self._capabilities

    def rescan(self) -> CapabilityMap:
        self._capabilities = DeviceCapabilityScanner().scan()
        return self._capabilities

    # ------------------------------------------------------------------ #

    @staticmethod
    def _safe_path(raw: str, allowed_roots: list[str] = None) -> str:
        """
        Resolve and validate a path.
        Raises ValueError if path traversal is detected or path is outside
        allowed roots. Returns the resolved absolute path string.
        """
        if not raw:
            raise ValueError("Empty path")
        # Block obvious traversal attempts before any resolution
        if ".." in raw:
            raise ValueError("Path traversal not allowed")
        resolved = os.path.realpath(os.path.expanduser(raw))
        if allowed_roots:
            if not any(
                resolved.startswith(os.path.realpath(r) + os.sep)
                or resolved == os.path.realpath(r)
                for r in allowed_roots
            ):
                raise ValueError(
                    f"Path '{resolved}' is outside allowed directories"
                )
        return resolved

    # ------------------------------------------------------------------ #

    def _check_policy(self, task: str) -> tuple:
        """Returns (allowed: bool, needs_approval: bool)"""
        if self._policy is None:
            return True, False
        try:
            from prism_policy import PolicyEngine
            verdict, reason = self._policy.evaluate(
                user="default", category="device",
                provider="local", estimated_cost=0.0, action=task)
            if verdict == PolicyEngine.Verdict.REJECT:
                return False, False
            if verdict == PolicyEngine.Verdict.ESCALATE:
                return False, True
            return True, False
        except Exception:
            return True, False

    # ------------------------------------------------------------------ #

    def execute(
        self,
        task:    str,
        params:  Optional[dict] = None,
        dry_run: bool = False,
        approval_override: bool = False,
    ) -> DeviceTaskResult:
        params = params or {}
        t0 = time.monotonic()

        # Safety: check for dangerous patterns in raw task text
        if _is_dangerous(task):
            return _fail(
                f"Task refused: dangerous pattern detected.",
                elapsed_ms=(time.monotonic() - t0) * 1000,
            )

        task_type, _extracted = _classify_task(task)

        # Dry run: describe without executing
        if dry_run:
            resolution = self._resolver.resolve(task_type, task, self._capabilities)
            elapsed = (time.monotonic() - t0) * 1000
            return DeviceTaskResult(
                success=True,
                output=(
                    f"[DRY RUN] Would execute '{task_type}' using {resolution.method}. "
                    f"{resolution.description or resolution.tool_name}"
                ),
                tool_used=resolution.method,
                elapsed_ms=elapsed,
            )

        if not approval_override:
            allowed, needs_approval = self._check_policy(task)
            if not allowed and not needs_approval:
                return DeviceTaskResult(
                    False, "", error=f"Policy denied: {task_type}",
                    elapsed_ms=(time.monotonic() - t0) * 1000,
                )
            if needs_approval:
                import json as _j
                return DeviceTaskResult(
                    False, f"Approval required for: {task}",
                    tool_used="pending_approval",
                    needs_approval=True,
                    undo_command=_j.dumps({"task": task, "params": params}),
                    elapsed_ms=(time.monotonic() - t0) * 1000,
                )

        result = self._dispatch(task_type, task, params)
        result.elapsed_ms = (time.monotonic() - t0) * 1000
        return result

    # ------------------------------------------------------------------ #

    def _dispatch(self, task_type: str, task: str, params: dict) -> DeviceTaskResult:
        lowered = task.lower().strip()

        _home = str(Path.home())

        if task_type == "list_files":
            m = re.search(
                r"(?:list\s+files?\s+(?:in\s+)?|what(?:'s| is)\s+(?:on|in)\s+my\s+)([\w./~-]+)",
                lowered,
            )
            raw_path = m.group(1) if m else params.get("path", _home)
            try:
                path_str = self._safe_path(raw_path)
            except ValueError as exc:
                return _fail(f"Invalid path: {exc}")
            return _exec_list_files(path_str)

        if task_type == "read_file":
            m = re.search(r"read\s+file\s+(\S+)", lowered)
            raw_path = m.group(1) if m else params.get("path", "")
            if not raw_path:
                return _fail("No file path provided.")
            try:
                path_str = self._safe_path(raw_path)
            except ValueError as exc:
                return _fail(f"Invalid path: {exc}")
            return _exec_read_file(path_str)

        if task_type == "copy_file":
            src_raw, dst_raw = _split_src_dst(lowered, r"copy\s+")
            if not (src_raw and dst_raw):
                src_raw = params.get("src", "")
                dst_raw = params.get("dst", "")
            if src_raw and dst_raw:
                try:
                    src = self._safe_path(src_raw)
                    dst = self._safe_path(dst_raw)
                except ValueError as exc:
                    return _fail(f"Invalid path: {exc}")
                return _exec_copy_file(src, dst)
            return _fail("Could not parse copy source/destination.")

        if task_type == "move_file":
            src_raw, dst_raw = _split_src_dst(lowered, r"(?:move|rename)\s+")
            if not (src_raw and dst_raw):
                src_raw = params.get("src", "")
                dst_raw = params.get("dst", "")
            if src_raw and dst_raw:
                try:
                    src = self._safe_path(src_raw)
                    dst = self._safe_path(dst_raw)
                except ValueError as exc:
                    return _fail(f"Invalid path: {exc}")
                return _exec_move_file(src, dst)
            return _fail("Could not parse move source/destination.")

        if task_type == "delete_file":
            m = re.search(r"(?:delete|remove)\s+(?:file\s+)?(\S+)", lowered)
            raw_path = m.group(1) if m else params.get("path", "")
            if not raw_path:
                return _fail("No file path provided.")
            try:
                path_str = self._safe_path(raw_path)
            except ValueError as exc:
                return _fail(f"Invalid path: {exc}")
            return _exec_delete_file(path_str)

        if task_type == "find_file":
            m = re.search(r"find\s+files?\s+(?:named\s+)?(\S+)", lowered)
            pattern = m.group(1) if m else params.get("pattern", "*")
            return _exec_find_file(pattern)

        if task_type == "search_in_files":
            # strip the leading keyword
            rest = re.sub(r"^(?:search\s+(?:for\s+|in\s+)?|grep\s+)", "", lowered)
            return _exec_search_in_files(rest or task)

        if task_type == "compress_zip":
            # "compress/zip <src> to <dst>" or "compress/zip <src>"
            m = re.search(r"(?:compress|zip)\s+(\S+)", lowered)
            if m:
                src_part = m.group(1)
                dst_part = ""
                to_idx = lowered.find(" to ", m.start(1))
                if to_idx != -1:
                    dst_part = lowered[to_idx + 4:].strip().split()[0] if lowered[to_idx + 4:].strip() else ""
                try:
                    src_part = self._safe_path(src_part)
                    if dst_part:
                        dst_part = self._safe_path(dst_part)
                except ValueError as exc:
                    return _fail(f"Invalid path: {exc}")
                return _exec_compress_zip(src_part, dst_part)
            return _fail("Could not parse compress/zip path.")

        if task_type in ("git_commit", "git_push", "git_pull", "git_status"):
            sub = task_type.replace("git_", "")
            return _exec_git(sub, params)

        if task_type == "run_command":
            m = re.search(r"(?:run\s+(?:command|script)|execute)\s+(.+)", lowered)
            cmd = m.group(1).strip() if m else task
            return _exec_run_command(cmd)

        if task_type == "open_app":
            opener = (
                "open" if sys.platform == "darwin"
                else "xdg-open" if sys.platform.startswith("linux")
                else "start"
            )
            m = re.search(r"open\s+(?:app(?:lication)?\s+)?(\S+)", lowered)
            target = m.group(1) if m else params.get("target", "")
            if not target:
                return _fail("No application specified.")
            try:
                result = subprocess.run(
                    [opener, target],
                    capture_output=True, text=True, timeout=30,
                )
                output = (result.stdout + result.stderr).strip()
                return DeviceTaskResult(
                    success=result.returncode == 0,
                    output=output or "(launched)",
                    tool_used=opener,
                    command_run=f"{opener} {target}",
                    error="" if result.returncode == 0 else result.stderr[:200],
                )
            except Exception as exc:
                return _fail(str(exc))

        if task_type == "install_package":
            m = re.search(r"install\s+(?:package|app)\s+(\S+)", lowered)
            pkg = m.group(1) if m else params.get("package", "")
            if not pkg:
                return _fail("No package name specified.")
            # Validate package name: only alphanum, dash, underscore, dot, brackets
            if not re.fullmatch(r"[\w.\-\[\]]+", pkg):
                return _fail(f"Invalid package name: {pkg!r}")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", pkg],
                    capture_output=True, text=True, timeout=120,
                )
                output = (result.stdout + result.stderr).strip()
                return DeviceTaskResult(
                    success=result.returncode == 0,
                    output=output or "(installed)",
                    tool_used="pip",
                    command_run=f"pip install {pkg}",
                    error="" if result.returncode == 0 else result.stderr[:200],
                )
            except Exception as exc:
                return _fail(str(exc))

        # show files fallback
        if "show" in lowered and "file" in lowered:
            raw_path = params.get("path", _home)
            try:
                path_str = self._safe_path(raw_path)
            except ValueError as exc:
                return _fail(f"Invalid path: {exc}")
            return _exec_list_files(path_str)

        return DeviceTaskResult(
            success=False,
            output="",
            error=f"Don't know how to handle task: '{task[:100]}'",
            tool_used="none",
        )
