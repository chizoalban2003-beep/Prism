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
# Execution helpers (stdlib)
# ---------------------------------------------------------------------------

_TRASH_DIR = Path("~/.prism/trash").expanduser()


def _ensure_trash() -> Path:
    _TRASH_DIR.mkdir(parents=True, exist_ok=True)
    return _TRASH_DIR


def _exec_list_files(path_str: str) -> DeviceTaskResult:
    p = Path(path_str.strip()).expanduser()
    if not p.exists():
        return _fail(f"Path not found: {p}")
    if p.is_file():
        return DeviceTaskResult(
            success=True, output=str(p), tool_used="stdlib",
        )
    entries = sorted(p.iterdir())
    lines = []
    for entry in entries:
        kind = "D" if entry.is_dir() else "F"
        lines.append(f"[{kind}] {entry.name}")
    return DeviceTaskResult(
        success=True, output="\n".join(lines) or "(empty)",
        tool_used="stdlib", command_run=f"listdir({p})",
    )


def _exec_read_file(path_str: str) -> DeviceTaskResult:
    p = Path(path_str.strip()).expanduser()
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
    sp = Path(src.strip()).expanduser()
    dp = Path(dst.strip()).expanduser()
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
    sp = Path(src.strip()).expanduser()
    dp = Path(dst.strip()).expanduser()
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
    p = Path(path_str.strip()).expanduser()
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
    if len(parts) == 2 and Path(parts[0]).expanduser().exists():
        search_dir = Path(parts[0]).expanduser()
        pat = parts[1]
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
    """Search for a string in files under the current directory."""
    # Parse: "for <term> in <dir>" or "in <dir> for <term>"
    dir_match = re.search(r"in\s+([\w./~-]+)", query)
    term_match = re.search(r"(?:for\s+)?['\"]?([^'\"]+?)['\"]?\s*(?:in|$)", query)

    search_dir = Path(dir_match.group(1)).expanduser() if dir_match else Path.cwd()
    term = term_match.group(1).strip() if term_match else query.strip()

    if not search_dir.exists():
        search_dir = Path.cwd()

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

    src = Path(path_str.strip()).expanduser()
    if not src.exists():
        return _fail(f"Path not found: {src}")
    out = Path(dest_str.strip()).expanduser() if dest_str else src.with_suffix(".zip")
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
    """Run a shell command with safety checks."""
    if _is_dangerous(command):
        return _fail(f"Command refused: dangerous pattern detected in '{command}'")
    try:
        result = subprocess.run(
            command,
            shell=True,
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
    cmd = f"git {sub}"
    if sub == "commit":
        msg = params.get("message", "auto commit by PRISM")
        cmd = f'git add -A && git commit -m {shlex.quote(msg)}'
    if _is_dangerous(cmd):
        return _fail(f"Refused: {cmd}")
    return _exec_run_command(cmd)


# ---------------------------------------------------------------------------
# PrismDeviceAgent
# ---------------------------------------------------------------------------

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

    def execute(
        self,
        task:    str,
        params:  Optional[dict] = None,
        dry_run: bool = False,
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

        result = self._dispatch(task_type, task, params)
        result.elapsed_ms = (time.monotonic() - t0) * 1000
        return result

    # ------------------------------------------------------------------ #

    def _dispatch(self, task_type: str, task: str, params: dict) -> DeviceTaskResult:
        lowered = task.lower().strip()

        if task_type == "list_files":
            m = re.search(r"(?:list\s+files?\s+(?:in\s+)?|what(?:'s| is)\s+(?:on|in)\s+my\s+)([\w./~-]+)", lowered)
            path_str = m.group(1) if m else params.get("path", str(Path.home()))
            return _exec_list_files(path_str)

        if task_type == "read_file":
            m = re.search(r"read\s+file\s+([\S]+)", lowered)
            path_str = m.group(1) if m else params.get("path", "")
            if not path_str:
                return _fail("No file path provided.")
            return _exec_read_file(path_str)

        if task_type == "copy_file":
            m = re.search(r"copy\s+([\S]+)\s+to\s+([\S]+)", lowered)
            if m:
                return _exec_copy_file(m.group(1), m.group(2))
            src = params.get("src", "")
            dst = params.get("dst", "")
            if src and dst:
                return _exec_copy_file(src, dst)
            return _fail("Could not parse copy source/destination.")

        if task_type == "move_file":
            m = re.search(r"(?:move|rename)\s+([\S]+)\s+to\s+([\S]+)", lowered)
            if m:
                return _exec_move_file(m.group(1), m.group(2))
            src = params.get("src", "")
            dst = params.get("dst", "")
            if src and dst:
                return _exec_move_file(src, dst)
            return _fail("Could not parse move source/destination.")

        if task_type == "delete_file":
            m = re.search(r"(?:delete|remove)\s+(?:file\s+)?([\S]+)", lowered)
            path_str = m.group(1) if m else params.get("path", "")
            if not path_str:
                return _fail("No file path provided.")
            return _exec_delete_file(path_str)

        if task_type == "find_file":
            m = re.search(r"find\s+files?\s+(?:named\s+)?([\S]+)", lowered)
            pattern = m.group(1) if m else params.get("pattern", "*")
            return _exec_find_file(pattern)

        if task_type == "search_in_files":
            # strip the leading keyword
            rest = re.sub(r"^(search\s+(?:for\s+|in\s+)?|grep\s+)", "", lowered)
            return _exec_search_in_files(rest or task)

        if task_type == "compress_zip":
            m = re.search(r"(?:compress|zip)\s+([\S]+)(?:\s+to\s+([\S]+))?", lowered)
            if m:
                return _exec_compress_zip(m.group(1), m.group(2) or "")
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
            m = re.search(r"open\s+(?:app(?:lication)?\s+)?([\S]+)", lowered)
            target = m.group(1) if m else params.get("target", "")
            if not target:
                return _fail("No application specified.")
            return _exec_run_command(f"{opener} {shlex.quote(target)}")

        if task_type == "install_package":
            m = re.search(r"install\s+(?:package|app)\s+([\S]+)", lowered)
            pkg = m.group(1) if m else params.get("package", "")
            if not pkg:
                return _fail("No package name specified.")
            return _exec_run_command(f"pip install {shlex.quote(pkg)}")

        # show files fallback
        if "show" in lowered and "file" in lowered:
            return _exec_list_files(params.get("path", str(Path.home())))

        return DeviceTaskResult(
            success=False,
            output="",
            error=f"Don't know how to handle task: '{task[:100]}'",
            tool_used="none",
        )
