"""Safe device task executor — file operations via stdlib and subprocess via shlex, never shell=True."""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


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


class BuiltinTasks:
    """File operations using Python stdlib only. Always available."""

    @staticmethod
    def list_files(directory: str, pattern: str = "*") -> DeviceTaskResult:
        import glob
        matches = glob.glob(
            os.path.join(directory, "**", pattern), recursive=True)
        lines   = matches[:50]
        extra   = f"\n...and {len(matches)-50} more" if len(matches) > 50 else ""
        return DeviceTaskResult(True, "\n".join(lines) + extra,
                                tool_used="python_glob")

    @staticmethod
    def read_file(path: str, max_chars: int = 5000) -> DeviceTaskResult:
        try:
            content = Path(path).read_text(errors="replace")[:max_chars]
            return DeviceTaskResult(True, content, tool_used="python_io")
        except OSError as e:
            return DeviceTaskResult(False, "", error=str(e))

    @staticmethod
    def write_file(path: str, content: str) -> DeviceTaskResult:
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(content, encoding="utf-8")
            return DeviceTaskResult(
                True, f"Written: {path}",
                files_created=[path], tool_used="python_io",
                undo_command=f"del {path}" if sys.platform == "win32" else f"rm '{path}'",
            )
        except OSError as e:
            return DeviceTaskResult(False, "", error=str(e))

    @staticmethod
    def move_file(src: str, dst: str) -> DeviceTaskResult:
        import shutil as sh
        try:
            sh.move(src, dst)
            return DeviceTaskResult(
                True, f"Moved: {src} → {dst}",
                files_modified=[dst], tool_used="python_shutil",
                undo_command=f"mv '{dst}' '{src}'",
            )
        except OSError as e:
            return DeviceTaskResult(False, "", error=str(e))

    @staticmethod
    def copy_file(src: str, dst: str) -> DeviceTaskResult:
        import shutil as sh
        try:
            sh.copy2(src, dst)
            return DeviceTaskResult(
                True, f"Copied: {src} → {dst}",
                files_created=[dst], tool_used="python_shutil",
                undo_command=f"rm '{dst}'",
            )
        except OSError as e:
            return DeviceTaskResult(False, "", error=str(e))

    @staticmethod
    def trash_file(path: str) -> DeviceTaskResult:
        """Soft delete — moves to ~/.prism/trash/ with timestamp prefix."""
        import shutil as sh
        trash = Path.home() / ".prism" / "trash"
        trash.mkdir(parents=True, exist_ok=True)
        dest = trash / f"{int(time.time())}_{Path(path).name}"
        try:
            sh.move(path, str(dest))
            return DeviceTaskResult(
                True, f"Moved to trash: {dest}",
                tool_used="python_trash",
                undo_command=f"mv '{dest}' '{path}'",
            )
        except OSError as e:
            return DeviceTaskResult(False, "", error=str(e))

    @staticmethod
    def search_files(directory: str, query: str,
                     extension: str = "") -> DeviceTaskResult:
        import glob
        pattern = f"**/*.{extension}" if extension else "**/*"
        results: list[str] = []
        for fp in glob.glob(os.path.join(directory, pattern), recursive=True):
            if not os.path.isfile(fp):
                continue
            try:
                text = Path(fp).read_text(errors="replace")
                if query.lower() in text.lower():
                    for i, line in enumerate(text.splitlines(), 1):
                        if query.lower() in line.lower():
                            results.append(f"{fp}:{i}: {line.strip()[:80]}")
                            break
            except OSError:
                pass
            if len(results) >= 20:
                break
        out = "\n".join(results) if results else f"No matches for '{query}'"
        return DeviceTaskResult(bool(results), out, tool_used="python_search")


def which(name: str) -> Optional[str]:
    """Locate *name* on PATH.

    Bridge for bundled organs: the organ loader's AST safety visitor blocks
    `import shutil` inside organ files, so platform-detection helpers import
    this instead.
    """
    import shutil
    return shutil.which(name)


def current_uid() -> int:
    """Return the current user id (0 on platforms without getuid)."""
    return os.getuid() if hasattr(os, "getuid") else 0


def run_argv(args: list[str], timeout: int = 10) -> DeviceTaskResult:
    """Run an argv list with shell=False and captured output.

    Bridge for bundled organs, where `import subprocess` is blocked by the
    organ loader's AST safety visitor. Never raises: hard failures (command
    missing, timeout) come back as a DeviceTaskResult with empty tool_used;
    a nonzero exit sets success=False but keeps tool_used/output populated.
    """
    if not args:
        return DeviceTaskResult(False, "", error="Empty command")
    start = time.time()
    try:
        proc = subprocess.run(  # nosec B603 — shell=False, argv list only
            args, capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError:
        return DeviceTaskResult(
            False, "", error=f"Command not found: {args[0]}",
            command_run=" ".join(args),
        )
    except subprocess.TimeoutExpired:
        return DeviceTaskResult(
            False, "", error=f"Timed out after {timeout}s",
            command_run=" ".join(args),
        )
    except Exception as exc:
        return DeviceTaskResult(
            False, "", error=str(exc), command_run=" ".join(args),
        )
    output = (proc.stdout or "") + (proc.stderr or "")
    return DeviceTaskResult(
        success     = proc.returncode == 0,
        output      = output[:5000],
        tool_used   = args[0],
        command_run = " ".join(args),
        elapsed_ms  = (time.time() - start) * 1000,
        error       = (proc.stderr or "")[:500] if proc.returncode != 0 else "",
    )


class SafeSubprocess:
    """
    Subprocess wrapper that never uses shell=True.
    Parses commands with shlex so they run safely as argument lists.
    """

    def run(
        self,
        command: str,
        cwd:     Optional[str] = None,
        timeout: int = 30,
    ) -> DeviceTaskResult:
        try:
            args = shlex.split(command)
        except ValueError as e:
            return DeviceTaskResult(False, "", error=f"Command parse error: {e}")

        start = time.time()
        try:
            proc = subprocess.run(  # nosec B603 — shell=False enforced by module contract (see docstring)
                args,
                capture_output = True,
                text           = True,
                cwd            = cwd or str(Path.home()),
                timeout        = timeout,
            )
            elapsed = (time.time() - start) * 1000
            output  = (proc.stdout or "") + (proc.stderr or "")
            return DeviceTaskResult(
                success     = proc.returncode == 0,
                output      = output[:5000],
                tool_used   = args[0] if args else "unknown",
                command_run = command,
                elapsed_ms  = elapsed,
                error       = proc.stderr[:500] if proc.returncode != 0 else "",
            )
        except FileNotFoundError:
            return DeviceTaskResult(
                False, "",
                error=f"Command not found: {command.split()[0]}",
                command_run=command,
            )
        except subprocess.TimeoutExpired:
            return DeviceTaskResult(
                False, "",
                error=f"Timed out after {timeout}s",
                command_run=command,
            )

    def run_python_code(self, code: str) -> DeviceTaskResult:
        """Run a small Python snippet in an isolated subprocess."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(code)
            tmp = f.name
        try:
            return self.run(f"{sys.executable} {tmp}")
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
