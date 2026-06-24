from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AcquiredTool:
    tool_id:      str
    name:         str
    description:  str
    code:         str           # synthesised Python module source
    requirements: list[str]     # pip packages needed
    entry_fn:     str           # function name to call: fn(task: str, params: dict) -> str
    created_at:   float = field(default_factory=time.time)
    use_count:    int   = 0
    last_result:  str   = ""


# ── AST-based safety check ────────────────────────────────────────────────────
# The actual checker lives in prism_organ_loader (_SafetyVisitor + _is_safe).
# _is_safe_code delegates there so there is exactly one definition of "unsafe".


def _is_safe_code(code: str) -> tuple[bool, str]:
    """
    AST-based safety check for synthesised tool code.

    Single-sourced from :func:`prism_organ_loader._is_safe` (strict mode) so the
    autonomous engine and the organ loader can never drift apart on what counts
    as unsafe. Strict mode also blocks arbitrary file writes
    (write_text/write_bytes/write), which is the right posture for LLM-generated
    tools. Cannot be bypassed by string obfuscation.
    """
    from prism_organ_loader import _is_safe
    return _is_safe(code, strict=True)


class PrismAutonomous:
    """
    Autonomous tool acquisition and execution engine.

    When PRISM is asked to do something it has no tool for, this engine:
      1. Uses LLM to understand what integration is needed
      2. Synthesises a Python executor module
      3. Safety-checks it
      4. Installs any pip requirements
      5. Dynamically loads and executes it
      6. Caches the tool for future reuse
      7. Pushes a notification when complete

    All synthesised tools are stored in ~/.prism/tools/ and persist
    across sessions — PRISM accumulates capability over time.
    """

    TOOL_DIR = Path("~/.prism/tools").expanduser()

    def __init__(self, llm_router=None, device_agent=None,
                  policy_engine=None, push=None,
                  task_queue=None):
        self._router  = llm_router
        self._device  = device_agent
        self._policy  = policy_engine
        self._push    = push
        self._queue   = task_queue
        self._tools:  dict[str, AcquiredTool] = {}
        self.TOOL_DIR.mkdir(parents=True, exist_ok=True)
        self._load_cached_tools()

    # ── Public API ────────────────────────────────────────────────────────────

    def can_handle(self, task: str) -> bool:
        """True if a cached tool exists that matches this task."""
        return bool(self._find_cached_tool(task))

    def execute_async(self, task: str, params: dict,
                       on_complete=None) -> str:
        """
        Submit task for autonomous background execution.
        Returns task_id immediately. Calls on_complete(result_card) when done.
        """
        task_id = str(uuid.uuid4())[:8]
        if self._queue:
            def _run():
                return self._execute_sync(task, params, task_id)
            self._queue.submit_single(f"Autonomous: {task[:50]}", _run)
        else:
            import threading
            t = threading.Thread(
                target=self._execute_sync,
                args=(task, params, task_id),
                daemon=True)
            t.start()
        return task_id

    def execute_sync(self, task: str, params: dict) -> str:
        """Blocking execution. Returns result string."""
        return self._execute_sync(task, params, str(uuid.uuid4())[:8])

    # ── Core execution pipeline ───────────────────────────────────────────────

    def _execute_sync(self, task: str, params: dict, task_id: str) -> str:
        logger.info("[autonomous] Starting task %s: %s", task_id, task[:60])

        # Step 1: Check cache
        tool = self._find_cached_tool(task)

        # Step 2: If no cached tool, synthesise one
        if not tool:
            logger.info("[autonomous] No cached tool — synthesising")
            tool = self._synthesise_tool(task)
            if not tool:
                msg = f"Could not synthesise a tool for: {task[:60]}"
                self._notify(msg, success=False)
                return msg

        # Step 3: Install requirements
        if tool.requirements:
            ok = self._install_requirements(tool.requirements)
            if not ok:
                msg = f"Failed to install requirements: {tool.requirements}"
                self._notify(msg, success=False)
                return msg

        # Step 4: Execute
        try:
            result = self._run_tool(tool, task, params)
            tool.use_count += 1
            tool.last_result = result[:200]
            self._save_tool(tool)
            self._notify(f"Done: {task[:50]}\n\n{result[:300]}", success=True)
            return result
        except Exception as e:
            msg = f"Tool execution failed: {e}"
            logger.warning("[autonomous] %s", msg)
            self._notify(msg, success=False)
            return msg

    # ── Tool synthesis ────────────────────────────────────────────────────────

    def _synthesise_tool(self, task: str) -> Optional[AcquiredTool]:
        if not self._router:
            return None

        prompt = f"""You are writing a Python tool for a personal assistant.

Task the user wants done: "{task}"

Write a self-contained Python module with:
1. A function called `execute(task: str, params: dict) -> str` that performs the task
2. Use only stdlib + common packages (requests, json, urllib, sqlite3, pathlib, datetime)
3. Return a plain text result string describing what was done
4. Handle errors gracefully — never raise, always return a string
5. No file writes, no shell commands, no eval/exec
6. If you need an API key, read it from params dict (e.g. params.get("api_key",""))

Also list any pip requirements (stdlib only = empty list).

Return ONLY valid JSON:
{{
  "name": "short_tool_name",
  "description": "one sentence what it does",
  "requirements": ["package1"],
  "code": "import ...\\n\\ndef execute(task, params):\\n    ..."
}}"""

        raw, _ = self._router.call(prompt, min_capability=2, max_tokens=1200, json_mode=True)
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data  = json.loads(clean)
        except Exception as e:
            logger.warning("[autonomous] JSON parse failed: %s", e)
            return None

        code = data.get("code", "")
        if not code or "def execute" not in code:
            logger.warning("[autonomous] No execute() function in synthesised code")
            return None

        safe, reason = _is_safe_code(code)
        if not safe:
            logger.warning("[autonomous] Unsafe code blocked: %s", reason)
            return None

        tool = AcquiredTool(
            tool_id      = hashlib.sha256(task.encode()).hexdigest()[:10],
            name         = data.get("name", "custom_tool"),
            description  = data.get("description", task[:80]),
            code         = code,
            requirements = data.get("requirements", []),
            entry_fn     = "execute",
        )
        self._save_tool(tool)
        logger.info("[autonomous] Synthesised tool: %s", tool.name)
        return tool

    # ── Tool execution ────────────────────────────────────────────────────────

    def _run_tool(self, tool: AcquiredTool, task: str, params: dict) -> str:
        """
        Execute synthesised tool in an isolated subprocess.
        - 30-second hard timeout
        - stdout captured as result
        - stderr logged but not surfaced to user
        - Clean temp file after execution
        """
        # Write tool code to temp file
        tool_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
            dir=str(self.TOOL_DIR), prefix=f"tool_{tool.tool_id}_")
        try:
            tool_file.write(tool.code)
            tool_file.flush()
            tool_path = tool_file.name
        finally:
            tool_file.close()

        # Write a runner script that imports the tool and calls execute()
        runner_code = f"""
import sys, json
sys.path.insert(0, {repr(str(self.TOOL_DIR))})
import importlib.util, traceback

spec   = importlib.util.spec_from_file_location("_tool", {repr(tool_path)})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
task   = {repr(task)}
params = {repr(params)}
try:
    result = module.execute(task, params)
    print(str(result))
except Exception as e:
    print(f"ERROR: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
        runner_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
            dir=str(self.TOOL_DIR), prefix="runner_")
        try:
            runner_file.write(runner_code)
            runner_file.flush()
            runner_path = runner_file.name
        finally:
            runner_file.close()

        try:
            proc = subprocess.run(  # nosec B603 — isolated sandbox subprocess, no shell, argv is [python, tempfile]
                [sys.executable, runner_path],
                capture_output=True, text=True,
                timeout=30,
            )
            if proc.returncode != 0:
                err = proc.stderr.strip() or "Unknown error"
                raise RuntimeError(f"Tool subprocess failed: {err}")
            return proc.stdout.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            raise RuntimeError("Tool timed out after 30 seconds") from None
        finally:
            for p in (tool_path, runner_path):
                try:
                    os.unlink(p)
                except Exception:
                    pass

    # ── Requirement installation ──────────────────────────────────────────────

    # Packages an LLM-synthesised tool may pull from PyPI. Anything outside
    # this set is refused — the LLM does not get to pick package names, since
    # a typo-squatted or attacker-named package is one `pip install` away from
    # arbitrary code execution at install time. Extend deliberately, not
    # reflexively. Keep names lowercase to match the comparison below.
    _ALLOWED_PYPI = frozenset({
        "requests", "httpx", "urllib3",
        "pyyaml", "tomli",
        "pillow",
        "beautifulsoup4", "lxml", "feedparser",
        "python-dateutil", "pytz",
        "markdown",
    })
    # Stdlib modules the LLM commonly lists in `requirements` even though
    # they are not pip-installable. Skipped silently.
    _STDLIB_NAMES = frozenset({
        "json", "re", "os", "sys", "time", "datetime", "pathlib", "urllib",
        "sqlite3", "hashlib", "uuid", "threading", "logging", "collections",
        "itertools", "functools", "typing", "dataclasses", "io", "math",
        "base64", "html", "random",
    })

    def _install_requirements(self, reqs: list[str]) -> bool:
        # Normalise + drop stdlib; reject anything not on the explicit allow-list.
        to_install: list[str] = []
        for raw in reqs:
            name = raw.strip().lower()
            if not name or name in self._STDLIB_NAMES:
                continue
            # Strip extras / version pins for the membership check.
            bare = re.split(r"[<>=!\[]", name, maxsplit=1)[0].strip()
            if bare not in self._ALLOWED_PYPI:
                logger.warning(
                    "[autonomous] refusing pip install %r: not on PyPI allow-list", raw,
                )
                return False
            to_install.append(name)
        if not to_install:
            return True
        try:
            subprocess.run(  # nosec B603 — pip install, no shell; packages restricted to _ALLOWED_PYPI
                [sys.executable, "-m", "pip", "install", "--quiet"] + to_install,
                check=True, timeout=60,
                capture_output=True)
            logger.info("[autonomous] Installed: %s", to_install)
            return True
        except subprocess.CalledProcessError as e:
            logger.warning("[autonomous] pip install failed: %s", e)
            return False
        except Exception as e:
            logger.warning("[autonomous] install error: %s", e)
            return False

    # ── Cache management ──────────────────────────────────────────────────────

    # Words that appear in many task descriptions but carry no topical meaning.
    # Without this filter, "hello world" matched world_cup_predictor on the
    # word "world", and a SHA256 hash request returned a FIFA prediction.
    _FUZZY_STOPWORDS = frozenset({
        "world", "hello", "thing", "stuff", "something", "anything",
        "please", "could", "would", "should", "needed", "value", "values",
        "result", "results", "input", "output", "thanks", "today",
        "calculate", "compute", "generate", "create", "make", "build",
        "find", "show", "list", "tell", "about",
    })

    def _find_cached_tool(self, task: str) -> Optional[AcquiredTool]:
        # Exact hash match first
        tid = hashlib.sha256(task.encode()).hexdigest()[:10]
        if tid in self._tools:
            return self._tools[tid]

        # Fuzzy: require at least two distinctive task keywords to appear in
        # the tool's name or description. Generic stopwords don't count.
        task_lower = task.lower()
        task_words = {
            w.strip(".,?!:;\"'()[]")
            for w in re.split(r"\s+", task_lower)
            if len(w) > 4
        }
        keywords = {w for w in task_words if w and w not in self._FUZZY_STOPWORDS}
        if len(keywords) < 2:
            return None

        best_tool: Optional[AcquiredTool] = None
        best_score = 0
        for tool in self._tools.values():
            haystack = f"{tool.name} {tool.description}".lower()
            score = sum(1 for kw in keywords if kw in haystack)
            if score >= 2 and score > best_score:
                best_tool = tool
                best_score = score
        return best_tool

    def _save_tool(self, tool: AcquiredTool) -> None:
        path = self.TOOL_DIR / f"{tool.tool_id}.json"
        data = {
            "tool_id":      tool.tool_id,
            "name":         tool.name,
            "description":  tool.description,
            "code":         tool.code,
            "requirements": tool.requirements,
            "entry_fn":     tool.entry_fn,
            "created_at":   tool.created_at,
            "use_count":    tool.use_count,
            "last_result":  tool.last_result,
        }
        path.write_text(json.dumps(data, indent=2))
        self._tools[tool.tool_id] = tool

    def _load_cached_tools(self) -> None:
        for path in self.TOOL_DIR.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                tool = AcquiredTool(**{
                    k: data[k] for k in AcquiredTool.__dataclass_fields__
                    if k in data
                })
                self._tools[tool.tool_id] = tool
            except Exception as e:
                logger.debug("Failed to load cached tool %s: %s", path.name, e)
        logger.info("[autonomous] Loaded %d cached tools", len(self._tools))

    def list_tools(self) -> list[AcquiredTool]:
        return sorted(self._tools.values(),
                       key=lambda t: t.use_count, reverse=True)

    # ── Notification ──────────────────────────────────────────────────────────

    def _notify(self, message: str, success: bool = True) -> None:
        if self._push and self._push.configured:
            priority = "default" if success else "high"
            tags     = ["white_check_mark"] if success else ["x"]
            self._push.send(
                title    = "PRISM — Task complete" if success else "PRISM — Task failed",
                body     = message,
                priority = priority,
                tags     = tags,
            )
        logger.info("[autonomous] %s: %s", "OK" if success else "FAIL", message[:80])
