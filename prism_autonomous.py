from __future__ import annotations
import hashlib
import importlib.util
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


# ── Safety patterns — never execute code containing these ─────────────────────
_BLOCKED_PATTERNS = [
    r"os\.system\s*\(",
    r"subprocess\.call\s*\(\s*['\"]",   # shell=True style
    r"__import__\s*\(\s*['\"]os['\"]",
    r"shutil\.rmtree",
    r"os\.remove\s*\(",
    r"open\s*\([^,]+,\s*['\"]w",        # file write without explicit path var
    r"eval\s*\(",
    r"exec\s*\(",
    r"socket\.connect",
    r"\.chmod\s*\(",
]

def _is_safe_code(code: str) -> tuple[bool, str]:
    """Return (safe, reason). Blocks dangerous patterns."""
    for pat in _BLOCKED_PATTERNS:
        if re.search(pat, code):
            return False, f"Blocked pattern: {pat}"
    return True, ""


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
        # Write code to temp file and import it
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
            dir=str(self.TOOL_DIR), prefix=f"tool_{tool.tool_id}_")
        try:
            tmp.write(tool.code)
            tmp.flush()
            tmp_path = tmp.name
        finally:
            tmp.close()

        try:
            spec   = importlib.util.spec_from_file_location(
                f"prism_tool_{tool.tool_id}", tmp_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            fn     = getattr(module, tool.entry_fn)
            result = fn(task, params)
            return str(result)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    # ── Requirement installation ──────────────────────────────────────────────

    def _install_requirements(self, reqs: list[str]) -> bool:
        stdlib = {"json","re","os","sys","time","datetime","pathlib",
                  "urllib","sqlite3","hashlib","uuid","threading","logging",
                  "collections","itertools","functools","typing","dataclasses"}
        to_install = [r for r in reqs if r.lower() not in stdlib]
        if not to_install:
            return True
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet"] + to_install,
                check=True, timeout=60,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info("[autonomous] Installed: %s", to_install)
            return True
        except subprocess.CalledProcessError as e:
            logger.warning("[autonomous] pip install failed: %s", e)
            return False
        except Exception as e:
            logger.warning("[autonomous] install error: %s", e)
            return False

    # ── Cache management ──────────────────────────────────────────────────────

    def _find_cached_tool(self, task: str) -> Optional[AcquiredTool]:
        # Exact hash match first
        tid = hashlib.sha256(task.encode()).hexdigest()[:10]
        if tid in self._tools:
            return self._tools[tid]
        # Fuzzy: check description similarity
        task_lower = task.lower()
        for tool in self._tools.values():
            if any(w in tool.description.lower()
                   for w in task_lower.split() if len(w) > 4):
                return tool
        return None

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
                logger.debug("Failed to load cached tool %s: %s", path, e)
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
