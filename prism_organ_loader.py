"""
prism_organ_loader.py
=====================
Dynamic organ discovery, loading, and synthesis for PRISM.

Organs are thin Python modules — each handles one intent and returns a
PrismCard. They live in two directories:

  ./organs/           bundled organs shipped with PRISM (version-controlled)
  ~/.prism/organs/    synthesized organs, persisted across sessions

Organ interface
---------------
    ORGAN_META = {
        "intent":      "unique_intent_name",
        "description": "one-line description shown to the LLM router",
        "version":     "1.0",
    }

    # Optional — declares risk level so the policy node is self-extending.
    # Organs that omit this fall back to the legacy HIGH_RISK hardcoded set.
    ORGAN_POLICY = {
        "risk_level":        "low",      # "low" | "medium" | "high" | "critical"
        "requires_approval": False,      # True → policy node flags before repeat
        "irreversible":      False,      # True → extra warning in chain context
        "max_per_session":   None,       # int → hard cap per session; None = unlimited
    }

    def execute(intent: str, message: str, ctx: dict):
        from prism_responses import text_card
        ...
        return text_card(result_string, intent)

On load, OrganLoader validates each file with the same AST safety visitor
used in prism_autonomous. Unsafe files are logged and skipped; they never
reach exec().

On synthesize(), the LLM generates a complete organ file, the safety
visitor runs before any code executes, and the file is saved to
~/.prism/organs/ for reuse in future sessions.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional  # noqa: F401

logger = logging.getLogger(__name__)

BUNDLED_DIR = Path(__file__).parent / "organs"
USER_DIR    = Path("~/.prism/organs").expanduser()

# ── AST safety ───────────────────────────────────────────────────────────────

_BLOCKED_IMPORTS = {
    "os", "subprocess", "shutil", "socket", "ctypes",
    "multiprocessing", "importlib", "builtins", "pty",
}
_BLOCKED_CALLS = {"eval", "exec", "compile", "__import__", "breakpoint", "open"}
_BLOCKED_ATTRS = {
    "system", "popen", "remove", "unlink", "rmtree", "chmod", "chown",
    "rename", "symlink", "fork", "spawn", "execv", "execve", "kill",
    # "replace" intentionally omitted — str.replace() is safe and commonly used
    # Note: write_text/write_bytes are NOT blocked here because bundled organs
    # in ./organs/ are version-controlled and legitimately write user files.
    # Synthesized organs (prism_autonomous.py) have their own stricter checker.
}


class _SafetyVisitor(ast.NodeVisitor):
    def __init__(self):
        self.violations: list[str] = []

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name.split(".")[0] in _BLOCKED_IMPORTS:
                self.violations.append(f"blocked import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if (node.module or "").split(".")[0] in _BLOCKED_IMPORTS:
            self.violations.append(f"blocked from-import: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_CALLS:
            self.violations.append(f"blocked call: {node.func.id}()")
        elif isinstance(node.func, ast.Attribute) and node.func.attr in _BLOCKED_ATTRS:
            self.violations.append(f"blocked attr call: .{node.func.attr}()")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr in _BLOCKED_ATTRS:
            self.violations.append(f"blocked attribute: .{node.attr}")
        self.generic_visit(node)


def _is_safe(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"
    v = _SafetyVisitor()
    v.visit(tree)
    return (False, "; ".join(v.violations)) if v.violations else (True, "")


# ── Synthesis prompt ──────────────────────────────────────────────────────────

_SYNTHESIS_PROMPT = """\
You are writing a PRISM organ module. An organ handles exactly one intent.

Intent to implement: "{intent}"
Example user request: "{message}"

Return ONLY valid JSON with this shape:
{{
  "intent":      "{intent}",
  "description": "one sentence what this organ does",
  "code": "complete Python source as a single string"
}}

The code string MUST contain exactly this structure:

ORGAN_META = {{
    "intent":      "{intent}",
    "description": "...",
    "version":     "1.0",
}}

def execute(intent: str, message: str, ctx: dict):
    from prism_responses import text_card
    # implementation
    return text_card(result_string, intent)

Constraints:
- Allowed: json, re, datetime, pathlib, urllib.request, urllib.parse,
  urllib.error, base64, hashlib, math, random, time, collections, html
- FORBIDDEN: os, subprocess, shutil, socket, ctypes, eval, exec, open(write)
- Never raise — catch all exceptions and return text_card with error message
- API keys: read from ctx.get("{intent}_key", "") or ctx.get("api_key", "")
- Keep code under 80 lines
"""


# ── OrganLoader ───────────────────────────────────────────────────────────────


class OrganLoader:
    """
    Discovers, loads, synthesizes, and registers PRISM logic organs.

    Usage
    -----
        loader = OrganLoader(llm_router=router)
        fn = loader.get("weather_check")
        if fn:
            card = fn("weather_check", "London weather", {})

        # Synthesize and register a new organ for an unknown intent:
        success = loader.synthesize("stock_price", "what is AAPL stock price?")
    """

    def __init__(
        self,
        bundled_dir: Optional[Path] = None,
        user_dir: Optional[Path]    = None,
        llm_router: Any             = None,
    ):
        self._bundled = Path(bundled_dir) if bundled_dir else BUNDLED_DIR
        self._user    = Path(user_dir).expanduser() if user_dir else USER_DIR
        self._router  = llm_router
        # {intent: (execute_fn, organ_meta_dict)}
        self._organs:  dict[str, tuple[Callable, dict]] = {}
        self._disabled: set[str] = set()
        self._organ_sources: dict[str, str] = {}  # intent → "bundled" | "user"
        self._user.mkdir(parents=True, exist_ok=True)
        self._load_all()

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, intent: str) -> Optional[Callable]:
        """Return the execute function for intent, or None if not loaded."""
        if intent in self._disabled:
            return None
        entry = self._organs.get(intent)
        return entry[0] if entry else None

    def get_organ_policy(self, intent: str) -> dict:
        """Return the ORGAN_POLICY dict for intent, or {} if not declared."""
        entry = self._organs.get(intent)
        if entry is None:
            return {}
        fn = entry[0]
        return getattr(fn, "_organ_policy", {})

    def get_organ_capabilities(self, intent: str) -> list:
        """Return the capabilities list declared in ORGAN_META, or [] if absent."""
        entry = self._organs.get(intent)
        if entry is None:
            return []
        fn = entry[0]
        return list(getattr(fn, "_organ_meta", {}).get("capabilities", []))

    def known_intents(self) -> dict[str, str]:
        """Return {intent: description} for every loaded organ."""
        return {k: v[1].get("description", k) for k, v in self._organs.items()}

    def list_organs(self) -> list[str]:
        """Return sorted list of loaded organ intent names."""
        return sorted(self._organs.keys())

    def enable(self, intent: str) -> bool:
        """Enable a previously disabled organ. Returns True if it was disabled."""
        if intent in self._disabled:
            self._disabled.discard(intent)
            return True
        return False

    def disable(self, intent: str) -> bool:
        """Disable an organ without unloading it. Returns True if found and disabled."""
        if intent in self._organs:
            self._disabled.add(intent)
            return True
        return False

    def is_enabled(self, intent: str) -> bool:
        return intent in self._organs and intent not in self._disabled

    def organ_details(self, intent: str) -> dict | None:
        """Return full metadata dict for intent, or None if not found."""
        entry = self._organs.get(intent)
        if entry is None:
            return None
        fn, meta = entry
        policy = getattr(fn, "_organ_policy", {})
        return {
            "intent":      intent,
            "description": meta.get("description", intent),
            "version":     meta.get("version", "1.0"),
            "source":      self._organ_sources.get(intent, "bundled"),
            "enabled":     intent not in self._disabled,
            "risk_level":  policy.get("risk_level", "unknown"),
            "requires_approval": policy.get("requires_approval", False),
            "irreversible":      policy.get("irreversible", False),
            "max_per_session":   policy.get("max_per_session", None),
            "capabilities": meta.get("capabilities", []),
        }

    def list_organ_details(self) -> list[dict]:
        """Return organ_details for all known intents, sorted by intent name."""
        return [self.organ_details(i) for i in sorted(self._organs.keys())]

    def reload(self) -> int:
        """Re-scan bundled and user dirs. Returns number of organs now loaded."""
        self._organs.clear()
        self._organ_sources.clear()
        # Keep disabled set intact so user's choices persist across reload
        self._load_all()
        return len(self._organs)

    def delete_user_organ(self, intent: str) -> bool:
        """
        Delete a user-synthesized organ from disk and unregister it.
        Bundled organs cannot be deleted (returns False).
        Returns True on success.
        """
        if self._organ_sources.get(intent) != "user":
            return False
        path = self._user / f"{intent}.py"
        path.unlink(missing_ok=True)
        self._organs.pop(intent, None)
        self._organ_sources.pop(intent, None)
        self._disabled.discard(intent)
        # Also remove from LOGIC_REGISTRY if present
        try:
            from prism_composer import LOGIC_REGISTRY
            LOGIC_REGISTRY.pop(intent, None)
        except ImportError:
            pass
        return True

    def execute_parallel(
        self,
        intents: list[str],
        message: str,
        ctx: dict,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """
        Run multiple organs concurrently. Only organs whose ORGAN_POLICY marks
        irreversible=False and requires_approval=False are eligible; others are
        silently skipped (call execute() for those individually).

        Returns {intent: card_or_error_dict}.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from concurrent.futures import TimeoutError as _FTimeout

        safe = [
            i for i in intents
            if not self.get_organ_policy(i).get("irreversible", False)
            and not self.get_organ_policy(i).get("requires_approval", False)
            and self.get(i) is not None
        ]
        if not safe:
            return {}

        results: dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=min(len(safe), 4)) as pool:
            futures = {pool.submit(self.get(i), i, message, ctx): i for i in safe}
            try:
                for future in as_completed(futures, timeout=timeout):
                    intent = futures[future]
                    try:
                        results[intent] = future.result(timeout=5.0)
                    except _FTimeout:
                        results[intent] = {"error": "timeout", "output": ""}
                    except Exception as exc:
                        results[intent] = {"error": str(exc), "output": ""}
            except _FTimeout:
                for future, intent in futures.items():
                    if intent not in results:
                        results[intent] = {"error": "overall timeout", "output": ""}
        return results

    def synthesize(self, intent: str, message: str) -> bool:
        """
        Ask the LLM to write a new organ for this intent, safety-check it,
        save to ~/.prism/organs/<intent>.py, and register it immediately.

        Returns True if the organ was successfully synthesized and registered.
        """
        if not self._router:
            logger.warning("[organ_loader] No router — cannot synthesize %s", intent)
            return False

        # L1 constitution: block synthesis of forbidden capabilities
        try:
            from prism_constitution import get_guard
            guard = get_guard()
            required = guard.required_capabilities(intent)
            blocked = [c for c in required if not guard.may_synthesize(c)]
            if blocked:
                logger.warning(
                    "[organ_loader] Synthesis blocked for %s — capabilities %s "
                    "are forbidden by constitution", intent, blocked)
                return False
        except Exception:
            pass

        prompt = _SYNTHESIS_PROMPT.format(intent=intent, message=message[:300])
        try:
            raw, _ = self._router.call(
                prompt, min_capability=2, max_tokens=1400, json_mode=True)
        except Exception as exc:
            logger.warning("[organ_loader] LLM call failed for %s: %s", intent, exc)
            return False

        data = self._parse_json(raw)
        if data is None:
            return False

        code = data.get("code", "")
        if not code or "def execute" not in code or "ORGAN_META" not in code:
            logger.warning("[organ_loader] Synthesized code missing interface for %s", intent)
            return False

        safe, reason = _is_safe(code)
        if not safe:
            logger.warning("[organ_loader] Unsafe organ blocked (%s): %s", intent, reason)
            return False

        out_path = self._user / f"{intent}.py"
        out_path.write_text(code)
        logger.info("[organ_loader] Synthesized organ saved: %s", out_path)

        fn = self._load_file(out_path)
        if fn is None:
            out_path.unlink(missing_ok=True)
            return False

        meta = {
            "intent":      intent,
            "description": data.get("description", intent),
            "version":     "1.0",
        }
        self._register(intent, fn, meta, source="user")
        return True

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_all(self):
        for directory, source in ((self._bundled, "bundled"), (self._user, "user")):
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.py")):
                if path.name.startswith("_"):
                    continue
                fn = self._load_file(path)
                if fn is None:
                    continue
                meta   = getattr(fn, "_organ_meta", {})
                intent = meta.get("intent") or path.stem
                self._register(intent, fn, meta, source=source)

    def _load_file(self, path: Path) -> Optional[Callable]:
        code = path.read_text()
        safe, reason = _is_safe(code)
        if not safe:
            logger.warning("[organ_loader] Skipping unsafe organ %s: %s",
                           path.name, reason)
            return None
        try:
            spec   = importlib.util.spec_from_file_location(
                f"_organ_{path.stem}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as exc:
            logger.warning("[organ_loader] Failed to load %s: %s", path.name, exc)
            return None
        fn = getattr(module, "execute", None)
        if not callable(fn):
            logger.warning("[organ_loader] No execute() in %s", path.name)
            return None
        fn._organ_meta   = getattr(module, "ORGAN_META", {})    # type: ignore[attr-defined]
        fn._organ_policy = getattr(module, "ORGAN_POLICY", {})  # type: ignore[attr-defined]
        return fn

    def _register(self, intent: str, fn: Callable, meta: dict, source: str = "bundled"):
        self._organs[intent] = (fn, meta)
        self._organ_sources[intent] = source
        # Dynamically extend LOGIC_REGISTRY so the chain router sees the new organ
        try:
            from prism_composer import LOGIC_REGISTRY
            if intent not in LOGIC_REGISTRY:
                LOGIC_REGISTRY[intent] = meta.get("description", intent)
                logger.debug("[organ_loader] Added %s to LOGIC_REGISTRY", intent)
        except ImportError:
            pass

    @staticmethod
    def _parse_json(raw: str) -> Optional[dict]:
        try:
            clean = (
                raw.strip()
                   .lstrip("```json")
                   .lstrip("```")
                   .rstrip("```")
                   .strip()
            )
            return json.loads(clean)
        except Exception:
            return None
