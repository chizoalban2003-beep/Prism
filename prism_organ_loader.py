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
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

BUNDLED_DIR = Path(__file__).parent / "organs"
USER_DIR    = Path("~/.prism/organs").expanduser()

# ── AST safety ───────────────────────────────────────────────────────────────

_BLOCKED_IMPORTS = {
    "os", "subprocess", "shutil", "socket", "ctypes",
    "multiprocessing", "importlib", "builtins", "pty",
}
_BLOCKED_CALLS = {"eval", "exec", "compile", "__import__", "breakpoint"}
_BLOCKED_ATTRS = {
    "system", "popen", "remove", "unlink", "rmtree", "chmod", "chown",
    "rename", "replace", "symlink", "fork", "spawn", "execv", "execve", "kill",
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
        self._user.mkdir(parents=True, exist_ok=True)
        self._load_all()

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, intent: str) -> Optional[Callable]:
        """Return the execute function for intent, or None if not loaded."""
        entry = self._organs.get(intent)
        return entry[0] if entry else None

    def get_organ_policy(self, intent: str) -> dict:
        """Return the ORGAN_POLICY dict for intent, or {} if not declared."""
        entry = self._organs.get(intent)
        if entry is None:
            return {}
        fn = entry[0]
        return getattr(fn, "_organ_policy", {})

    def known_intents(self) -> dict[str, str]:
        """Return {intent: description} for every loaded organ."""
        return {k: v[1].get("description", k) for k, v in self._organs.items()}

    def list_organs(self) -> list[str]:
        """Return sorted list of loaded organ intent names."""
        return sorted(self._organs.keys())

    def synthesize(self, intent: str, message: str) -> bool:
        """
        Ask the LLM to write a new organ for this intent, safety-check it,
        save to ~/.prism/organs/<intent>.py, and register it immediately.

        Returns True if the organ was successfully synthesized and registered.
        """
        if not self._router:
            logger.warning("[organ_loader] No router — cannot synthesize %s", intent)
            return False

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
        self._register(intent, fn, meta)
        return True

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_all(self):
        for directory in (self._bundled, self._user):
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
                self._register(intent, fn, meta)

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

    def _register(self, intent: str, fn: Callable, meta: dict):
        self._organs[intent] = (fn, meta)
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
