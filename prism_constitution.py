"""
prism_constitution.py
=====================
L1 Constitution — loaded once at startup, never hot-reloaded.

Enforces absolute limits that no L2 ORGAN_POLICY or user instruction can
override. The three-layer model:

  L1  constitution.yaml      — immutable at runtime (this module)
  L2  ORGAN_POLICY per organ — organ-level risk / approval / rate limits
  L3  BudContext per task    — scoped capabilities for one execution

Usage
-----
    from prism_constitution import ConstitutionGuard
    guard = ConstitutionGuard()               # loads constitution.yaml once
    ok, reason = guard.check("shell_run", ["subprocess"])
    required   = guard.required_capabilities("web_search")
    is_safe    = guard.may_synthesize("internet_read")
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CONSTITUTION_PATH = Path(__file__).parent / "constitution.yaml"


def _load_yaml(path: Path) -> dict:
    """Load a YAML file without requiring PyYAML — falls back to a tiny parser."""
    try:
        import yaml  # type: ignore
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        pass
    # Minimal YAML parser with look-ahead to distinguish list vs dict children.
    import re
    with open(path, encoding="utf-8") as f:
        sig_lines = [
            (len(raw) - len(raw.lstrip()), raw.rstrip().lstrip())
            for raw in f
            if raw.strip() and not raw.lstrip().startswith("#")
        ]

    def _coerce(s: str):
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            pass
        return s.strip('"').strip("'")

    data: dict = {}
    current_list: Optional[list] = None
    indent_stack: list[tuple[int, dict]] = [(0, data)]

    for idx, (indent, stripped) in enumerate(sig_lines):
        # List item
        if stripped.startswith("- "):
            if current_list is not None:
                current_list.append(_coerce(stripped[2:].strip()))
            continue

        m = re.match(r'^([\w][\w_-]*)\s*:\s*(.*)', stripped)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()

        # Pop stack back to current indent level
        while len(indent_stack) > 1 and indent <= indent_stack[-1][0]:
            indent_stack.pop()
        parent = indent_stack[-1][1]

        if val == "":
            # Look ahead to decide: list or nested dict?
            next_stripped = sig_lines[idx + 1][1] if idx + 1 < len(sig_lines) else ""
            if next_stripped.startswith("- "):
                lst: list = []
                parent[key] = lst
                current_list = lst
            else:
                nested: dict = {}
                parent[key] = nested
                indent_stack.append((indent, nested))
                current_list = None
        elif val.startswith("[") and val.endswith("]"):
            parent[key] = [_coerce(v.strip()) for v in val[1:-1].split(",") if v.strip()]
            current_list = None
        else:
            parent[key] = _coerce(val)
            current_list = None
    return data


class ConstitutionGuard:
    """
    Singleton-style guard loaded from constitution.yaml.

    All public methods are safe to call even if the YAML failed to load —
    they degrade gracefully to permissive defaults so PRISM still works.
    """

    def __init__(self, path: Optional[Path] = None):
        self._path = path or _CONSTITUTION_PATH
        self._data: dict = {}
        self._loaded = False
        self._load()

    def _load(self) -> None:
        try:
            self._data = _load_yaml(self._path)
            self._loaded = True
            logger.info("[constitution] Loaded from %s", self._path)
        except Exception as exc:
            logger.warning("[constitution] Could not load %s: %s — permissive mode", self._path, exc)

    # ── Public API ─────────────────────────────────────────────────────────────

    def check(self, intent: str, declared_capabilities: list[str]) -> tuple[bool, str]:
        """
        Return (True, "") if the organ is allowed to execute, or
        (False, reason) if the constitution blocks it.
        """
        required = self.required_capabilities(intent)
        if not required:
            return True, ""
        missing = [c for c in required if c not in declared_capabilities]
        if missing:
            return (
                False,
                f"Intent '{intent}' requires capabilities {required} but organ "
                f"only declares {declared_capabilities}. Missing: {missing}.",
            )
        return True, ""

    def required_capabilities(self, intent: str) -> list[str]:
        """Return the capabilities required by the constitution for this intent."""
        caps = self._data.get("capability_requirements", {})
        return caps.get(intent, [])

    def may_synthesize(self, capability: str) -> bool:
        """Return False if the constitution forbids synthesizing organs with this capability."""
        blocked = (
            self._data
            .get("absolute_limits", {})
            .get("never_synthesize_capabilities", [])
        )
        return capability not in blocked

    def may_synthesize_intent(self, intent: str) -> tuple[bool, str]:
        """Return (False, matched_pattern) when the intent name matches a
        forbidden pattern, otherwise (True, "").

        Complements ``may_synthesize(capability)``: that gate refuses
        synthesis when the intent's *declared* required capabilities are
        forbidden, but only catches intents the constitution already
        knows. This gate catches LLM-coined alias names (``run_shell``,
        ``spawn_process``, etc.) before any LLM call is made.
        """
        patterns = (
            self._data
            .get("absolute_limits", {})
            .get("never_synthesize_intent_patterns", [])
        )
        needle = (intent or "").lower()
        for pat in patterns:
            if not isinstance(pat, str) or not pat:
                continue
            if pat.lower() in needle:
                return False, pat
        return True, ""

    def max_synthesis_per_session(self) -> int:
        return int(
            self._data.get("absolute_limits", {})
            .get("max_synthesis_per_session", 10)
        )

    def max_organs_per_session(self) -> int:
        return int(
            self._data.get("absolute_limits", {})
            .get("max_organs_per_session", 200)
        )

    def is_never_log(self, intent: str) -> bool:
        never = (
            self._data.get("absolute_limits", {})
            .get("never_log_intents", [])
        )
        return intent in never

    def capability_risk(self, capability: str) -> str:
        caps = self._data.get("capabilities", {})
        return caps.get(capability, {}).get("risk", "low")

    @property
    def loaded(self) -> bool:
        return self._loaded


# Module-level singleton — import and use directly
_guard: Optional[ConstitutionGuard] = None


def get_guard() -> ConstitutionGuard:
    global _guard
    if _guard is None:
        _guard = ConstitutionGuard()
    return _guard
