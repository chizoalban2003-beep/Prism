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
from functools import lru_cache
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
    # Minimal YAML parser: handle the flat/list structure we use
    import re
    data: dict = {}
    current_key: Optional[str] = None
    current_list: Optional[list] = None
    list_key: Optional[str] = None
    indent_stack: list[tuple[int, dict]] = [(0, data)]

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip()
            if not line or line.lstrip().startswith("#"):
                continue
            stripped = line.lstrip()
            indent = len(line) - len(stripped)

            # List item
            if stripped.startswith("- "):
                val = stripped[2:].strip()
                if current_list is not None:
                    current_list.append(val)
                continue

            # Key: value or Key:
            m = re.match(r'^(\w[\w_]*)\s*:\s*(.*)', stripped)
            if not m:
                continue
            key, val = m.group(1), m.group(2).strip()

            # Pop stack to current indent level
            while len(indent_stack) > 1 and indent <= indent_stack[-1][0]:
                indent_stack.pop()
            parent = indent_stack[-1][1]

            if val == "":
                # Nested dict or upcoming list
                nested: dict = {}
                parent[key] = nested
                indent_stack.append((indent, nested))
                current_list = None
            elif val.startswith("[") and val.endswith("]"):
                inner = val[1:-1]
                parent[key] = [v.strip() for v in inner.split(",") if v.strip()]
                current_list = None
            else:
                # Try int/float, then string
                try:
                    parent[key] = int(val)
                except ValueError:
                    try:
                        parent[key] = float(val)
                    except ValueError:
                        parent[key] = val.strip('"').strip("'")
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

    def max_synthesis_per_session(self) -> int:
        return int(
            self._data.get("absolute_limits", {})
            .get("max_synthesis_per_session", 10)
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
