"""
prism_state.py
==============
Shared state module for Prism ASGI.

All route files import _state from here to avoid circular imports with prism_asgi.py.
prism_daemon calls _set_state() after building PrismAgent to inject dependencies.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

_state: dict[str, Any] = {}


def _set_state(**kwargs: Any) -> None:
    """Called by prism_daemon after building PrismAgent to inject dependencies."""
    _state.update(kwargs)


def _get_agent():
    """Return the current PrismAgent from shared state."""
    return _state.get("agent")


def _safe_dict(obj) -> dict:
    """Convert a dataclass (or any object) to a dict safely."""
    try:
        return asdict(obj)
    except TypeError:
        pass
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return {}
