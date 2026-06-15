"""
prism_state.py
==============
Shared state module for Prism ASGI.

All route files import _state from here to avoid circular imports with prism_asgi.py.
prism_daemon calls _set_state() after building PrismAgent to inject dependencies.

Thread safety
-------------
Single-key reads (``_state.get("agent")``) are atomic under CPython's GIL and
can be done directly. Writes and any read-modify-write sequence must go
through ``set_state()`` / ``update_state()`` (or ``with state_lock():``) so
that 11 background workers + ASGI request threads cannot race against each
other or the daemon's startup ``_set_state`` call.
"""
from __future__ import annotations

import threading
from dataclasses import asdict
from typing import Any

_state: dict[str, Any] = {}
state_lock = threading.RLock()


def _set_state(**kwargs: Any) -> None:
    """Called by prism_daemon after building PrismAgent to inject dependencies."""
    with state_lock:
        _state.update(kwargs)


def set_state(key: str, value: Any) -> None:
    """Atomic single-key write. Use from route handlers / background workers."""
    with state_lock:
        _state[key] = value


def update_state(**kwargs: Any) -> None:
    """Atomic multi-key update."""
    with state_lock:
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
