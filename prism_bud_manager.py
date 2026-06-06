"""
prism_bud_manager.py
====================
BudManager — ephemeral execution contexts for PRISM organ calls.

A Bud is a short-lived, capability-scoped wrapper around one organ execution.
It carries a unique token, receives only the ctx fields it needs, and is
decommissioned (token revoked, metrics logged) as soon as execution finishes.

Three-layer context model
-------------------------
  Full ctx (PrismAgent)           — all credentials, subsystem handles
  Scoped ctx (BudHandle)          — only capabilities the organ declared
  Organ execute(intent, msg, ctx) — sees scoped ctx only

Lifecycle
---------
  handle = mgr.spawn(intent, message, full_ctx, declared_capabilities)
  card   = mgr.execute(handle, organ_fn)   ← decommissions automatically
  # or manually: mgr.decommission(handle)

Usage
-----
    from prism_bud_manager import BudManager
    mgr = BudManager(constitution_guard=guard)
    handle = mgr.spawn("web_search", msg, full_ctx, ["internet_read"])
    card   = mgr.execute(handle, organ_fn)
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class BudStatus(str, Enum):
    PENDING      = "pending"
    RUNNING      = "running"
    COMPLETED    = "completed"
    FAILED       = "failed"
    DECOMMISSIONED = "decommissioned"


# ── Capability → ctx key mapping ──────────────────────────────────────────────
# Each capability grants access to specific ctx keys.  Organs only receive
# keys their declared capabilities grant.  The token field is always included.
_CAPABILITY_CTX_KEYS: dict[str, list[str]] = {
    "internet_read":    [],                         # no extra ctx needed
    "internet_write":   ["twilio_config"],
    "filesystem_read":  [],
    "filesystem_write": [],
    "subprocess":       ["shell_runner"],
    "telephony":        ["twilio_config", "contacts"],
    "system_ui":        [],
    "notifications":    [],
    "smart_home":       [],
    "spectrum_control": [],  # uses module singleton; no extra ctx keys needed
    # Always included regardless of capability
    "_always": [
        "organ_loader", "policy_engine", "router",
        "email", "calendar", "tasks", "contacts",
        "standing_instructions", "history", "persona_context",
        "memory_context", "perception", "perception_summary",
        "context_profile",
    ],
}


def _scoped_ctx(full_ctx: dict, capabilities: list[str]) -> dict:
    """Return a shallow copy of full_ctx with only keys the capabilities grant."""
    allowed_keys: set[str] = set(_CAPABILITY_CTX_KEYS.get("_always", []))
    for cap in capabilities:
        allowed_keys.update(_CAPABILITY_CTX_KEYS.get(cap, []))
    # Always pass through approval flags so the organ gate works
    approval_keys = {k for k in full_ctx if k.startswith("_approved_")}
    allowed_keys |= approval_keys
    return {k: v for k, v in full_ctx.items() if k in allowed_keys}


# ── BudHandle ─────────────────────────────────────────────────────────────────

@dataclass
class BudHandle:
    bud_id: str
    intent: str
    message: str
    capabilities: list[str]
    scoped_ctx: dict
    spawned_at: float = field(default_factory=time.monotonic)
    status: BudStatus = BudStatus.PENDING
    completed_at: Optional[float] = None
    error: Optional[str] = None

    @property
    def elapsed_ms(self) -> float:
        end = self.completed_at or time.monotonic()
        return (end - self.spawned_at) * 1000


# ── BudManager ────────────────────────────────────────────────────────────────

class BudManager:
    """
    Manages the spawn → execute → decommission lifecycle for organ Buds.

    Thread-safe: each Bud is independent, _active_buds is updated under
    the assumption of single-thread usage per agent instance (CPython GIL
    makes dict operations atomic for simple add/del).
    """

    def __init__(self, constitution_guard=None):
        self._guard = constitution_guard
        self._active_buds: dict[str, BudHandle] = {}
        self._session_count = 0  # total buds spawned this session
        self._session_synthesis = 0  # synthesis buds this session

    # ── Public API ─────────────────────────────────────────────────────────────

    def spawn(
        self,
        intent: str,
        message: str,
        full_ctx: dict,
        declared_capabilities: list[str],
    ) -> BudHandle:
        """
        Create a scoped execution context for one organ call.
        Returns a BudHandle; does NOT execute the organ yet.
        """
        bud_id = uuid.uuid4().hex[:12]
        scoped = _scoped_ctx(full_ctx, declared_capabilities)
        scoped["_bud_id"]           = bud_id
        scoped["_bud_capabilities"] = list(declared_capabilities)
        handle = BudHandle(
            bud_id       = bud_id,
            intent       = intent,
            message      = message,
            capabilities = declared_capabilities,
            scoped_ctx   = scoped,
        )
        self._active_buds[bud_id] = handle
        self._session_count += 1
        logger.debug(
            "[bud:%s] spawned  intent=%s  caps=%s",
            bud_id[:6], intent, declared_capabilities,
        )
        return handle

    def execute(self, handle: BudHandle, organ_fn: Callable) -> Any:
        """
        Run organ_fn inside the Bud's scoped context.
        Decommissions the Bud on completion (success or failure).
        """
        handle.status = BudStatus.RUNNING
        try:
            result = organ_fn(handle.intent, handle.message, handle.scoped_ctx)
            handle.status = BudStatus.COMPLETED
            return result
        except Exception as exc:
            handle.status = BudStatus.FAILED
            handle.error = str(exc)
            logger.warning("[bud:%s] failed: %s", handle.bud_id[:6], exc)
            raise
        finally:
            handle.completed_at = time.monotonic()
            self.decommission(handle)

    def decommission(self, handle: BudHandle) -> None:
        """Revoke the Bud's scoped token and remove from active set."""
        if handle.bud_id in self._active_buds:
            del self._active_buds[handle.bud_id]
        if handle.status not in (BudStatus.COMPLETED, BudStatus.FAILED):
            handle.status = BudStatus.DECOMMISSIONED
        # Clear the bud token from the scoped ctx
        handle.scoped_ctx.pop("_bud_id", None)
        handle.scoped_ctx.pop("_bud_capabilities", None)
        logger.debug(
            "[bud:%s] decommissioned  status=%s  elapsed=%.0fms",
            handle.bud_id[:6], handle.status, handle.elapsed_ms,
        )

    def active_count(self) -> int:
        return len(self._active_buds)

    def session_stats(self) -> dict:
        return {
            "total_spawned":        self._session_count,
            "currently_active":     self.active_count(),
            "synthesis_this_session": self._session_synthesis,
        }

    def record_synthesis(self) -> None:
        """Call when an organ is synthesized via a Bud."""
        self._session_synthesis += 1

    def synthesis_allowed(self) -> bool:
        """Check L1 session synthesis limit."""
        if self._guard is None:
            return True
        return self._session_synthesis < self._guard.max_synthesis_per_session()
