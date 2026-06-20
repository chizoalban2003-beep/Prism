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

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path.home() / ".prism" / "bud_executions.db"
_SCHEMA_VERSION = 1


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
    # Email/calendar/contacts handles + the LLM router are only needed by
    # internet_write organs (email_send, calendar_write); they are NOT in
    # _always, so a read-only or unrelated organ never sees your mailbox.
    "internet_write":   ["twilio_config", "email", "calendar", "contacts"],
    "filesystem_read":  [],
    "filesystem_write": [],
    "subprocess":       ["shell_runner"],
    "telephony":        ["twilio_config", "contacts"],
    "system_ui":        [],
    "notifications":    [],
    "smart_home":       [],
    "spectrum_control": [],  # uses module singleton; no extra ctx keys needed
    # Always included regardless of capability. Deliberately minimal
    # (least privilege): only infra handles that capability-less organs
    # legitimately need (policy_inspect→organ_loader, policy_update→
    # policy_engine, task_reminder→tasks, veax_control→router,
    # canary_check→memory_graph). The user's conversation history, persona,
    # perception, and recalled memories are NOT exposed to organs — no organ
    # reads them, and they are sensitive.
    "_always": [
        "organ_loader", "policy_engine", "router",
        "tasks", "memory_graph",
    ],
}


def _scoped_ctx(full_ctx: dict, capabilities: list[str]) -> dict:
    """Return a shallow copy of full_ctx with only keys the capabilities grant."""
    allowed_keys: set[str] = set(_CAPABILITY_CTX_KEYS.get("_always", []))
    for cap in capabilities:
        allowed_keys.update(_CAPABILITY_CTX_KEYS.get(cap, []))
    # Always pass through approval flags so the organ gate works, plus
    # explicit per-call tool arguments (benign data, not a credential/handle).
    approval_keys = {k for k in full_ctx if k.startswith("_approved_")}
    allowed_keys |= approval_keys
    allowed_keys.add("mcp_arguments")
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

    def __init__(
        self,
        constitution_guard=None,
        db_path: Optional[Path] = None,
    ):
        self._guard = constitution_guard
        self._active_buds: dict[str, BudHandle] = {}
        self._session_count = 0
        self._session_synthesis = 0
        self._session_intent_counts: dict[str, int] = {}
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

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
        self._session_intent_counts[intent] = (
            self._session_intent_counts.get(intent, 0) + 1
        )
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
        """Revoke the Bud's scoped token, remove from active set, and persist log."""
        if handle.bud_id in self._active_buds:
            del self._active_buds[handle.bud_id]
        if handle.status not in (BudStatus.COMPLETED, BudStatus.FAILED):
            handle.status = BudStatus.DECOMMISSIONED
        handle.scoped_ctx.pop("_bud_id", None)
        handle.scoped_ctx.pop("_bud_capabilities", None)
        logger.debug(
            "[bud:%s] decommissioned  status=%s  elapsed=%.0fms",
            handle.bud_id[:6], handle.status, handle.elapsed_ms,
        )
        self._persist(handle)

    def active_count(self) -> int:
        return len(self._active_buds)

    def session_intent_count(self, intent: str) -> int:
        """Number of times this intent has been spawned in the current session."""
        return self._session_intent_counts.get(intent, 0)

    def session_organ_total(self) -> int:
        """Total organ Buds spawned in the current session (all intents)."""
        return self._session_count

    def organ_budget_exceeded(self) -> bool:
        """True when the L1 max_organs_per_session ceiling has been reached."""
        if self._guard is None:
            return False
        try:
            return self._session_count >= self._guard.max_organs_per_session()
        except Exception:
            return False

    def session_stats(self) -> dict:
        total_all_time = self._session_count
        try:
            if self._conn:
                row = self._conn.execute("SELECT COUNT(*) FROM bud_executions").fetchone()
                total_all_time = row[0] if row else self._session_count
        except Exception:
            pass
        return {
            "total_spawned":          self._session_count,
            "currently_active":       self.active_count(),
            "synthesis_this_session": self._session_synthesis,
            "total_all_time":         total_all_time,
        }

    def record_synthesis(self) -> None:
        """Call when an organ is synthesized via a Bud."""
        self._session_synthesis += 1

    def synthesis_allowed(self) -> bool:
        """Check L1 session synthesis limit."""
        if self._guard is None:
            return True
        return self._session_synthesis < self._guard.max_synthesis_per_session()

    def execution_history(
        self,
        intent: Optional[str] = None,
        limit: int = 100,
        days: float = 7.0,
    ) -> list[dict]:
        """
        Return recent bud execution records from the persistent log.

        Parameters
        ----------
        intent : filter to a specific intent; None returns all intents
        limit  : max rows returned
        days   : only return executions within this many days
        """
        if self._conn is None:
            return []
        try:
            since = time.time() - days * 86400
            if intent:
                rows = self._conn.execute(
                    "SELECT bud_id, intent, status, duration_ms, error, "
                    "capabilities, timestamp FROM bud_executions "
                    "WHERE intent = ? AND timestamp >= ? "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (intent, since, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT bud_id, intent, status, duration_ms, error, "
                    "capabilities, timestamp FROM bud_executions "
                    "WHERE timestamp >= ? "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (since, limit),
                ).fetchall()
            return [
                {
                    "bud_id":       r[0],
                    "intent":       r[1],
                    "status":       r[2],
                    "duration_ms":  r[3],
                    "error":        r[4],
                    "capabilities": json.loads(r[5]) if r[5] else [],
                    "timestamp":    r[6],
                }
                for r in rows
            ]
        except Exception as exc:
            logger.debug("[BudManager] execution_history query failed: %s", exc)
            return []

    def intent_stats(self, intent: str) -> dict:
        """
        Aggregate stats for a specific organ intent from the persistent log.

        Returns
        -------
        dict with keys: intent, total, success_rate, avg_duration_ms, last_seen
        """
        if self._conn is None:
            return {"intent": intent, "total": 0, "success_rate": 0.0,
                    "avg_duration_ms": 0.0, "last_seen": None}
        try:
            row = self._conn.execute(
                "SELECT COUNT(*), "
                "SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END), "
                "AVG(duration_ms), MAX(timestamp) "
                "FROM bud_executions WHERE intent = ?",
                (intent,),
            ).fetchone()
            total      = row[0] or 0
            successes  = row[1] or 0
            avg_dur    = row[2] or 0.0
            last_seen  = row[3]
            return {
                "intent":          intent,
                "total":           total,
                "success_rate":    round(successes / max(total, 1), 4),
                "avg_duration_ms": round(avg_dur, 2),
                "last_seen":       last_seen,
            }
        except Exception as exc:
            logger.debug("[BudManager] intent_stats query failed: %s", exc)
            return {"intent": intent, "total": 0, "success_rate": 0.0,
                    "avg_duration_ms": 0.0, "last_seen": None}

    # ── Persistence ───────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                timeout=5,
            )
            conn.execute("PRAGMA journal_mode=WAL")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version < _SCHEMA_VERSION:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS bud_executions (
                        bud_id       TEXT    NOT NULL,
                        intent       TEXT    NOT NULL,
                        status       TEXT    NOT NULL,
                        duration_ms  REAL    NOT NULL,
                        error        TEXT,
                        capabilities TEXT,
                        timestamp    REAL    NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_be_intent
                        ON bud_executions(intent);
                    CREATE INDEX IF NOT EXISTS idx_be_ts
                        ON bud_executions(timestamp);
                """)
                conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                conn.commit()
            self._conn = conn
        except Exception as exc:
            logger.warning("[BudManager] DB init failed, running without persistence: %s", exc)
            self._conn = None

    def _persist(self, handle: BudHandle) -> None:
        if self._conn is None:
            return
        try:
            self._conn.execute(
                "INSERT INTO bud_executions "
                "(bud_id, intent, status, duration_ms, error, capabilities, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    handle.bud_id,
                    handle.intent,
                    handle.status.value,
                    handle.elapsed_ms,
                    handle.error,
                    json.dumps(handle.capabilities),
                    time.time(),
                ),
            )
            self._conn.commit()
        except Exception as exc:
            logger.debug("[BudManager] persist failed: %s", exc)
