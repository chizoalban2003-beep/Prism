from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class IdentitySignal:
    """Normalised pattern signal published by a sub-agent."""

    source: str
    signal_id: str
    value: float
    confidence: float
    timestamp: float


class IdentityBus:
    """
    Publish/subscribe signal bus between sub-agents.
    SQLite-backed. Sub-agents publish signals; subscribers receive them.
    No raw data crosses the bus — only normalised pattern signals (0-1 floats).
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT, signal_id TEXT, value REAL, confidence REAL,
        timestamp REAL, consumed INTEGER DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_signal ON signals(signal_id, consumed);
    """

    def __init__(self, db_path: str = "~/.prism/identity_bus.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._subscribers: dict[str, list[Callable[[IdentitySignal], None]]] = {}
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(self.SCHEMA)

    def publish(self, signal: IdentitySignal) -> None:
        """
        Store signal in DB. Notify any registered in-memory subscribers.
        Thread-safe (each call opens its own connection).
        """
        value = max(0.0, min(1.0, float(signal.value)))
        confidence = max(0.0, min(1.0, float(signal.confidence)))
        timestamp = float(signal.timestamp or time.time())
        stored = IdentitySignal(
            source=signal.source,
            signal_id=signal.signal_id,
            value=value,
            confidence=confidence,
            timestamp=timestamp,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO signals (source, signal_id, value, confidence, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (stored.source, stored.signal_id, stored.value, stored.confidence, stored.timestamp),
            )
        for callback in list(self._subscribers.get(stored.signal_id, [])):
            try:
                callback(stored)
            except Exception:
                logger.exception("IdentityBus subscriber failed for %s", stored.signal_id)

    def subscribe(self, signal_id: str, callback: Callable[[IdentitySignal], None]) -> None:
        """Register an in-memory subscriber for a signal_id."""
        self._subscribers.setdefault(signal_id, []).append(callback)

    def latest(self, signal_id: str, n: int = 5) -> list[IdentitySignal]:
        """Return the n most recent signals for a given signal_id."""
        limit = max(1, int(n))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT source, signal_id, value, confidence, timestamp
                FROM signals
                WHERE signal_id = ?
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                (signal_id, limit),
            ).fetchall()
        return [
            IdentitySignal(
                source=row["source"],
                signal_id=row["signal_id"],
                value=float(row["value"]),
                confidence=float(row["confidence"]),
                timestamp=float(row["timestamp"]),
            )
            for row in rows
        ]

    def aggregate(self, signal_id: str, window_days: int = 30) -> float:
        """
        Weighted average of signal values over the window.
        Recent signals weighted more heavily (exponential decay, half-life=7 days).
        Returns 0.5 if no signals exist.
        """
        now = time.time()
        cutoff = now - (max(1, int(window_days)) * 86400)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT value, confidence, timestamp
                FROM signals
                WHERE signal_id = ? AND timestamp >= ?
                ORDER BY timestamp DESC, id DESC
                """,
                (signal_id, cutoff),
            ).fetchall()
        if not rows:
            return 0.5
        weighted_sum = 0.0
        total_weight = 0.0
        for row in rows:
            age_days = max(0.0, (now - float(row["timestamp"])) / 86400.0)
            recency_weight = 0.5 ** (age_days / 7.0)
            confidence_weight = max(0.05, float(row["confidence"]))
            weight = recency_weight * confidence_weight
            weighted_sum += float(row["value"]) * weight
            total_weight += weight
        if total_weight <= 0:
            return 0.5
        return max(0.0, min(1.0, weighted_sum / total_weight))

    def cross_domain_profile(self) -> dict[str, float]:
        """
        Returns latest aggregated value for each known signal_id.
        Used by DigitalIdentity to build the cross-domain vector.
        """
        known = [
            "risk_override_tendency",
            "time_pressure_response",
            "data_reliance",
            "consistency_score",
            "aggression_index",
        ]
        return {sid: self.aggregate(sid) for sid in known}
