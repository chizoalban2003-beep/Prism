from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

KNOWN_SIGNALS = [
    "risk_override_tendency",
    "time_pressure_response",
    "data_reliance",
    "consistency_score",
    "aggression_index",
]


@dataclass
class IdentitySignal:
    """Normalised pattern signal published by a sub-agent."""

    source: str
    signal_id: str
    value: float
    confidence: float
    timestamp: float


class IdentityBus:
    """Publish/subscribe signal bus backed by SQLite."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT,
        signal_id TEXT,
        value REAL,
        confidence REAL,
        timestamp REAL,
        consumed INTEGER DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_signal ON signals(signal_id, consumed);
    """

    def __init__(self, db_path: str = "~/.prism/bus.db"):
        requested_path = Path(db_path).expanduser()
        legacy_path = requested_path.with_name("identity_bus.db")
        self.db_path = (
            legacy_path
            if requested_path == Path("~/.prism/bus.db").expanduser()
            and not requested_path.exists()
            and legacy_path.exists()
            else requested_path
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._subs: dict[str, list[Callable[[IdentitySignal], None]]] = {}
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def publish(self, signal: IdentitySignal) -> None:
        stored = IdentitySignal(
            source=signal.source,
            signal_id=signal.signal_id,
            value=max(0.0, min(1.0, float(signal.value))),
            confidence=max(0.0, min(1.0, float(signal.confidence))),
            timestamp=float(signal.timestamp or time.time()),
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO signals(source,signal_id,value,confidence,timestamp) VALUES(?,?,?,?,?)",
                (
                    stored.source,
                    stored.signal_id,
                    stored.value,
                    stored.confidence,
                    stored.timestamp,
                ),
            )
        for callback in list(self._subs.get(stored.signal_id, [])):
            try:
                callback(stored)
            except Exception:
                logger.exception("IdentityBus subscriber failed for %s", stored.signal_id)

    def subscribe(self, signal_id: str, callback: Callable[[IdentitySignal], None]) -> None:
        self._subs.setdefault(signal_id, []).append(callback)

    def latest(self, signal_id: str, n: int = 5) -> list[IdentitySignal]:
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
        cutoff = time.time() - (max(1, int(window_days)) * 86400)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT value, timestamp FROM signals WHERE signal_id = ? AND timestamp > ?",
                (signal_id, cutoff),
            ).fetchall()
        if not rows:
            return 0.5
        now = time.time()
        decay = 7 * 86400
        weights = [2 ** ((float(row["timestamp"]) - now) / decay) for row in rows]
        total_weight = sum(weights)
        if total_weight <= 0:
            return 0.5
        score = sum(float(row["value"]) * weight for row, weight in zip(rows, weights)) / total_weight
        return max(0.0, min(1.0, score))

    def cross_domain_profile(self) -> dict[str, float]:
        return {signal_id: self.aggregate(signal_id) for signal_id in KNOWN_SIGNALS}
