"""
PRISM Three-Layered Observability Metrics

Layer 1 — Operational (muted/informational):
    wal_replays, pipeline_restarts, dedup_events, commits_total

Layer 2 — Warning (proactive alerts):
    reconciliation_latency (Lr) — time from buffer write to cold commit
    Alert fires when rolling 5-min mean Lr > 60 s

Layer 3 — Critical (emergency):
    drift_magnitude (Dm) — buffer_seq_id minus graph_seq_id
    Alert fires when Dm is growing AND Lr is already over threshold

All metrics are persisted to a SQLite table so they survive restarts.
The module is a singleton; import and call anywhere in the codebase.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_DEFAULT_PATH = Path.home() / ".prism" / "metrics.db"

# Layer 2 threshold: warn when mean reconciliation latency exceeds this
LR_WARN_THRESHOLD_S = 60.0

# Layer 3 threshold: critical when Dm grows by more than this per check
DM_GROWTH_THRESHOLD = 10


class PrismMetrics:
    """
    Thread-safe metrics store.  All increments are fire-and-forget.

    Usage::

        from prism_metrics import metrics   # global singleton
        metrics.inc("wal_replays")
        metrics.record_latency(2.4)
        report = metrics.report()
    """

    def __init__(self, db_path: Path | str = _DEFAULT_PATH) -> None:
        self._db = Path(db_path)
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()
        self._setup()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _setup(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS counters (
                name  TEXT PRIMARY KEY,
                value INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS latency_log (
                ts    REAL NOT NULL,
                value REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_lat_ts ON latency_log(ts);
            CREATE TABLE IF NOT EXISTS dm_log (
                ts    REAL NOT NULL,
                value INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_dm_ts ON dm_log(ts);
            CREATE TABLE IF NOT EXISTS canary_log (
                ts            REAL NOT NULL,
                duration_ms   REAL NOT NULL,
                success       INTEGER NOT NULL DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS ix_can_ts ON canary_log(ts);
        """)
        self._conn.commit()

    # ── Layer 1: counters ─────────────────────────────────────────────────────

    def inc(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO counters(name, value) VALUES(?,?)"
                " ON CONFLICT(name) DO UPDATE SET value = value + ?",
                (name, amount, amount),
            )
            self._conn.commit()

    def get(self, name: str) -> int:
        row = self._conn.execute(
            "SELECT value FROM counters WHERE name=?", (name,)
        ).fetchone()
        return row[0] if row else 0

    def all_counters(self) -> dict[str, int]:
        rows = self._conn.execute("SELECT name, value FROM counters").fetchall()
        return {r[0]: r[1] for r in rows}

    # ── Layer 2: reconciliation latency (Lr) ──────────────────────────────────

    def record_latency(self, seconds: float) -> None:
        """Record one reconciliation cycle's latency."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO latency_log(ts, value) VALUES(?,?)",
                (time.time(), seconds),
            )
            self._conn.commit()

    def mean_latency(self, window_s: float = 300.0) -> float | None:
        """Rolling mean Lr over the last window_s seconds. None if no data."""
        cutoff = time.time() - window_s
        rows = self._conn.execute(
            "SELECT value FROM latency_log WHERE ts > ?", (cutoff,)
        ).fetchall()
        if not rows:
            return None
        return sum(r[0] for r in rows) / len(rows)

    def lr_alert(self, window_s: float = 300.0) -> bool:
        """True if rolling mean Lr exceeds the warning threshold."""
        lr = self.mean_latency(window_s)
        return lr is not None and lr > LR_WARN_THRESHOLD_S

    # ── Layer 3: drift magnitude (Dm) ─────────────────────────────────────────

    def record_dm(self, pending: int) -> None:
        """Record current Dm (pending WAL entries = Ψ)."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO dm_log(ts, value) VALUES(?,?)",
                (time.time(), pending),
            )
            self._conn.commit()

    def dm_trend(self, last_n: int = 5) -> float:
        """
        Linear trend of the last N Dm samples.
        Positive = Dm growing (bad). Negative = shrinking (healthy).
        Returns 0.0 if fewer than 2 samples.
        """
        rows = self._conn.execute(
            "SELECT value FROM dm_log ORDER BY ts DESC LIMIT ?", (last_n,)
        ).fetchall()
        vals = [r[0] for r in rows]
        if len(vals) < 2:
            return 0.0
        # Simple slope: last minus first over count
        return (vals[0] - vals[-1]) / max(len(vals) - 1, 1)

    def critical_alert(self, window_s: float = 300.0) -> bool:
        """
        True when Dm is growing AND Lr is already over threshold —
        meaning self-healing mechanisms have failed.
        """
        return self.dm_trend() >= DM_GROWTH_THRESHOLD and self.lr_alert(window_s)

    # ── Canary performance tracking ───────────────────────────────────────────

    def record_canary(self, duration_ms: float, success: bool = True) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO canary_log(ts, duration_ms, success) VALUES(?,?,?)",
                (time.time(), duration_ms, int(success)),
            )
            self._conn.commit()

    def canary_stats(self, last_n: int = 30) -> dict[str, Any]:
        """Return mean/max canary latency and success rate over last N runs."""
        rows = self._conn.execute(
            "SELECT duration_ms, success FROM canary_log"
            " ORDER BY ts DESC LIMIT ?",
            (last_n,),
        ).fetchall()
        if not rows:
            return {"mean_ms": None, "max_ms": None, "success_rate": None, "n": 0}
        durations = [r[0] for r in rows]
        successes = [r[1] for r in rows]
        return {
            "mean_ms":      round(sum(durations) / len(durations), 2),
            "max_ms":       round(max(durations), 2),
            "success_rate": round(sum(successes) / len(successes), 3),
            "n":            len(rows),
        }

    def performance_rho(self, last_n: int = 30) -> float | None:
        """
        ρ = slope of canary duration over time (ms/run).
        Positive slope = degradation. None if insufficient data.
        """
        rows = self._conn.execute(
            "SELECT ts, duration_ms FROM canary_log ORDER BY ts ASC LIMIT ?",
            (last_n,),
        ).fetchall()
        if len(rows) < 2:
            return None
        # Simple least-squares slope
        n = len(rows)
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        den = sum((x - x_mean) ** 2 for x in xs)
        return num / den if den > 1e-12 else 0.0

    # ── Full report ───────────────────────────────────────────────────────────

    def report(self, window_s: float = 300.0) -> dict[str, Any]:
        lr = self.mean_latency(window_s)
        return {
            "layer1_counters":    self.all_counters(),
            "layer2_lr_mean_s":   round(lr, 3) if lr is not None else None,
            "layer2_lr_alert":    self.lr_alert(window_s),
            "layer3_dm_trend":    round(self.dm_trend(), 3),
            "layer3_critical":    self.critical_alert(window_s),
            "canary":             self.canary_stats(),
            "performance_rho":    self.performance_rho(),
        }

    def close(self) -> None:
        self._conn.close()

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def prune(self, older_than_days: int = 30) -> dict[str, int]:
        cutoff = time.time() - older_than_days * 86400
        with self._lock:
            r1 = self._conn.execute(
                "DELETE FROM latency_log WHERE ts < ?", (cutoff,)
            ).rowcount
            r2 = self._conn.execute(
                "DELETE FROM dm_log WHERE ts < ?", (cutoff,)
            ).rowcount
            r3 = self._conn.execute(
                "DELETE FROM canary_log WHERE ts < ?", (cutoff,)
            ).rowcount
            self._conn.commit()
        return {"latency_rows": r1, "dm_rows": r2, "canary_rows": r3}


# ── Global singleton ──────────────────────────────────────────────────────────

metrics = PrismMetrics()
