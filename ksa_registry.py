"""
ksa_registry.py
===============
Kinetic State Agent — Snapshot Registry

A SQLite-backed registry that stores, versions, and retrieves KineticSnapshot
matrices keyed by task name. Every successful task run can persist its lever
configuration here. The self-optimising loop reads from here to hot-swap the
best known configuration for a given task.

Schema summary:
    snapshots       — one row per snapshot version per task
    task_metrics    — per-run performance telemetry (feeds optimisation loop)

Public API:
    registry = SnapshotRegistry("ksa_state.db")

    registry.save(task_name, system)             → version int
    registry.load(task_name) → ThreeBarSystem    (current best version)
    registry.record_outcome(task_name, version, metrics)
    registry.promote(task_name, version)         → make a version "current"
    registry.rollback(task_name)                 → revert to previous version
    registry.best_version(task_name)             → version with best score
    registry.list_tasks()                        → summary of all tasks
    registry.history(task_name)                  → all versions + metrics
    registry.prune(task_name, keep=5)            → trim old versions
    registry.delete_task(task_name)              → remove all versions
"""

from __future__ import annotations

import json
import os
import sqlite3

# Import the lever system from the same package
# (assumes ksa_lever.py is on sys.path or in the same directory)
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))
from ksa_lever import ThreeBarSystem

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PerformanceMetrics:
    """
    Telemetry captured after a task run.
    Used by the optimisation loop to decide which snapshot version is best.
    """
    execution_time_ms: float  = 0.0   # wall-clock time for the task
    cpu_peak_pct:      float  = 0.0   # peak CPU % during execution
    ram_peak_mb:       float  = 0.0   # peak RAM usage in MB
    success:           bool   = True  # did the task complete without error?
    override_fired:    bool   = False # did the Balancer override trigger?
    notes:             str    = ""    # free-text annotation

    def score(self) -> float:
        """
        Composite performance score (higher = better).

        Formula balances three signals:
          - Speed:    inverse of execution time (faster = better)
          - Stability: penalty if override fired (instability = bad)
          - Success:  hard multiplier of 0 if the task failed

        Returns a float in [0, ∞). Intended for relative ranking only.
        """
        if not self.success:
            return 0.0
        speed_score     = 1000.0 / max(self.execution_time_ms, 1.0)
        stability_bonus = 0.0 if self.override_fired else 1.0
        return round((speed_score + stability_bonus) * 1.0, 6)

    def to_dict(self) -> dict:
        return {
            "execution_time_ms": self.execution_time_ms,
            "cpu_peak_pct":      self.cpu_peak_pct,
            "ram_peak_mb":       self.ram_peak_mb,
            "success":           self.success,
            "override_fired":    self.override_fired,
            "notes":             self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PerformanceMetrics:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SnapshotRecord:
    """A single row from the snapshots table, fully hydrated."""
    id:            int
    task_name:     str
    version:       int
    snapshot:      dict              # the raw S = {W, F, L, arm_lengths, ...}
    created_at:    str
    is_current:    bool
    metrics:       Optional[PerformanceMetrics] = None
    score:         Optional[float]              = None

    def to_system(self) -> ThreeBarSystem:
        """Instantiate and hydrate a ThreeBarSystem from this record."""
        three_bar = ThreeBarSystem()
        three_bar.hydrate(self.snapshot)
        return three_bar


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class SnapshotRegistry:
    """
    Manages all KineticSnapshot storage, versioning, and retrieval.

    Thread-safety: SQLite WAL mode is enabled. Each public method opens
    and closes its own connection, so the registry is safe to use from
    multiple threads as long as they share the same db path.
    """

    SCHEMA = """
    PRAGMA journal_mode = WAL;
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS snapshots (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        task_name     TEXT    NOT NULL,
        version       INTEGER NOT NULL,
        snapshot_json TEXT    NOT NULL,
        created_at    TEXT    NOT NULL,
        is_current    INTEGER NOT NULL DEFAULT 1,
        UNIQUE (task_name, version)
    );

    CREATE TABLE IF NOT EXISTS task_metrics (
        id                 INTEGER PRIMARY KEY AUTOINCREMENT,
        task_name          TEXT    NOT NULL,
        version            INTEGER NOT NULL,
        recorded_at        TEXT    NOT NULL,
        execution_time_ms  REAL    NOT NULL DEFAULT 0,
        cpu_peak_pct       REAL    NOT NULL DEFAULT 0,
        ram_peak_mb        REAL    NOT NULL DEFAULT 0,
        success            INTEGER NOT NULL DEFAULT 1,
        override_fired     INTEGER NOT NULL DEFAULT 0,
        score              REAL    NOT NULL DEFAULT 0,
        notes              TEXT    NOT NULL DEFAULT '',
        FOREIGN KEY (task_name, version)
            REFERENCES snapshots (task_name, version)
            ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_snap_task
        ON snapshots (task_name, is_current);
    CREATE INDEX IF NOT EXISTS idx_metrics_task
        ON task_metrics (task_name, version);
    """

    def __init__(self, db_path: str = "ksa_state.db"):
        self.db_path = Path(db_path)
        # Ensure the parent directory exists before sqlite3.connect — otherwise
        # the first-ever command on a fresh home (e.g. `ksa status`) dies with
        # "unable to open database file".
        parent = self.db_path.parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Internal helpers ────────────────────────────────────────────────────

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(self.SCHEMA)

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def _next_version(self, conn: sqlite3.Connection, task_name: str) -> int:
        row = conn.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 FROM snapshots WHERE task_name = ?",
            (task_name,)
        ).fetchone()
        return row[0]

    def _get_record(
        self,
        conn: sqlite3.Connection,
        task_name: str,
        version: Optional[int] = None
    ) -> Optional[sqlite3.Row]:
        if version is None:
            return conn.execute(
                "SELECT * FROM snapshots WHERE task_name = ? AND is_current = 1",
                (task_name,)
            ).fetchone()
        return conn.execute(
            "SELECT * FROM snapshots WHERE task_name = ? AND version = ?",
            (task_name, version)
        ).fetchone()

    def _best_metrics(
        self,
        conn: sqlite3.Connection,
        task_name: str,
        version: int
    ) -> Optional[PerformanceMetrics]:
        """Return the most recent metrics row for a given task/version."""
        row = conn.execute(
            """SELECT * FROM task_metrics
               WHERE task_name = ? AND version = ?
               ORDER BY recorded_at DESC LIMIT 1""",
            (task_name, version)
        ).fetchone()
        if row is None:
            return None
        return PerformanceMetrics(
            execution_time_ms = row["execution_time_ms"],
            cpu_peak_pct      = row["cpu_peak_pct"],
            ram_peak_mb       = row["ram_peak_mb"],
            success           = bool(row["success"]),
            override_fired    = bool(row["override_fired"]),
            notes             = row["notes"],
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def save(self, task_name: str, system: ThreeBarSystem) -> int:
        """
        Serialise the system's current state and save it as a new snapshot
        version for the given task. Automatically marks this version as
        `is_current`, demoting any previous current version.

        Returns the new version number.
        """
        snapshot_json = json.dumps(system.snapshot())
        with self._conn() as conn:
            version = self._next_version(conn, task_name)
            # Demote all existing current versions for this task
            conn.execute(
                "UPDATE snapshots SET is_current = 0 WHERE task_name = ?",
                (task_name,)
            )
            conn.execute(
                """INSERT INTO snapshots
                   (task_name, version, snapshot_json, created_at, is_current)
                   VALUES (?, ?, ?, ?, 1)""",
                (task_name, version, snapshot_json, self._now())
            )
        return version

    def load(self, task_name: str, version: Optional[int] = None) -> ThreeBarSystem:
        """
        Hydrate and return a ThreeBarSystem from the registry.

        If version is None, loads the current (most recently promoted) version.
        Raises KeyError if the task or version is not found.
        """
        with self._conn() as conn:
            row = self._get_record(conn, task_name, version)
        if row is None:
            label = f"version {version}" if version else "current version"
            raise KeyError(f"No snapshot found for task '{task_name}' ({label})")
        snapshot = json.loads(row["snapshot_json"])
        sys = ThreeBarSystem()
        sys.hydrate(snapshot)
        return sys

    def record_outcome(
        self,
        task_name:  str,
        version:    int,
        metrics:    PerformanceMetrics,
    ) -> None:
        """
        Store a performance telemetry row for a given task/version.
        Can be called multiple times; all runs are retained for trend analysis.
        """
        with self._conn() as conn:
            # Verify the snapshot exists
            row = self._get_record(conn, task_name, version)
            if row is None:
                raise KeyError(
                    f"Cannot record outcome: snapshot '{task_name}' v{version} not found."
                )
            conn.execute(
                """INSERT INTO task_metrics
                   (task_name, version, recorded_at,
                    execution_time_ms, cpu_peak_pct, ram_peak_mb,
                    success, override_fired, score, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    task_name, version, self._now(),
                    metrics.execution_time_ms,
                    metrics.cpu_peak_pct,
                    metrics.ram_peak_mb,
                    int(metrics.success),
                    int(metrics.override_fired),
                    metrics.score(),
                    metrics.notes,
                )
            )

    def promote(self, task_name: str, version: int) -> None:
        """
        Explicitly mark a specific version as `is_current`.
        Useful after manual inspection or A/B comparison.
        """
        with self._conn() as conn:
            row = self._get_record(conn, task_name, version)
            if row is None:
                raise KeyError(f"Snapshot '{task_name}' v{version} not found.")
            conn.execute(
                "UPDATE snapshots SET is_current = 0 WHERE task_name = ?",
                (task_name,)
            )
            conn.execute(
                "UPDATE snapshots SET is_current = 1 WHERE task_name = ? AND version = ?",
                (task_name, version)
            )

    def rollback(self, task_name: str) -> int:
        """
        Revert to the previous version (current_version - 1).
        Returns the version number now marked as current.
        Raises ValueError if there is no previous version.
        """
        with self._conn() as conn:
            current = self._get_record(conn, task_name)
            if current is None:
                raise KeyError(f"No snapshot found for task '{task_name}'.")
            prev_version = current["version"] - 1
            prev = self._get_record(conn, task_name, prev_version)
            if prev is None:
                raise ValueError(
                    f"Cannot rollback '{task_name}': already at version 1."
                )
            conn.execute(
                "UPDATE snapshots SET is_current = 0 WHERE task_name = ?",
                (task_name,)
            )
            conn.execute(
                "UPDATE snapshots SET is_current = 1 "
                "WHERE task_name = ? AND version = ?",
                (task_name, prev_version)
            )
        return prev_version

    def best_version(self, task_name: str) -> Optional[int]:
        """
        Return the version number with the highest average performance score
        across all recorded outcomes. Returns None if no metrics exist yet.

        This is the core feed for the self-optimising loop:
            v = registry.best_version(task_name)
            if v:
                registry.promote(task_name, v)
        """
        with self._conn() as conn:
            row = conn.execute(
                """SELECT version, AVG(score) as avg_score
                   FROM task_metrics
                   WHERE task_name = ? AND success = 1
                   GROUP BY version
                   ORDER BY avg_score DESC
                   LIMIT 1""",
                (task_name,)
            ).fetchone()
        return int(row["version"]) if row else None

    def auto_promote_best(self, task_name: str) -> Optional[int]:
        """
        Convenience: find the best version and promote it as current.
        Returns the promoted version, or None if no metrics exist.
        This is the one-liner for the self-optimisation loop.
        """
        best = self.best_version(task_name)
        if best is not None:
            self.promote(task_name, best)
        return best

    def list_tasks(self) -> list[dict]:
        """
        Return a summary list of all tasks in the registry:
            task_name, current_version, total_versions, last_updated, best_score
        """
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT
                     s.task_name,
                     s.version              AS current_version,
                     s.created_at           AS last_updated,
                     COUNT(s2.version)      AS total_versions,
                     COALESCE(MAX(m.score), NULL) AS best_score
                   FROM snapshots s
                   JOIN snapshots s2 ON s2.task_name = s.task_name
                   LEFT JOIN task_metrics m ON m.task_name = s.task_name
                   WHERE s.is_current = 1
                   GROUP BY s.task_name
                   ORDER BY s.task_name"""
            ).fetchall()
        return [dict(r) for r in rows]

    def history(self, task_name: str) -> list[SnapshotRecord]:
        """
        Return all snapshot versions for a task, newest first,
        each annotated with its best metrics.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM snapshots WHERE task_name = ? ORDER BY version DESC",
                (task_name,)
            ).fetchall()
            records = []
            for row in rows:
                metrics = self._best_metrics(conn, task_name, row["version"])
                records.append(SnapshotRecord(
                    id         = row["id"],
                    task_name  = row["task_name"],
                    version    = row["version"],
                    snapshot   = json.loads(row["snapshot_json"]),
                    created_at = row["created_at"],
                    is_current = bool(row["is_current"]),
                    metrics    = metrics,
                    score      = metrics.score() if metrics else None,
                ))
        return records

    def prune(self, task_name: str, keep: int = 5) -> int:
        """
        Delete old versions for a task, keeping only the `keep` most recent.
        Never deletes the current version. Returns the number of rows deleted.
        """
        with self._conn() as conn:
            # Get IDs to keep (newest `keep` versions, always include current)
            rows = conn.execute(
                """SELECT id FROM snapshots WHERE task_name = ?
                   ORDER BY version DESC LIMIT ?""",
                (task_name, keep)
            ).fetchall()
            keep_ids = [r["id"] for r in rows]
            if not keep_ids:
                return 0
            placeholders = ",".join("?" * len(keep_ids))
            result = conn.execute(
                f"DELETE FROM snapshots WHERE task_name = ? AND id NOT IN ({placeholders})",
                [task_name] + keep_ids
            )
        return result.rowcount

    def delete_task(self, task_name: str) -> None:
        """Remove all snapshots and metrics for a task. Irreversible."""
        with self._conn() as conn:
            conn.execute("DELETE FROM snapshots WHERE task_name = ?", (task_name,))
            conn.execute("DELETE FROM task_metrics WHERE task_name = ?", (task_name,))

    # ── Debug / inspection ──────────────────────────────────────────────────

    def __repr__(self) -> str:
        tasks = self.list_tasks()
        return (
            f"SnapshotRegistry(db='{self.db_path}', "
            f"tasks={len(tasks)}, "
            f"task_names={[t['task_name'] for t in tasks]})"
        )
