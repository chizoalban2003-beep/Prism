"""
PRISM Write-Ahead Log
Every mutation to the memory graph is durably logged here before being
applied to the cold layer. Guarantees replay-based recovery after crashes.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

_DEFAULT_PATH = Path.home() / ".prism" / "wal.db"


class PrismWAL:
    def __init__(self, db_path: Path | str = _DEFAULT_PATH):
        self._db = Path(db_path)
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()
        self._setup()

    def _setup(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS wal (
                seq_id    TEXT PRIMARY KEY,
                op        TEXT NOT NULL,
                payload   TEXT NOT NULL,
                ts        REAL NOT NULL,
                committed INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS ix_wal_committed ON wal(committed);
            CREATE INDEX IF NOT EXISTS ix_wal_ts        ON wal(ts);
        """)
        self._conn.commit()

    def append(self, op: str, payload: dict[str, Any]) -> str:
        """Append a mutation. Returns seq_id."""
        seq_id = uuid.uuid4().hex
        with self._lock:
            self._conn.execute(
                "INSERT INTO wal(seq_id, op, payload, ts, committed) VALUES (?,?,?,?,0)",
                (seq_id, op, json.dumps(payload), time.time()),
            )
            self._conn.commit()
        return seq_id

    def mark_committed(self, seq_id: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE wal SET committed=1 WHERE seq_id=?", (seq_id,))
            self._conn.commit()

    def pending(self) -> list[dict[str, Any]]:
        """Return all uncommitted entries ordered by ts."""
        rows = self._conn.execute(
            "SELECT seq_id, op, payload, ts FROM wal WHERE committed=0 ORDER BY ts"
        ).fetchall()
        return [
            {"seq_id": r[0], "op": r[1], "payload": json.loads(r[2]), "ts": r[3]}
            for r in rows
        ]

    def pending_count(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM wal WHERE committed=0"
        ).fetchone()[0]

    def append_batch(self, entries: list[tuple[str, dict[str, Any]]]) -> list[str]:
        """Append multiple mutations in a single transaction. Returns list of seq_ids."""
        if not entries:
            return []
        now = time.time()
        seq_ids = [uuid.uuid4().hex for _ in entries]
        rows = [
            (sid, op, json.dumps(payload), now, 0)
            for sid, (op, payload) in zip(seq_ids, entries)
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT INTO wal(seq_id, op, payload, ts, committed) VALUES (?,?,?,?,?)", rows
            )
            self._conn.commit()
        return seq_ids

    def mark_committed_batch(self, seq_ids: list[str]) -> None:
        """Mark multiple entries committed in a single transaction."""
        if not seq_ids:
            return
        placeholders = ",".join("?" * len(seq_ids))
        with self._lock:
            self._conn.execute(
                f"UPDATE wal SET committed=1 WHERE seq_id IN ({placeholders})", seq_ids
            )
            self._conn.commit()

    def drain_committed(self, older_than_days: int = 7) -> int:
        """Delete committed entries older than N days. Returns count removed."""
        cutoff = time.time() - older_than_days * 86400
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM wal WHERE committed=1 AND ts < ?", (cutoff,)
            )
            self._conn.commit()
        return cur.rowcount

    def replay(self, memory_graph: Any) -> int:
        """
        Apply all uncommitted WAL entries to *memory_graph* on daemon startup.

        Called once before the HTTP server accepts requests so that a crash
        between WAL append and shadow-pipeline commit leaves no data loss.
        Returns the number of entries replayed.
        """
        entries = self.pending()
        if not entries:
            return 0
        replayed = 0
        for entry in entries:
            try:
                op      = entry["op"]
                payload = entry["payload"]
                if hasattr(memory_graph, "apply_wal_entry"):
                    memory_graph.apply_wal_entry(op, payload)
                elif op == "add" and hasattr(memory_graph, "add"):
                    memory_graph.add(**payload)
                elif op == "update" and hasattr(memory_graph, "update"):
                    memory_graph.update(**payload)
                elif op == "delete" and hasattr(memory_graph, "delete"):
                    memory_graph.delete(**payload)
                self.mark_committed(entry["seq_id"])
                replayed += 1
            except Exception as exc:  # noqa: BLE001
                import logging
                logging.getLogger(__name__).error(
                    "WAL replay failed for seq_id=%s op=%s: %s",
                    entry["seq_id"], entry["op"], exc,
                )
        return replayed

    def close(self) -> None:
        self._conn.close()
