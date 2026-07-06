"""
prism_pipelines.py
==================
Persistent pipelines (gap 2 of docs/command-centre-assessment.md).

A *pipeline* is a named, saved natural-language instruction that the user
can re-run on demand or on a schedule. Running one feeds its text to the
tool loop (multistep=True), so the same LLM→policy→organ machinery that
handles a one-off "check the weather and note what to wear" now backs a
durable "morning_brief" the user defined once. This is the persistent-
pipe layer the command-centre vision needs: connections between apps
(organs + MCP tools) saved as a reusable trajectory rather than retyped.

Storage is a single SQLite table under ~/.prism. Scheduling reuses the
proactive loop's cadence — a scheduled pipeline fires in the daemon and
its result is delivered as a proactive notification, exactly like a
reminder, so nothing new has to reach the UI.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_DEFAULT_DB = Path.home() / ".prism" / "pipelines.db"


@dataclass
class Pipeline:
    name:          str
    instruction:   str
    schedule_secs: int = 0        # 0 = manual-only; >0 = run every N seconds
    enabled:       bool = True
    created_at:    float = 0.0
    last_run:      float = 0.0
    run_count:     int = 0


class PipelineStore:
    def __init__(self, db_path: str | Path = _DEFAULT_DB) -> None:
        self._db = Path(db_path).expanduser()
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db, timeout=30.0) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS pipelines(
                name          TEXT PRIMARY KEY,
                instruction   TEXT NOT NULL,
                schedule_secs INTEGER DEFAULT 0,
                enabled       INTEGER DEFAULT 1,
                created_at    REAL,
                last_run      REAL DEFAULT 0,
                run_count     INTEGER DEFAULT 0)""")

    @staticmethod
    def _norm(name: str) -> str:
        return " ".join((name or "").strip().lower().split())

    def save(self, name: str, instruction: str,
             schedule_secs: int = 0) -> Pipeline:
        """Create or replace a pipeline. Returns the stored row."""
        key = self._norm(name)
        if not key:
            raise ValueError("pipeline name is empty")
        if not (instruction or "").strip():
            raise ValueError("pipeline instruction is empty")
        now = time.time()
        with sqlite3.connect(self._db, timeout=30.0) as c:
            existing = c.execute(
                "SELECT created_at, run_count FROM pipelines WHERE name=?",
                (key,)).fetchone()
            created = existing[0] if existing else now
            runs = existing[1] if existing else 0
            c.execute(
                "INSERT OR REPLACE INTO pipelines"
                "(name, instruction, schedule_secs, enabled, created_at,"
                " last_run, run_count) VALUES(?,?,?,?,?,?,?)",
                (key, instruction.strip(), int(max(0, schedule_secs)), 1,
                 created, 0.0, runs))
        return self.get(key)  # type: ignore[return-value]

    def get(self, name: str) -> Optional[Pipeline]:
        with sqlite3.connect(self._db, timeout=30.0) as c:
            row = c.execute(
                "SELECT name, instruction, schedule_secs, enabled, "
                "created_at, last_run, run_count FROM pipelines WHERE name=?",
                (self._norm(name),)).fetchone()
        return self._row(row) if row else None

    def list_all(self) -> list[Pipeline]:
        with sqlite3.connect(self._db, timeout=30.0) as c:
            rows = c.execute(
                "SELECT name, instruction, schedule_secs, enabled, "
                "created_at, last_run, run_count FROM pipelines "
                "ORDER BY created_at").fetchall()
        return [self._row(r) for r in rows]

    def delete(self, name: str) -> bool:
        with sqlite3.connect(self._db, timeout=30.0) as c:
            cur = c.execute("DELETE FROM pipelines WHERE name=?",
                            (self._norm(name),))
            return (cur.rowcount or 0) > 0

    def set_enabled(self, name: str, enabled: bool) -> bool:
        with sqlite3.connect(self._db, timeout=30.0) as c:
            cur = c.execute("UPDATE pipelines SET enabled=? WHERE name=?",
                            (1 if enabled else 0, self._norm(name)))
            return (cur.rowcount or 0) > 0

    def mark_run(self, name: str) -> None:
        with sqlite3.connect(self._db, timeout=30.0) as c:
            c.execute(
                "UPDATE pipelines SET last_run=?, run_count=run_count+1 "
                "WHERE name=?", (time.time(), self._norm(name)))

    def due(self, now: Optional[float] = None) -> list[Pipeline]:
        """Enabled, scheduled pipelines whose interval has elapsed."""
        now = now or time.time()
        out = []
        for p in self.list_all():
            if p.enabled and p.schedule_secs > 0 and \
                    now - p.last_run >= p.schedule_secs:
                out.append(p)
        return out

    @staticmethod
    def _row(row) -> Pipeline:
        return Pipeline(
            name=row[0], instruction=row[1], schedule_secs=int(row[2]),
            enabled=bool(row[3]), created_at=row[4] or 0.0,
            last_run=row[5] or 0.0, run_count=int(row[6] or 0))


# ── Natural-language parsing for the chat intents ────────────────────────

def parse_save(message: str) -> tuple[str, str, int]:
    """Parse 'save pipeline <name>: <instruction> [every <N> <unit>]'.

    Returns (name, instruction, schedule_secs). Raises ValueError when the
    name/instruction separator is missing.
    """
    import re
    m = re.search(r"(?:save|create|define|make)\s+(?:a\s+)?"
                  r"(?:pipeline|routine|workflow|pipe)\s+(.+)", message,
                  re.IGNORECASE)
    body = m.group(1) if m else message
    # Trailing "every N <unit>" → schedule.
    schedule_secs = 0
    sm = re.search(r"\bevery\s+(\d+)\s*(second|sec|minute|min|hour|hr|day)s?\b",
                   body, re.IGNORECASE)
    if sm:
        n = int(sm.group(1))
        unit = sm.group(2).lower()
        mult = {"second": 1, "sec": 1, "minute": 60, "min": 60,
                "hour": 3600, "hr": 3600, "day": 86400}[unit]
        schedule_secs = n * mult
        body = body[:sm.start()].rstrip(" ,;")
    # Also accept a leading "daily/hourly" adverb.
    dm = re.search(r"\b(daily|hourly|every day|every hour)\b", body, re.IGNORECASE)
    if dm and not schedule_secs:
        schedule_secs = 86400 if "day" in dm.group(1).lower() else 3600
        body = (body[:dm.start()] + body[dm.end():]).strip(" ,;")
    if ":" not in body:
        raise ValueError(
            "Use: save pipeline <name>: <what to do> [every <N> minutes]")
    name, instruction = body.split(":", 1)
    return name.strip(), instruction.strip(), schedule_secs


def human_schedule(secs: int) -> str:
    if secs <= 0:
        return "manual"
    if secs % 86400 == 0:
        return f"every {secs // 86400}d"
    if secs % 3600 == 0:
        return f"every {secs // 3600}h"
    if secs % 60 == 0:
        return f"every {secs // 60}m"
    return f"every {secs}s"
