"""
prism_session_manager.py
========================
SQLite-backed, thread-safe named conversation session store.

Stored at ~/.prism/sessions.db by default.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Session:
    session_id: str
    name: str
    description: str = ""
    tags: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    message_count: int = 0


@dataclass
class MessageRecord:
    message_id: str
    session_id: str
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: float = field(default_factory=time.time)


class SessionManager:
    def __init__(self, db_path: str = "~/.prism/sessions.db") -> None:
        resolved = Path(db_path).expanduser()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(resolved)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id   TEXT PRIMARY KEY,
                    name         TEXT NOT NULL,
                    description  TEXT NOT NULL DEFAULT '',
                    tags         TEXT NOT NULL DEFAULT '[]',
                    created_at   REAL NOT NULL,
                    updated_at   REAL NOT NULL,
                    message_count INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role       TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    timestamp  REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session_id
                ON messages (session_id)
            """)
            conn.commit()

    # -------------------------------------------------------------------------
    # Session CRUD
    # -------------------------------------------------------------------------

    def create_session(
        self,
        name: str,
        description: str = "",
        tags: list = None,
    ) -> Session:
        tags_list = tags or []
        session_id = uuid.uuid4().hex[:16]
        now = time.time()
        tags_json = json.dumps(tags_list)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO sessions
                        (session_id, name, description, tags, created_at, updated_at, message_count)
                    VALUES (?, ?, ?, ?, ?, ?, 0)
                    """,
                    (session_id, name, description, tags_json, now, now),
                )
                conn.commit()
        return Session(
            session_id=session_id,
            name=name,
            description=description,
            tags=tags_list,
            created_at=now,
            updated_at=now,
            message_count=0,
        )

    def get_session(self, session_id: str) -> Optional[Session]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            return None
        return Session(
            session_id=row["session_id"],
            name=row["name"],
            description=row["description"],
            tags=json.loads(row["tags"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            message_count=row["message_count"],
        )

    def list_sessions(self, limit: int = 50, offset: int = 0) -> list[Session]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [
            Session(
                session_id=r["session_id"],
                name=r["name"],
                description=r["description"],
                tags=json.loads(r["tags"]),
                created_at=r["created_at"],
                updated_at=r["updated_at"],
                message_count=r["message_count"],
            )
            for r in rows
        ]

    def update_session(
        self,
        session_id: str,
        name: str = None,
        description: str = None,
        tags: list = None,
    ) -> Optional[Session]:
        existing = self.get_session(session_id)
        if existing is None:
            return None

        new_name = name if name is not None else existing.name
        new_desc = description if description is not None else existing.description
        new_tags = tags if tags is not None else existing.tags
        now = time.time()

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE sessions
                    SET name = ?, description = ?, tags = ?, updated_at = ?
                    WHERE session_id = ?
                    """,
                    (new_name, new_desc, json.dumps(new_tags), now, session_id),
                )
                conn.commit()

        return Session(
            session_id=session_id,
            name=new_name,
            description=new_desc,
            tags=new_tags,
            created_at=existing.created_at,
            updated_at=now,
            message_count=existing.message_count,
        )

    def delete_session(self, session_id: str) -> bool:
        existing = self.get_session(session_id)
        if existing is None:
            return False
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                conn.commit()
        return True

    # -------------------------------------------------------------------------
    # Message store
    # -------------------------------------------------------------------------

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> Optional[MessageRecord]:
        existing = self.get_session(session_id)
        if existing is None:
            return None
        message_id = uuid.uuid4().hex[:16]
        now = time.time()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO messages (message_id, session_id, role, content, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (message_id, session_id, role, content, now),
                )
                conn.execute(
                    """
                    UPDATE sessions
                    SET message_count = message_count + 1, updated_at = ?
                    WHERE session_id = ?
                    """,
                    (now, session_id),
                )
                conn.commit()
        return MessageRecord(
            message_id=message_id,
            session_id=session_id,
            role=role,
            content=content,
            timestamp=now,
        )

    def get_history(
        self,
        session_id: str,
        n: int = 50,
        offset: int = 0,
    ) -> list[MessageRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
                LIMIT ? OFFSET ?
                """,
                (session_id, n, offset),
            ).fetchall()
        return [
            MessageRecord(
                message_id=r["message_id"],
                session_id=r["session_id"],
                role=r["role"],
                content=r["content"],
                timestamp=r["timestamp"],
            )
            for r in rows
        ]

    def clear_history(self, session_id: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) as cnt FROM messages WHERE session_id = ?",
                (session_id,),
            )
            count = cursor.fetchone()["cnt"]
        if count == 0:
            return 0
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                conn.execute(
                    "UPDATE sessions SET message_count = 0 WHERE session_id = ?",
                    (session_id,),
                )
                conn.commit()
        return count


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_manager: Optional[SessionManager] = None
_manager_lock = threading.Lock()


def get_session_manager(db_path: str = "~/.prism/sessions.db") -> SessionManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = SessionManager(db_path)
    return _manager


def reset_session_manager(db_path: str = "~/.prism/sessions.db") -> SessionManager:
    global _manager
    with _manager_lock:
        _manager = SessionManager(db_path)
    return _manager
