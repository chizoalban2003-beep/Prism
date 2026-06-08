"""
prism_mobile_sync.py
====================
Mobile client management for PRISM — REST sync layer for native iOS/Android clients.

Provides:
- MobileSyncManager: SQLite-backed manager for device registration, health data
  ingestion, push token storage, and pending notification retrieval.
- HMAC-SHA256 sync token generation and verification (no external auth libraries).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SYNC_TOKEN_TTL = 86400  # 24 hours


class MobileSyncManager:
    """Manages mobile client registration, sync state, and health data ingestion."""

    def __init__(
        self,
        secret_key: str = "",
        db_path: str = "~/.prism/mobile.db",
    ) -> None:
        self._secret = secret_key.encode() if secret_key else b"prism-default-secret"
        self._db = Path(db_path).expanduser()
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db))

    def _init_db(self) -> None:
        con = self._connect()
        con.executescript("""
            CREATE TABLE IF NOT EXISTS clients (
                device_id     TEXT PRIMARY KEY,
                name          TEXT NOT NULL DEFAULT '',
                platform      TEXT NOT NULL DEFAULT '',
                push_token    TEXT NOT NULL DEFAULT '',
                last_sync     REAL NOT NULL DEFAULT 0.0,
                registered_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS health_data (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id  TEXT NOT NULL,
                metric     TEXT NOT NULL,
                value      REAL NOT NULL,
                unit       TEXT NOT NULL DEFAULT '',
                timestamp  REAL NOT NULL,
                synced_at  REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS notifications (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id  TEXT NOT NULL,
                payload    TEXT NOT NULL,
                created_at REAL NOT NULL,
                delivered  INTEGER NOT NULL DEFAULT 0
            );
        """)
        con.commit()
        con.close()

    # ------------------------------------------------------------------
    # Token generation / verification
    # ------------------------------------------------------------------

    def _make_token(self, device_id: str, issued_at: float) -> str:
        """Produce a deterministic HMAC-SHA256 token for (device_id, issued_at)."""
        msg = f"{device_id}:{issued_at:.0f}".encode()
        return hmac.new(self._secret, msg, hashlib.sha256).hexdigest()

    def _token_record(self, device_id: str) -> str:
        """Return a token string that embeds the issued-at timestamp."""
        issued_at_int = int(time.time())
        raw = self._make_token(device_id, float(issued_at_int))
        # Encode as "<issued_at_int>.<hex_digest>" so we can verify expiry later
        return f"{issued_at_int}.{raw}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_client(self, device_id: str, name: str, platform: str) -> str:
        """Register or update a mobile client. Returns a sync_token."""
        now = time.time()
        con = self._connect()
        con.execute(
            """
            INSERT INTO clients (device_id, name, platform, last_sync, registered_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(device_id) DO UPDATE SET
                name          = excluded.name,
                platform      = excluded.platform,
                registered_at = CASE WHEN registered_at = 0 THEN excluded.registered_at
                                     ELSE registered_at END
            """,
            (device_id, name, platform, 0.0, now),
        )
        con.commit()
        con.close()
        token = self._token_record(device_id)
        logger.info("[mobile_sync] Registered client %s (%s/%s)", device_id, name, platform)
        return token

    def verify_token(self, device_id: str, token: str) -> bool:
        """Return True if token is valid (correct HMAC) and not expired."""
        try:
            parts = token.split(".", 1)
            if len(parts) != 2:
                return False
            issued_at = float(parts[0])
            hex_digest = parts[1]
        except (ValueError, AttributeError):
            return False

        # Expiry check
        if time.time() - issued_at > SYNC_TOKEN_TTL:
            return False

        # Constant-time HMAC comparison
        expected = self._make_token(device_id, issued_at)
        return hmac.compare_digest(expected, hex_digest)

    def ingest_health_data(self, device_id: str, metrics: list[dict]) -> int:
        """
        Insert health metrics for a device.

        Each metric dict should have: metric, value, unit, timestamp.
        Returns the count of successfully inserted rows.
        """
        now = time.time()
        rows: list[tuple[Any, ...]] = []
        for m in metrics:
            try:
                rows.append((
                    device_id,
                    str(m["metric"]),
                    float(m["value"]),
                    str(m.get("unit", "")),
                    float(m.get("timestamp", now)),
                    now,
                ))
            except (KeyError, ValueError, TypeError) as exc:
                logger.debug("[mobile_sync] Skipping bad metric %s: %s", m, exc)

        if not rows:
            return 0

        con = self._connect()
        con.executemany(
            """
            INSERT INTO health_data (device_id, metric, value, unit, timestamp, synced_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        con.commit()
        # Update last_sync timestamp for the device
        con.execute(
            "UPDATE clients SET last_sync = ? WHERE device_id = ?",
            (now, device_id),
        )
        con.commit()
        con.close()
        logger.debug("[mobile_sync] Ingested %d health metrics for %s", len(rows), device_id)
        return len(rows)

    def get_pending_notifications(self, device_id: str) -> list[dict]:
        """Return undelivered push payloads for this device, marking them as delivered."""
        con = self._connect()
        rows = con.execute(
            """
            SELECT id, payload, created_at FROM notifications
            WHERE device_id = ? AND delivered = 0
            ORDER BY created_at ASC
            """,
            (device_id,),
        ).fetchall()

        if rows:
            ids = [r[0] for r in rows]
            placeholders = ",".join("?" * len(ids))
            con.execute(
                f"UPDATE notifications SET delivered = 1 WHERE id IN ({placeholders})",
                ids,
            )
            con.commit()
        con.close()

        result = []
        for row_id, payload_json, created_at in rows:
            try:
                payload = json.loads(payload_json)
            except (json.JSONDecodeError, TypeError):
                payload = {"raw": payload_json}
            result.append({"id": row_id, "payload": payload, "created_at": created_at})
        return result

    def register_push_token(self, device_id: str, push_token: str) -> None:
        """Update push_token for this device."""
        con = self._connect()
        con.execute(
            "UPDATE clients SET push_token = ? WHERE device_id = ?",
            (push_token, device_id),
        )
        con.commit()
        con.close()
        logger.info("[mobile_sync] Push token updated for %s", device_id)

    def sync_state(self, device_id: str) -> dict:
        """Return sync metadata for the device: last_sync, pending_count, agent_status."""
        con = self._connect()
        row = con.execute(
            "SELECT last_sync FROM clients WHERE device_id = ?",
            (device_id,),
        ).fetchone()
        pending_count = con.execute(
            "SELECT COUNT(*) FROM notifications WHERE device_id = ? AND delivered = 0",
            (device_id,),
        ).fetchone()[0]
        con.close()

        last_sync = row[0] if row else 0.0
        return {
            "last_sync":     last_sync,
            "pending_count": pending_count,
            "agent_status":  "online",
        }

    def queue_notification(self, device_id: str, payload: dict) -> None:
        """Queue a push notification payload for delivery to this device."""
        con = self._connect()
        con.execute(
            "INSERT INTO notifications (device_id, payload, created_at) VALUES (?, ?, ?)",
            (device_id, json.dumps(payload, default=str), time.time()),
        )
        con.commit()
        con.close()
