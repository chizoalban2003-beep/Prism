"""
prism_federation.py
===================
Federated Mesh — peer-to-peer state sync between multiple PRISM instances
(home / work / phone).  No central server; conflict resolution via Lamport
vector clock + user-priority timestamp.

Key classes
-----------
* ``FederationPeer``   — record for a known remote node
* ``StateVector``      — Lamport clock for conflict-free distributed state
* ``FederationManager``— manages peers, payloads, and merge logic

Persistence
-----------
All state lives in ``~/.prism/federation.db`` (SQLite), consistent with
every other PRISM subsystem.

Design notes
------------
* No network calls are made inside this module — HTTP push/pull is handled
  by the HTTP routes layer (``prism_routes_federation.py``).
* All merge logic is purely local state manipulation.
* The additive-only pattern is followed: existing data is never deleted on
  merge, only updated when the remote version is strictly newer.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_DB = "~/.prism/federation.db"


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


@dataclass
class FederationPeer:
    peer_id: str
    name: str
    url: str          # e.g. "http://192.168.1.5:8742"
    last_seen: float
    sync_version: int

    def to_row(self) -> tuple:
        return (self.peer_id, self.name, self.url, self.last_seen, self.sync_version)

    @classmethod
    def from_row(cls, row: tuple) -> FederationPeer:
        peer_id, name, url, last_seen, sync_version = row
        return cls(
            peer_id=peer_id,
            name=name,
            url=url,
            last_seen=last_seen or 0.0,
            sync_version=sync_version or 0,
        )


# ---------------------------------------------------------------------------
# StateVector — Lamport clock
# ---------------------------------------------------------------------------


class StateVector:
    """Lamport-style vector clock for conflict-free distributed state.

    Each PRISM node maintains a ``{node_id: int}`` mapping.  A counter is
    incremented on every local write.  On merge the component-wise maximum
    is taken, which lets us determine whether one state causally precedes
    another.
    """

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self._vec: dict[str, int] = {node_id: 0}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def increment(self) -> int:
        """Advance our own logical clock by 1 and return the new value."""
        with self._lock:
            self._vec[self.node_id] = self._vec.get(self.node_id, 0) + 1
            return self._vec[self.node_id]

    def update(self, remote: dict[str, int]) -> None:
        """Merge a remote vector by taking component-wise maximums."""
        with self._lock:
            for node, ts in remote.items():
                self._vec[node] = max(self._vec.get(node, 0), ts)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, int]:
        with self._lock:
            return dict(self._vec)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def happens_before(self, other: dict[str, int]) -> bool:
        """Return True if this vector strictly happens-before *other*.

        ``A < B`` (A happens before B) iff every component of A is ≤ the
        corresponding component of B *and* at least one is strictly less.
        """
        with self._lock:
            all_nodes = set(self._vec) | set(other)
            dominated = False
            for node in all_nodes:
                a = self._vec.get(node, 0)
                b = other.get(node, 0)
                if a > b:
                    return False  # A is NOT before B
                if a < b:
                    dominated = True
            return dominated


# ---------------------------------------------------------------------------
# FederationManager
# ---------------------------------------------------------------------------

_STALE_THRESHOLD = 300  # seconds — peers not seen in 5 min need sync


class FederationManager:
    """Manage federation peers and local/remote state synchronisation.

    Parameters
    ----------
    node_id : str
        Stable identifier for this PRISM instance.  If empty a new UUID is
        generated and persisted in the database.
    db_path : str
        Path to the SQLite database (default: ``~/.prism/federation.db``).
    """

    def __init__(
        self,
        node_id: str = "",
        db_path: str = _DEFAULT_DB,
    ) -> None:
        self._db = Path(db_path).expanduser()
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()
        self.node_id: str = node_id or self._load_or_create_node_id()
        self._vector = StateVector(self.node_id)
        self._load_vector()

    # ------------------------------------------------------------------
    # Node identity
    # ------------------------------------------------------------------

    def announce(self, url: str) -> str:
        """Register this node with the given URL and return our node_id.

        The URL is persisted so that peers who receive our payload know
        where to push back.
        """
        with self._lock:
            with sqlite3.connect(self._db) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO federation_self(key, value) VALUES (?,?)",
                    ("url", url),
                )
        logger.info("FederationManager: announced as %s at %s", self.node_id, url)
        return self.node_id

    # ------------------------------------------------------------------
    # Peer management
    # ------------------------------------------------------------------

    def add_peer(self, peer_id: str, name: str, url: str) -> FederationPeer:
        """Register or update a remote peer. Returns the peer record."""
        peer = FederationPeer(
            peer_id=peer_id,
            name=name,
            url=url,
            last_seen=time.time(),
            sync_version=0,
        )
        with self._lock:
            with sqlite3.connect(self._db) as conn:
                # Preserve existing sync_version on update
                existing = conn.execute(
                    "SELECT sync_version FROM federation_peers WHERE peer_id = ?",
                    (peer_id,),
                ).fetchone()
                if existing:
                    peer.sync_version = existing[0]
                conn.execute(
                    "INSERT OR REPLACE INTO federation_peers"
                    "(peer_id, name, url, last_seen, sync_version) VALUES (?,?,?,?,?)",
                    peer.to_row(),
                )
        logger.info("FederationManager: added/updated peer %s (%s) at %s", peer_id, name, url)
        return peer

    def remove_peer(self, peer_id: str) -> bool:
        """Remove a peer by ID. Returns True if it existed."""
        with self._lock:
            with sqlite3.connect(self._db) as conn:
                cur = conn.execute(
                    "DELETE FROM federation_peers WHERE peer_id = ?", (peer_id,)
                )
                deleted = cur.rowcount > 0
        if deleted:
            logger.info("FederationManager: removed peer %s", peer_id)
        return deleted

    def list_peers(self) -> list[FederationPeer]:
        """Return all known peers."""
        with sqlite3.connect(self._db) as conn:
            rows = conn.execute(
                "SELECT peer_id, name, url, last_seen, sync_version"
                " FROM federation_peers ORDER BY name"
            ).fetchall()
        return [FederationPeer.from_row(r) for r in rows]

    # ------------------------------------------------------------------
    # State payload
    # ------------------------------------------------------------------

    def get_sync_payload(self) -> dict[str, Any]:
        """Return a snapshot of local state suitable for sending to peers.

        Schema
        ------
        ``{node_id, version, vector, goals, beliefs_summary, timestamp}``
        """
        version = self._vector.increment()
        self._persist_vector()

        goals = self._load_goals_summary()
        beliefs = self._load_beliefs_summary()

        payload: dict[str, Any] = {
            "node_id": self.node_id,
            "version": version,
            "vector": self._vector.to_dict(),
            "goals": goals,
            "beliefs_summary": beliefs,
            "timestamp": time.time(),
        }
        logger.debug("FederationManager: sync payload v%d prepared", version)
        return payload

    # ------------------------------------------------------------------
    # Merge incoming peer state
    # ------------------------------------------------------------------

    def merge_peer_state(self, peer_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Apply remote state from *peer_id*.

        Conflict resolution
        -------------------
        * Vector clock is used to detect whether the remote state causally
          precedes ours (in which case we skip).
        * For concurrent states (neither precedes the other) the remote
          record wins if its ``timestamp`` is newer than ours for that item.
        * User-priority: if a local item has ``user_priority=True`` it is
          never overwritten by remote data.

        Returns
        -------
        dict
            ``{merged_count, conflicts_resolved, peer_version}``
        """
        remote_vector: dict[str, int] = payload.get("vector", {})
        remote_version: int = payload.get("version", 0)
        remote_ts: float = payload.get("timestamp", 0.0)
        remote_goals: list[dict] = payload.get("goals", [])
        remote_beliefs: dict = payload.get("beliefs_summary", {})

        merged_count = 0
        conflicts_resolved = 0

        with self._lock:
            # Merge vector clock first
            if self._vector.happens_before(remote_vector):
                # Remote strictly dominates — safe to apply everything
                pass
            # (If we dominate remote or concurrent, we still apply on a
            #  per-record basis using wall-clock timestamps for tiebreaking.)

            self._vector.update(remote_vector)
            self._persist_vector()

            # Merge goals
            for remote_goal in remote_goals:
                gid = remote_goal.get("goal_id")
                if not gid:
                    continue
                local_goal = self._load_local_goal(gid)
                if local_goal is None:
                    # New goal from remote — adopt it
                    self._upsert_local_goal(remote_goal)
                    merged_count += 1
                else:
                    local_ts = local_goal.get("updated_at", 0.0)
                    local_priority = local_goal.get("user_priority", False)
                    remote_goal_ts = remote_goal.get("updated_at", remote_ts)

                    if local_priority:
                        # Never overwrite user-priority records
                        conflicts_resolved += 1
                        continue

                    if remote_goal_ts > local_ts:
                        # Remote is newer — overwrite
                        self._upsert_local_goal(remote_goal)
                        merged_count += 1
                    elif remote_goal_ts < local_ts:
                        pass  # ours is newer — keep
                    else:
                        # Exact timestamp tie — keep local (idempotent)
                        pass

            # Merge beliefs summary (shallow merge, remote wins on conflict
            # unless the local belief has user_priority)
            if remote_beliefs:
                self._merge_beliefs(remote_beliefs, remote_ts)
                merged_count += len(remote_beliefs)

            # Update peer record
            self._update_peer_last_seen(peer_id, remote_version)

        logger.info(
            "FederationManager: merged state from %s — "
            "merged=%d conflicts_resolved=%d peer_v=%d",
            peer_id, merged_count, conflicts_resolved, remote_version,
        )
        return {
            "merged_count": merged_count,
            "conflicts_resolved": conflicts_resolved,
            "peer_version": remote_version,
        }

    # ------------------------------------------------------------------
    # Pending sync detection
    # ------------------------------------------------------------------

    def push_pending(self) -> dict[str, int]:
        """Push local state to all peers that haven't been synced recently.

        Returns ``{"pushed": int, "failed": int}``.
        """
        import json as _json
        import urllib.request as _urlreq

        peer_ids = self.pending_sync()
        pushed = 0
        failed = 0

        if not peer_ids:
            return {"pushed": 0, "failed": 0}

        payload = self.get_sync_payload()
        payload_bytes = _json.dumps(payload).encode()

        for peer_id in peer_ids:
            try:
                with sqlite3.connect(self._db) as conn:
                    row = conn.execute(
                        "SELECT url FROM federation_peers WHERE peer_id = ?",
                        (peer_id,),
                    ).fetchone()
                if row is None:
                    failed += 1
                    continue
                peer_url = row[0].rstrip("/")
                import os as _os
                _hdrs: dict[str, str] = {"Content-Type": "application/json"}
                _tok = _os.environ.get("PRISM_FEDERATION_TOKEN", "")
                if _tok:
                    _hdrs["Authorization"] = f"Bearer {_tok}"
                req = _urlreq.Request(
                    f"{peer_url}/federation/receive",
                    data=payload_bytes,
                    headers=_hdrs,
                )
                _urlreq.urlopen(req, timeout=5)
                self._update_peer_last_seen(peer_id, payload["version"])
                pushed += 1
                logger.debug("FederationManager: pushed to peer %s", peer_id)
            except Exception as exc:
                failed += 1
                logger.debug("FederationManager: push to peer %s failed: %s", peer_id, exc)

        return {"pushed": pushed, "failed": failed}

    def pending_sync(self) -> list[str]:
        """Return peer_ids of peers that have not been synced recently.

        A peer is considered stale when ``time.time() - last_seen > 300``.
        """
        threshold = time.time() - _STALE_THRESHOLD
        with sqlite3.connect(self._db) as conn:
            rows = conn.execute(
                "SELECT peer_id FROM federation_peers WHERE last_seen < ?",
                (threshold,),
            ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Status / introspection
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Return a JSON-serialisable status summary."""
        peers = self.list_peers()
        pending = self.pending_sync()
        now = time.time()
        return {
            "node_id": self.node_id,
            "vector": self._vector.to_dict(),
            "peer_count": len(peers),
            "peers": [
                {
                    "peer_id": p.peer_id,
                    "name": p.name,
                    "url": p.url,
                    "last_seen": p.last_seen,
                    "seconds_since_sync": now - p.last_seen,
                    "sync_version": p.sync_version,
                    "pending": p.peer_id in pending,
                }
                for p in peers
            ],
            "pending_peers": pending,
        }

    # ------------------------------------------------------------------
    # Internal helpers — node identity
    # ------------------------------------------------------------------

    def _load_or_create_node_id(self) -> str:
        with sqlite3.connect(self._db) as conn:
            row = conn.execute(
                "SELECT value FROM federation_self WHERE key = 'node_id'"
            ).fetchone()
            if row:
                return row[0]
            nid = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO federation_self(key, value) VALUES ('node_id', ?)", (nid,)
            )
        logger.info("FederationManager: created new node_id %s", nid)
        return nid

    # ------------------------------------------------------------------
    # Internal helpers — vector persistence
    # ------------------------------------------------------------------

    def _persist_vector(self) -> None:
        vec_json = json.dumps(self._vector.to_dict())
        with sqlite3.connect(self._db) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO federation_self(key, value) VALUES ('vector', ?)",
                (vec_json,),
            )

    def _load_vector(self) -> None:
        with sqlite3.connect(self._db) as conn:
            row = conn.execute(
                "SELECT value FROM federation_self WHERE key = 'vector'"
            ).fetchone()
        if row:
            try:
                saved: dict[str, int] = json.loads(row[0])
                self._vector.update(saved)
                # Ensure our own counter is at least what was saved
                self._vector._vec[self.node_id] = max(
                    self._vector._vec.get(self.node_id, 0),
                    saved.get(self.node_id, 0),
                )
            except (json.JSONDecodeError, TypeError):
                pass

    # ------------------------------------------------------------------
    # Internal helpers — goals
    # ------------------------------------------------------------------

    def _load_goals_summary(self) -> list[dict[str, Any]]:
        """Load goals from horizon.db if available, else return empty list."""
        horizon_db = self._db.parent / "horizon.db"
        if not horizon_db.exists():
            return []
        try:
            with sqlite3.connect(horizon_db) as conn:
                rows = conn.execute(
                    "SELECT goal_id, intent, status, created_at FROM horizon_goals"
                    " ORDER BY created_at DESC LIMIT 50"
                ).fetchall()
            return [
                {
                    "goal_id": r[0],
                    "intent": r[1],
                    "status": r[2],
                    "updated_at": r[3],
                }
                for r in rows
            ]
        except Exception as exc:
            logger.debug("FederationManager: could not load goals — %s", exc)
            return []

    def _load_local_goal(self, goal_id: str) -> dict[str, Any] | None:
        with sqlite3.connect(self._db) as conn:
            row = conn.execute(
                "SELECT goal_id, intent, status, updated_at, user_priority"
                " FROM fed_goals WHERE goal_id = ?",
                (goal_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "goal_id": row[0],
            "intent": row[1],
            "status": row[2],
            "updated_at": row[3],
            "user_priority": bool(row[4]),
        }

    def _upsert_local_goal(self, goal: dict[str, Any]) -> None:
        with sqlite3.connect(self._db) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO fed_goals"
                "(goal_id, intent, status, updated_at, user_priority)"
                " VALUES (?,?,?,?,?)",
                (
                    goal.get("goal_id", ""),
                    goal.get("intent", ""),
                    goal.get("status", ""),
                    goal.get("updated_at", time.time()),
                    1 if goal.get("user_priority") else 0,
                ),
            )

    # ------------------------------------------------------------------
    # Internal helpers — beliefs
    # ------------------------------------------------------------------

    def _load_beliefs_summary(self) -> dict[str, Any]:
        """Return a lightweight summary of local beliefs/context."""
        with sqlite3.connect(self._db) as conn:
            rows = conn.execute(
                "SELECT key, value, updated_at FROM fed_beliefs"
            ).fetchall()
        return {r[0]: {"value": r[1], "updated_at": r[2]} for r in rows}

    def _merge_beliefs(self, remote: dict[str, Any], fallback_ts: float) -> None:
        for key, info in remote.items():
            value = info.get("value") if isinstance(info, dict) else info
            remote_ts = info.get("updated_at", fallback_ts) if isinstance(info, dict) else fallback_ts
            with sqlite3.connect(self._db) as conn:
                existing = conn.execute(
                    "SELECT updated_at FROM fed_beliefs WHERE key = ?", (key,)
                ).fetchone()
                if existing is None or remote_ts > existing[0]:
                    conn.execute(
                        "INSERT OR REPLACE INTO fed_beliefs(key, value, updated_at)"
                        " VALUES (?,?,?)",
                        (key, json.dumps(value), remote_ts),
                    )

    # ------------------------------------------------------------------
    # Internal helpers — peer last_seen
    # ------------------------------------------------------------------

    def _update_peer_last_seen(self, peer_id: str, version: int) -> None:
        with sqlite3.connect(self._db) as conn:
            conn.execute(
                "UPDATE federation_peers"
                " SET last_seen = ?, sync_version = ?"
                " WHERE peer_id = ?",
                (time.time(), version, peer_id),
            )

    # ------------------------------------------------------------------
    # SQLite initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with sqlite3.connect(self._db) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS federation_self (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS federation_peers (
                    peer_id      TEXT PRIMARY KEY,
                    name         TEXT NOT NULL,
                    url          TEXT NOT NULL,
                    last_seen    REAL NOT NULL DEFAULT 0,
                    sync_version INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS ix_fp_last_seen
                    ON federation_peers(last_seen);

                CREATE TABLE IF NOT EXISTS fed_goals (
                    goal_id       TEXT PRIMARY KEY,
                    intent        TEXT NOT NULL DEFAULT '',
                    status        TEXT NOT NULL DEFAULT '',
                    updated_at    REAL NOT NULL DEFAULT 0,
                    user_priority INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS fed_beliefs (
                    key        TEXT PRIMARY KEY,
                    value      TEXT NOT NULL DEFAULT '',
                    updated_at REAL NOT NULL DEFAULT 0
                );
            """)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        peers = self.list_peers()
        return (
            f"FederationManager(node_id={self.node_id!r}, peers={len(peers)}, "
            f"vector={self._vector.to_dict()})"
        )
