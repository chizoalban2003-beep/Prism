"""
prism_multi_user.py
===================
Multi-User Home Hub — per-user isolated memory graphs + shared household
organ bus routing.

Classes
-------
UserProfile     — dataclass representing a registered household member
UserRegistry    — manage registration, lookup, and per-user resources
HouseholdBus    — fan-out / point-to-point signal routing across users

Design notes
------------
- Each user gets an isolated PrismMemoryGraph backed by their own SQLite DB
  at ``<base_dir>/<user_id>/memory_graph.db``.
- PrismSoul instances are lazy-initialised on first access.
- HouseholdBus wraps an optional OrganBus for persistence/audit;
  without one it still delivers signals to all registered user handlers
  via a simple in-process deque.
- Integrates with prism_asgi via prism_routes_users.
"""
from __future__ import annotations

import collections
import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from prism_memory_graph import PrismMemoryGraph
from prism_organ_bus import OrganBus, OrganSignal

__all__ = [
    "UserProfile",
    "UserRegistry",
    "HouseholdBus",
]

_24H = 86_400.0  # seconds


# ---------------------------------------------------------------------------
# UserProfile
# ---------------------------------------------------------------------------


@dataclass
class UserProfile:
    """Persistent record for a single household member."""

    user_id: str
    name: str
    role: str          # "admin" | "member" | "guest"
    db_path: str       # absolute path to this user's memory_graph.db
    soul_path: str     # absolute path to this user's soul.md
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "UserProfile":
        return cls(
            user_id=d["user_id"],
            name=d["name"],
            role=d["role"],
            db_path=d["db_path"],
            soul_path=d["soul_path"],
            created_at=d.get("created_at", time.time()),
            last_active=d.get("last_active", time.time()),
        )


# ---------------------------------------------------------------------------
# UserRegistry
# ---------------------------------------------------------------------------

_VALID_ROLES = {"admin", "member", "guest"}


class UserRegistry:
    """
    Registry of household members.

    Files on disk
    -------------
    ``<base_dir>/registry.json``   — index of all profiles
    ``<base_dir>/<user_id>/``      — per-user directory
        memory_graph.db            — isolated cold-layer SQLite
        wal.db                     — per-user WAL
        soul.md                    — living soul document

    Thread safety
    -------------
    All public methods are protected by a single ``threading.Lock``.
    """

    def __init__(self, base_dir: str = "~/.prism/users") -> None:
        self._base = Path(base_dir).expanduser()
        self._base.mkdir(parents=True, exist_ok=True)
        self._registry_file = self._base / "registry.json"
        self._lock = threading.Lock()

        # In-process caches (user_id → object)
        self._profiles: dict[str, UserProfile] = {}
        self._memories: dict[str, PrismMemoryGraph] = {}
        self._souls: dict[str, Any] = {}          # PrismSoul | None

        self._load_registry()

    # ── Persistence ────────────────────────────────────────────────────────

    def _load_registry(self) -> None:
        if not self._registry_file.exists():
            return
        try:
            raw: list[dict] = json.loads(self._registry_file.read_text())
            for d in raw:
                p = UserProfile.from_dict(d)
                self._profiles[p.user_id] = p
        except Exception:
            pass  # corrupt registry — start fresh

    def _save_registry(self) -> None:
        data = [p.to_dict() for p in self._profiles.values()]
        self._registry_file.write_text(json.dumps(data, indent=2))

    # ── Public API ─────────────────────────────────────────────────────────

    def register(
        self,
        user_id: str,
        name: str,
        role: str = "member",
    ) -> UserProfile:
        """
        Register a new user.  Raises ``ValueError`` on duplicate ``user_id``
        or invalid ``role``.
        """
        if role not in _VALID_ROLES:
            raise ValueError(
                f"role must be one of {sorted(_VALID_ROLES)!r}, got {role!r}"
            )

        with self._lock:
            if user_id in self._profiles:
                raise ValueError(f"User {user_id!r} is already registered")

            user_dir = self._base / user_id
            user_dir.mkdir(parents=True, exist_ok=True)

            profile = UserProfile(
                user_id=user_id,
                name=name,
                role=role,
                db_path=str(user_dir / "memory_graph.db"),
                soul_path=str(user_dir / "soul.db"),
            )
            self._profiles[user_id] = profile
            self._save_registry()

        return profile

    def get(self, user_id: str) -> UserProfile | None:
        """Return the profile for ``user_id``, or ``None`` if not found."""
        with self._lock:
            return self._profiles.get(user_id)

    def list_users(self) -> list[UserProfile]:
        """Return a snapshot of all registered user profiles."""
        with self._lock:
            return list(self._profiles.values())

    def remove(self, user_id: str) -> bool:
        """
        Unregister a user.  Closes any open memory graph.
        Returns ``True`` if the user existed, ``False`` if not found.
        """
        with self._lock:
            if user_id not in self._profiles:
                return False
            del self._profiles[user_id]

            # Close and evict in-process resources
            mem = self._memories.pop(user_id, None)
            if mem is not None:
                try:
                    mem.close()
                except Exception:
                    pass

            self._souls.pop(user_id, None)
            self._save_registry()
        return True

    def touch(self, user_id: str) -> None:
        """Update ``last_active`` timestamp for ``user_id`` (no-op if unknown)."""
        with self._lock:
            p = self._profiles.get(user_id)
            if p is not None:
                p.last_active = time.time()
                self._save_registry()

    def get_memory(self, user_id: str) -> PrismMemoryGraph:
        """
        Return the ``PrismMemoryGraph`` for ``user_id``.
        Lazily initialises on first call.  Raises ``KeyError`` if unknown.
        """
        with self._lock:
            if user_id not in self._profiles:
                raise KeyError(f"Unknown user: {user_id!r}")

            if user_id not in self._memories:
                p = self._profiles[user_id]
                db   = Path(p.db_path)
                wal  = db.parent / "wal.db"
                self._memories[user_id] = PrismMemoryGraph(
                    db_path=db, wal_path=wal
                )
            return self._memories[user_id]

    def get_soul(self, user_id: str) -> Any | None:
        """
        Return the ``PrismSoul`` for ``user_id``, lazily initialised.
        Returns ``None`` if PrismSoul is unavailable (import error).
        Raises ``KeyError`` if the user is unknown.
        """
        with self._lock:
            if user_id not in self._profiles:
                raise KeyError(f"Unknown user: {user_id!r}")

            if user_id in self._souls:
                return self._souls[user_id]

        # Lazy import — prism_soul has heavy optional deps
        soul: Any | None = None
        try:
            from prism_soul import PrismSoul  # noqa: PLC0415

            p = self._profiles[user_id]
            soul = PrismSoul(db_path=str(Path(p.soul_path)))
        except Exception:
            pass

        with self._lock:
            self._souls[user_id] = soul
        return soul


# ---------------------------------------------------------------------------
# HouseholdBus
# ---------------------------------------------------------------------------

_MAX_SIGNAL_HISTORY = 200


class HouseholdBus:
    """
    Fan-out / point-to-point signal routing for a multi-user household.

    When an ``OrganBus`` is supplied the bus uses it for persistence and
    anomaly detection.  Without one, signals are tracked only in-process
    (useful for tests and lightweight deployments).
    """

    def __init__(
        self,
        registry: UserRegistry,
        organ_bus: OrganBus | None = None,
    ) -> None:
        self._registry = registry
        self._bus = organ_bus
        self._lock = threading.Lock()
        # Recent signal log: deque of {signal, results, ts}
        self._history: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=_MAX_SIGNAL_HISTORY
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def broadcast(self, signal: OrganSignal) -> dict[str, Any]:
        """
        Emit ``signal`` to every registered user's organ bus segment.

        Returns a dict mapping ``user_id → delivery result``.
        If the underlying ``OrganBus`` is present it handles routing;
        otherwise the payload is delivered directly.
        """
        users = self._registry.list_users()
        results: dict[str, Any] = {}

        for profile in users:
            results[profile.user_id] = self._deliver(profile.user_id, signal)

        self._record(signal, results)
        return results

    def route_to(self, user_id: str, signal: OrganSignal) -> Any:
        """
        Route ``signal`` to a single user.  Updates ``last_active``.
        Raises ``KeyError`` if ``user_id`` is not registered.
        """
        if self._registry.get(user_id) is None:
            raise KeyError(f"Unknown user: {user_id!r}")

        result = self._deliver(user_id, signal)
        self._record(signal, {user_id: result})
        return result

    def active_users(self) -> list[str]:
        """
        Return user IDs with ``last_active`` within the last 24 hours,
        sorted by most-recent first.
        """
        cutoff = time.time() - _24H
        profiles = self._registry.list_users()
        recent = sorted(
            (p for p in profiles if p.last_active >= cutoff),
            key=lambda p: p.last_active,
            reverse=True,
        )
        return [p.user_id for p in recent]

    def signal_history(self, n: int = 20) -> list[dict[str, Any]]:
        """Return the *n* most recent signal records (newest first)."""
        with self._lock:
            records = list(self._history)
        return list(reversed(records))[:n]

    # ── Internal helpers ───────────────────────────────────────────────────

    def _deliver(self, user_id: str, signal: OrganSignal) -> Any:
        """Deliver ``signal`` for a single user and update last_active."""
        self._registry.touch(user_id)

        if self._bus is not None:
            try:
                records = self._bus.emit(signal)
                return [
                    {
                        "receiver": r.receiver,
                        "success": r.success,
                        "duration_ms": r.duration_ms,
                    }
                    for r in records
                ]
            except Exception as exc:
                return {"error": str(exc)}

        # No OrganBus — direct pass
        return {"delivered": True, "user_id": user_id, "signal_id": signal.signal_id}

    def _record(
        self, signal: OrganSignal, results: dict[str, Any]
    ) -> None:
        entry: dict[str, Any] = {
            "signal_id": signal.signal_id,
            "source": signal.source,
            "signal_type": signal.signal_type,
            "payload": signal.payload,
            "priority": signal.priority,
            "ts": signal.timestamp,
            "results": results,
        }
        with self._lock:
            self._history.append(entry)
