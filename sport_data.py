"""
sport_data.py
=============
KDE Moment Platform — Sport Data Connectors

Provides a local-disk-cached HTTP connector for the StatsBomb open-data
repository.  Uses the Python standard library only (urllib, json, hashlib,
pathlib).

Public API
----------
StatsBombConnector — fetch + cache StatsBomb JSON files from GitHub
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

STATSBOMB_BASE = (
    "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
)

_JsonData = dict | list


class StatsBombConnector:
    """
    HTTP fetch + local disk cache for StatsBomb open-data.

    Files are stored as ``<cache_dir>/<md5(url)>.json``.
    A cached file is considered fresh for *ttl_hours* hours.
    On HTTP failure the connector falls back to a stale cache if one exists.

    Usage::

        conn   = StatsBombConnector()
        events = conn.get_match_events(3788741)
        for ev in events:
            print(ev["type"]["name"])
    """

    def __init__(
        self,
        cache_dir: str = "~/.kde/statsbomb_cache",
        ttl_hours: int = 24,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser()
        self.ttl_sec   = ttl_hours * 3600
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core fetch / cache layer
    # ------------------------------------------------------------------

    def _cache_path(self, url: str) -> Path:
        key = hashlib.md5(url.encode()).hexdigest()  # nosec B324 — cache key, not crypto
        return self.cache_dir / f"{key}.json"

    def _is_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        return (time.time() - path.stat().st_mtime) < self.ttl_sec

    def fetch(self, url: str) -> _JsonData:
        """
        Return parsed JSON from *url*.

        Checks the local cache first.  On a miss (or stale entry) fetches
        from the network and writes the result to cache.  If the network is
        unavailable and a stale cache entry exists, it is returned with a
        warning.
        """
        path = self._cache_path(url)
        if self._is_fresh(path):
            logger.debug("cache hit: %s", url)
            return json.loads(path.read_text(encoding="utf-8"))

        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                raw  = resp.read()
                data = json.loads(raw)
            path.write_text(json.dumps(data), encoding="utf-8")
            logger.debug("fetched and cached: %s", url)
            return data
        except urllib.error.URLError as exc:
            if path.exists():
                logger.warning(
                    "HTTP error (%s); returning stale cache for %s", exc, url
                )
                return json.loads(path.read_text(encoding="utf-8"))
            raise ConnectionError(f"Cannot fetch {url}: {exc}") from exc

    # ------------------------------------------------------------------
    # StatsBomb convenience methods
    # ------------------------------------------------------------------

    def get_competitions(self) -> list[dict]:
        """Return the full competitions list."""
        url    = f"{STATSBOMB_BASE}/competitions.json"
        result = self.fetch(url)
        return result if isinstance(result, list) else []

    def get_matches(self, competition_id: int, season_id: int) -> list[dict]:
        """Return all matches for a given competition and season."""
        url    = f"{STATSBOMB_BASE}/matches/{competition_id}/{season_id}.json"
        result = self.fetch(url)
        return result if isinstance(result, list) else []

    def get_match_events(self, match_id: int) -> list[dict]:
        """Return the event stream for a single match."""
        url    = f"{STATSBOMB_BASE}/events/{match_id}.json"
        result = self.fetch(url)
        return result if isinstance(result, list) else []

    def get_match_freeze_frames(self, match_id: int) -> dict[str, dict]:
        """
        Return a mapping of event_id → freeze-frame dict for every shot
        event in the match that contains a ``freeze_frame`` array.

        The returned structure per entry is::

            {
                "freeze_frame": [<player entries>],
                "event":        <the full shot event dict>,
            }
        """
        events: list[dict] = self.get_match_events(match_id)
        frames: dict[str, dict] = {}
        for ev in events:
            ev_id   = ev.get("id")
            shot    = ev.get("shot", {})
            if not ev_id or not shot:
                continue
            ff = shot.get("freeze_frame")
            if ff:
                frames[str(ev_id)] = {"freeze_frame": ff, "event": ev}
        return frames

    def get_lineups(self, match_id: int) -> list[dict]:
        """Return lineups for both teams in a match."""
        url    = f"{STATSBOMB_BASE}/lineups/{match_id}.json"
        result = self.fetch(url)
        return result if isinstance(result, list) else []
