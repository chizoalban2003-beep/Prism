"""
duel_analyzer.py
================
KDE Moment Platform — 1v1 Duel Analysis

Provides tools for extracting, storing, and analysing head-to-head duel
data from match events (compatible with StatsBomb format).

Public API
----------
DuelRecord    — snapshot of a single 1v1 duel
DuelExtractor — parses StatsBomb-style event lists → DuelRecord objects
DuelNetwork   — historical matchup network between players
DuelAnalyzer  — high-level analyser: processes matches and evaluates duels
"""

from __future__ import annotations

import math
import uuid as _uuid_mod
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# DuelRecord
# ---------------------------------------------------------------------------

@dataclass
class DuelRecord:
    """Snapshot of a single 1v1 duel extracted from a match event."""
    duel_id:       str
    match_id:      str
    timestamp:     float       # seconds from kick-off
    attacker:      str         # player who initiated the duel
    defender:      str         # opposing player
    attacker_team: str
    defender_team: str
    location_x:   float        # pitch x (StatsBomb yards: 0–120)
    location_y:   float        # pitch y (StatsBomb yards: 0–80)
    attacker_won:  Optional[bool]   # True / False / None (inconclusive)
    duel_type:     str          # "ground", "aerial", "tackle", "interception"
    duration:      float = 0.0  # seconds (if available)
    notes:         str   = ""


# ---------------------------------------------------------------------------
# DuelExtractor
# ---------------------------------------------------------------------------

class DuelExtractor:
    """Parses StatsBomb event dicts and emits DuelRecord objects."""

    DUEL_EVENT_TYPES = {"Duel", "Tackle", "Interception"}

    # StatsBomb outcome names that indicate the acting player won/lost
    _WIN_TERMS  = {"won", "success", "complete", "goal"}
    _LOSE_TERMS = {"lost", "fail", "incomplete", "out", "blocked"}

    def extract(self, events: list[dict], match_id: str) -> list[DuelRecord]:
        """Return a DuelRecord for every applicable event in *events*."""
        records: list[DuelRecord] = []
        for ev in events:
            ev_type = ev.get("type", {})
            if isinstance(ev_type, dict):
                type_name = ev_type.get("name", "")
            else:
                type_name = str(ev_type)
            if type_name not in self.DUEL_EVENT_TYPES:
                continue
            rec = self._parse_event(ev, match_id)
            if rec is not None:
                records.append(rec)
        return records

    def _parse_event(self, event: dict, match_id: str) -> Optional[DuelRecord]:
        player_info  = event.get("player", {})
        team_info    = event.get("team", {})
        loc          = event.get("location", [0.0, 0.0])
        duel_data    = event.get("duel", {})

        player_name  = player_info.get("name", "Unknown") if isinstance(player_info, dict) else str(player_info)
        team_name    = team_info.get("name", "Unknown")   if isinstance(team_info,   dict) else str(team_info)

        # Counterpart (defender) name
        counterpart = duel_data.get("counterpart", {}) if isinstance(duel_data, dict) else {}
        opponent    = counterpart.get("name", "Unknown") if isinstance(counterpart, dict) else str(counterpart)
        opponent_team = counterpart.get("team", {})
        opponent_team_name = (
            opponent_team.get("name", "")
            if isinstance(opponent_team, dict) else str(opponent_team)
        )

        # Duel type
        duel_type_raw = duel_data.get("type", {}) if isinstance(duel_data, dict) else {}
        duel_type_name = (
            duel_type_raw.get("name", "ground")
            if isinstance(duel_type_raw, dict) else str(duel_type_raw)
        ).lower()

        # Outcome
        outcome_raw = duel_data.get("outcome", {}) if isinstance(duel_data, dict) else {}
        outcome_name = (
            outcome_raw.get("name", "")
            if isinstance(outcome_raw, dict) else str(outcome_raw)
        ).lower()

        won: Optional[bool] = None
        if any(t in outcome_name for t in self._WIN_TERMS):
            won = True
        elif any(t in outcome_name for t in self._LOSE_TERMS):
            won = False

        # Timestamp: StatsBomb uses "HH:MM:SS.mmm" strings or plain floats
        raw_ts = event.get("timestamp", 0.0)
        timestamp = self._parse_timestamp(raw_ts)

        return DuelRecord(
            duel_id       = str(_uuid_mod.uuid4()),
            match_id      = match_id,
            timestamp     = timestamp,
            attacker      = player_name,
            defender      = opponent,
            attacker_team = team_name,
            defender_team = opponent_team_name,
            location_x    = float(loc[0]) if len(loc) > 0 else 0.0,
            location_y    = float(loc[1]) if len(loc) > 1 else 0.0,
            attacker_won  = won,
            duel_type     = duel_type_name,
        )

    @staticmethod
    def _parse_timestamp(raw) -> float:
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, str) and ":" in raw:
            # "HH:MM:SS.mmm"
            parts = raw.split(":")
            try:
                h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
                return h * 3600.0 + m * 60.0 + s
            except (IndexError, ValueError):
                pass
        return 0.0


# ---------------------------------------------------------------------------
# DuelNetwork
# ---------------------------------------------------------------------------

class DuelNetwork:
    """
    Historical matchup network between players.

    Stores an edge (attacker, defender) → win/total counters and provides
    head-to-head win-rate lookups.  Unknown matchups default to 0.50.
    """

    def __init__(self) -> None:
        # (attacker_name, defender_name) → {"total": int, "won": int}
        self._edges: dict[tuple[str, str], dict[str, int]] = {}

    def add_record(self, rec: DuelRecord) -> None:
        """Incorporate one DuelRecord into the network."""
        key = (rec.attacker, rec.defender)
        if key not in self._edges:
            self._edges[key] = {"total": 0, "won": 0}
        self._edges[key]["total"] += 1
        if rec.attacker_won:
            self._edges[key]["won"] += 1

    def win_rate(self, attacker: str, defender: str) -> float:
        """Return probability that *attacker* wins against *defender* (0–1)."""
        key = (attacker, defender)
        e = self._edges.get(key)
        if not e or e["total"] == 0:
            return 0.50
        return e["won"] / e["total"]

    def head_to_head(self, attacker: str, defender: str) -> dict:
        """Return a summary dict for the (attacker, defender) matchup."""
        key = (attacker, defender)
        e = self._edges.get(key, {"total": 0, "won": 0})
        total = e["total"]
        won   = e["won"]
        return {
            "attacker":  attacker,
            "defender":  defender,
            "total":     total,
            "won":       won,
            "lost":      total - won,
            "win_rate":  won / total if total > 0 else 0.50,
        }

    def player_attack_stats(self, player: str) -> dict:
        """Return aggregate attacking-duel stats for *player*."""
        total = won = 0
        for (att, _), edge in self._edges.items():
            if att == player:
                total += edge["total"]
                won   += edge["won"]
        return {
            "player":    player,
            "total":     total,
            "won":       won,
            "lost":      total - won,
            "win_rate":  won / total if total > 0 else 0.50,
        }


# ---------------------------------------------------------------------------
# DuelAnalyzer
# ---------------------------------------------------------------------------

class DuelAnalyzer:
    """
    High-level analyser that processes match events into DuelRecord objects
    and maintains a DuelNetwork for win-probability queries.

    Usage::

        analyzer = DuelAnalyzer()
        records  = analyzer.process_match(events, "match-42")
        prob     = analyzer.expected_outcome("Messi", "Boateng", location_x=105.0)
    """

    def __init__(self) -> None:
        self.network   = DuelNetwork()
        self.extractor = DuelExtractor()

    def process_match(
        self,
        events:   list[dict],
        match_id: str,
    ) -> list[DuelRecord]:
        """Extract duel records from *events* and add them to the network."""
        records = self.extractor.extract(events, match_id)
        for rec in records:
            self.network.add_record(rec)
        return records

    def expected_outcome(
        self,
        attacker:   str,
        defender:   str,
        location_x: float = 60.0,
    ) -> float:
        """
        Return the probability (0–1) that *attacker* wins this duel.

        Adds a small position bonus when deep in the attacking third
        (x > 80 yards) where attackers historically perform better.
        """
        base   = self.network.win_rate(attacker, defender)
        bonus  = max(0.0, (location_x - 80.0) / 80.0) * 0.05
        return min(0.99, max(0.01, base + bonus))
