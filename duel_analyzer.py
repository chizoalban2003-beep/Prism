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

import uuid as _uuid_mod
from dataclasses import dataclass
from typing import Optional

from decision_spectrum import DecisionBeam, Factor, SpectrumFulcrum
from sport_spectrum import ALL_SPORTS, SportDecisionModel, DuelModel


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
    zone_label:          str   = ""
    pitch_zone_norm:     float = 0.0
    defensive_press:     float = 0.0
    xg_at_location:      float = 0.0
    coupling:            float = 0.0
    predicted_winner:    str   = ""
    attacker_fulcrum:    float = 0.0
    defender_fulcrum:    float = 0.0
    model_confidence:    float = 0.0
    model_correct:       bool  = False


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
    Upgraded: uses DuelModel from sport_spectrum for physics-based predictions
    alongside the existing DuelNetwork statistics.
    """

    def __init__(self, sport: str = "Football") -> None:
        self.network = DuelNetwork()
        self.extractor = DuelExtractor()
        self.sport = sport
        self._records: list[DuelRecord] = []
        config = ALL_SPORTS.get(sport)
        self._model = SportDecisionModel(config) if config else None
        self._duel_model = DuelModel(self._model, coupling_strength=0.7) if self._model else None

    def process_match(self, events: list[dict], match_id: str) -> list[DuelRecord]:
        """Extract, predict, and store duel records."""
        records = self.extractor.extract(events, match_id)
        for rec in records:
            self._enrich(rec)
            self.network.add_record(rec)
            self._records.append(rec)
        return records

    def _enrich(self, rec: DuelRecord) -> None:
        """Add zone, normalised coords, and spectrum prediction to one DuelRecord."""
        x = rec.location_x
        rec.pitch_zone_norm = min(1.0, x / 120.0)
        rec.zone_label = ("own_third" if rec.pitch_zone_norm < 0.33
                          else "middle" if rec.pitch_zone_norm < 0.67
                          else "final_third" if rec.pitch_zone_norm < 0.88 else "box")
        rec.defensive_press = max(0.0, 1.0 - rec.location_y / 80.0)
        rec.xg_at_location = rec.pitch_zone_norm * 0.4
        config = ALL_SPORTS.get("Football")
        if config is None:
            return
        model = SportDecisionModel(config)
        duel = DuelModel(model, coupling_strength=0.65)
        names = [p.name for p in config.profiles]
        try:
            ctx = {
                "pitch_zone": rec.pitch_zone_norm,
                "press": rec.defensive_press,
                "xg": rec.xg_at_location,
            }
            result = duel.simulate(
                names[7],
                names[1],
                ctx,
                {"pitch_zone": rec.pitch_zone_norm},
            )
            rec.predicted_winner = result.advantage
            rec.attacker_fulcrum = result.attacker_fulcrum
            rec.defender_fulcrum = result.defender_fulcrum
            rec.model_confidence = max(result.attacker_activation, result.defender_activation)
            actual = ("attacker" if rec.attacker_won else
                      "defender" if rec.attacker_won is False else "draw")
            rec.model_correct = rec.predicted_winner in (actual, "contested")
        except Exception:
            pass

    def expected_outcome(
        self,
        attacker: str,
        defender: str,
        location_x: float = 60.0,
        attacker_profile: str = None,
        defender_profile: str = None,
    ) -> float:
        """
        Blends historical win rate with physics model prediction.
        Returns probability (0-1) that attacker wins.
        """
        historical = self.network.win_rate(attacker, defender)

        if (
            self._duel_model is None
            or attacker_profile is None
            or defender_profile is None
        ):
            bonus = max(0.0, (location_x - 80.0) / 80.0) * 0.05
            return min(0.99, max(0.01, historical + bonus))

        try:
            ctx = {"pitch_zone": min(1.0, location_x / 120.0), "press": 0.4}
            result = self._duel_model.simulate(attacker_profile, defender_profile, ctx, {})
            physics = (
                1.0 if result.advantage == "attacker" else
                0.0 if result.advantage == "defender" else
                0.5
            )
            total = self.network._edges.get((attacker, defender), {}).get("total", 0)
            hist_weight = min(0.6, 0.1 * total)
            blended = hist_weight * historical + (1 - hist_weight) * physics
            return min(1.0, max(0.0, blended))
        except Exception:
            return historical

    def model_accuracy(self) -> float:
        """Fraction of enriched records where model_correct is True."""
        enriched = [r for r in self._all_records() if r.predicted_winner]
        if not enriched:
            return 0.0
        return sum(1 for r in enriched if r.model_correct) / len(enriched)

    def _all_records(self) -> list[DuelRecord]:
        return list(self._records)
