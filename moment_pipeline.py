"""
moment_pipeline.py
==================
KDE Moment Platform — Live + Batch Data Pipelines

Connects MomentAnalyzer to real data sources and produces MomentResult
objects from StatsBomb batch events or from live 25 Hz tracking frames.

Public API
----------
PipelineStats            — counters collected during a pipeline run
StatsBombMomentPipeline  — batch pipeline for StatsBomb JSON event files
LiveMomentPipeline       — real-time pipeline for 25 Hz tracking frames
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

from duel_analyzer import DuelAnalyzer, DuelExtractor
from moment_analyzer import (
    ActionOutcome,
    Moment,
    MomentAnalyzer,
    MomentResult,
    NearbyPlayer,
)

logger = logging.getLogger(__name__)

# Default assumed player speed for arrival-time calculation (yards per second)
_DEFAULT_SPEED_YDS = 7.5
# Default pitch dimensions for live tracking normalisation (metres)
_PITCH_LENGTH_M = 105.0
_PITCH_WIDTH_M  = 68.0


# ---------------------------------------------------------------------------
# PipelineStats
# ---------------------------------------------------------------------------

@dataclass
class PipelineStats:
    moments_processed: int   = 0
    moments_analyzed:  int   = 0
    calibrations:      int   = 0
    errors:            int   = 0
    elapsed_sec:       float = 0.0

    @property
    def throughput(self) -> float:
        """Moments processed per second."""
        return self.moments_processed / max(self.elapsed_sec, 0.01)


# ---------------------------------------------------------------------------
# Role → focal-base mapping
# ---------------------------------------------------------------------------

_ROLE_FOCAL_BASE: dict[str, float] = {
    "Striker":       0.65,
    "Forward":       0.65,
    "Winger":        0.60,
    "Attacking Mid": 0.58,
    "Midfielder":    0.50,
    "Wide Mid":      0.52,
    "Defender":      0.35,
    "Centre Back":   0.32,
    "Full Back":     0.38,
    "Goalkeeper":    0.20,
}

_DEFAULT_FOCAL_BASE = 0.50


def _role_to_base(role: str) -> float:
    return _ROLE_FOCAL_BASE.get(role, _DEFAULT_FOCAL_BASE)


# ---------------------------------------------------------------------------
# StatsBombMomentPipeline
# ---------------------------------------------------------------------------

class StatsBombMomentPipeline:
    """
    Batch pipeline: processes StatsBomb JSON event files and extracts
    MomentResult objects from key action events.

    Event-type → moment_type mapping
    ---------------------------------
    "Shot"    → ("Football", "1v1_keeper")   when x > SHOT_MOMENT_THRESHOLD_X
    "Dribble" → ("Football", "winger_cross") when in wide final third
                ("Football", "1v1_keeper")   when central final third
    "Carry"   → ("Football", "winger_cross") when in wide final third
    "Duel"    → extracted by DuelExtractor (not a MomentAnalyzer moment)

    Usage::

        pipeline = StatsBombMomentPipeline(
            analyzer    = MomentAnalyzer(),
            profile_map = {"Lionel Messi": "Winger"},
            sport       = "Football",
        )
        results = pipeline.process_match(events, freeze_frames, match_id="3788741")
    """

    # StatsBomb pitch: 120 × 80 yards
    SHOT_MOMENT_THRESHOLD_X = 85.0   # only shots from x > 85 yards → keeper moment
    WIDE_Y_LOW              =  8.0   # y < 8 or y > 72 → wide area
    WIDE_Y_HIGH             = 72.0
    CARRY_WIDE_X_MIN        = 85.0

    def __init__(
        self,
        analyzer:    MomentAnalyzer,
        profile_map: dict[str, str],   # player_name → role_name
        sport:       str = "Football",
    ) -> None:
        self.analyzer    = analyzer
        self.profile_map = profile_map
        self.sport       = sport
        self._duel_extractor = DuelExtractor()

    # ------------------------------------------------------------------
    # Public: process one match
    # ------------------------------------------------------------------

    def process_match(
        self,
        events:          list[dict],
        freeze_frames:   Optional[dict[str, dict]] = None,
        match_id:        str  = "unknown",
        record_outcomes: bool = True,
    ) -> list[MomentResult]:
        """
        Process all events in one StatsBomb match.

        Steps per event:
          1. Filter to action events (Shot, Dribble, Carry in wide zone).
          2. Build a Moment from event location, player, and freeze frame.
          3. Extract NearbyPlayer objects from the freeze frame.
          4. Call analyzer.analyze(moment) → MomentResult.
          5. If record_outcomes: extract the actual outcome and calibrate.
          6. Return all MomentResult objects.
        """
        if freeze_frames is None:
            freeze_frames = {}

        results: list[MomentResult] = []

        for ev in events:
            ev_type = self._event_type(ev)
            ev_id   = ev.get("id", "")
            ff_data = freeze_frames.get(str(ev_id), {})
            ff      = ff_data.get("freeze_frame", []) if isinstance(ff_data, dict) else []

            moment: Optional[Moment] = None
            try:
                if ev_type == "Shot":
                    moment = self._build_moment_from_shot(ev, ff, match_id)
                elif ev_type in ("Dribble", "Carry"):
                    moment = self._build_moment_from_dribble(ev, ff, match_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("moment build error for event %s: %s", ev_id, exc)
                continue

            if moment is None:
                continue

            try:
                result = self.analyzer.analyze(moment)
                results.append(result)

                if record_outcomes:
                    outcome = self._extract_outcome(ev)
                    if outcome is not None:
                        self.analyzer.calibrate(moment, outcome)
            except KeyError as exc:
                logger.debug("no config for moment %s: %s", moment.moment_type, exc)
            except Exception as exc:  # noqa: BLE001
                logger.warning("analysis error for event %s: %s", ev_id, exc)

        return results

    # ------------------------------------------------------------------
    # Public: process a full season
    # ------------------------------------------------------------------

    def process_season(
        self,
        match_ids:     list[int],
        cache_dir:     str = "~/.kde/statsbomb_cache",
        on_match_done: Optional[Callable[[str, list[MomentResult]], None]] = None,
    ) -> tuple[list[MomentResult], PipelineStats]:
        """
        Process multiple matches in sequence.

        Loads each match's events + freeze frames from cache or HTTP via
        StatsBombConnector.  Calls *on_match_done* after each match.
        Returns (all_results, stats).
        """
        # Lazy import to avoid circular deps when sport_data is not needed
        from sport_data import StatsBombConnector  # noqa: PLC0415

        connector   = StatsBombConnector(cache_dir=cache_dir)
        all_results: list[MomentResult] = []
        stats       = PipelineStats()
        t0          = time.monotonic()

        for mid in match_ids:
            try:
                events        = connector.get_match_events(mid)
                freeze_frames = connector.get_match_freeze_frames(mid)
                results       = self.process_match(
                    events, freeze_frames, match_id=str(mid)
                )
                all_results.extend(results)
                stats.moments_processed += len(events)
                stats.moments_analyzed  += len(results)
                if on_match_done is not None:
                    on_match_done(str(mid), results)
            except Exception as exc:  # noqa: BLE001
                logger.error("season pipeline error for match %s: %s", mid, exc)
                stats.errors += 1

        stats.elapsed_sec = time.monotonic() - t0
        return all_results, stats

    # ------------------------------------------------------------------
    # Internal: moment builders
    # ------------------------------------------------------------------

    def _build_moment_from_shot(
        self,
        event:        dict,
        freeze_frame: list,
        match_id:     str,
    ) -> Optional[Moment]:
        """Build a Football 1v1_keeper Moment from a StatsBomb Shot event."""
        loc = event.get("location", [])
        if len(loc) < 2:
            return None

        x_raw, y_raw = float(loc[0]), float(loc[1])
        if x_raw < self.SHOT_MOMENT_THRESHOLD_X:
            return None   # not deep enough for a keeper moment

        pitch_x = x_raw / 120.0
        pitch_y = y_raw / 80.0

        player      = event.get("player", {})
        team        = event.get("team", {})
        player_name = player.get("name", "Unknown") if isinstance(player, dict) else str(player)
        team_name   = team.get("name",   "Unknown") if isinstance(team,   dict) else str(team)
        role        = self.profile_map.get(player_name, "Forward")

        shot_data = event.get("shot", {})
        xg_raw    = shot_data.get("statsbomb_xg", 0.0) if isinstance(shot_data, dict) else 0.0

        opponents, teammates = self._extract_nearby_players(
            freeze_frame, [x_raw, y_raw], focal_is_teammate=True
        )

        # First opponent flagged as goalkeeper (if any) → primary
        gk = next((p for p in opponents if p.is_goalkeeper), None)
        others = [p for p in opponents if not p.is_goalkeeper]

        return Moment(
            moment_id           = str(uuid.uuid4()),
            match_id            = match_id,
            sport               = self.sport,
            moment_type         = "1v1_keeper",
            timestamp           = DuelExtractor._parse_timestamp(event.get("timestamp", 0.0)),
            focal_player        = player_name,
            focal_profile       = role,
            focal_team          = team_name,
            focal_base          = _role_to_base(role),
            pitch_x             = pitch_x,
            pitch_y             = pitch_y,
            primary_opponent    = gk,
            secondary_opponents = others,
            teammates           = teammates,
            xg_raw              = xg_raw,
        )

    def _build_moment_from_dribble(
        self,
        event:        dict,
        freeze_frame: list,
        match_id:     str,
    ) -> Optional[Moment]:
        """Build a Football Moment from a Dribble or Carry event."""
        loc = event.get("location", [])
        if len(loc) < 2:
            return None

        x_raw, y_raw = float(loc[0]), float(loc[1])
        if x_raw <= self.SHOT_MOMENT_THRESHOLD_X:
            return None   # skip mid-pitch actions

        pitch_x  = x_raw / 120.0
        pitch_y  = y_raw / 80.0
        is_wide  = y_raw < self.WIDE_Y_LOW or y_raw > self.WIDE_Y_HIGH
        mtype    = "winger_cross" if is_wide else "1v1_keeper"

        player      = event.get("player", {})
        team        = event.get("team", {})
        player_name = player.get("name", "Unknown") if isinstance(player, dict) else str(player)
        team_name   = team.get("name",   "Unknown") if isinstance(team,   dict) else str(team)
        role        = self.profile_map.get(player_name, "Winger" if is_wide else "Forward")

        opponents, teammates = self._extract_nearby_players(
            freeze_frame, [x_raw, y_raw], focal_is_teammate=True
        )
        primary = opponents[0] if opponents else None
        others  = opponents[1:]

        return Moment(
            moment_id           = str(uuid.uuid4()),
            match_id            = match_id,
            sport               = self.sport,
            moment_type         = mtype,
            timestamp           = DuelExtractor._parse_timestamp(event.get("timestamp", 0.0)),
            focal_player        = player_name,
            focal_profile       = role,
            focal_team          = team_name,
            focal_base          = _role_to_base(role),
            pitch_x             = pitch_x,
            pitch_y             = pitch_y,
            primary_opponent    = primary,
            secondary_opponents = others,
            teammates           = teammates,
        )

    def _extract_nearby_players(
        self,
        freeze_frame:      list,
        event_location:    list[float],
        focal_is_teammate: bool,
    ) -> tuple[list[NearbyPlayer], list[NearbyPlayer]]:
        """
        Parse *freeze_frame* entries into sorted (opponents, teammates) lists.

        distance     = Euclidean distance from *event_location* in yards.
        arrival_time = distance / _DEFAULT_SPEED_YDS (seconds).
        """
        ex, ey = event_location[0], event_location[1]
        opponents: list[NearbyPlayer] = []
        teammates: list[NearbyPlayer] = []

        for entry in freeze_frame:
            loc     = entry.get("location", [0.0, 0.0])
            px, py  = float(loc[0]) if len(loc) > 0 else 0.0, \
                      float(loc[1]) if len(loc) > 1 else 0.0
            dist    = math.sqrt((px - ex) ** 2 + (py - ey) ** 2)
            arrival = dist / _DEFAULT_SPEED_YDS

            player_info = entry.get("player", {})
            pname       = player_info.get("name", "Unknown") \
                          if isinstance(player_info, dict) else str(player_info)

            position_info = entry.get("position", {})
            pos_name      = position_info.get("name", "") \
                            if isinstance(position_info, dict) else str(position_info)
            is_gk = "goalkeeper" in pos_name.lower() or "gk" in pos_name.lower()

            is_teammate = bool(entry.get("teammate", False))

            np = NearbyPlayer(
                name=pname, team="", distance=dist,
                arrival_time=arrival, x=px, y=py,
                is_goalkeeper=is_gk,
            )
            if is_teammate:
                teammates.append(np)
            else:
                opponents.append(np)

        opponents.sort(key=lambda p: p.distance)
        teammates.sort(key=lambda p: p.distance)
        return opponents, teammates

    # ------------------------------------------------------------------
    # Internal: outcome extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _event_type(event: dict) -> str:
        ev_type = event.get("type", {})
        if isinstance(ev_type, dict):
            return ev_type.get("name", "")
        return str(ev_type)

    @staticmethod
    def _extract_outcome(event: dict) -> Optional[ActionOutcome]:
        """Map a StatsBomb event to an ActionOutcome for calibration."""
        ev_type = StatsBombMomentPipeline._event_type(event)

        if ev_type == "Shot":
            shot = event.get("shot", {})
            outcome_name = (
                shot.get("outcome", {}).get("name", "")
                if isinstance(shot, dict) else ""
            )
            success = outcome_name == "Goal"
            return ActionOutcome(
                action_taken=outcome_name or "shot",
                success=success,
                xg_delta=shot.get("statsbomb_xg", 0.0) if isinstance(shot, dict) else 0.0,
            )

        if ev_type == "Dribble":
            dribble = event.get("dribble", {})
            outcome_name = (
                dribble.get("outcome", {}).get("name", "")
                if isinstance(dribble, dict) else ""
            )
            success = "complete" in outcome_name.lower()
            return ActionOutcome(action_taken=outcome_name or "dribble", success=success)

        return None


# ---------------------------------------------------------------------------
# LiveMomentPipeline
# ---------------------------------------------------------------------------

class LiveMomentPipeline:
    """
    Real-time pipeline: processes live tracking frames (25 Hz) and emits a
    MomentResult when a key moment is detected.

    Detection criteria (all three must hold for MIN_FRAMES consecutive frames):
      1. A player with the ball is in the attacking zone (pitch_x > DETECTION_THRESHOLD_X).
      2. The nearest opponent is within DETECTION_DIST metres.
      3. The situation persists for ≥ MIN_FRAMES frames (0.4 s at 25 Hz).

    Frame format::

        {
            "timestamp": float,            # seconds
            "players": [{
                "name":     str,
                "team":     str,
                "x":        float,         # metres from own goal line
                "y":        float,         # metres from left touchline
                "speed":    float,         # m/s
                "has_ball": bool,
            }]
        }

    Usage::

        pipe = LiveMomentPipeline(analyzer, sport="Football")
        pipe.on_moment(lambda r: dashboard.update(r))
        for frame in tracking_stream:
            pipe.feed_frame(frame)
    """

    DETECTION_THRESHOLD_X = 0.72   # normalised pitch_x threshold
    DETECTION_DIST        = 5.0    # metres to nearest opponent
    MIN_FRAMES            = 10     # consecutive frames before emitting

    def __init__(
        self,
        analyzer:    MomentAnalyzer,
        sport:       str = "Football",
        profile_map: Optional[dict[str, str]] = None,
    ) -> None:
        self.analyzer    = analyzer
        self.sport       = sport
        self.profile_map = profile_map or {}

        self._callbacks:       list[Callable[[MomentResult], None]] = []
        self._emitted_moments: dict[str, Moment] = {}  # moment_id → Moment

        # Detection state
        self._candidate_player: Optional[str]  = None
        self._candidate_team:   Optional[str]  = None
        self._frame_count:      int            = 0
        self._last_frame:       Optional[dict] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def on_moment(self, callback: Callable[[MomentResult], None]) -> None:
        """Register a callback that receives MomentResult objects as emitted."""
        self._callbacks.append(callback)

    def feed_frame(self, frame: dict) -> Optional[MomentResult]:
        """
        Feed one tracking frame.

        Returns a MomentResult if a moment was detected and analysed;
        otherwise returns None.  Registered callbacks are also invoked.
        """
        self._last_frame = frame
        players = frame.get("players", [])

        ball_carrier = next(
            (p for p in players if p.get("has_ball", False)), None
        )
        if ball_carrier is None:
            self._reset_state()
            return None

        bx       = float(ball_carrier.get("x", 0.0))
        by       = float(ball_carrier.get("y", 0.0))
        pitch_x  = bx / _PITCH_LENGTH_M
        bname    = ball_carrier.get("name", "Unknown")
        bteam    = ball_carrier.get("team", "Unknown")

        # Zone check
        if pitch_x <= self.DETECTION_THRESHOLD_X:
            self._reset_state()
            return None

        # Nearest opponent distance check
        opponents = [
            p for p in players
            if not p.get("has_ball", False) and p.get("team") != bteam
        ]
        if not opponents:
            self._reset_state()
            return None

        def _dist(p: dict) -> float:
            ox, oy = float(p.get("x", 0.0)), float(p.get("y", 0.0))
            return math.sqrt((ox - bx) ** 2 + (oy - by) ** 2)

        nearest_opp = min(opponents, key=_dist)
        nearest_dist = _dist(nearest_opp)

        if nearest_dist > self.DETECTION_DIST:
            self._reset_state()
            return None

        # Accumulate consecutive frames
        if bname == self._candidate_player:
            self._frame_count += 1
        else:
            self._candidate_player = bname
            self._candidate_team   = bteam
            self._frame_count      = 1

        if self._frame_count < self.MIN_FRAMES:
            return None

        # Moment detected — build and analyse
        moment = self._build_live_moment(
            frame, ball_carrier, nearest_opp, opponents, players, pitch_x
        )
        self._reset_state()

        try:
            result = self.analyzer.analyze(moment)
        except KeyError as exc:
            logger.debug("live pipeline: no config for moment type: %s", exc)
            return None

        self._emitted_moments[moment.moment_id] = moment
        for cb in self._callbacks:
            try:
                cb(result)
            except Exception as exc:  # noqa: BLE001
                logger.warning("live moment callback error: %s", exc)

        return result

    def feed_calibration(
        self,
        moment_id:    str,
        action_taken: str,
        success:      bool,
    ) -> None:
        """
        Called by the operator/analyst after a moment resolves.
        Triggers MomentAnalyzer.calibrate() for the stored moment.
        """
        moment = self._emitted_moments.get(moment_id)
        if moment is None:
            logger.warning("feed_calibration: unknown moment_id %s", moment_id)
            return
        outcome = ActionOutcome(action_taken=action_taken, success=success)
        self.analyzer.calibrate(moment, outcome)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        self._candidate_player = None
        self._candidate_team   = None
        self._frame_count      = 0

    def _build_live_moment(
        self,
        frame:       dict,
        carrier:     dict,
        nearest_opp: dict,
        all_opponents: list[dict],
        all_players:   list[dict],
        pitch_x:     float,
    ) -> Moment:
        bx = float(carrier.get("x", 0.0))
        by = float(carrier.get("y", 0.0))
        pitch_y = by / _PITCH_WIDTH_M

        player_name = carrier.get("name", "Unknown")
        team_name   = carrier.get("team", "Unknown")
        role        = self.profile_map.get(player_name, "Forward")

        # Determine moment type from position on pitch
        is_wide = (pitch_y < 0.15 or pitch_y > 0.85)
        mtype   = "winger_cross" if is_wide else "1v1_keeper"

        def _make_np(p: dict) -> NearbyPlayer:
            ox, oy = float(p.get("x", 0.0)), float(p.get("y", 0.0))
            dist   = math.sqrt((ox - bx) ** 2 + (oy - by) ** 2)
            return NearbyPlayer(
                name=p.get("name", "Unknown"),
                team=p.get("team", ""),
                distance=dist,
                arrival_time=dist / _DEFAULT_SPEED_YDS,
                x=ox, y=oy,
                speed=float(p.get("speed", 0.0)),
            )

        primary_opp = _make_np(nearest_opp)
        secondary   = sorted(
            [_make_np(p) for p in all_opponents if p is not nearest_opp],
            key=lambda np: np.distance,
        )
        teammates = sorted(
            [
                _make_np(p)
                for p in all_players
                if p.get("team") == team_name and not p.get("has_ball", False)
            ],
            key=lambda np: np.distance,
        )

        return Moment(
            moment_id           = str(uuid.uuid4()),
            match_id            = "live",
            sport               = self.sport,
            moment_type         = mtype,
            timestamp           = float(frame.get("timestamp", 0.0)),
            focal_player        = player_name,
            focal_profile       = role,
            focal_team          = team_name,
            focal_base          = _role_to_base(role),
            pitch_x             = pitch_x,
            pitch_y             = pitch_y,
            primary_opponent    = primary_opp,
            secondary_opponents = secondary,
            teammates           = teammates,
        )
