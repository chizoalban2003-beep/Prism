"""
tests/test_moment_pipeline.py
=============================
Tests for moment_pipeline.py — batch (StatsBomb) and live pipelines.
"""

from __future__ import annotations

import math
from typing import Optional

from moment_analyzer import MomentAnalyzer
from moment_pipeline import (
    LiveMomentPipeline,
    StatsBombMomentPipeline,
)

# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------

def _make_shot_event(
    event_id: str = "evt-001",
    player:   str = "Test Striker",
    team:     str = "Team A",
    x: float = 100.0,
    y: float = 40.0,
    outcome:  str = "Off T",
    xg:       float = 0.18,
    freeze_frame: Optional[list] = None,
) -> dict:
    shot: dict = {
        "outcome":      {"name": outcome},
        "statsbomb_xg": xg,
    }
    if freeze_frame is not None:
        shot["freeze_frame"] = freeze_frame

    return {
        "id":        event_id,
        "type":      {"name": "Shot"},
        "timestamp": "00:35:12.000",
        "player":    {"name": player},
        "team":      {"name": team},
        "location":  [x, y],
        "shot":      shot,
    }


def _make_pipeline(
    profile_map: Optional[dict] = None,
    sport: str = "Football",
) -> StatsBombMomentPipeline:
    return StatsBombMomentPipeline(
        analyzer=MomentAnalyzer(),
        profile_map=profile_map or {"Test Striker": "Striker"},
        sport=sport,
    )


def _freeze_entry(
    name: str,
    x:    float,
    y:    float,
    teammate: bool = False,
    position: str  = "Defender",
) -> dict:
    return {
        "location":  [x, y],
        "player":    {"name": name},
        "position":  {"name": position},
        "teammate":  teammate,
    }


# ---------------------------------------------------------------------------
# test_shot_builds_moment
# ---------------------------------------------------------------------------

def test_shot_builds_moment():
    """A StatsBomb Shot event with x > 85 must produce a Moment with pitch_x > 0."""
    pipe  = _make_pipeline()
    ev    = _make_shot_event(x=100.0, y=40.0, freeze_frame=[])
    results = pipe.process_match([ev], {}, match_id="m1")
    assert len(results) >= 1
    moment = results[0].moment
    assert moment.pitch_x > 0, f"Expected pitch_x > 0, got {moment.pitch_x}"
    assert moment.sport == "Football"
    assert moment.moment_type == "1v1_keeper"


# ---------------------------------------------------------------------------
# test_gk_from_freeze_frame
# ---------------------------------------------------------------------------

def test_gk_from_freeze_frame():
    """A freeze frame containing a Goalkeeper entry must set primary_opponent.name."""
    gk_name = "Manuel Neuer"
    ff = [
        _freeze_entry(gk_name, 119.0, 40.0, teammate=False, position="Goalkeeper"),
        _freeze_entry("Defender 1", 108.0, 38.0, teammate=False, position="Defender"),
    ]
    ev      = _make_shot_event(x=100.0, y=40.0, freeze_frame=ff)
    pipe    = _make_pipeline()
    results = pipe.process_match([ev], {str(ev["id"]): {"freeze_frame": ff}}, "m1")
    assert results, "Expected at least one MomentResult"
    primary = results[0].moment.primary_opponent
    assert primary is not None, "primary_opponent should be set"
    assert primary.name == gk_name, (
        f"Expected GK '{gk_name}', got '{primary.name}'"
    )


# ---------------------------------------------------------------------------
# test_defenders_sorted_nearest_first
# ---------------------------------------------------------------------------

def test_defenders_sorted_nearest_first():
    """secondary_opponents must be sorted by ascending distance."""
    ff = [
        _freeze_entry("Goalkeeper",  119.0, 40.0, teammate=False, position="Goalkeeper"),
        _freeze_entry("Far Defender",  98.0, 50.0, teammate=False),
        _freeze_entry("Near Defender", 92.0, 42.0, teammate=False),
    ]
    ev      = _make_shot_event(x=88.0, y=40.0, freeze_frame=ff)
    pipe    = _make_pipeline()
    results = pipe.process_match([ev], {str(ev["id"]): {"freeze_frame": ff}}, "m1")
    assert results
    secondary = results[0].moment.secondary_opponents
    assert len(secondary) >= 2, "Expected at least 2 secondary opponents"
    assert secondary[0].distance <= secondary[1].distance, (
        f"secondary_opponents not sorted: {secondary[0].distance} > {secondary[1].distance}"
    )


# ---------------------------------------------------------------------------
# test_arrival_time
# ---------------------------------------------------------------------------

def test_arrival_time():
    """
    A defender 10 yards away from the shot location at the default speed
    (7.5 yards/s) should have arrival_time ≈ 1.33 s.
    """
    # Shot at (100, 40). Defender at (110, 40) → distance = 10 yards.
    ff = [
        _freeze_entry("Defender", 110.0, 40.0, teammate=False),
    ]
    ev      = _make_shot_event(x=100.0, y=40.0, freeze_frame=ff)
    pipe    = _make_pipeline()
    results = pipe.process_match([ev], {str(ev["id"]): {"freeze_frame": ff}}, "m1")
    assert results
    # No GK in freeze frame; nearest defender becomes primary
    moment = results[0].moment
    player = (
        moment.primary_opponent
        or (moment.secondary_opponents[0] if moment.secondary_opponents else None)
    )
    assert player is not None, "Expected a NearbyPlayer"
    expected = 10.0 / 7.5   # ≈ 1.3333
    assert math.isclose(player.arrival_time, expected, rel_tol=0.01), (
        f"Expected arrival_time ≈ {expected:.4f}, got {player.arrival_time:.4f}"
    )


# ---------------------------------------------------------------------------
# test_goal_outcome_calibrates_success
# ---------------------------------------------------------------------------

def test_goal_outcome_calibrates_success():
    """
    A Shot event with outcome 'Goal' must result in ActionOutcome.success=True
    being passed to calibrate(), raising the player's success count.
    """
    player_name = "Goal Scorer"
    ev      = _make_shot_event(
        player=player_name,
        x=100.0, y=40.0,
        outcome="Goal",
        xg=0.35,
        freeze_frame=[],
    )
    analyzer = MomentAnalyzer()
    pipe     = StatsBombMomentPipeline(
        analyzer=analyzer,
        profile_map={player_name: "Striker"},
    )
    pipe.process_match([ev], {}, "m1", record_outcomes=True)

    stats = analyzer.player_stats(player_name)
    assert stats["success"] >= 1, (
        f"Expected success >= 1 after a 'Goal' outcome, got {stats}"
    )


# ---------------------------------------------------------------------------
# test_process_match_returns_list
# ---------------------------------------------------------------------------

def test_process_match_returns_list():
    """process_match with a valid shot event must return a non-empty list."""
    events = [_make_shot_event(x=100.0, y=40.0, freeze_frame=[])]
    pipe   = _make_pipeline()
    results = pipe.process_match(events, {}, "m1")
    assert isinstance(results, list), "Expected a list"
    assert len(results) > 0, "Expected non-empty list of MomentResults"
    from moment_analyzer import MomentResult
    assert all(isinstance(r, MomentResult) for r in results)


# ---------------------------------------------------------------------------
# LiveMomentPipeline tests
# ---------------------------------------------------------------------------

def _make_frame(
    name:     str,
    x:        float,
    y:        float,
    has_ball: bool,
    team:     str,
    opp_x:   Optional[float] = None,
    opp_y:   Optional[float] = None,
    opp_team: str = "Team B",
    ts:       float = 0.0,
) -> dict:
    players = [
        {
            "name":     name,
            "team":     team,
            "x":        x,
            "y":        y,
            "speed":    5.0,
            "has_ball": has_ball,
        }
    ]
    if opp_x is not None:
        players.append({
            "name":     "Opponent",
            "team":     opp_team,
            "x":        opp_x,
            "y":        opp_y if opp_y is not None else y,
            "speed":    4.0,
            "has_ball": False,
        })
    return {"timestamp": ts, "players": players}


def test_live_detects_entering_zone():
    """
    Feeding MIN_FRAMES+1 frames with a ball carrier at x=80 m (pitch_x≈0.76)
    and an opponent within 5 m must emit a MomentResult.
    """
    analyzer = MomentAnalyzer()
    pipe     = LiveMomentPipeline(analyzer, sport="Football")

    emitted: list = []
    pipe.on_moment(emitted.append)

    # Ball carrier at x=80m, opponent at x=82m (distance ≈ 2m) → inside zone
    # Feed MIN_FRAMES frames so the pipeline accumulates enough to fire.
    n_frames = LiveMomentPipeline.MIN_FRAMES
    result   = None
    for i in range(n_frames):
        r = pipe.feed_frame(
            _make_frame(
                name="Striker", x=80.0, y=34.0, has_ball=True, team="A",
                opp_x=82.0, opp_y=34.0,
                ts=float(i) / 25.0,
            )
        )
        if r is not None:
            result = r

    assert result is not None, "Expected a MomentResult after MIN_FRAMES frames"
    assert len(emitted) >= 1, "Callback must have been called"
    assert result.moment.pitch_x > LiveMomentPipeline.DETECTION_THRESHOLD_X


def test_live_no_moment_at_own_end():
    """
    A player at x=20 m (own half, pitch_x≈0.19) must never trigger a moment,
    regardless of opponent proximity.
    """
    analyzer = MomentAnalyzer()
    pipe     = LiveMomentPipeline(analyzer, sport="Football")

    emitted: list = []
    pipe.on_moment(emitted.append)

    n_frames = LiveMomentPipeline.MIN_FRAMES + 5
    for i in range(n_frames):
        result = pipe.feed_frame(
            _make_frame(
                name="Defender", x=20.0, y=34.0, has_ball=True, team="A",
                opp_x=21.0, opp_y=34.0,
                ts=float(i) / 25.0,
            )
        )
        assert result is None, (
            f"Unexpected moment at own end (frame {i}, "
            f"pitch_x={20.0/105.0:.3f})"
        )

    assert len(emitted) == 0, "No callbacks expected for own-half possession"
