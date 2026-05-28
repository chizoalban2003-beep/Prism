from __future__ import annotations

import pytest

from moment_analyzer import ActionOutcome, Moment, MomentAnalyzer, NearbyPlayer


def _make_moment() -> Moment:
    return Moment(
        moment_id="m-upgrade",
        match_id="match-1",
        sport="Football",
        moment_type="1v1_keeper",
        timestamp=12.0,
        focal_player="Haaland",
        focal_profile="Striker",
        focal_team="City",
        focal_base=0.72,
        pitch_x=0.91,
        pitch_y=0.45,
        primary_opponent=NearbyPlayer(
            name="Keeper",
            team="Away",
            distance=3.0,
            arrival_time=1.2,
            is_goalkeeper=True,
        ),
        secondary_opponents=[
            NearbyPlayer(name="CB", team="Away", distance=4.0, arrival_time=1.5),
        ],
        fatigue=0.1,
        confidence=0.8,
        score_pressure=0.2,
        xg_raw=0.42,
    )


def test_analyze_returns_activations():
    analyzer = MomentAnalyzer()
    result = analyzer.analyze(_make_moment())

    assert isinstance(result.activations, list)
    assert result.activations


def test_activations_sum_near_one():
    analyzer = MomentAnalyzer()
    result = analyzer.analyze(_make_moment())

    assert sum(act for _, act, _ in result.activations) == pytest.approx(1.0)


def test_calibrate_updates_fulcrum():
    analyzer = MomentAnalyzer()
    moment = _make_moment()
    result = analyzer.analyze(moment)
    fulcrum = analyzer._get_fulcrum(moment.focal_player, moment.moment_type)
    before = next(f.weight for f in fulcrum.factors if f.name == "_focal_anchor")

    analyzer.calibrate(
        moment,
        ActionOutcome(action_taken=result.recommended, success=True, xg_delta=0.2),
    )

    after = next(f.weight for f in fulcrum.factors if f.name == "_focal_anchor")
    assert after != before


def test_player_stats_compat():
    analyzer = MomentAnalyzer()
    moment = _make_moment()
    result = analyzer.analyze(moment)
    analyzer.calibrate(
        moment,
        ActionOutcome(action_taken=result.recommended, success=True, xg_delta=0.1),
    )

    stats = analyzer.player_stats(moment.focal_player)

    assert stats["total"] == 1
    assert stats["success"] == 1
    assert stats["success_rate"] == 1.0


def test_xg_contextual_nonzero_for_shot():
    analyzer = MomentAnalyzer()
    result = analyzer.analyze(_make_moment())

    assert result.xg_contextual > 0.0
