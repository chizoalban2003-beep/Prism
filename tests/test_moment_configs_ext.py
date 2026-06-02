"""
tests/test_moment_configs_ext.py
================================
Tests for moment_configs_ext.py — extended sport moment configurations.
"""

from __future__ import annotations

import pytest

import moment_configs_ext as ext
from moment_analyzer import ALL_MOMENT_CONFIGS, Moment, MomentAnalyzer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def registered():
    """Register extended configs before each test and clean up after."""
    # Store original keys so we can remove extended ones after the test
    before = set(ALL_MOMENT_CONFIGS.keys())
    ext.register_extended_configs()
    yield
    # Teardown: remove any keys that weren't present originally
    for key in list(ALL_MOMENT_CONFIGS.keys()):
        if key not in before:
            del ALL_MOMENT_CONFIGS[key]


def _make_moment(sport: str, moment_type: str, **kwargs) -> Moment:
    defaults = dict(
        moment_id="m1",
        match_id="match1",
        sport=sport,
        moment_type=moment_type,
        timestamp=0.0,
        focal_player="Test Player",
        focal_profile="Forward",
        focal_team="Team A",
        focal_base=0.50,
        pitch_x=0.70,
        pitch_y=0.50,
        fatigue=0.0,
        confidence=0.50,
        score_pressure=0.0,
        xg_raw=0.0,
        custom_factors={},
    )
    defaults.update(kwargs)
    return Moment(**defaults)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

def test_register_extended_configs():
    """All 11 extended keys must be present after registration."""
    expected = {
        ("Tennis",      "serve_deuce"),
        ("Tennis",      "serve_ad"),
        ("Tennis",      "baseline_rally"),
        ("Tennis",      "net_approach"),
        ("Basketball",  "pick_roll_read"),
        ("Basketball",  "post_up"),
        ("Basketball",  "fast_break"),
        ("Rugby Union", "ball_carrier_contact"),
        ("Rugby Union", "breakdown_contest"),
        ("Rugby Union", "lineout_call"),
        ("Cricket",     "batting_delivery"),
    }
    for key in expected:
        assert key in ALL_MOMENT_CONFIGS, f"Missing config: {key}"


def test_option_positions_ascending():
    """options[0].position < options[-1].position for every extended config."""
    for key, cfg in ext.EXTENDED_CONFIGS.items():
        positions = [o.position for o in cfg.options]
        assert positions[0] < positions[-1], (
            f"{key}: first position {positions[0]} is not less than "
            f"last {positions[-1]}"
        )


def test_bandwidth_in_range():
    """0.12 <= bandwidth <= 0.22 for every extended config."""
    for key, cfg in ext.EXTENDED_CONFIGS.items():
        assert 0.12 <= cfg.bandwidth <= 0.22, (
            f"{key}: bandwidth {cfg.bandwidth} outside [0.12, 0.22]"
        )


def test_base_probs_valid():
    """All base_prob values must be in [0.20, 0.98]."""
    for key, cfg in ext.EXTENDED_CONFIGS.items():
        for opt in cfg.options:
            assert 0.20 <= opt.base_prob <= 0.98, (
                f"{key} / {opt.name!r}: base_prob {opt.base_prob} "
                f"outside [0.20, 0.98]"
            )


def test_option_count():
    """Every extended config must have between 6 and 10 options."""
    for key, cfg in ext.EXTENDED_CONFIGS.items():
        n = len(cfg.options)
        assert 6 <= n <= 10, f"{key}: {n} options (expected 6–10)"


# ---------------------------------------------------------------------------
# Decision tests
# ---------------------------------------------------------------------------

def test_tennis_serve_flat():
    """
    Aggressive player + open court → 'Flat down T' must be in the top 2
    recommended options for the serve_deuce config.
    """
    analyzer = MomentAnalyzer()
    moment   = _make_moment(
        sport="Tennis",
        moment_type="serve_deuce",
        focal_base=0.55,    # moderately aggressive player
        confidence=0.70,
        custom_factors={"open_court": True},
    )
    result = analyzer.analyze(moment)
    # Sort options by score descending and get top-2 names
    top2 = sorted(result.option_scores, key=result.option_scores.__getitem__, reverse=True)[:2]
    assert "Flat down T" in top2, (
        f"'Flat down T' not in top 2. Top 2: {top2}. "
        f"Focal position: {result.focal_position:.3f}. "
        f"Scores: {result.option_scores}"
    )


def test_basketball_pick_roll():
    """
    Open roll man + low shot clock → 'Hit roll man alley-oop' must be in top 2.
    """
    analyzer = MomentAnalyzer()
    moment   = _make_moment(
        sport="Basketball",
        moment_type="pick_roll_read",
        focal_base=0.50,
        confidence=0.70,
        score_pressure=0.40,    # urgency = low shot clock
        custom_factors={"open_roll_man": True},
    )
    result = analyzer.analyze(moment)
    top2 = sorted(result.option_scores, key=result.option_scores.__getitem__, reverse=True)[:2]
    assert "Hit roll man alley-oop" in top2, (
        f"'Hit roll man alley-oop' not in top 2. Top 2: {top2}. "
        f"Focal position: {result.focal_position:.3f}. "
        f"Scores: {result.option_scores}"
    )


def test_rugby_carrier_offload():
    """
    Good support + opponent 22m area + fast defensive line →
    'Offload in tackle' must be in the top 3 options.
    """
    analyzer = MomentAnalyzer()
    moment   = _make_moment(
        sport="Rugby Union",
        moment_type="ball_carrier_contact",
        focal_base=0.50,
        confidence=0.75,
        score_pressure=0.30,    # urgency from fast defensive line
        pitch_x=0.78,           # inside opponent 22m area
        custom_factors={"support_close": True},
    )
    result = analyzer.analyze(moment)
    top3 = sorted(result.option_scores, key=result.option_scores.__getitem__, reverse=True)[:3]
    assert "Offload in tackle" in top3, (
        f"'Offload in tackle' not in top 3. Top 3: {top3}. "
        f"Focal position: {result.focal_position:.3f}. "
        f"Scores: {result.option_scores}"
    )
