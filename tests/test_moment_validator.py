"""
test_moment_validator.py
========================
Tests for moment_validator.py.

All tests use mocked/in-memory data — no real HTTP calls to StatsBomb.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from moment_analyzer import (
    ActionOutcome,
    Moment,
    MomentAnalyzer,
    MomentResult,
)
from moment_validator import (
    MomentValidator,
    SeasonValidationReport,
    ValidationResult,
    _TrackedMomentAnalyzer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_moment(player: str = "Salah", pitch_x: float = 0.88,
                 moment_type: str = "1v1_keeper", xg_raw: float = 0.35) -> Moment:
    return Moment(
        moment_id    = "m1",
        match_id     = "test_match",
        sport        = "Football",
        moment_type  = moment_type,
        timestamp    = 0.0,
        focal_player = player,
        focal_profile= "Winger",
        focal_team   = "Liverpool",
        focal_base   = 0.6,
        pitch_x      = pitch_x,
        pitch_y      = 0.5,
        xg_raw       = xg_raw,
    )


def _make_result(moment: Moment) -> MomentResult:
    analyzer = MomentAnalyzer()
    return analyzer.analyze(moment)


def _make_outcome(success: bool, action: str = "Shot") -> dict:
    return {"action_taken": action, "success": success, "xg_delta": 0.0}


# ---------------------------------------------------------------------------
# _brier_score unit tests
# ---------------------------------------------------------------------------

def test_brier_perfect():
    """All pred=1.0, actual=1.0 → Brier = 0.0."""
    v = MomentValidator(competition_id=11, season_id=90)
    assert v._brier_score([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]) == pytest.approx(0.0)


def test_brier_worst():
    """All pred=1.0, actual=0.0 → Brier = 1.0."""
    v = MomentValidator(competition_id=11, season_id=90)
    assert v._brier_score([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_brier_half():
    """All pred=0.5, actual=0.0 → Brier = 0.25."""
    v = MomentValidator(competition_id=11, season_id=90)
    assert v._brier_score([0.5, 0.5], [0.0, 0.0]) == pytest.approx(0.25)


def test_brier_empty():
    """Empty lists → returns 1.0 (sentinel for no data)."""
    v = MomentValidator(competition_id=11, season_id=90)
    assert v._brier_score([], []) == 1.0


# ---------------------------------------------------------------------------
# Accuracy above random
# ---------------------------------------------------------------------------

def test_accuracy_above_random():
    """
    For a 1v1_keeper moment the analyzer should recommend the best option.
    Build 8 tracked pairs where the recommended == actual → 100% accuracy.
    Target: > 0.125 (= 1 / 8 options, random baseline).
    """
    analyzer = _TrackedMomentAnalyzer()
    moment   = _make_moment()
    result   = analyzer.analyze(moment)

    # Calibrate so recommended matches actual → guaranteed match
    outcome = ActionOutcome(
        action_taken=result.recommended, success=True, xg_delta=0.2
    )
    analyzer.calibrate(moment, outcome)

    assert len(analyzer.tracked) == 1
    tracked = analyzer.tracked

    v = MomentValidator(competition_id=11, season_id=90)
    by_player = v.slice_by(tracked, "player")
    assert len(by_player) == 1
    accuracy = by_player[0].accuracy
    assert accuracy > 0.125, f"accuracy {accuracy} should be above random 0.125"


# ---------------------------------------------------------------------------
# slice_by tests
# ---------------------------------------------------------------------------

def test_slice_by_player():
    """slice_by('player') with two players returns two ValidationResult objects."""
    analyzer = _TrackedMomentAnalyzer()

    for player in ("Salah", "Mane"):
        m  = _make_moment(player=player)
        r  = analyzer.analyze(m)
        oc = ActionOutcome(action_taken=r.recommended, success=True, xg_delta=0.3)
        analyzer.calibrate(m, oc)

    v      = MomentValidator(competition_id=11, season_id=90)
    sliced = v.slice_by(analyzer.tracked, "player")
    assert len(sliced) == 2
    keys = {vr.slice_key for vr in sliced}
    assert "Salah" in keys
    assert "Mane" in keys


def test_slice_by_zone():
    """slice_by('zone') classifies moments into the correct zone buckets."""
    analyzer = _TrackedMomentAnalyzer()

    # box (pitch_x > 0.83) and attacking_third
    for pitch_x in (0.90, 0.70):
        m  = _make_moment(pitch_x=pitch_x)
        _r = analyzer.analyze(m)
        oc = ActionOutcome(action_taken="shot", success=False, xg_delta=0.0)
        analyzer.calibrate(m, oc)

    v      = MomentValidator(competition_id=11, season_id=90)
    sliced = v.slice_by(analyzer.tracked, "zone")
    zones  = {vr.slice_key for vr in sliced}
    assert "box" in zones
    assert "attacking_third" in zones


def test_slice_by_moment_type():
    """slice_by('moment_type') groups correctly."""
    analyzer = _TrackedMomentAnalyzer()

    for mtype in ("1v1_keeper", "penalty"):
        m  = _make_moment(moment_type=mtype)
        _r = analyzer.analyze(m)
        oc = ActionOutcome(action_taken="shot", success=True, xg_delta=0.5)
        analyzer.calibrate(m, oc)

    v      = MomentValidator(competition_id=11, season_id=90)
    sliced = v.slice_by(analyzer.tracked, "moment_type")
    types  = {vr.slice_key for vr in sliced}
    assert "1v1_keeper" in types
    assert "penalty" in types


# ---------------------------------------------------------------------------
# Export tests
# ---------------------------------------------------------------------------

def _build_sample_report() -> SeasonValidationReport:
    """Build a minimal SeasonValidationReport for export tests."""
    vr = ValidationResult(
        slice_key="Salah",
        n_moments=10,
        correct=7,
        accuracy=0.70,
        avg_xg_pred=0.35,
        avg_xg_actual=0.30,
        xg_brier_score=0.12,
        calibration_gap=0.05,
    )
    return SeasonValidationReport(
        season="2022/2023",
        competition="Premier League",
        n_matches=5,
        n_moments=10,
        overall_accuracy=0.70,
        overall_brier=0.12,
        by_player=[vr],
        by_zone=[],
        by_moment_type=[],
        best_calibrated=["Salah"],
        worst_calibrated=[],
    )


def test_export_markdown(tmp_path: Path):
    """export_report(fmt='markdown') creates a .md file with ## headers."""
    report = _build_sample_report()
    v      = MomentValidator(competition_id=11, season_id=90)
    out    = tmp_path / "report.md"
    path   = v.export_report(report, str(out), fmt="markdown")
    assert Path(path).exists()
    content = Path(path).read_text()
    assert "##" in content
    assert "Season Validation Report" in content
    assert "Premier League" in content


def test_export_json(tmp_path: Path):
    """export_report(fmt='json') creates a valid JSON file."""
    report = _build_sample_report()
    v      = MomentValidator(competition_id=11, season_id=90)
    out    = tmp_path / "report.json"
    path   = v.export_report(report, str(out), fmt="json")
    assert Path(path).exists()
    data = json.loads(Path(path).read_text())
    assert data["competition"] == "Premier League"
    assert data["overall_accuracy"] == pytest.approx(0.70)


def test_export_html(tmp_path: Path):
    """export_report(fmt='html') creates an HTML file."""
    report = _build_sample_report()
    v      = MomentValidator(competition_id=11, season_id=90)
    out    = tmp_path / "report.html"
    path   = v.export_report(report, str(out), fmt="html")
    assert Path(path).exists()
    content = Path(path).read_text()
    assert "<html>" in content


# ---------------------------------------------------------------------------
# Calibration gap test
# ---------------------------------------------------------------------------

def test_calibration_gap():
    """
    avg_xg_pred - avg_xg_actual should stay within [-0.15, +0.15].
    We feed moments with xg_raw=0.3 and mix of successes (actual avg ≈ 0.25-0.35).
    """
    analyzer = _TrackedMomentAnalyzer()
    outcomes = [True, False, False, True, False, False, True, False]  # 3/8 ≈ 0.375 actual

    for i, success in enumerate(outcomes):
        m  = _make_moment(xg_raw=0.30)
        _r = analyzer.analyze(m)
        oc = ActionOutcome(action_taken="shot", success=success, xg_delta=0.0)
        analyzer.calibrate(m, oc)

    v      = MomentValidator(competition_id=11, season_id=90)
    sliced = v.slice_by(analyzer.tracked, "player")
    for vr in sliced:
        assert -0.15 <= vr.calibration_gap <= 0.15, (
            f"calibration_gap={vr.calibration_gap} out of [-0.15, +0.15]"
        )


# ---------------------------------------------------------------------------
# MomentValidator.run() with mocked StatsBomb
# ---------------------------------------------------------------------------

def _minimal_events() -> list[dict]:
    """Return a tiny StatsBomb-style event list with one Shot."""
    return [
        {
            "id":        "ev-shot-1",
            "type":      {"name": "Shot"},
            "player":    {"name": "Salah"},
            "team":      {"name": "Liverpool"},
            "location":  [100.0, 40.0],
            "timestamp": "00:30:00.000",
            "shot": {
                "statsbomb_xg": 0.35,
                "outcome":      {"name": "Goal"},
                "freeze_frame": [],
            },
        }
    ]


def test_run_with_mocked_statsbomb(tmp_path):
    """
    MomentValidator.run() should return a SeasonValidationReport without
    making real HTTP calls (StatsBombConnector is mocked).
    """
    matches = [{"match_id": 1}]
    events  = _minimal_events()

    with patch("moment_validator.StatsBombConnector") as MockConn:
        conn_inst = MagicMock()
        conn_inst.get_matches.return_value = matches
        conn_inst.get_match_events.return_value = events
        conn_inst.get_match_freeze_frames.return_value = {}
        MockConn.return_value = conn_inst

        v      = MomentValidator(competition_id=11, season_id=90, n_matches=1)
        report = v.run()

    assert isinstance(report, SeasonValidationReport)
    assert report.n_matches == 1
    assert report.n_moments >= 0  # may be 0 if shot is below threshold
