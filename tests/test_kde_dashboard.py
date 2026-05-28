"""
test_kde_dashboard.py
=====================
Tests for kde_dashboard.py

pytest + tmp_path. No Ollama calls.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kde_dashboard import HTMLReportGenerator, TerminalDashboard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_platform_mock():
    """Create a mock PredictionPlatform."""
    from prediction_engine import PredictionPlatform, InjuryRiskPrediction, MatchPrediction, TacticalPrediction
    platform = MagicMock(spec=PredictionPlatform)

    # injury predictor
    ip = MagicMock()
    ip.predict.return_value = MagicMock(
        spec         = InjuryRiskPrediction,
        athlete_name = "Player A",
        risk_level   = "low",
        days_to_risk = 30,
        recommendations = ["Maintain current load"],
        confidence   = 0.8,
        prediction   = "low risk",
        subject      = "Player A",
        distribution = {"low": 0.8, "moderate": 0.15, "high": 0.05},
        risk         = 0.1,
        risk_adj     = 0.1,
        fulcrum      = 0.5,
        key_factors  = [],
        expected_value = 0.1,
    )
    platform.injury = ip

    # match predictor
    mp = MagicMock()
    mp.predict.return_value = MagicMock(
        spec             = MatchPrediction,
        subject          = "Arsenal vs Chelsea",
        prediction       = "Arsenal win",
        confidence       = 0.62,
        p_home_win       = 0.55,
        p_draw           = 0.25,
        p_away_win       = 0.20,
        predicted_margin = 1.2,
        distribution     = {"home_win": 0.55, "draw": 0.25, "away_win": 0.20},
        risk             = 0.3,
        risk_adj         = 0.28,
        fulcrum          = 0.5,
        key_factors      = [("form", 0.7, "positive")],
        expected_value   = 1.0,
        home_team        = "Arsenal",
        away_team        = "Chelsea",
    )
    platform.match = mp

    platform.pre_match_brief.return_value = {
        "match_prediction": mp.predict.return_value,
        "tactical_analysis": MagicMock(
            spec                  = TacticalPrediction,
            home_team             = "Arsenal",
            away_team             = "Chelsea",
            matchup_summary       = "Tight match expected",
            home_advantage        = "Strong home form",
            away_advantage        = "Counter-attack",
            home_predicted_style  = "Possession",
            away_predicted_style  = "Low block",
            key_duels             = [],
        ),
        "squad_risk":       [],
        "squad_performance": [],
        "generated_at":     "2026-05-28T09:00:00",
    }
    return platform


def _make_plan():
    from sports_pro import DailyPlan, DailyTask
    return DailyPlan(
        primary_focus = "Speed Training",
        activation    = 0.75,
        fulcrum       = 0.52,
        tasks         = [
            DailyTask(time_slot="07:00", duration_min=30, category="warmup", title="Morning run", notes=""),
        ],
        warnings  = [],
        rationale = "Focus on speed",
    )


# ---------------------------------------------------------------------------
# HTMLReportGenerator
# ---------------------------------------------------------------------------

def test_html_report_is_valid_html():
    """HTML report output must start with <!DOCTYPE html>."""
    gen = HTMLReportGenerator(platform=_make_platform_mock())
    html = gen.pre_match_brief("Arsenal", "Chelsea")
    assert html.strip().startswith("<!DOCTYPE html>"), \
        f"Expected <!DOCTYPE html>, got: {html[:60]}"


def test_pre_match_brief_contains_teams():
    gen  = HTMLReportGenerator(platform=_make_platform_mock())
    html = gen.pre_match_brief("Arsenal", "Chelsea")
    assert "Arsenal" in html
    assert "Chelsea" in html


def test_squad_risk_overview_is_valid_html():
    gen = HTMLReportGenerator(platform=_make_platform_mock())
    squad = [
        {"name": "Player A", "recovery_score": 0.8, "load_7d": 0.4, "muscle_soreness": 0.2},
        {"name": "Player B", "recovery_score": 0.5, "load_7d": 0.8, "muscle_soreness": 0.6},
    ]
    html = gen.squad_risk_overview(squad)
    assert html.strip().startswith("<!DOCTYPE html>")
    assert "Player A" in html
    assert "Player B" in html


def test_weekly_performance_is_valid_html():
    gen  = HTMLReportGenerator()
    html = gen.weekly_performance("TestAthlete")
    assert html.strip().startswith("<!DOCTYPE html>")
    assert "TestAthlete" in html


def test_season_dashboard_is_valid_html():
    gen  = HTMLReportGenerator()
    html = gen.season_dashboard("TestAthlete", history=[])
    assert html.strip().startswith("<!DOCTYPE html>")


def test_save_writes_file(tmp_path: Path):
    gen  = HTMLReportGenerator()
    html = gen.weekly_performance("TestAthlete")
    path = gen.save(html, str(tmp_path / "report.html"))
    assert Path(path).exists()
    assert Path(path).read_text().startswith("<!DOCTYPE html>")


# ---------------------------------------------------------------------------
# TerminalDashboard
# ---------------------------------------------------------------------------

def test_morning_brief_prints_to_stdout(capsys):
    dash = TerminalDashboard(use_colour=False)
    plan = _make_plan()
    dash.morning_brief(plan, wearable="HRV: 65ms")
    captured = capsys.readouterr()
    assert "Speed Training" in captured.out
    assert "Morning run" in captured.out


def test_match_prediction_prints_bars(capsys):
    dash = TerminalDashboard(use_colour=False)
    platform = _make_platform_mock()
    pred = platform.match.predict.return_value
    dash.match_prediction(pred)
    captured = capsys.readouterr()
    assert "Arsenal" in captured.out or "prediction" in captured.out.lower()


def test_squad_risk_prints_players(capsys):
    dash = TerminalDashboard(use_colour=False)
    platform = _make_platform_mock()
    risks = [
        MagicMock(
            athlete_name    = "Player A",
            risk_level      = "low",
            days_to_risk    = 30,
            recommendations = ["Maintain load"],
        ),
        MagicMock(
            athlete_name    = "Player B",
            risk_level      = "high",
            days_to_risk    = 3,
            recommendations = ["Rest immediately"],
        ),
    ]
    dash.squad_risk(risks)
    captured = capsys.readouterr()
    assert "Player A" in captured.out
    assert "Player B" in captured.out
    assert "low" in captured.out
    assert "high" in captured.out
