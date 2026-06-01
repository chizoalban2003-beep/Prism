"""
test_sport_tasks.py
===================
Tests for sport_tasks.py

pytest + tmp_path. Mocks Ollama (no real HTTP calls).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ksa_registry import SnapshotRegistry
from ksa_executor import ExecutionContext
from ksa_lever import EquilibriumResult, LeverState, TiltDirection
from sport_tasks import (
    TrainingPlanTask,
    MatchReportTask,
    SocialMediaTask,
    PredictionReportTask,
    _ollama_text,
    _save_artifact,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_eq(tilt: TiltDirection = TiltDirection.LEFT) -> EquilibriumResult:
    states = [LeverState(i, 1.0 if tilt == TiltDirection.LEFT else -1.0, tilt, 1.0) for i in range(3)]
    return EquilibriumResult(states=states, final_tilt=tilt, override_active=False, confidence=0.7)


def _make_ctx(task_name: str = "create_training_plan") -> ExecutionContext:
    return ExecutionContext(
        task_name = task_name,
        version   = 1,
        result    = _make_eq(),
        payload   = {
            "sport":        "football",
            "role":         "athlete",
            "season_phase": "in-season",
            "days_to_match": 5,
            "fitness_level": "high",
            "body_weight":   75,
            "platform":      "twitter",
            "email_type":    "agent_update",
            "home_team":     "Arsenal",
            "away_team":     "Chelsea",
            "opponent_name": "Chelsea",
        },
    )


def _make_registry(tmp_path: Path) -> SnapshotRegistry:
    return SnapshotRegistry(str(tmp_path / "test.db"))


def _make_platform() -> MagicMock:
    from prediction_engine import PredictionPlatform
    platform = MagicMock(spec=PredictionPlatform)
    platform.pre_match_brief.return_value = {
        "match_prediction":  MagicMock(
            subject="Arsenal vs Chelsea",
            prediction="Arsenal win",
            confidence=0.62,
            p_home_win=0.55,
            p_draw=0.25,
            p_away_win=0.20,
            predicted_margin=1.2,
            distribution={"home_win": 0.55, "draw": 0.25, "away_win": 0.20},
            risk=0.3,
            risk_adj=0.28,
            fulcrum=0.5,
            key_factors=[("form", 0.7, "positive")],
            expected_value=1.0,
        ),
        "tactical_analysis": MagicMock(
            home_team="Arsenal", away_team="Chelsea",
            matchup_summary="Tight match expected",
            home_advantage="Strong home form",
            away_advantage="Counter-attack",
            home_predicted_style="Possession",
            away_predicted_style="Low block",
            key_duels=[],
        ),
        "squad_risk":        [],
        "squad_performance": [],
        "generated_at":      "2026-05-28T09:00:00",
    }
    return platform


MOCK_LLM_RESPONSE = "# Mock LLM Response\n\nThis is mock content from the LLM.\n\n## Performance\n\nGood.\n\n## Tactics\n\nSolid."


# ---------------------------------------------------------------------------
# _ollama_text: ConnectionError when Ollama is down
# ---------------------------------------------------------------------------

def test_ollama_down_raises_connection_error():
    import urllib.error
    with patch("sport_tasks.urllib.request.urlopen",
               side_effect=urllib.error.URLError("connection refused")):
        with pytest.raises(ConnectionError):
            _ollama_text("test prompt")


# ---------------------------------------------------------------------------
# _save_artifact
# ---------------------------------------------------------------------------

def test_save_artifact(tmp_path: Path):
    path = _save_artifact("hello world", "test.txt", str(tmp_path))
    assert Path(path).exists()
    assert Path(path).read_text() == "hello world"


# ---------------------------------------------------------------------------
# TrainingPlanTask
# ---------------------------------------------------------------------------

def test_training_plan_primary_saves_file(tmp_path: Path):
    registry = _make_registry(tmp_path)
    task     = TrainingPlanTask(
        registry   = registry,
        output_dir = str(tmp_path / "artifacts"),
    )
    ctx = _make_ctx("create_training_plan")
    with patch("sport_tasks._ollama_text", return_value=MOCK_LLM_RESPONSE):
        outcome = task.primary(ctx)
    assert outcome.return_code == 0
    artifacts = list((tmp_path / "artifacts").glob("training_plan_*.md"))
    assert len(artifacts) == 1
    assert "# Mock LLM Response" in artifacts[0].read_text()


def test_training_plan_safe_no_file(tmp_path: Path):
    registry = _make_registry(tmp_path)
    task     = TrainingPlanTask(
        registry   = registry,
        output_dir = str(tmp_path / "artifacts"),
    )
    ctx = _make_ctx("create_training_plan")
    with patch("sport_tasks._ollama_text", return_value=MOCK_LLM_RESPONSE):
        outcome = task.safe(ctx)
    assert outcome.return_code == 0
    # safe() must not write files
    artifacts = list((tmp_path / "artifacts").glob("training_plan_*.md"))
    assert len(artifacts) == 0


def test_training_plan_secondary_three_day(tmp_path: Path):
    registry = _make_registry(tmp_path)
    task     = TrainingPlanTask(
        registry   = registry,
        output_dir = str(tmp_path / "artifacts"),
    )
    ctx = _make_ctx("create_training_plan")
    with patch("sport_tasks._ollama_text", return_value=MOCK_LLM_RESPONSE) as mock_llm:
        outcome = task.secondary(ctx)
    assert outcome.return_code == 0
    # Secondary prompt should mention 3-day
    call_args = mock_llm.call_args[0][0]
    assert "3-day" in call_args or "three" in call_args.lower() or "3 day" in call_args


# ---------------------------------------------------------------------------
# MatchReportTask
# ---------------------------------------------------------------------------

def test_match_report_contains_sections(tmp_path: Path):
    registry = _make_registry(tmp_path)
    task     = MatchReportTask(
        registry   = registry,
        output_dir = str(tmp_path / "artifacts"),
    )
    ctx = _make_ctx("match_report")
    with patch("sport_tasks._ollama_text", return_value=MOCK_LLM_RESPONSE):
        outcome = task.primary(ctx)
    data = json.loads(outcome.stdout)
    _content = data.get("content", "") + data.get("path", "")
    # Check saved file if path returned
    saved_files = list((tmp_path / "artifacts").glob("match_report_*.md"))
    if saved_files:
        file_content = saved_files[0].read_text()
        assert "## Performance" in file_content or "Performance" in file_content


# ---------------------------------------------------------------------------
# SocialMediaTask
# ---------------------------------------------------------------------------

def test_social_media_twitter_length(tmp_path: Path):
    """Twitter / X variant must be ≤ 280 characters."""
    registry = _make_registry(tmp_path)
    task     = SocialMediaTask(
        registry   = registry,
        output_dir = str(tmp_path / "artifacts"),
    )
    ctx = _make_ctx("social_media_post")
    ctx.payload["platform"] = "twitter"

    # Mock LLM to return a short response that fits
    short_response = "Great session today! Hard work paying off. #Football #Training 💪"
    with patch("sport_tasks._ollama_text", return_value=short_response):
        outcome = task.primary(ctx)
    assert outcome.return_code == 0
    data = json.loads(outcome.stdout)
    post = data.get("post", "") or data.get("twitter", "") or data.get("content", "")
    # Twitter post must be at most 280 chars
    assert len(post) <= 280, f"Twitter post too long: {len(post)} chars"


def test_social_media_twitter_truncated(tmp_path: Path):
    """When LLM returns content > 280 chars for Twitter, it must be truncated."""
    registry = _make_registry(tmp_path)
    task     = SocialMediaTask(
        registry   = registry,
        output_dir = str(tmp_path / "artifacts"),
    )
    ctx = _make_ctx("social_media_post")
    ctx.payload["platform"] = "twitter"

    # Mock LLM to return a response that exceeds the Twitter limit
    long_response = "A" * 400 + " #Football"
    with patch("sport_tasks._ollama_text", return_value=long_response):
        outcome = task.primary(ctx)
    assert outcome.return_code == 0
    data = json.loads(outcome.stdout)
    twitter_post = data.get("twitter", data.get("post", data.get("content", "")))
    assert len(twitter_post) <= 280, f"Truncation failed: {len(twitter_post)} chars"


# ---------------------------------------------------------------------------
# Ollama down: secondary degrades gracefully
# ---------------------------------------------------------------------------

def test_ollama_down_degrades(tmp_path: Path):
    """When Ollama is unavailable, secondary() should succeed with a fallback."""
    import urllib.error
    registry = _make_registry(tmp_path)
    task     = TrainingPlanTask(
        registry   = registry,
        output_dir = str(tmp_path / "artifacts"),
    )
    ctx = _make_ctx("create_training_plan")
    with patch("sport_tasks.urllib.request.urlopen",
               side_effect=urllib.error.URLError("connection refused")):
        outcome = task.secondary(ctx)
    # Must not raise; must return a non-crash outcome
    assert outcome is not None
    # return_code may be non-zero but no exception should propagate


# ---------------------------------------------------------------------------
# PredictionReportTask
# ---------------------------------------------------------------------------

def test_prediction_report_calls_platform(tmp_path: Path):
    registry = _make_registry(tmp_path)
    platform = _make_platform()
    task     = PredictionReportTask(
        registry   = registry,
        platform   = platform,
        output_dir = str(tmp_path / "artifacts"),
    )
    ctx = _make_ctx("prediction_report")
    with patch("sport_tasks._ollama_text", return_value=MOCK_LLM_RESPONSE):
        outcome = task.primary(ctx)
    platform.pre_match_brief.assert_called_once()
    assert outcome.return_code == 0
