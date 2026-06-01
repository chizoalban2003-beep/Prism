"""
test_kde_agent.py
=================
Tests for kde_agent.py

pytest + tmp_path. Mocks device hub, media processor, vision analyzer, ffmpeg.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sports_pro import Role
from kde_agent import KDEAgent, KDEConfig, TaskResult
from device_hub import DeviceHub, DeviceType
from media_processor import MediaProcessor, VideoRecord, Frame, MediaMetrics
from vision_analyzer import VisionAnalyzer
from daily_workflow import MorningBrief, SessionLog, EveningReview


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(tmp_path: Path) -> KDEConfig:
    return KDEConfig(
        db_path=str(tmp_path / "kde.db"),
        media_dir=str(tmp_path / "media"),
        ollama_host="http://localhost:11434",
        ollama_model="llava",
        text_model="mistral",
        ffmpeg_path="ffmpeg",
        poll_interval=30,
        auto_watch=False,
    )


def _make_agent(tmp_path: Path) -> KDEAgent:
    """Create a minimal KDEAgent with mocked sub-components."""
    cfg   = _config(tmp_path)
    agent = KDEAgent.setup(name="TestAthlete", role=Role.ATHLETE, sport="Football", team="Test FC", config=cfg)

    # Patch device hub
    hub = MagicMock(spec=DeviceHub)
    hub.list_devices.return_value = []
    hub.list_files.return_value = []
    hub.ingest_folder.return_value = []
    hub.start_watching.return_value = None
    hub.stop_watching.return_value  = None
    agent._hub = hub

    # Patch media processor
    mp = MagicMock(spec=MediaProcessor)
    mp.probe.return_value = VideoRecord("v1", "/f.mp4", 120.0, 30.0, 1920, 1080, "h264", 50.0, "gopro", "")
    mp.extract_frames.return_value = [Frame("f1", "v1", 0.0, "/f.jpg", "abc")]
    mp.frame_to_base64.return_value = "abc"
    mp.extract_metrics.return_value = MediaMetrics("v1", 90.0, 3.5, 6.0, 1500.0, 145.0, 180.0)
    mp.create_highlight_reel.return_value = str(tmp_path / "reel.mp4")
    agent._mp = mp

    # Patch vision analyzer (offline)
    va = MagicMock(spec=VisionAnalyzer)
    va.is_available.return_value = False
    agent._va = va

    # Re-wire the workflow with new mocked components
    from daily_workflow import DailyWorkflow
    agent._workflow = DailyWorkflow(
        assistant       = agent._assistant,
        device_hub      = hub,
        media_processor = mp,
        vision_analyzer = va,
        profile_name    = "TestAthlete",
    )

    return agent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent(tmp_path):
    return _make_agent(tmp_path)


# ---------------------------------------------------------------------------
# Setup / Profile
# ---------------------------------------------------------------------------

class TestSetup:

    def test_setup_creates_profile(self, tmp_path):
        cfg   = _config(tmp_path)
        agent = KDEAgent.setup(name="Bob", role=Role.COACH, sport="Basketball", config=cfg)
        assert agent._profile.name  == "Bob"
        assert agent._profile.role  == Role.COACH
        assert agent._profile.sport == "Basketball"

    def test_setup_default_team(self, tmp_path):
        cfg   = _config(tmp_path)
        agent = KDEAgent.setup(name="Carol", role=Role.ANALYST, sport="Tennis", config=cfg)
        assert agent._profile.name == "Carol"

    def test_profile_registered_in_assistant(self, tmp_path):
        cfg   = _config(tmp_path)
        agent = KDEAgent.setup(name="Dave", role=Role.ATHLETE, sport="Rugby", config=cfg)
        _, p = agent._assistant.get_profile("Dave")
        assert p.name == "Dave"


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

class TestStatus:

    def test_status_has_all_keys(self, agent):
        s = agent.status()
        for key in ("profile", "devices", "ollama_available", "ffmpeg_available",
                    "plans_this_month", "sessions_this_month", "artifacts_stored"):
            assert key in s, f"Missing key: {key}"

    def test_status_profile_matches(self, agent):
        s = agent.status()
        assert s["profile"] == "TestAthlete"

    def test_ollama_available_false(self, agent):
        s = agent.status()
        assert s["ollama_available"] is False


# ---------------------------------------------------------------------------
# ask() routing
# ---------------------------------------------------------------------------

class TestAskRouting:

    def test_ask_routes_video(self, agent):
        result = agent.ask("analyse my session video from yesterday")
        assert isinstance(result, TaskResult)
        assert result.task in ("video_analysis", "error")
        assert result.method in ("keyword", "direct", "error")

    def test_ask_routes_highlight(self, agent):
        result = agent.ask("make a highlight reel from this week")
        assert isinstance(result, TaskResult)
        assert result.task in ("highlight_reel", "error")

    def test_ask_routes_report(self, agent):
        result = agent.ask("generate a performance report for my coach")
        assert isinstance(result, TaskResult)
        assert result.task in ("performance_report", "error")

    def test_ask_routes_morning(self, agent):
        result = agent.ask("morning briefing")
        assert isinstance(result, TaskResult)
        assert result.task == "morning_briefing"
        assert result.success

    def test_ask_routes_evening(self, agent):
        result = agent.ask("evening review")
        assert isinstance(result, TaskResult)
        assert result.task == "evening_review"
        assert result.success

    def test_ask_routes_status(self, agent):
        result = agent.ask("status")
        assert isinstance(result, TaskResult)
        assert result.task == "status"
        assert result.success

    def test_ask_routes_reflect(self, agent):
        result = agent.ask("reflect")
        assert isinstance(result, TaskResult)
        assert result.task == "reflect"
        assert result.success

    def test_ask_unknown_prompt_returns_task_result(self, agent):
        # An unknown prompt should not raise; it falls through gracefully
        result = agent.ask("xyzzy flurble grob")
        assert isinstance(result, TaskResult)

    def test_ask_has_elapsed_ms(self, agent):
        result = agent.ask("status")
        assert result.elapsed_ms >= 0

    def test_ask_empty_string_is_graceful(self, agent):
        result = agent.ask("")
        assert isinstance(result, TaskResult)


# ---------------------------------------------------------------------------
# Device management
# ---------------------------------------------------------------------------

class TestDeviceManagement:

    def test_add_device_returns_id(self, agent):
        agent._hub.register_device.return_value = "fake-device-id"
        device_id = agent.add_device(
            name="GoPro12",
            device_type=DeviceType.GOPRO,
            watch_path="/fake/gopro",
        )
        assert isinstance(device_id, str)
        assert len(device_id) > 0

    def test_sync_devices_returns_dict(self, agent):
        result = agent.sync_devices()
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Daily lifecycle
# ---------------------------------------------------------------------------

class TestDailyLifecycle:

    def test_morning_briefing_returns_brief(self, agent):
        brief = agent.morning_briefing()
        assert isinstance(brief, MorningBrief)
        assert brief.plan
        assert len(brief.priority_tasks) <= 3

    def test_morning_briefing_with_manual_values(self, agent):
        brief = agent.morning_briefing(hrv_ms=65.0, sleep_hrs=7.5, soreness=2, energy=8)
        assert isinstance(brief, MorningBrief)

    def test_log_session_returns_session_log(self, agent):
        log = agent.log_session(rpe=6)
        assert isinstance(log, SessionLog)
        assert log.rpe == 6
        assert log.sport == "Football"

    def test_evening_review_returns_review(self, agent):
        review = agent.evening_review(day_rating=4.0)
        assert isinstance(review, EveningReview)
        assert review.sleep_target_hrs > 0


# ---------------------------------------------------------------------------
# Reflect
# ---------------------------------------------------------------------------

class TestReflect:

    def test_reflect_shows_drift(self, agent):
        # Log a session and rate the day to trigger drift
        agent.log_session(rpe=7)
        agent.evening_review(day_rating=5.0)
        r = agent.reflect()
        assert "fixed_fulcrum" in r

    def test_reflect_returns_dict(self, agent):
        r = agent.reflect()
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------

class TestGenerateReport:

    def test_generate_report_weekly(self, agent):
        report = agent.generate_report(report_type="weekly")
        assert isinstance(report, str)
        assert "# " in report  # markdown heading

    def test_generate_report_to_path(self, agent, tmp_path):
        out = str(tmp_path / "report.md")
        report = agent.generate_report(report_type="weekly", output_path=out)
        assert isinstance(report, str)
        assert Path(out).exists()
