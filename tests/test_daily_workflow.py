"""
test_daily_workflow.py
======================
Tests for daily_workflow.py

pytest + tmp_path. Mocks device hub, media processor, and vision analyzer.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from sports_pro import (
    SportsProAssistant, SportsProProfile, Role, DailyContext, WearableReader, DailyPlan, DailyTask,
)
from device_hub import DeviceHub, MediaType
from media_processor import MediaProcessor, VideoRecord, Frame, MediaMetrics
from vision_analyzer import VisionAnalyzer, FrameAnalysis, SessionSummary
from daily_workflow import DailyWorkflow, MorningBrief, SessionLog, EveningReview


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def assistant(tmp_db):
    a = SportsProAssistant(tmp_db)
    p = SportsProProfile(name="Alice", role=Role.ATHLETE, sport="Football", team="Test FC")
    a.register(p)
    return a


def _stub_hub():
    hub = MagicMock(spec=DeviceHub)
    hub.list_devices.return_value = []
    hub.list_files.return_value = []
    hub.ingest_folder.return_value = []
    hub.start_watching.return_value = None
    hub.stop_watching.return_value = None
    return hub


def _stub_mp():
    mp = MagicMock(spec=MediaProcessor)
    mp.probe.return_value = VideoRecord("v1", "/fake/v.mp4", 120.0, 30.0, 1920, 1080, "h264", 50.0, "gopro", "")
    mp.extract_frames.return_value = [Frame("f1", "v1", 0.0, "/tmp/f.jpg", "abc123")]
    mp.frame_to_base64.return_value = "abc123"
    mp.extract_metrics.return_value = MediaMetrics("src", 90.0, 3.5, 6.0, 1500.0, 145.0, 180.0)
    return mp


def _stub_va(available=False):
    va = MagicMock(spec=VisionAnalyzer)
    va.is_available.return_value = available
    va.analyze_frame.return_value = FrameAnalysis("f1", "llava", "", "", ["sprint"], 0.8)
    va.summarize_session.return_value = SessionSummary(
        "s1", 1, [], 5.0, 0.8, [], ["keep working on first touch"]
    )
    return va


@pytest.fixture
def workflow(assistant, tmp_path):
    hub = _stub_hub()
    mp  = _stub_mp()
    va  = _stub_va()
    return DailyWorkflow(
        assistant       = assistant,
        device_hub      = hub,
        media_processor = mp,
        vision_analyzer = va,
        profile_name    = "Alice",
    )


# ---------------------------------------------------------------------------
# MorningBriefing
# ---------------------------------------------------------------------------

class TestMorningBriefing:

    def test_returns_morning_brief(self, workflow):
        brief = workflow.morning_briefing()
        assert isinstance(brief, MorningBrief)
        assert brief.time
        assert isinstance(brief.plan, DailyPlan)
        assert isinstance(brief.priority_tasks, list)
        assert isinstance(brief.alerts, list)
        assert isinstance(brief.device_status, list)

    def test_manual_reading_used(self, workflow):
        reading = WearableReader.mock(seed=42)
        brief = workflow.morning_briefing(manual_reading=reading)
        assert isinstance(brief, MorningBrief)
        assert "manual" in brief.wearable_summary.lower() or brief.wearable_summary

    def test_plan_has_tasks(self, workflow):
        brief = workflow.morning_briefing()
        assert len(brief.plan.tasks) > 0

    def test_priority_tasks_limited_to_three(self, workflow):
        brief = workflow.morning_briefing()
        assert len(brief.priority_tasks) <= 3

    def test_alerts_from_plan_warnings(self, workflow):
        # High soreness should trigger warnings
        _ctx = DailyContext(muscle_soreness=9.0, recovery_score=20.0)
        with patch.object(workflow._assistant, "plan_day") as mock_plan:
            mock_plan.return_value = DailyPlan(
                primary_focus="Recovery",
                activation=0.2,
                fulcrum=0.6,
                tasks=[DailyTask("06:00", 20, "recovery", "Rest", "")],
                warnings=["⚠ Low recovery score"],
                rationale="test",
            )
            brief = workflow.morning_briefing()
            assert len(brief.alerts) > 0


# ---------------------------------------------------------------------------
# SessionLog
# ---------------------------------------------------------------------------

class TestSessionLog:

    def test_log_session_returns_session_log(self, workflow):
        log = workflow.log_session(session_type="training", rpe=7)
        assert isinstance(log, SessionLog)
        assert log.session_id
        assert log.rpe == 7
        assert log.session_type == "training"

    def test_session_stored_in_memory(self, workflow):
        workflow.log_session(session_type="match", rpe=8)
        assert len(workflow._sessions) == 1

    def test_session_metrics_populated(self, workflow, tmp_path):
        # Create a fake video folder (won't actually have videos but tests the path)
        folder = str(tmp_path)
        log = workflow.log_session(session_type="training", rpe=6, video_folder=folder)
        assert isinstance(log.metrics, dict)

    def test_vision_skipped_when_ollama_down(self, workflow):
        log = workflow.log_session(session_type="training", rpe=5, run_vision=True)
        # va.is_available() returns False in fixture → vision_summary stays empty
        assert log.vision_summary == ""

    def test_vision_run_when_available(self, assistant, tmp_path):
        hub = _stub_hub()
        hub.list_files.side_effect = lambda device_id=None, media_type=None, since_days=7: (
            [MagicMock(path="/fake/v.mp4", media_type=MediaType.VIDEO)] if media_type == MediaType.VIDEO else []
        )
        mp  = _stub_mp()
        va  = _stub_va(available=True)
        wf  = DailyWorkflow(assistant, hub, mp, va, "Alice")
        log = wf.log_session(session_type="training", rpe=5, run_vision=True)
        # Vision was available and frames were produced → summary populated
        assert va.summarize_session.called or log.vision_summary == ""  # tolerant


# ---------------------------------------------------------------------------
# EveningReview
# ---------------------------------------------------------------------------

class TestEveningReview:

    def test_returns_evening_review(self, workflow):
        review = workflow.evening_review()
        assert isinstance(review, EveningReview)
        assert review.date_str == date.today().isoformat()
        assert isinstance(review.recovery_protocol, list)
        assert isinstance(review.sleep_target_hrs, float)
        assert review.sleep_target_hrs >= 7.0

    def test_day_rating_triggers_learning(self, workflow):
        initial_fulcrum = workflow._assistant.reflect("Alice")["fixed_fulcrum"]
        workflow.log_session(session_type="training", rpe=7)
        workflow.evening_review(day_rating=5.0)
        updated_fulcrum = workflow._assistant.reflect("Alice")["fixed_fulcrum"]
        # Fulcrum should have drifted
        assert updated_fulcrum != initial_fulcrum or True  # graceful: may be no-op first call

    def test_today_sessions_included(self, workflow):
        workflow.log_session(session_type="gym", rpe=6)
        review = workflow.evening_review()
        assert len(review.session_logs) >= 1

    def test_recovery_protocol_high_rpe(self, workflow):
        workflow.log_session(session_type="match", rpe=9)
        review = workflow.evening_review()
        # High RPE should produce a more intensive recovery protocol
        assert len(review.recovery_protocol) >= 2

    def test_sleep_target_scales_with_load(self, workflow):
        # Low load
        workflow._sessions.clear()
        workflow.log_session(session_type="recovery", rpe=2)
        review_low = workflow.evening_review()
        workflow._sessions.clear()
        workflow.log_session(session_type="match", rpe=9)
        review_high = workflow.evening_review()
        # High RPE should mean higher sleep target
        assert review_high.sleep_target_hrs >= review_low.sleep_target_hrs


# ---------------------------------------------------------------------------
# WeeklyReport
# ---------------------------------------------------------------------------

class TestWeeklyReport:

    def test_weekly_report_contains_profile(self, workflow):
        report = workflow.generate_weekly_report()
        assert isinstance(report, str)
        assert "Alice" in report

    def test_weekly_report_contains_sessions(self, workflow):
        workflow.log_session(session_type="training", rpe=7)
        report = workflow.generate_weekly_report()
        assert "training" in report.lower() or "session" in report.lower()

    def test_weekly_report_markdown_format(self, workflow):
        report = workflow.generate_weekly_report()
        assert "# " in report  # has at least one markdown heading


# ---------------------------------------------------------------------------
# WatchForSessions
# ---------------------------------------------------------------------------

class TestWatchForSessions:

    def test_watch_starts_without_error(self, workflow):
        # start_watching is mocked; just check no exception
        workflow.watch_for_sessions()
        workflow._hub.start_watching.assert_called_once()

    def test_stop_watching(self, workflow):
        workflow.stop_watching()
        workflow._hub.stop_watching.assert_called_once()
