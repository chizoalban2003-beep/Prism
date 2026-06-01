"""
test_sport_executor.py
======================
Tests for sport_executor.py

pytest + tmp_path. Mocks ffmpeg and Ollama calls.
"""
from __future__ import annotations

import json
import os
import uuid
from unittest.mock import MagicMock


from ksa_lever import TiltDirection
from ksa_registry import SnapshotRegistry
from device_hub import DeviceHub, DeviceType, MediaType, IngestedFile
from media_processor import MediaProcessor, VideoRecord, Frame, Clip, MediaMetrics
from vision_analyzer import VisionAnalyzer, TechniqueReport, FrameAnalysis, TacticalContext, SessionSummary
from sport_executor import (
    VideoAnalysisExecutor,
    HighlightReelExecutor,
    PerformanceReportExecutor,
    WearableSyncExecutor,
    SessionLogExecutor,
)
from ksa_executor import ExecutionContext
from ksa_lever import EquilibriumResult, LeverState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_eq(tilt=TiltDirection.LEFT):
    states = [LeverState(i, 1.0 if tilt == TiltDirection.LEFT else -1.0, tilt, 1.0) for i in range(3)]
    return EquilibriumResult(states=states, final_tilt=tilt, override_active=False, confidence=0.7)


def _make_ctx(task="video_analysis", tilt=TiltDirection.LEFT):
    return ExecutionContext(
        task_name="video_analysis",
        version=1,
        result=_make_eq(tilt),
        working_dir=".",
        payload={"sport": "football", "role": "athlete"},
    )


def _make_ingested(path: str, media_type=MediaType.VIDEO) -> IngestedFile:
    return IngestedFile(
        file_id=str(uuid.uuid4()),
        device_id="dev1",
        device_type=DeviceType.GOPRO,
        media_type=media_type,
        path=path,
        filename=os.path.basename(path),
        size_bytes=1024 * 1024,
        sha256="abc",
        ingested_at="2026-01-01",
        metadata={},
    )


def _make_video_record(path: str = "/fake/video.mp4") -> VideoRecord:
    return VideoRecord(
        file_id="v1",
        path=path,
        duration_sec=120.0,
        fps=30.0,
        width=1920,
        height=1080,
        codec="h264",
        size_mb=50.0,
        device_type="gopro",
        recorded_at="2026-01-01",
    )


def _make_frame(frame_id="f1", path="/tmp/frame.jpg") -> Frame:
    return Frame(frame_id=frame_id, video_id="v1", timestamp=0.0, path=path, base64="abc123")


def _stub_hub(videos=None, gps=None, data=None):
    hub = MagicMock(spec=DeviceHub)
    hub.list_files.side_effect = lambda device_id=None, media_type=None, since_days=7: (
        videos if media_type == MediaType.VIDEO else
        gps if media_type == MediaType.GPS else
        data if media_type == MediaType.DATA else []
    ) or []
    hub.list_devices.return_value = []
    hub.ingest_folder.return_value = []
    return hub


def _stub_mp(frames=None, record=None, metrics=None, clip=None, b64="abc"):
    mp = MagicMock(spec=MediaProcessor)
    mp.probe.return_value = record or _make_video_record()
    mp.extract_frames.return_value = frames or [_make_frame()]
    mp.frame_to_base64.return_value = b64
    mp.extract_metrics.return_value = metrics or MediaMetrics("src", 120.0)
    mp.extract_clip.return_value = clip or Clip("c1", "src", 0, 30, "/tmp/clip.mp4")
    mp.create_highlight_reel.return_value = "/tmp/reel.mp4"
    return mp


def _stub_va(available=False):
    va = MagicMock(spec=VisionAnalyzer)
    va.is_available.return_value = available
    va.analyze_frame.return_value = FrameAnalysis(
        frame_id="f1", model="llava", prompt_used="",
        raw_response="", tags=["sprint"], quality_score=0.8,
    )
    va.analyze_technique.return_value = TechniqueReport(
        video_id="v1", sport="football", profile="athlete",
        n_frames=3, key_findings=["good technique"], improvements=[],
        strengths=["speed"], overall_score=0.8,
    )
    va.detect_tactical_situation.return_value = TacticalContext(
        formation_guess="4-4-2", ball_zone="midfield", pressure_level=0.5,
        space_available=0.6, phase="build-up", tags=[],
    )
    va.summarize_session.return_value = SessionSummary(
        session_id="s1", n_clips=1, highlights=[], load_estimate=5.0,
        technique_score=0.8, tactical_insights=[], recommendations=["keep it up"],
    )
    return va


# ---------------------------------------------------------------------------
# VideoAnalysisExecutor
# ---------------------------------------------------------------------------

class TestVideoAnalysisExecutor:

    def test_safe_no_side_effects(self, tmp_path):
        hub     = _stub_hub(videos=[_make_ingested("/fake/v.mp4")])
        mp      = _stub_mp()
        va      = _stub_va(available=False)
        reg     = SnapshotRegistry(str(tmp_path / "db.db"))
        exe     = VideoAnalysisExecutor(hub, mp, va, reg)
        ctx     = _make_ctx()
        outcome = exe.safe(ctx)
        assert outcome.return_code == 0
        assert outcome.action_taken == "safe"
        # No calls to probe or Ollama
        mp.probe.assert_not_called()
        va.analyze_technique.assert_not_called()
        data = json.loads(outcome.stdout)
        assert "available_videos" in data

    def test_secondary_extracts_frames_only(self, tmp_path):
        vid     = _make_ingested("/fake/v.mp4")
        hub     = _stub_hub(videos=[vid])
        mp      = _stub_mp()
        va      = _stub_va(available=False)
        exe     = VideoAnalysisExecutor(hub, mp, va)
        ctx     = _make_ctx()
        outcome = exe.secondary(ctx)
        assert outcome.return_code == 0
        assert outcome.action_taken == "secondary"
        mp.extract_frames.assert_called_once()
        va.analyze_technique.assert_not_called()
        data = json.loads(outcome.stdout)
        assert "frame_count" in data

    def test_primary_runs_vision(self, tmp_path):
        vid     = _make_ingested("/fake/v.mp4")
        hub     = _stub_hub(videos=[vid])
        mp      = _stub_mp()
        va      = _stub_va(available=True)
        exe     = VideoAnalysisExecutor(hub, mp, va)
        ctx     = _make_ctx()
        outcome = exe.primary(ctx)
        assert outcome.return_code == 0
        va.analyze_technique.assert_called_once()
        data = json.loads(outcome.stdout)
        assert "overall_score" in data or "key_findings" in data

    def test_primary_no_videos_returns_error(self):
        hub = _stub_hub(videos=[])
        mp  = _stub_mp()
        va  = _stub_va(available=True)
        exe = VideoAnalysisExecutor(hub, mp, va)
        ctx = _make_ctx()
        outcome = exe.primary(ctx)
        assert outcome.return_code == 1

    def test_primary_ollama_down_returns_error(self):
        vid = _make_ingested("/fake/v.mp4")
        hub = _stub_hub(videos=[vid])
        mp  = _stub_mp()
        va  = _stub_va(available=False)
        exe = VideoAnalysisExecutor(hub, mp, va)
        ctx = _make_ctx()
        outcome = exe.primary(ctx)
        assert outcome.return_code == 1

    def test_records_outcome_metrics(self, tmp_path):
        vid = _make_ingested("/fake/v.mp4")
        hub = _stub_hub(videos=[vid])
        mp  = _stub_mp()
        va  = _stub_va(available=True)
        exe = VideoAnalysisExecutor(hub, mp, va)
        ctx = _make_ctx()
        outcome = exe.primary(ctx)
        assert outcome.metrics.execution_time_ms >= 0
        assert isinstance(outcome.metrics.success, bool)


# ---------------------------------------------------------------------------
# HighlightReelExecutor
# ---------------------------------------------------------------------------

class TestHighlightReelExecutor:

    def test_safe_lists_clips(self, tmp_path):
        vid  = _make_ingested("/fake/v.mp4")
        hub  = _stub_hub(videos=[vid])
        mp   = _stub_mp()
        exe  = HighlightReelExecutor(hub, mp, output_dir=str(tmp_path))
        ctx  = _make_ctx("highlight_reel")
        out  = exe.safe(ctx)
        assert out.return_code == 0
        data = json.loads(out.stdout)
        assert "clips_available" in data

    def test_secondary_no_encoding(self, tmp_path):
        vid  = _make_ingested("/fake/v.mp4")
        hub  = _stub_hub(videos=[vid])
        mp   = _stub_mp()
        exe  = HighlightReelExecutor(hub, mp, output_dir=str(tmp_path))
        ctx  = _make_ctx("highlight_reel")
        out  = exe.secondary(ctx)
        assert out.return_code == 0
        mp.create_highlight_reel.assert_not_called()

    def test_primary_creates_reel(self, tmp_path):
        vid  = _make_ingested("/fake/v.mp4")
        hub  = _stub_hub(videos=[vid])
        mp   = _stub_mp()
        exe  = HighlightReelExecutor(hub, mp, output_dir=str(tmp_path))
        ctx  = _make_ctx("highlight_reel")
        out  = exe.primary(ctx)
        assert out.return_code == 0
        mp.create_highlight_reel.assert_called_once()
        data = json.loads(out.stdout)
        assert "reel_path" in data

    def test_primary_no_videos(self, tmp_path):
        hub = _stub_hub(videos=[])
        mp  = _stub_mp()
        exe = HighlightReelExecutor(hub, mp, output_dir=str(tmp_path))
        ctx = _make_ctx("highlight_reel")
        out = exe.primary(ctx)
        assert out.return_code == 1


# ---------------------------------------------------------------------------
# PerformanceReportExecutor
# ---------------------------------------------------------------------------

class TestPerformanceReportExecutor:

    def test_safe_returns_sources(self, tmp_path):
        hub = _stub_hub()
        mp  = _stub_mp()
        exe = PerformanceReportExecutor(hub, mp, output_dir=str(tmp_path))
        ctx = _make_ctx("performance_report")
        out = exe.safe(ctx)
        assert out.return_code == 0
        data = json.loads(out.stdout)
        assert "data_sources" in data

    def test_primary_returns_markdown(self, tmp_path):
        vid = _make_ingested("/fake/v.mp4")
        hub = _stub_hub(videos=[vid])
        mp  = _stub_mp()
        exe = PerformanceReportExecutor(hub, mp, output_dir=str(tmp_path))
        ctx = _make_ctx("performance_report")
        out = exe.primary(ctx)
        assert out.return_code == 0
        data = json.loads(out.stdout)
        assert "report" in data
        assert "# Performance Report" in data["report"]

    def test_secondary_plain_text(self, tmp_path):
        hub = _stub_hub()
        mp  = _stub_mp()
        exe = PerformanceReportExecutor(hub, mp, output_dir=str(tmp_path))
        ctx = _make_ctx("performance_report")
        out = exe.secondary(ctx)
        assert out.return_code == 0
        data = json.loads(out.stdout)
        assert "summary" in data


# ---------------------------------------------------------------------------
# WearableSyncExecutor
# ---------------------------------------------------------------------------

class TestWearableSyncExecutor:

    def test_safe_no_side_effects(self, tmp_path):
        hub = MagicMock(spec=DeviceHub)
        hub.list_devices.return_value = []
        exe = WearableSyncExecutor(hub)
        ctx = _make_ctx("wearable_sync")
        out = exe.safe(ctx)
        assert out.return_code == 0
        hub.ingest_folder.assert_not_called()
        data = json.loads(out.stdout)
        assert "would_sync" in data

    def test_primary_syncs_devices(self, tmp_path):
        device = MagicMock()
        device.name     = "MyWatch"
        device.enabled  = True
        device.watch_path = str(tmp_path)
        device.device_id  = "d1"
        hub = MagicMock(spec=DeviceHub)
        hub.list_devices.return_value = [device]
        hub.ingest_folder.return_value = []
        exe = WearableSyncExecutor(hub)
        ctx = _make_ctx("wearable_sync")
        out = exe.primary(ctx)
        assert out.return_code == 0
        hub.ingest_folder.assert_called_once()


# ---------------------------------------------------------------------------
# SessionLogExecutor
# ---------------------------------------------------------------------------

class TestSessionLogExecutor:

    def test_safe_describes_content(self, tmp_path):
        hub = _stub_hub()
        mp  = _stub_mp()
        va  = _stub_va(available=False)
        exe = SessionLogExecutor(hub, mp, va)
        ctx = _make_ctx("session_log")
        out = exe.safe(ctx)
        assert out.return_code == 0
        data = json.loads(out.stdout)
        assert "would_include" in data

    def test_secondary_manual_fields(self, tmp_path):
        hub = _stub_hub()
        mp  = _stub_mp()
        va  = _stub_va(available=False)
        exe = SessionLogExecutor(hub, mp, va)
        ctx = _make_ctx("session_log")
        out = exe.secondary(ctx)
        assert out.return_code == 0
        data = json.loads(out.stdout)
        assert "session_id" in data
        assert "session_type" in data

    def test_primary_creates_full_record(self, tmp_path):
        vid = _make_ingested("/fake/v.mp4")
        hub = _stub_hub(videos=[vid])
        mp  = _stub_mp()
        va  = _stub_va(available=False)
        exe = SessionLogExecutor(hub, mp, va)
        ctx = _make_ctx("session_log")
        ctx.payload["rpe"] = 7
        out = exe.primary(ctx)
        assert out.return_code == 0
        data = json.loads(out.stdout)
        assert "session_id" in data
        assert data["rpe"] == 7
