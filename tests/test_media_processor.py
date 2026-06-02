"""
tests/test_media_processor.py
==============================
Tests for media_processor.py

Covers:
  - probe → VideoRecord (mocked ffprobe subprocess)
  - extract_frames → ffmpeg called with correct args
  - frame_to_base64 → correct base64
  - extract_clip → ffmpeg called, Clip returned
  - create_highlight_reel → concat list written, ffmpeg called
  - extract_metrics (gpx) → distance/speed/duration populated
  - image_to_base64 → valid base64 string
  - graceful degradation when ffmpeg missing
"""

from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from media_processor import (
    Clip,
    Frame,
    MediaMetrics,
    MediaProcessor,
    VideoRecord,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FFPROBE_JSON = json.dumps({
    "streams": [
        {
            "codec_type":    "video",
            "codec_name":    "h264",
            "r_frame_rate":  "30000/1001",
            "width":         1920,
            "height":        1080,
        }
    ],
    "format": {
        "duration": "63.5",
        "size":     "52428800",
        "tags":     {"creation_time": "2024-01-15T10:00:00Z"},
    },
})

FAKE_VIDEO_RECORD = VideoRecord(
    file_id="vid001",
    path="/tmp/video.mp4",
    duration_sec=63.5,
    fps=29.97,
    width=1920,
    height=1080,
    codec="h264",
    size_mb=50.0,
    device_type="gopro",
    recorded_at="2024-01-15T10:00:00Z",
)


def _make_completed_process(stdout: str = "", returncode: int = 0):
    p = MagicMock(spec=subprocess.CompletedProcess)
    p.returncode = returncode
    p.stdout     = stdout.encode() if isinstance(stdout, str) else stdout
    p.stderr     = b""
    return p


def _make_processor(tmp_path: Path, ffmpeg_ok: bool = True) -> MediaProcessor:
    mp = MediaProcessor(
        output_dir   = str(tmp_path / "media"),
        ffmpeg_path  = "ffmpeg",
        ffprobe_path = "ffprobe",
        frame_rate   = 1.0,
        thumb_width  = 320,
    )
    mp._ffmpeg_ok = ffmpeg_ok
    return mp


# ---------------------------------------------------------------------------
# probe → VideoRecord
# ---------------------------------------------------------------------------

class TestProbe:
    def test_probe_returns_video_record(self, tmp_path):
        mp = _make_processor(tmp_path)
        with patch.object(mp, "_run", return_value=_make_completed_process(FFPROBE_JSON)):
            record = mp.probe("/fake/video.mp4")
        assert isinstance(record, VideoRecord)
        assert record.codec        == "h264"
        assert record.width        == 1920
        assert record.height       == 1080
        assert record.duration_sec == pytest.approx(63.5)
        assert record.fps          == pytest.approx(29.97, abs=0.01)
        assert record.size_mb      == pytest.approx(50.0, abs=0.1)

    def test_probe_raises_if_ffmpeg_unavailable(self, tmp_path):
        mp = _make_processor(tmp_path, ffmpeg_ok=False)
        with pytest.raises(RuntimeError, match="ffmpeg"):
            mp.probe("/fake/video.mp4")

    def test_probe_raises_on_ffprobe_error(self, tmp_path):
        mp = _make_processor(tmp_path)
        with patch.object(
            mp, "_run",
            return_value=_make_completed_process("", returncode=1)
        ):
            with pytest.raises(RuntimeError):
                mp.probe("/fake/video.mp4")


# ---------------------------------------------------------------------------
# extract_frames
# ---------------------------------------------------------------------------

class TestExtractFrames:
    def test_extract_frames_calls_ffmpeg(self, tmp_path):
        mp       = _make_processor(tmp_path)
        frame_dir = tmp_path / "media" / "frames" / FAKE_VIDEO_RECORD.file_id
        frame_dir.mkdir(parents=True)

        # Pre-create fake frames so glob finds them
        (frame_dir / "frame_000001.jpg").write_bytes(b"\xFF\xD8\xFF" + b"\x00" * 100)
        (frame_dir / "frame_000002.jpg").write_bytes(b"\xFF\xD8\xFF" + b"\x00" * 100)

        called_cmds: list[list[str]] = []

        def fake_run(cmd, **kw):
            called_cmds.append(cmd)
            return _make_completed_process()

        with patch.object(mp, "_run", side_effect=fake_run):
            frames = mp.extract_frames(FAKE_VIDEO_RECORD, rate=1.0)

        assert any("ffmpeg" in cmd[0] for cmd in called_cmds)
        assert len(frames) == 2
        assert all(isinstance(f, Frame) for f in frames)

    def test_extract_frames_returns_empty_when_no_ffmpeg(self, tmp_path):
        mp = _make_processor(tmp_path, ffmpeg_ok=False)
        frames = mp.extract_frames(FAKE_VIDEO_RECORD)
        assert frames == []

    def test_extract_frames_start_end(self, tmp_path):
        mp        = _make_processor(tmp_path)
        frame_dir  = tmp_path / "media" / "frames" / FAKE_VIDEO_RECORD.file_id
        frame_dir.mkdir(parents=True)
        (frame_dir / "frame_000001.jpg").write_bytes(b"\xFF\xD8\xFF" + b"\x00" * 50)

        captured: dict = {}

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            return _make_completed_process()

        with patch.object(mp, "_run", side_effect=fake_run):
            mp.extract_frames(FAKE_VIDEO_RECORD, rate=2.0, start_sec=10.0, end_sec=20.0)

        cmd_str = " ".join(captured.get("cmd", []))
        assert "-ss" in cmd_str
        assert "10.0" in cmd_str


# ---------------------------------------------------------------------------
# frame_to_base64
# ---------------------------------------------------------------------------

class TestFrameToBase64:
    def test_frame_to_base64(self, tmp_path):
        jpg = tmp_path / "frame.jpg"
        jpg.write_bytes(b"\xFF\xD8\xFF" + b"\xAB" * 100)
        frame = Frame(
            frame_id="f1", video_id="v1", timestamp=0.0, path=str(jpg)
        )
        mp  = _make_processor(tmp_path)
        b64 = mp.frame_to_base64(frame)
        assert isinstance(b64, str)
        # Verify it decodes correctly
        decoded = base64.b64decode(b64)
        assert decoded == b"\xFF\xD8\xFF" + b"\xAB" * 100


# ---------------------------------------------------------------------------
# extract_clip
# ---------------------------------------------------------------------------

class TestExtractClip:
    def test_extract_clip_returns_clip(self, tmp_path):
        mp = _make_processor(tmp_path)

        def fake_run(cmd, **kw):
            # Simulate ffmpeg creating the output file
            out_arg = cmd[-1]
            Path(out_arg).write_bytes(b"\x00" * 256)
            return _make_completed_process()

        with patch.object(mp, "_run", side_effect=fake_run):
            clip = mp.extract_clip(FAKE_VIDEO_RECORD, start_sec=5.0, end_sec=15.0, label="sprint")

        assert isinstance(clip, Clip)
        assert clip.start_sec == 5.0
        assert clip.end_sec   == 15.0
        assert clip.label     == "sprint"
        assert clip.source_id == FAKE_VIDEO_RECORD.file_id

    def test_extract_clip_raises_when_ffmpeg_fails(self, tmp_path):
        mp = _make_processor(tmp_path)
        with patch.object(
            mp, "_run",
            return_value=_make_completed_process("", returncode=1)
        ):
            with pytest.raises(RuntimeError):
                mp.extract_clip(FAKE_VIDEO_RECORD, 0.0, 10.0)

    def test_extract_clip_raises_without_ffmpeg(self, tmp_path):
        mp = _make_processor(tmp_path, ffmpeg_ok=False)
        with pytest.raises(RuntimeError):
            mp.extract_clip(FAKE_VIDEO_RECORD, 0.0, 5.0)


# ---------------------------------------------------------------------------
# create_highlight_reel (concat)
# ---------------------------------------------------------------------------

class TestCreateHighlightReel:
    def _make_clips(self, tmp_path: Path, n: int = 2) -> list[Clip]:
        clips = []
        for i in range(n):
            p = tmp_path / f"clip_{i}.mp4"
            p.write_bytes(b"\x00" * 128)
            clips.append(
                Clip(
                    clip_id=f"c{i}", source_id="src",
                    start_sec=float(i * 10), end_sec=float(i * 10 + 10),
                    output_path=str(p),
                )
            )
        return clips

    def test_highlight_reel_writes_concat_list_and_calls_ffmpeg(self, tmp_path):
        mp    = _make_processor(tmp_path)
        clips = self._make_clips(tmp_path)
        out   = str(tmp_path / "reel.mp4")

        called: list[list[str]] = []

        def fake_run(cmd, **kw):
            called.append(cmd)
            return _make_completed_process()

        with patch.object(mp, "_run", side_effect=fake_run):
            result = mp.create_highlight_reel(clips, out)

        assert result == out
        assert any("concat" in " ".join(cmd) for cmd in called)
        # Concat list file should have been written
        assert (tmp_path / "media" / "concat_list.txt").exists()

    def test_highlight_reel_with_title(self, tmp_path):
        mp    = _make_processor(tmp_path)
        clips = self._make_clips(tmp_path)
        out   = str(tmp_path / "reel_titled.mp4")

        cmds: list[list[str]] = []

        def fake_run(cmd, **kw):
            cmds.append(cmd)
            return _make_completed_process()

        with patch.object(mp, "_run", side_effect=fake_run):
            mp.create_highlight_reel(clips, out, title_text="Training Highlights")

        full_cmd = " ".join(cmds[0]) if cmds else ""
        assert "drawtext" in full_cmd

    def test_highlight_reel_raises_on_empty_clips(self, tmp_path):
        mp = _make_processor(tmp_path)
        with pytest.raises(ValueError):
            mp.create_highlight_reel([], str(tmp_path / "reel.mp4"))


# ---------------------------------------------------------------------------
# extract_metrics from GPX
# ---------------------------------------------------------------------------

GPX_CONTENT = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">
  <trk>
    <trkseg>
      <trkpt lat="51.5074" lon="-0.1278">
        <ele>10.0</ele>
        <time>2024-01-15T09:00:00Z</time>
      </trkpt>
      <trkpt lat="51.5175" lon="-0.1278">
        <ele>12.0</ele>
        <time>2024-01-15T09:10:00Z</time>
      </trkpt>
      <trkpt lat="51.5276" lon="-0.1278">
        <ele>11.0</ele>
        <time>2024-01-15T09:20:00Z</time>
      </trkpt>
    </trkseg>
  </trk>
</gpx>"""


class TestGPXMetrics:
    def test_gpx_metrics_distance_positive(self, tmp_path):
        gpx = tmp_path / "run.gpx"
        gpx.write_text(GPX_CONTENT)
        mp      = _make_processor(tmp_path)
        metrics = mp.extract_metrics(str(gpx))
        assert metrics.distance_m > 0

    def test_gpx_metrics_duration(self, tmp_path):
        gpx = tmp_path / "run.gpx"
        gpx.write_text(GPX_CONTENT)
        mp      = _make_processor(tmp_path)
        metrics = mp.extract_metrics(str(gpx))
        # 2 intervals × 10 min = 1200s
        assert metrics.duration_sec == pytest.approx(1200.0, abs=5.0)

    def test_gpx_metrics_avg_speed(self, tmp_path):
        gpx = tmp_path / "run.gpx"
        gpx.write_text(GPX_CONTENT)
        mp      = _make_processor(tmp_path)
        metrics = mp.extract_metrics(str(gpx))
        assert metrics.avg_speed_ms > 0

    def test_gpx_returns_media_metrics_type(self, tmp_path):
        gpx = tmp_path / "run.gpx"
        gpx.write_text(GPX_CONTENT)
        mp      = _make_processor(tmp_path)
        metrics = mp.extract_metrics(str(gpx))
        assert isinstance(metrics, MediaMetrics)

    def test_gpx_invalid_returns_empty(self, tmp_path):
        bad = tmp_path / "bad.gpx"
        bad.write_text("this is not xml")
        mp      = _make_processor(tmp_path)
        metrics = mp.extract_metrics(str(bad))
        assert metrics.distance_m == 0.0


# ---------------------------------------------------------------------------
# CSV metrics
# ---------------------------------------------------------------------------

class TestCSVMetrics:
    def test_csv_metrics_extracts_hr_and_speed(self, tmp_path):
        csv_file = tmp_path / "session.csv"
        csv_file.write_text(
            "heart_rate,speed\n145,3.2\n150,3.5\n148,3.1\n"
        )
        mp      = _make_processor(tmp_path)
        metrics = mp.extract_metrics(str(csv_file))
        assert metrics.avg_hr    > 0
        assert metrics.avg_speed_ms > 0

    def test_csv_empty_returns_zero(self, tmp_path):
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("heart_rate,speed\n")
        mp      = _make_processor(tmp_path)
        metrics = mp.extract_metrics(str(csv_file))
        assert metrics.avg_hr == 0.0


# ---------------------------------------------------------------------------
# image_to_base64
# ---------------------------------------------------------------------------

class TestImageToBase64:
    def test_image_to_base64(self, tmp_path):
        img  = Image.new("RGB", (64, 64), color=(100, 150, 200))
        path = str(tmp_path / "test.png")
        img.save(path)
        mp  = _make_processor(tmp_path)
        b64 = mp.image_to_base64(path)
        assert isinstance(b64, str)
        # Verify decode doesn't raise
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0

    def test_resize_image(self, tmp_path):
        img  = Image.new("RGB", (1280, 960), color=(255, 0, 0))
        path = str(tmp_path / "large.jpg")
        img.save(path)
        mp       = _make_processor(tmp_path)
        new_path = mp.resize_image(path, max_width=320)
        resized  = Image.open(new_path)
        assert resized.width == 320

    def test_resize_image_no_op_when_smaller(self, tmp_path):
        img  = Image.new("RGB", (200, 150))
        path = str(tmp_path / "small.jpg")
        img.save(path)
        mp       = _make_processor(tmp_path)
        new_path = mp.resize_image(path, max_width=640)
        resized  = Image.open(new_path)
        assert resized.width == 200


# ---------------------------------------------------------------------------
# Graceful degradation without ffmpeg
# ---------------------------------------------------------------------------

class TestNoFfmpeg:
    def test_extract_frames_no_ffmpeg(self, tmp_path):
        mp     = _make_processor(tmp_path, ffmpeg_ok=False)
        frames = mp.extract_frames(FAKE_VIDEO_RECORD)
        assert frames == []

    def test_extract_clip_raises(self, tmp_path):
        mp = _make_processor(tmp_path, ffmpeg_ok=False)
        with pytest.raises(RuntimeError):
            mp.extract_clip(FAKE_VIDEO_RECORD, 0, 5)

    def test_highlight_reel_raises(self, tmp_path):
        mp    = _make_processor(tmp_path, ffmpeg_ok=False)
        clips = [Clip("c1", "src", 0, 5, "/tmp/a.mp4")]
        with pytest.raises(RuntimeError):
            mp.create_highlight_reel(clips, "/tmp/out.mp4")

    def test_metrics_from_video_returns_empty(self, tmp_path):
        mp = _make_processor(tmp_path, ffmpeg_ok=False)
        # mp4 extension triggers _metrics_from_video
        f = tmp_path / "vid.mp4"
        f.write_bytes(b"\x00" * 10)
        metrics = mp.extract_metrics(str(f))
        assert metrics.duration_sec == 0.0
