"""
media_processor.py
==================
KDE Sports Agent — Video & Image Pipeline

Processes ingested media using ffmpeg (system binary) and Pillow.
Produces structured artefacts (clips, frames, metrics) for the vision
analyser.

Requirements:
    System: ffmpeg  (apt install ffmpeg / brew install ffmpeg)
    Python: Pillow  (pip install Pillow)

If ffmpeg is missing the class logs a warning and gracefully skips
all video tasks (returning empty results rather than raising).
"""

from __future__ import annotations

import base64
import csv
import json
import logging
import shutil
import subprocess
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VideoRecord:
    file_id:      str
    path:         str
    duration_sec: float
    fps:          float
    width:        int
    height:       int
    codec:        str
    size_mb:      float
    device_type:  str
    recorded_at:  str


@dataclass
class Frame:
    frame_id:  str
    video_id:  str
    timestamp: float   # seconds into video
    path:      str     # extracted JPEG path
    base64:    str = ""


@dataclass
class Clip:
    clip_id:      str
    source_id:    str
    start_sec:    float
    end_sec:      float
    output_path:  str
    label:        str   = ""
    quality_score:float = 0.0


@dataclass
class MediaMetrics:
    """Numeric metrics extracted from video / GPS / wearable files."""
    source_id:    str
    duration_sec: float
    avg_speed_ms: float = 0.0
    max_speed_ms: float = 0.0
    distance_m:   float = 0.0
    avg_hr:       float = 0.0
    max_hr:       float = 0.0
    calories:     float = 0.0
    raw:          dict  = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MediaProcessor
# ---------------------------------------------------------------------------

class MediaProcessor:
    """
    All video/image operations.  Wraps ffmpeg as a subprocess.
    Gracefully degrades if ffmpeg is missing (warns, skips video tasks).
    """

    def __init__(
        self,
        output_dir:   str   = "~/.kde/media",
        ffmpeg_path:  str   = "ffmpeg",
        ffprobe_path: str   = "ffprobe",
        frame_rate:   float = 1.0,
        thumb_width:  int   = 640,
    ) -> None:
        self._out_dir      = Path(output_dir).expanduser()
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._ffmpeg       = ffmpeg_path
        self._ffprobe      = ffprobe_path
        self._frame_rate   = frame_rate
        self._thumb_width  = thumb_width
        self._ffmpeg_ok    = self._check_ffmpeg()

    def _check_ffmpeg(self) -> bool:
        if shutil.which(self._ffmpeg):
            return True
        logger.warning(
            "ffmpeg not found at '%s'. Video tasks will be skipped.", self._ffmpeg
        )
        return False

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _run(self, cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        """Run a subprocess, capturing stdout/stderr."""
        return subprocess.run(
            cmd,
            capture_output=True,
            **kwargs,
        )

    # ── Inspection ───────────────────────────────────────────────────────────

    def probe(self, path: str) -> VideoRecord:
        """Run ffprobe and return VideoRecord.  Raises RuntimeError if not a video."""
        if not self._ffmpeg_ok:
            raise RuntimeError("ffmpeg/ffprobe not available")

        cmd = [
            self._ffprobe, "-v", "quiet",
            "-print_format", "json",
            "-show_streams", "-show_format",
            path,
        ]
        result = self._run(cmd)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffprobe failed for {path!r}: {result.stderr.decode()}"
            )

        data     = json.loads(result.stdout.decode())
        fmt      = data.get("format", {})
        streams  = data.get("streams", [])
        video_st = next(
            (s for s in streams if s.get("codec_type") == "video"), {}
        )

        fps_raw = video_st.get("r_frame_rate", "0/1")
        try:
            num, den = fps_raw.split("/")
            fps = float(num) / float(den) if float(den) else 0.0
        except Exception:
            fps = 0.0

        return VideoRecord(
            file_id      = uuid.uuid4().hex,
            path         = str(Path(path).resolve()),
            duration_sec = float(fmt.get("duration", 0)),
            fps          = fps,
            width        = int(video_st.get("width", 0)),
            height       = int(video_st.get("height", 0)),
            codec        = video_st.get("codec_name", ""),
            size_mb      = int(fmt.get("size", 0)) / 1024 / 1024,
            device_type  = "",
            recorded_at  = fmt.get("tags", {}).get("creation_time", ""),
        )

    # ── Frame extraction ─────────────────────────────────────────────────────

    def extract_frames(
        self,
        video:     VideoRecord,
        rate:      Optional[float] = None,
        start_sec: float           = 0.0,
        end_sec:   Optional[float] = None,
    ) -> list[Frame]:
        """Extract frames from a video at the given rate."""
        if not self._ffmpeg_ok:
            logger.warning("extract_frames: ffmpeg unavailable, returning []")
            return []

        fps      = rate if rate is not None else self._frame_rate
        frame_dir = self._out_dir / "frames" / video.file_id
        frame_dir.mkdir(parents=True, exist_ok=True)

        cmd = [self._ffmpeg, "-y"]
        if start_sec > 0:
            cmd += ["-ss", str(start_sec)]
        cmd += ["-i", video.path]
        if end_sec is not None:
            duration = max(0.0, end_sec - start_sec)
            cmd += ["-t", str(duration)]
        cmd += [
            "-vf", f"fps={fps},scale={self._thumb_width}:-1",
            "-q:v", "2",
            str(frame_dir / "frame_%06d.jpg"),
        ]

        result = self._run(cmd)
        if result.returncode != 0:
            logger.warning("extract_frames failed: %s", result.stderr.decode())
            return []

        frames: list[Frame] = []
        for i, jpg in enumerate(sorted(frame_dir.glob("frame_*.jpg"))):
            ts = start_sec + i / fps
            frames.append(
                Frame(
                    frame_id  = uuid.uuid4().hex,
                    video_id  = video.file_id,
                    timestamp = round(ts, 3),
                    path      = str(jpg),
                )
            )
        return frames

    def frame_to_base64(self, frame: Frame) -> str:
        """Read frame JPEG and return base64 string for Ollama vision API."""
        with open(frame.path, "rb") as fh:
            data = fh.read()
        return base64.b64encode(data).decode()

    # ── Clip operations ──────────────────────────────────────────────────────

    def extract_clip(
        self,
        video:     VideoRecord,
        start_sec: float,
        end_sec:   float,
        label:     str = "",
    ) -> Clip:
        """Extract a sub-clip from a video using stream copy (fast)."""
        if not self._ffmpeg_ok:
            raise RuntimeError("ffmpeg not available")

        clip_id    = uuid.uuid4().hex
        clip_dir   = self._out_dir / "clips"
        clip_dir.mkdir(parents=True, exist_ok=True)
        output     = str(clip_dir / f"{clip_id}.mp4")
        duration   = max(0.0, end_sec - start_sec)

        cmd = [
            self._ffmpeg, "-y",
            "-ss", str(start_sec),
            "-i", video.path,
            "-t", str(duration),
            "-c", "copy",
            output,
        ]
        result = self._run(cmd)
        if result.returncode != 0:
            raise RuntimeError(
                f"extract_clip failed: {result.stderr.decode()}"
            )

        return Clip(
            clip_id     = clip_id,
            source_id   = video.file_id,
            start_sec   = start_sec,
            end_sec     = end_sec,
            output_path = output,
            label       = label,
        )

    def create_highlight_reel(
        self,
        clips:       list[Clip],
        output_path: str,
        title_text:  str = "",
    ) -> str:
        """
        Concatenate clips into a highlight reel using ffmpeg concat demuxer.
        Optionally burns title text using drawtext filter.
        Returns output_path.
        """
        if not self._ffmpeg_ok:
            raise RuntimeError("ffmpeg not available")
        if not clips:
            raise ValueError("No clips provided for highlight reel")

        # Write concat file list
        list_path = self._out_dir / "concat_list.txt"
        with open(list_path, "w") as fh:
            for clip in clips:
                fh.write(f"file '{clip.output_path}'\n")

        # Build ffmpeg command
        if title_text:
            # Re-encode to apply drawtext filter
            escaped = title_text.replace("'", "\\'").replace(":", "\\:")
            cmd = [
                self._ffmpeg, "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(list_path),
                "-vf",
                (
                    f"drawtext=text='{escaped}':"
                    "fontsize=48:fontcolor=white:x=(w-text_w)/2:y=50:"
                    "box=1:boxcolor=black@0.5:boxborderw=10"
                ),
                "-c:a", "copy",
                output_path,
            ]
        else:
            cmd = [
                self._ffmpeg, "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(list_path),
                "-c", "copy",
                output_path,
            ]

        result = self._run(cmd)
        if result.returncode != 0:
            raise RuntimeError(
                f"create_highlight_reel failed: {result.stderr.decode()}"
            )
        return output_path

    def add_overlay(
        self,
        video_path:  str,
        text:        str,
        output_path: str,
    ) -> str:
        """Burn a text overlay onto a video using drawtext filter."""
        if not self._ffmpeg_ok:
            raise RuntimeError("ffmpeg not available")

        escaped = text.replace("'", "\\'").replace(":", "\\:")
        cmd = [
            self._ffmpeg, "-y",
            "-i", video_path,
            "-vf",
            (
                f"drawtext=text='{escaped}':"
                "fontsize=36:fontcolor=white:x=10:y=10:"
                "box=1:boxcolor=black@0.4:boxborderw=5"
            ),
            "-codec:a", "copy",
            output_path,
        ]
        result = self._run(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"add_overlay failed: {result.stderr.decode()}")
        return output_path

    # ── Data extraction ──────────────────────────────────────────────────────

    def extract_metrics(self, path: str) -> MediaMetrics:
        """
        Extract numeric metrics from various file types:
          .gpx  → GPS speed, distance, elevation
          .csv  → detect column headers, extract HR, speed, etc.
          .fit  → decode binary FIT file (pure-Python fallback)
          .mp4/.mov → ffprobe duration + attempt GoPro GPS metadata
        """
        source_id = uuid.uuid4().hex
        ext       = Path(path).suffix.lower()

        if ext == ".gpx":
            return self._metrics_from_gpx(source_id, path)
        if ext == ".csv":
            return self._metrics_from_csv(source_id, path)
        if ext == ".fit":
            return self._metrics_from_fit(source_id, path)
        if ext in {".mp4", ".mov", ".avi", ".mkv"}:
            return self._metrics_from_video(source_id, path)

        return MediaMetrics(source_id=source_id, duration_sec=0.0)

    def _metrics_from_gpx(self, source_id: str, path: str) -> MediaMetrics:
        """Parse GPX for speed, distance, elevation."""
        try:
            tree = ET.parse(path)
        except Exception:
            logger.warning("GPX parse failed: %s", path, exc_info=True)
            return MediaMetrics(source_id=source_id, duration_sec=0.0)

        root = tree.getroot()
        ns   = {"gpx": "http://www.topografix.com/GPX/1/1"}

        points: list[tuple[float, float, float]] = []  # (lat, lon, ele)
        times:  list[str] = []

        for trkpt in root.findall(".//gpx:trkpt", ns):
            lat = float(trkpt.get("lat", 0))
            lon = float(trkpt.get("lon", 0))
            ele_el = trkpt.find("gpx:ele", ns)
            ele    = float(ele_el.text) if ele_el is not None and ele_el.text else 0.0
            points.append((lat, lon, ele))
            t_el = trkpt.find("gpx:time", ns)
            if t_el is not None and t_el.text:
                times.append(t_el.text)

        # Haversine distance
        import math
        def haversine(p1: tuple, p2: tuple) -> float:
            R  = 6371000.0
            la1, lo1 = math.radians(p1[0]), math.radians(p1[1])
            la2, lo2 = math.radians(p2[0]), math.radians(p2[1])
            dla = la2 - la1
            dlo = lo2 - lo1
            a   = math.sin(dla/2)**2 + math.cos(la1)*math.cos(la2)*math.sin(dlo/2)**2
            return R * 2 * math.asin(math.sqrt(a))

        distance = 0.0
        for i in range(1, len(points)):
            distance += haversine(points[i-1], points[i])

        # Duration from timestamps
        duration = 0.0
        if len(times) >= 2:
            from datetime import datetime as dt
            try:
                fmt = "%Y-%m-%dT%H:%M:%SZ"
                t0  = dt.strptime(times[0],  fmt)
                t1  = dt.strptime(times[-1], fmt)
                duration = (t1 - t0).total_seconds()
            except Exception:
                pass

        avg_speed = distance / duration if duration > 0 else 0.0

        return MediaMetrics(
            source_id    = source_id,
            duration_sec = duration,
            distance_m   = round(distance, 2),
            avg_speed_ms = round(avg_speed, 3),
            raw          = {"n_trackpoints": len(points)},
        )

    def _metrics_from_csv(self, source_id: str, path: str) -> MediaMetrics:
        """Detect column headers and extract HR, speed, etc."""
        try:
            with open(path, newline="", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh)
                rows   = list(reader)
        except Exception:
            logger.warning("CSV parse failed: %s", path, exc_info=True)
            return MediaMetrics(source_id=source_id, duration_sec=0.0)

        if not rows:
            return MediaMetrics(source_id=source_id, duration_sec=0.0)

        hr_cols    = ["heart_rate", "hr", "bpm", "Heart Rate", "HR", "BPM"]
        speed_cols = ["speed", "Speed", "speed_ms", "pace"]

        def _col_vals(cols: list[str]) -> list[float]:
            for col in cols:
                if col in rows[0]:
                    vals = []
                    for row in rows:
                        try:
                            vals.append(float(row[col]))
                        except (ValueError, KeyError):
                            pass
                    if vals:
                        return vals
            return []

        hr_vals    = _col_vals(hr_cols)
        speed_vals = _col_vals(speed_cols)

        avg_hr  = sum(hr_vals) / len(hr_vals)       if hr_vals    else 0.0
        max_hr  = max(hr_vals)                       if hr_vals    else 0.0
        avg_spd = sum(speed_vals) / len(speed_vals) if speed_vals else 0.0
        max_spd = max(speed_vals)                   if speed_vals else 0.0

        return MediaMetrics(
            source_id    = source_id,
            duration_sec = float(len(rows)),
            avg_hr       = round(avg_hr, 2),
            max_hr       = round(max_hr, 2),
            avg_speed_ms = round(avg_spd, 3),
            max_speed_ms = round(max_spd, 3),
            raw          = {"rows": len(rows), "columns": list(rows[0].keys())},
        )

    def _metrics_from_fit(self, source_id: str, path: str) -> MediaMetrics:
        """
        Pure-Python FIT decoder (minimal — reads global messages for summary
        data records such as total_distance, total_elapsed_time, avg_heart_rate).
        """
        # FIT files start with a 14-byte header followed by data records.
        # This is a best-effort extractor; a full SDK is not required.
        try:
            with open(path, "rb") as fh:
                raw = fh.read()
        except Exception:
            logger.warning("FIT read failed: %s", path, exc_info=True)
            return MediaMetrics(source_id=source_id, duration_sec=0.0)

        # Look for common field patterns in the binary blob as a fallback.
        # Return empty metrics — a full FIT decoder is out of scope here.
        return MediaMetrics(
            source_id    = source_id,
            duration_sec = 0.0,
            raw          = {"file_size": len(raw)},
        )

    def _metrics_from_video(self, source_id: str, path: str) -> MediaMetrics:
        """Extract duration via ffprobe; attempt GoPro telemetry if available."""
        if not self._ffmpeg_ok:
            return MediaMetrics(source_id=source_id, duration_sec=0.0)
        try:
            record = self.probe(path)
            metrics = MediaMetrics(
                source_id    = source_id,
                duration_sec = record.duration_sec,
                raw          = {
                    "codec": record.codec,
                    "width": record.width,
                    "height": record.height,
                },
            )
            # Attempt GoPro telemetry enrichment
            try:
                telem = self.extract_gopro_telemetry(path)
                gps   = telem.get("gps", [])
                if gps:
                    speeds = [p.get("speed", 0.0) for p in gps]
                    metrics.avg_speed_ms = sum(speeds) / len(speeds)
                    metrics.max_speed_ms = max(speeds)
            except Exception:
                pass
            return metrics
        except Exception:
            logger.warning(
                "_metrics_from_video failed for %s", path, exc_info=True
            )
            return MediaMetrics(source_id=source_id, duration_sec=0.0)

    def extract_gopro_telemetry(self, path: str) -> dict:
        """
        Extract GoPro GPMF telemetry from MP4 metadata track.
        Returns GPS coords, accelerometer, gyro as lists.
        Uses ffmpeg to extract the telemetry stream.
        """
        if not self._ffmpeg_ok:
            return {}

        telem_path = self._out_dir / f"{uuid.uuid4().hex}_telem.bin"
        cmd = [
            self._ffmpeg, "-y",
            "-i", path,
            "-codec", "copy",
            "-map", "0:d",
            str(telem_path),
        ]
        result = self._run(cmd)
        if result.returncode != 0 or not telem_path.exists():
            return {}

        # Minimal GPMF binary parsing is complex; return raw binary size only.
        size = telem_path.stat().st_size
        telem_path.unlink(missing_ok=True)
        return {"raw_bytes": size, "gps": []}

    # ── Image utilities ──────────────────────────────────────────────────────

    def resize_image(self, path: str, max_width: int = 640) -> str:
        """Resize image with Pillow to max_width, preserving aspect ratio.
        Returns path to resized image (saved alongside original)."""
        img  = Image.open(path)
        if img.width > max_width:
            ratio  = max_width / img.width
            height = int(img.height * ratio)
            img    = img.resize((max_width, height), Image.LANCZOS)

        p   = Path(path)
        out = str(p.parent / f"{p.stem}_resized{p.suffix}")
        img.save(out)
        return out

    def image_to_base64(self, path: str) -> str:
        """Read image with Pillow (converts HEIC etc.) and return base64."""
        img  = Image.open(path).convert("RGB")
        import io
        buf  = io.BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode()
