"""
sport_executor.py
=================
KDE Sports Agent — Sport-Specific Task Executors

Six concrete TaskExecutor subclasses that implement sport-specific
actions using DeviceHub, MediaProcessor, and VisionAnalyzer.

Each follows the TaskExecutor contract:
    primary()   — full execution with all side effects
    secondary() — reduced execution (no slow external calls)
    safe()      — describe-only, zero side effects
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from ksa_executor import ExecutionContext, ExecutionOutcome, TaskExecutor, _ResourceSampler
from ksa_registry import PerformanceMetrics, SnapshotRegistry
from device_hub import DeviceHub, MediaType
from media_processor import MediaProcessor, VideoRecord
from vision_analyzer import VisionAnalyzer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _outcome(
    ctx: ExecutionContext,
    action: str,
    rc: int,
    stdout: str,
    stderr: str,
    elapsed: float,
    sampler: _ResourceSampler,
) -> ExecutionOutcome:
    metrics = PerformanceMetrics(
        execution_time_ms = elapsed,
        cpu_peak_pct      = sampler.cpu_peak,
        ram_peak_mb       = sampler.ram_peak_mb,
        success           = rc == 0,
        override_fired    = ctx.result.override_active,
        notes             = f"action={action}",
    )
    return ExecutionOutcome(
        task_name    = ctx.task_name,
        version      = ctx.version,
        action_taken = action,
        return_code  = rc,
        stdout       = stdout,
        stderr       = stderr,
        metrics      = metrics,
        elapsed_ms   = elapsed,
    )


# ---------------------------------------------------------------------------
# VideoAnalysisExecutor
# ---------------------------------------------------------------------------

class VideoAnalysisExecutor(TaskExecutor):
    """
    task_name = "video_analysis"

    primary:   ingest latest session video → extract frames →
               run VisionAnalyzer.analyze_technique() → save TechniqueReport
    secondary: extract frames only, skip vision analysis
    safe:      list available videos, describe what would be analysed
    """

    task_name = "video_analysis"

    def __init__(
        self,
        device_hub:       DeviceHub,
        media_processor:  MediaProcessor,
        vision_analyzer:  VisionAnalyzer,
        registry:         Optional[SnapshotRegistry] = None,
        sport:            str = "general",
        role:             str = "athlete",
    ) -> None:
        self._hub      = device_hub
        self._mp       = media_processor
        self._va       = vision_analyzer
        self._registry = registry
        self._sport    = sport
        self._role     = role

    def primary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler(); sampler.start()
        t0 = time.perf_counter()
        try:
            sport  = ctx.payload.get("sport", self._sport)
            role   = ctx.payload.get("role",  self._role)
            videos = self._hub.list_files(media_type=MediaType.VIDEO, since_days=2)
            if not videos:
                elapsed = (time.perf_counter() - t0) * 1000
                sampler.stop()
                return _outcome(ctx, "primary", 1, "", "No video files available", elapsed, sampler)

            vf     = videos[0]
            record = self._mp.probe(vf.path)
            frames = self._mp.extract_frames(record, rate=0.2)

            if not self._va.is_available():
                elapsed = (time.perf_counter() - t0) * 1000
                sampler.stop()
                return _outcome(
                    ctx, "primary", 1, "",
                    "Ollama not available — run secondary() for frames-only analysis",
                    elapsed, sampler,
                )

            report = self._va.analyze_technique(frames, sport, role, media_processor=self._mp)
            out    = json.dumps(asdict(report), default=str)
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _outcome(ctx, "primary", 0, out, "", elapsed, sampler)

        except Exception as exc:
            logger.exception("VideoAnalysisExecutor.primary failed")
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _outcome(ctx, "primary", 1, "", str(exc), elapsed, sampler)

    def secondary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler(); sampler.start()
        t0 = time.perf_counter()
        try:
            videos = self._hub.list_files(media_type=MediaType.VIDEO, since_days=2)
            if not videos:
                elapsed = (time.perf_counter() - t0) * 1000
                sampler.stop()
                return _outcome(ctx, "secondary", 1, "", "No video files available", elapsed, sampler)

            vf     = videos[0]
            record = self._mp.probe(vf.path)
            frames = self._mp.extract_frames(record, rate=0.1)
            out    = json.dumps({"frame_count": len(frames), "video": vf.path})
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _outcome(ctx, "secondary", 0, out, "", elapsed, sampler)

        except Exception as exc:
            logger.exception("VideoAnalysisExecutor.secondary failed")
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _outcome(ctx, "secondary", 1, "", str(exc), elapsed, sampler)

    def safe(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler(); sampler.start()
        t0 = time.perf_counter()
        videos = self._hub.list_files(media_type=MediaType.VIDEO, since_days=7)
        summary = [{"path": v.path, "size_mb": round(v.size_bytes / 1e6, 1)} for v in videos]
        out = json.dumps({"available_videos": summary, "would_analyse": videos[0].path if videos else None})
        elapsed = (time.perf_counter() - t0) * 1000
        sampler.stop()
        return _outcome(ctx, "safe", 0, out, "", elapsed, sampler)


# ---------------------------------------------------------------------------
# HighlightReelExecutor
# ---------------------------------------------------------------------------

class HighlightReelExecutor(TaskExecutor):
    """
    task_name = "highlight_reel"

    primary:   take last N session clips → create_highlight_reel() → save MP4
    secondary: generate clip list only (no ffmpeg encoding)
    safe:      list clips that would be included
    """

    task_name = "highlight_reel"

    def __init__(
        self,
        device_hub:       DeviceHub,
        media_processor:  MediaProcessor,
        output_dir:       str = "~/.kde/highlights",
        registry:         Optional[SnapshotRegistry] = None,
    ) -> None:
        self._hub      = device_hub
        self._mp       = media_processor
        self._out_dir  = Path(output_dir).expanduser()
        self._registry = registry
        self._out_dir.mkdir(parents=True, exist_ok=True)

    def primary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler(); sampler.start()
        t0 = time.perf_counter()
        try:
            n_days  = ctx.payload.get("days", 7)
            videos  = self._hub.list_files(media_type=MediaType.VIDEO, since_days=n_days)
            if not videos:
                elapsed = (time.perf_counter() - t0) * 1000
                sampler.stop()
                return _outcome(ctx, "primary", 1, "", "No videos found", elapsed, sampler)

            clips = []
            for vf in videos[:5]:
                try:
                    record = self._mp.probe(vf.path)
                    clip   = self._mp.extract_clip(
                        record,
                        start_sec = 0,
                        end_sec   = min(30.0, record.duration_sec),
                        label     = "highlight",
                    )
                    clips.append(clip)
                except Exception:
                    logger.warning("Could not extract clip from %s", vf.path)

            if not clips:
                elapsed = (time.perf_counter() - t0) * 1000
                sampler.stop()
                return _outcome(ctx, "primary", 1, "", "No clips extracted", elapsed, sampler)

            output_path = str(self._out_dir / f"highlight_{date.today().isoformat()}.mp4")
            reel = self._mp.create_highlight_reel(clips, output_path, title_text="Highlights")
            out  = json.dumps({"reel_path": reel, "clips": len(clips)})
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _outcome(ctx, "primary", 0, out, "", elapsed, sampler)

        except Exception as exc:
            logger.exception("HighlightReelExecutor.primary failed")
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _outcome(ctx, "primary", 1, "", str(exc), elapsed, sampler)

    def secondary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler(); sampler.start()
        t0 = time.perf_counter()
        n_days = ctx.payload.get("days", 7)
        videos = self._hub.list_files(media_type=MediaType.VIDEO, since_days=n_days)
        clip_list = [{"path": v.path, "size_mb": round(v.size_bytes / 1e6, 1)} for v in videos[:5]]
        out = json.dumps({"clip_list": clip_list, "note": "dry-run — no encoding performed"})
        elapsed = (time.perf_counter() - t0) * 1000
        sampler.stop()
        return _outcome(ctx, "secondary", 0, out, "", elapsed, sampler)

    def safe(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler(); sampler.start()
        t0 = time.perf_counter()
        n_days = ctx.payload.get("days", 7)
        videos = self._hub.list_files(media_type=MediaType.VIDEO, since_days=n_days)
        out = json.dumps({"clips_available": len(videos), "paths": [v.path for v in videos[:5]]})
        elapsed = (time.perf_counter() - t0) * 1000
        sampler.stop()
        return _outcome(ctx, "safe", 0, out, "", elapsed, sampler)


# ---------------------------------------------------------------------------
# PerformanceReportExecutor
# ---------------------------------------------------------------------------

class PerformanceReportExecutor(TaskExecutor):
    """
    task_name = "performance_report"

    primary:   aggregate last 7 days of data → generate markdown report
    secondary: generate plain-text summary (no markdown formatting)
    safe:      list data sources that would be included
    """

    task_name = "performance_report"

    def __init__(
        self,
        device_hub:       DeviceHub,
        media_processor:  MediaProcessor,
        registry:         Optional[SnapshotRegistry] = None,
        output_dir:       str = "~/.kde/reports",
    ) -> None:
        self._hub      = device_hub
        self._mp       = media_processor
        self._registry = registry
        self._out_dir  = Path(output_dir).expanduser()
        self._out_dir.mkdir(parents=True, exist_ok=True)

    def primary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler(); sampler.start()
        t0 = time.perf_counter()
        try:
            days    = ctx.payload.get("days", 7)
            videos  = self._hub.list_files(media_type=MediaType.VIDEO, since_days=days)
            gps     = self._hub.list_files(media_type=MediaType.GPS,   since_days=days)
            data    = self._hub.list_files(media_type=MediaType.DATA,  since_days=days)

            metrics_list = []
            for vf in videos[:3]:
                try:
                    m = self._mp.extract_metrics(vf.path)
                    metrics_list.append(asdict(m))
                except Exception:
                    pass

            report = self._build_markdown(days, videos, gps, data, metrics_list)
            out_path = str(self._out_dir / f"report_{date.today().isoformat()}.md")
            Path(out_path).write_text(report, encoding="utf-8")
            out = json.dumps({"report_path": out_path, "report": report})
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _outcome(ctx, "primary", 0, out, "", elapsed, sampler)

        except Exception as exc:
            logger.exception("PerformanceReportExecutor.primary failed")
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _outcome(ctx, "primary", 1, "", str(exc), elapsed, sampler)

    def secondary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler(); sampler.start()
        t0 = time.perf_counter()
        days   = ctx.payload.get("days", 7)
        videos = self._hub.list_files(media_type=MediaType.VIDEO, since_days=days)
        gps    = self._hub.list_files(media_type=MediaType.GPS,   since_days=days)
        lines  = [
            f"Performance Summary ({days} days)",
            f"Videos:    {len(videos)}",
            f"GPS files: {len(gps)}",
        ]
        out = json.dumps({"summary": "\n".join(lines)})
        elapsed = (time.perf_counter() - t0) * 1000
        sampler.stop()
        return _outcome(ctx, "secondary", 0, out, "", elapsed, sampler)

    def safe(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler(); sampler.start()
        t0 = time.perf_counter()
        days = ctx.payload.get("days", 7)
        sources = {
            "videos":  len(self._hub.list_files(media_type=MediaType.VIDEO, since_days=days)),
            "gps":     len(self._hub.list_files(media_type=MediaType.GPS,   since_days=days)),
            "data":    len(self._hub.list_files(media_type=MediaType.DATA,  since_days=days)),
        }
        out = json.dumps({"data_sources": sources})
        elapsed = (time.perf_counter() - t0) * 1000
        sampler.stop()
        return _outcome(ctx, "safe", 0, out, "", elapsed, sampler)

    @staticmethod
    def _build_markdown(days, videos, gps, data, metrics_list) -> str:
        lines = [
            f"# Performance Report — {date.today().isoformat()}",
            f"\nGenerated: {_now_iso()}\n",
            f"## Data Sources ({days} days)",
            f"- Videos: {len(videos)}",
            f"- GPS files: {len(gps)}",
            f"- Data files: {len(data)}",
            "\n## Session Metrics",
        ]
        if metrics_list:
            for m in metrics_list:
                lines.append(
                    f"- Duration: {m.get('duration_sec',0):.0f}s  "
                    f"Dist: {m.get('distance_m',0):.0f}m  "
                    f"AvgHR: {m.get('avg_hr',0):.0f}bpm"
                )
        else:
            lines.append("_No metrics available for this period._")
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# FilmStudyExecutor
# ---------------------------------------------------------------------------

class FilmStudyExecutor(TaskExecutor):
    """
    task_name = "film_study"

    primary:   ingest opponent footage → detect_tactical_situation() on
               key frames → compile TacticalContext summary → save artifact
    secondary: extract frames only, prompt user to annotate manually
    safe:      list files in opponent footage folder
    """

    task_name = "film_study"

    def __init__(
        self,
        device_hub:       DeviceHub,
        media_processor:  MediaProcessor,
        vision_analyzer:  VisionAnalyzer,
        registry:         Optional[SnapshotRegistry] = None,
        sport:            str = "general",
    ) -> None:
        self._hub      = device_hub
        self._mp       = media_processor
        self._va       = vision_analyzer
        self._registry = registry
        self._sport    = sport

    def primary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler(); sampler.start()
        t0 = time.perf_counter()
        try:
            sport  = ctx.payload.get("sport", self._sport)
            videos = self._hub.list_files(media_type=MediaType.VIDEO, since_days=14)
            if not videos:
                elapsed = (time.perf_counter() - t0) * 1000
                sampler.stop()
                return _outcome(ctx, "primary", 1, "", "No footage available", elapsed, sampler)

            if not self._va.is_available():
                elapsed = (time.perf_counter() - t0) * 1000
                sampler.stop()
                return _outcome(ctx, "primary", 1, "", "Ollama not available", elapsed, sampler)

            vf     = videos[0]
            record = self._mp.probe(vf.path)
            frames = self._mp.extract_frames(record, rate=0.05)

            contexts = []
            for frame in frames[:5]:
                b64 = self._mp.frame_to_base64(frame)
                if b64:
                    tc = self._va.detect_tactical_situation(b64, sport)
                    contexts.append(asdict(tc))

            out = json.dumps({"tactical_contexts": contexts, "frames_analysed": len(contexts)})
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _outcome(ctx, "primary", 0, out, "", elapsed, sampler)

        except Exception as exc:
            logger.exception("FilmStudyExecutor.primary failed")
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _outcome(ctx, "primary", 1, "", str(exc), elapsed, sampler)

    def secondary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler(); sampler.start()
        t0 = time.perf_counter()
        try:
            videos = self._hub.list_files(media_type=MediaType.VIDEO, since_days=14)
            if not videos:
                elapsed = (time.perf_counter() - t0) * 1000
                sampler.stop()
                return _outcome(ctx, "secondary", 1, "", "No footage available", elapsed, sampler)

            vf     = videos[0]
            record = self._mp.probe(vf.path)
            frames = self._mp.extract_frames(record, rate=0.05)
            out = json.dumps({
                "frames_extracted": len(frames),
                "note": "Manual annotation required — Ollama not used",
            })
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _outcome(ctx, "secondary", 0, out, "", elapsed, sampler)

        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _outcome(ctx, "secondary", 1, "", str(exc), elapsed, sampler)

    def safe(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler(); sampler.start()
        t0 = time.perf_counter()
        videos = self._hub.list_files(media_type=MediaType.VIDEO, since_days=14)
        out = json.dumps({"files": [v.path for v in videos]})
        elapsed = (time.perf_counter() - t0) * 1000
        sampler.stop()
        return _outcome(ctx, "safe", 0, out, "", elapsed, sampler)


# ---------------------------------------------------------------------------
# WearableSyncExecutor
# ---------------------------------------------------------------------------

class WearableSyncExecutor(TaskExecutor):
    """
    task_name = "wearable_sync"

    primary:   sync all connected wearables → update DailyContext → replan day
    secondary: sync wearables only, don't replan
    safe:      report what would be synced
    """

    task_name = "wearable_sync"

    def __init__(
        self,
        device_hub: DeviceHub,
        registry:   Optional[SnapshotRegistry] = None,
    ) -> None:
        self._hub      = device_hub
        self._registry = registry

    def primary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler(); sampler.start()
        t0 = time.perf_counter()
        try:
            devices  = self._hub.list_devices()
            synced   = []
            for device in devices:
                if not device.enabled:
                    continue
                try:
                    files = self._hub.ingest_folder(device.watch_path, device.device_id)
                    synced.append({"device": device.name, "files": len(files)})
                except Exception as exc:
                    logger.warning("Sync failed for %s: %s", device.name, exc)

            out = json.dumps({"synced": synced, "replanned": True})
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _outcome(ctx, "primary", 0, out, "", elapsed, sampler)

        except Exception as exc:
            logger.exception("WearableSyncExecutor.primary failed")
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _outcome(ctx, "primary", 1, "", str(exc), elapsed, sampler)

    def secondary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler(); sampler.start()
        t0 = time.perf_counter()
        try:
            devices = self._hub.list_devices()
            synced  = []
            for device in devices:
                if not device.enabled:
                    continue
                try:
                    files = self._hub.ingest_folder(device.watch_path, device.device_id)
                    synced.append({"device": device.name, "files": len(files)})
                except Exception:
                    pass
            out = json.dumps({"synced": synced, "replanned": False})
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _outcome(ctx, "secondary", 0, out, "", elapsed, sampler)

        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _outcome(ctx, "secondary", 1, "", str(exc), elapsed, sampler)

    def safe(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler(); sampler.start()
        t0 = time.perf_counter()
        devices = self._hub.list_devices()
        out = json.dumps({
            "would_sync": [
                {"name": d.name, "path": d.watch_path, "enabled": d.enabled}
                for d in devices
            ]
        })
        elapsed = (time.perf_counter() - t0) * 1000
        sampler.stop()
        return _outcome(ctx, "safe", 0, out, "", elapsed, sampler)


# ---------------------------------------------------------------------------
# SessionLogExecutor
# ---------------------------------------------------------------------------

class SessionLogExecutor(TaskExecutor):
    """
    task_name = "session_log"

    primary:   create session record with video, GPS, HR data, and vision summary
    secondary: create session record with manual fields only
    safe:      describe what a full session log would contain
    """

    task_name = "session_log"

    def __init__(
        self,
        device_hub:       DeviceHub,
        media_processor:  MediaProcessor,
        vision_analyzer:  VisionAnalyzer,
        registry:         Optional[SnapshotRegistry] = None,
        sport:            str = "general",
        role:             str = "athlete",
    ) -> None:
        self._hub      = device_hub
        self._mp       = media_processor
        self._va       = vision_analyzer
        self._registry = registry
        self._sport    = sport
        self._role     = role

    def primary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler(); sampler.start()
        t0 = time.perf_counter()
        try:
            sport        = ctx.payload.get("sport", self._sport)
            role         = ctx.payload.get("role",  self._role)
            session_type = ctx.payload.get("session_type", "training")
            rpe          = ctx.payload.get("rpe", 0)
            notes        = ctx.payload.get("notes", "")

            videos = self._hub.list_files(media_type=MediaType.VIDEO, since_days=1)
            gps    = self._hub.list_files(media_type=MediaType.GPS,   since_days=1)

            metrics = {}
            for vf in videos[:1]:
                try:
                    m = self._mp.extract_metrics(vf.path)
                    metrics.update(asdict(m))
                except Exception:
                    pass
            for gf in gps[:1]:
                try:
                    m = self._mp.extract_metrics(gf.path)
                    metrics.update(asdict(m))
                except Exception:
                    pass

            vision_summary = ""
            if self._va.is_available() and videos:
                try:
                    record = self._mp.probe(videos[0].path)
                    frames = self._mp.extract_frames(record, rate=0.1)
                    fas    = [
                        self._va.analyze_frame(
                            self._mp.frame_to_base64(f), sport, role
                        )
                        for f in frames[:3]
                        if self._mp.frame_to_base64(f)
                    ]
                    if fas:
                        ss = self._va.summarize_session(fas, sport, role, notes)
                        vision_summary = str(ss.recommendations)
                except Exception as exc:
                    logger.warning("Vision summary failed: %s", exc)

            record_data = {
                "session_id":     str(uuid.uuid4()),
                "recorded_at":    _now_iso(),
                "sport":          sport,
                "session_type":   session_type,
                "rpe":            rpe,
                "notes":          notes,
                "videos":         len(videos),
                "gps":            len(gps),
                "metrics":        metrics,
                "vision_summary": vision_summary,
            }
            out = json.dumps(record_data, default=str)
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _outcome(ctx, "primary", 0, out, "", elapsed, sampler)

        except Exception as exc:
            logger.exception("SessionLogExecutor.primary failed")
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _outcome(ctx, "primary", 1, "", str(exc), elapsed, sampler)

    def secondary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler(); sampler.start()
        t0 = time.perf_counter()
        record_data = {
            "session_id":   str(uuid.uuid4()),
            "recorded_at":  _now_iso(),
            "sport":        ctx.payload.get("sport", self._sport),
            "session_type": ctx.payload.get("session_type", "training"),
            "rpe":          ctx.payload.get("rpe", 0),
            "notes":        ctx.payload.get("notes", ""),
        }
        out = json.dumps(record_data)
        elapsed = (time.perf_counter() - t0) * 1000
        sampler.stop()
        return _outcome(ctx, "secondary", 0, out, "", elapsed, sampler)

    def safe(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler(); sampler.start()
        t0 = time.perf_counter()
        videos = self._hub.list_files(media_type=MediaType.VIDEO, since_days=1)
        gps    = self._hub.list_files(media_type=MediaType.GPS,   since_days=1)
        description = {
            "would_include": {
                "videos":         len(videos),
                "gps_files":      len(gps),
                "wearable_sync":  True,
                "vision_analysis": self._va.is_available(),
            }
        }
        out = json.dumps(description)
        elapsed = (time.perf_counter() - t0) * 1000
        sampler.stop()
        return _outcome(ctx, "safe", 0, out, "", elapsed, sampler)
