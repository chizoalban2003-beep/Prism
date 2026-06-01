"""
daily_workflow.py
=================
KDE Sports Agent — Full Daily Lifecycle Orchestrator

Manages the complete practitioner day: morning briefing, session logging,
evening review, device watching, and weekly reporting.

Designed to run as a background service or be called directly.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from sports_pro import (
    SportsProAssistant,
    DailyPlan,
    WearableReading,
    WearableReader,
)
from device_hub import DeviceHub, DeviceType, MediaType
from media_processor import MediaProcessor
from vision_analyzer import VisionAnalyzer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MorningBrief:
    time:               str
    plan:               DailyPlan
    wearable_summary:   str
    device_status:      list[str]
    priority_tasks:     list[str]
    alerts:             list[str]
    match_intelligence: Optional[dict] = None


@dataclass
class SessionLog:
    session_id:     str
    start_time:     str
    end_time:       str
    sport:          str
    session_type:   str
    rpe:            int   = 0
    notes:          str   = ""
    video_paths:    list[str] = field(default_factory=list)
    gps_path:       str       = ""
    metrics:        dict      = field(default_factory=dict)
    vision_summary: str       = ""


@dataclass
class EveningReview:
    date_str:          str
    session_logs:      list[SessionLog]
    day_rating_prompt: str
    recovery_protocol: list[str]
    tomorrow_preview:  str
    sleep_target_hrs:  float


# ---------------------------------------------------------------------------
# DailyWorkflow
# ---------------------------------------------------------------------------

class DailyWorkflow:
    """
    The full daily lifecycle manager.

    Designed to be called at:
        morning_briefing()       — wake time
        log_session()            — after each training/match/gym block
        evening_review()         — end of day
        watch_for_sessions()     — continuous background monitoring
        generate_weekly_report() — weekly summary
    """

    def __init__(
        self,
        assistant:       SportsProAssistant,
        device_hub:      DeviceHub,
        media_processor: MediaProcessor,
        vision_analyzer: VisionAnalyzer,
        profile_name:    str,
    ) -> None:
        self._assistant  = assistant
        self._hub        = device_hub
        self._mp         = media_processor
        self._va         = vision_analyzer
        self._name       = profile_name
        self._sessions:  list[SessionLog] = []
        self._watch_thread: Optional[threading.Thread] = None
        self._stop_watch = threading.Event()

    # ── morning briefing ──────────────────────────────────────────────────

    def morning_briefing(
        self,
        manual_reading: Optional[WearableReading] = None,
    ) -> MorningBrief:
        """
        1. Sync enabled wearable devices
        2. Parse exports → WearableReading
        3. Plan the day
        4. List devices with pending data
        5. Extract top 3 priority tasks
        6. Check for warnings
        7. Return MorningBrief
        """
        now_str = datetime.now(tz=timezone.utc).isoformat()

        # --- Step 1 & 2: wearable reading ---
        reading = manual_reading
        wearable_summary = "Manual reading provided." if reading else ""

        if reading is None:
            reading, wearable_summary = self._auto_sync_wearables()

        # --- Step 3: plan the day ---
        try:
            plan = self._assistant.plan_day(self._name, reading=reading)
        except KeyError as exc:
            logger.error("Profile not found: %s", exc)
            from sports_pro import DailyPlan
            plan = DailyPlan(
                primary_focus = "Profile not registered",
                activation    = 0.5,
                fulcrum       = 0.5,
                tasks         = [],
                warnings      = [f"Profile '{self._name}' not found"],
                rationale     = "",
            )

        # --- Step 4: device status ---
        device_status = self._device_status()

        # --- Step 5: priority tasks ---
        priority_tasks = [t.title for t in plan.tasks[:3]]

        # --- Step 6: alerts ---
        alerts = list(plan.warnings)
        if not self._va.is_available():
            alerts.append("ℹ Ollama not running — vision analysis unavailable")

        # --- Step 7: match intelligence (when match is close) ---
        match_intelligence: Optional[dict] = None
        try:
            # Retrieve days_to_match from reading-derived context if available
            _days_to_match: float = 99.0
            if reading is not None:
                try:
                    _dctx = reading.to_daily_context()
                    _days_to_match = _dctx.days_to_match
                except Exception:
                    pass
            if _days_to_match <= 4:
                from prediction_engine import PredictionPlatform
                _plat = PredictionPlatform()
                match_intelligence = _plat.pre_match_brief(
                    "Home Team", "Away Team", "football", [], {}, {}
                )
        except Exception as _exc:
            logger.debug("match intelligence skipped: %s", _exc)

        return MorningBrief(
            time               = now_str,
            plan               = plan,
            wearable_summary   = wearable_summary,
            device_status      = device_status,
            priority_tasks     = priority_tasks,
            alerts             = alerts,
            match_intelligence = match_intelligence,
        )

    # ── session logging ───────────────────────────────────────────────────

    def log_session(
        self,
        session_type: str,
        rpe:          int,
        notes:        str   = "",
        video_folder: str   = None,
        gps_file:     str   = None,
        run_vision:   bool  = True,
    ) -> SessionLog:
        """
        1. Ingest video files from video_folder
        2. Run MediaProcessor.extract_metrics() on GPS/video
        3. If run_vision and Ollama available: extract frames → analyze_technique()
        4. Build SessionLog
        5. Save SessionLog as artifact
        6. Update daily load metrics
        """
        start_time = datetime.now(tz=timezone.utc).isoformat()
        session_id = str(uuid.uuid4())

        # Step 1: ingest video
        video_paths: list[str] = []
        if video_folder and Path(video_folder).expanduser().exists():
            folder = str(Path(video_folder).expanduser())
            try:
                # Use the first available device, or create a temporary one
                devices = self._hub.list_devices()
                device_id = devices[0].device_id if devices else "manual"
                ingested = self._hub.ingest_folder(folder, device_id)
                video_paths = [f.path for f in ingested if f.media_type == MediaType.VIDEO]
            except Exception as exc:
                logger.warning("Could not ingest video folder %s: %s", video_folder, exc)

        # Step 2: metrics
        metrics: dict = {}
        metric_sources = video_paths[:1]
        if gps_file and Path(gps_file).expanduser().exists():
            metric_sources = [str(Path(gps_file).expanduser())] + metric_sources

        for src in metric_sources:
            try:
                m = self._mp.extract_metrics(src)
                for k, v in {
                    "duration_sec": m.duration_sec,
                    "distance_m":   m.distance_m,
                    "avg_hr":       m.avg_hr,
                    "max_hr":       m.max_hr,
                    "avg_speed_ms": m.avg_speed_ms,
                    "max_speed_ms": m.max_speed_ms,
                }.items():
                    if v and v > 0:
                        metrics[k] = v
            except Exception as exc:
                logger.warning("Metrics extraction failed for %s: %s", src, exc)

        # Step 3: vision analysis
        vision_summary = ""
        if run_vision and self._va.is_available() and video_paths:
            try:
                _, profile = self._assistant.get_profile(self._name)
                sport = profile.sport if profile else "general"
                role  = profile.role.value if profile else "athlete"

                record = self._mp.probe(video_paths[0])
                frames = self._mp.extract_frames(record, rate=0.1)
                fas    = []
                for frame in frames[:4]:
                    b64 = self._mp.frame_to_base64(frame)
                    if b64:
                        fa = self._va.analyze_frame(b64, sport, role)
                        fas.append(fa)
                if fas:
                    ss = self._va.summarize_session(fas, sport, role, notes)
                    vision_summary = "; ".join(ss.recommendations[:3])
            except Exception as exc:
                logger.warning("Vision analysis failed: %s", exc)

        end_time = datetime.now(tz=timezone.utc).isoformat()

        session = SessionLog(
            session_id     = session_id,
            start_time     = start_time,
            end_time       = end_time,
            sport          = self._get_sport(),
            session_type   = session_type,
            rpe            = rpe,
            notes          = notes,
            video_paths    = video_paths,
            gps_path       = gps_file or "",
            metrics        = metrics,
            vision_summary = vision_summary,
        )

        # Step 5: persist
        self._sessions.append(session)
        self._persist_session(session)

        # Step 6: log load metric
        try:
            load_estimate = rpe * metrics.get("duration_sec", 3600) / 60.0
            self._assistant.log_metric(self._name, "session_load", load_estimate, "au")
        except Exception:
            pass

        return session

    # ── evening review ────────────────────────────────────────────────────

    def evening_review(
        self,
        day_rating: Optional[float] = None,
        notes:      str             = "",
    ) -> EveningReview:
        """
        1. Retrieve today's session logs
        2. If day_rating provided, call assistant.rate_day() (drives learning)
        3. Compute recovery protocol
        4. Generate tomorrow's preview
        5. Calculate sleep target
        6. Return EveningReview
        """
        date_str = date.today().isoformat()

        if day_rating is not None:
            try:
                self._assistant.rate_day(self._name, date_str, day_rating, notes)
            except Exception as exc:
                logger.warning("rate_day failed: %s", exc)

        today_sessions = [s for s in self._sessions if s.start_time.startswith(date_str[:10])]

        # Recovery protocol
        total_load  = sum(s.rpe * s.metrics.get("duration_sec", 3600) / 60 for s in today_sessions)
        avg_rpe     = (sum(s.rpe for s in today_sessions) / len(today_sessions)) if today_sessions else 0
        recovery    = self._recovery_protocol(avg_rpe, total_load)

        # Tomorrow preview
        tomorrow_preview = self._tomorrow_preview()

        # Sleep target
        sleep_target = min(10.0, max(7.0, 7.5 + (avg_rpe - 5) * 0.15))

        prompt = (
            f"How was today? Rate your day from 1 (very poor) to 5 (excellent). "
            f"You completed {len(today_sessions)} session(s) with avg RPE {avg_rpe:.1f}."
        )

        return EveningReview(
            date_str          = date_str,
            session_logs      = today_sessions,
            day_rating_prompt = prompt,
            recovery_protocol = recovery,
            tomorrow_preview  = tomorrow_preview,
            sleep_target_hrs  = round(sleep_target, 1),
        )

    # ── device watching ───────────────────────────────────────────────────

    def watch_for_sessions(
        self,
        on_session_complete: Optional[Callable] = None,
    ) -> None:
        """Start device_hub.start_watching() in a background thread."""

        def _on_file(ingested_file):
            logger.info("New file detected: %s", ingested_file.path)
            if on_session_complete:
                try:
                    on_session_complete(ingested_file)
                except Exception as exc:
                    logger.warning("on_session_complete callback failed: %s", exc)

        self._hub.start_watching(_on_file)
        logger.info("DailyWorkflow: watching for new sessions…")

    def stop_watching(self) -> None:
        self._hub.stop_watching()

    # ── weekly report ─────────────────────────────────────────────────────

    def generate_weekly_report(self, week_offset: int = 0) -> str:
        """
        Generate a markdown performance report for the specified week.
        week_offset=0 → this week, week_offset=-1 → last week.
        Saves to ~/.kde/reports/ and returns the markdown string.
        """
        history = self._assistant.history(self._name, days=7 + abs(week_offset) * 7)
        reflect = self._assistant.reflect(self._name)

        lines = [
            "# Weekly Performance Report",
            f"\nProfile: **{self._name}**  |  Generated: {_now_iso()}",
            "\n## Learning State",
            f"- Fixed fulcrum: {reflect.get('fixed_fulcrum', 'N/A')}",
            f"- Trend: {reflect.get('fulcrum_trend', 'N/A')}",
            f"- Avg day rating: {reflect.get('avg_day_rating', 'N/A')}",
            "\n## Sessions This Week",
        ]

        if self._sessions:
            for s in self._sessions[-7:]:
                lines.append(
                    f"- {s.start_time[:10]}  {s.session_type}  "
                    f"RPE {s.rpe}  dist {s.metrics.get('distance_m', 0):.0f}m"
                )
        else:
            lines.append("_No sessions logged this week._")

        lines += [
            "\n## Day Ratings",
        ]
        ratings = [h for h in history if h.get("type") == "rating"]
        if ratings:
            for r in ratings[:7]:
                lines.append(f"- {r['date']}: {r['rating']}/5  {r.get('notes', '')}")
        else:
            lines.append("_No ratings logged this week._")

        lines += [
            "\n## Recommendations",
            f"- Keep sleep above {reflect.get('fulcrum_trend', 'balanced')} targets",
            "- Maintain session log consistency",
        ]

        report = "\n".join(lines) + "\n"

        out_dir = Path("~/.kde/reports").expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"weekly_{date.today().isoformat()}_offset{week_offset}.md"
        try:
            out_path.write_text(report, encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not save weekly report: %s", exc)

        return report

    # ── private helpers ───────────────────────────────────────────────────

    def _auto_sync_wearables(self) -> tuple[Optional[WearableReading], str]:
        """Attempt to sync wearable devices and parse their exports."""
        devices = self._hub.list_devices()
        wearable_types = {
            DeviceType.WEARABLE_WHOOP,
            DeviceType.WEARABLE_GARMIN,
            DeviceType.WEARABLE_APPLE,
            DeviceType.WEARABLE_OURA,
        }

        reading      = None
        summary_parts: list[str] = []

        for device in devices:
            if not device.enabled or device.device_type not in wearable_types:
                continue
            try:
                if device.device_type == DeviceType.WEARABLE_APPLE and device.watch_path:
                    xml_files = list(Path(device.watch_path).expanduser().glob("*.xml"))
                    if xml_files:
                        data = self._hub.parse_apple_health(str(xml_files[0]))
                        reading = _apple_health_to_reading(data)
                        summary_parts.append(f"Apple Health: HRV {reading.hrv_ms:.0f}ms")
                elif device.device_type == DeviceType.WEARABLE_GARMIN and device.watch_path:
                    csv_files = list(Path(device.watch_path).expanduser().glob("*.csv"))
                    if csv_files:
                        data = self._hub.parse_garmin_csv(str(csv_files[0]))
                        reading = _garmin_to_reading(data)
                        summary_parts.append(f"Garmin: {len(csv_files)} file(s)")
            except Exception as exc:
                logger.warning("Wearable sync failed for %s: %s", device.name, exc)

        if reading is None:
            reading = WearableReader.mock(seed=int(time.time()) % 100)
            summary_parts.append("No wearable data — using mock values")

        return reading, " | ".join(summary_parts) if summary_parts else "No wearables connected"

    def _device_status(self) -> list[str]:
        status: list[str] = []
        for device in self._hub.list_devices():
            recent = self._hub.list_files(device_id=device.device_id, since_days=1)
            if recent:
                status.append(f"✓ {device.name}: {len(recent)} new file(s)")
            elif device.enabled:
                status.append(f"○ {device.name}: no new data")
        return status

    def _get_sport(self) -> str:
        try:
            _, profile = self._assistant.get_profile(self._name)
            return profile.sport if profile else "general"
        except Exception:
            return "general"

    def _persist_session(self, session: SessionLog) -> None:
        try:
            self._assistant.log_metric(
                self._name, "session_rpe", float(session.rpe), "rpe"
            )
        except Exception as exc:
            logger.debug("Could not persist session metric: %s", exc)

    def _recovery_protocol(self, avg_rpe: float, total_load: float) -> list[str]:
        protocol: list[str] = []
        if avg_rpe >= 8:
            protocol += [
                "Ice bath or contrast shower (10–15 min)",
                "Compression garments overnight",
                "Foam rolling — quads, hamstrings, calves (20 min)",
                "No alcohol tonight",
            ]
        elif avg_rpe >= 5:
            protocol += [
                "Warm shower + light stretching (15 min)",
                "Hydrate: 500ml water before bed",
                "Elevation of legs for 20 min",
            ]
        else:
            protocol += [
                "Light walk or yoga (optional)",
                "Normal hydration",
            ]
        protocol.append(f"Sleep target: based on today's load ({total_load:.0f} au)")
        return protocol

    def _tomorrow_preview(self) -> str:
        try:
            plan = self._assistant.plan_day(self._name)
            return f"Tomorrow: {plan.primary_focus} | {len(plan.tasks)} tasks planned"
        except Exception:
            return "Tomorrow's plan not yet available"


# ---------------------------------------------------------------------------
# Wearable data converters
# ---------------------------------------------------------------------------

def _apple_health_to_reading(data: dict) -> WearableReading:
    hrv  = float(data.get("hrv_ms_avg", 60.0) or 60.0)
    slp  = float(data.get("sleep_hrs", 7.5)   or 7.5)
    hr   = float(data.get("resting_hr", 60.0) or 60.0)
    return WearableReading(
        source          = "apple_health",
        hrv_ms          = hrv,
        hrv_baseline_ms = 60.0,
        sleep_hrs       = slp,
        sleep_score     = min(100.0, (slp / 9.0) * 100.0),
        body_battery    = min(100.0, max(0.0, 100.0 - (hr - 50) * 1.2)),
    )


def _garmin_to_reading(data: dict) -> WearableReading:
    hrv  = float(data.get("hrv_ms", 60.0)     or 60.0)
    slp  = float(data.get("sleep_hrs", 7.5)   or 7.5)
    batt = float(data.get("body_battery", 70.0)or 70.0)
    return WearableReading(
        source          = "garmin",
        hrv_ms          = hrv,
        hrv_baseline_ms = 60.0,
        sleep_hrs       = slp,
        sleep_score     = min(100.0, (slp / 9.0) * 100.0),
        body_battery    = batt,
    )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
