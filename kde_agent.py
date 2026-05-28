"""
kde_agent.py
============
KDE Sports Agent — Unified Top-Level Agent

KDEAgent is the single object a practitioner or integration instantiates.
Combines all layers and exposes a simple natural-language ask() interface.

Example usage:
    agent = KDEAgent.setup(name="Marcus", role=Role.ATHLETE, sport="Football", team="City FC")
    brief = agent.morning_briefing()
    result = agent.ask("analyse my session from yesterday")
    log = agent.log_session(rpe=7, video_folder="~/GoPro/session_2026_05_28")
    review = agent.evening_review(day_rating=4.0)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sports_pro import SportsProAssistant, SportsProProfile, Role, DailyContext
from device_hub import DeviceHub, Device, DeviceType
from media_processor import MediaProcessor
from vision_analyzer import VisionAnalyzer
from sport_executor import (
    VideoAnalysisExecutor,
    HighlightReelExecutor,
    PerformanceReportExecutor,
    FilmStudyExecutor,
    WearableSyncExecutor,
    SessionLogExecutor,
)
from daily_workflow import DailyWorkflow, MorningBrief, SessionLog, EveningReview
from ksa_router import MasterFulcrum
from ksa_registry import SnapshotRegistry
from ksa_fixes import LiveWeightInjector, GroundTruthOptimizer
from sport_tasks import (
    TrainingPlanTask,
    MatchReportTask,
    ScoutingReportTask,
    NutritionPlanTask,
    SocialMediaTask,
    EmailDraftTask,
    PerformanceDashboardTask,
    PredictionReportTask,
)
from prediction_engine import PredictionPlatform

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config & Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class KDEConfig:
    db_path:       str  = "~/.kde/kde.db"
    media_dir:     str  = "~/.kde/media"
    ollama_host:   str  = "http://localhost:11434"
    ollama_model:  str  = "llava"
    text_model:    str  = "mistral"
    ffmpeg_path:   str  = "ffmpeg"
    poll_interval: int  = 30
    auto_watch:    bool = True


@dataclass
class TaskResult:
    task:       str
    method:     str        # "keyword" | "llm" | "direct"
    output:     Any
    success:    bool
    elapsed_ms: float
    artifact_id: str = ""


# ---------------------------------------------------------------------------
# Intent routing map
# ---------------------------------------------------------------------------

# keyword fragments → task_name
INTENT_MAP: dict[str, str] = {
    # video analysis
    "analyse":        "video_analysis",
    "analyze":        "video_analysis",
    "analysis":       "video_analysis",
    "technique":      "video_analysis",
    "footage":        "video_analysis",
    "session video":  "video_analysis",
    # highlight reel
    "highlight":      "highlight_reel",
    "reel":           "highlight_reel",
    "compilation":    "highlight_reel",
    "edit":           "highlight_reel",
    # performance report
    "report":         "performance_report",
    "performance":    "performance_report",
    "stats":          "performance_report",
    "metrics":        "performance_report",
    "recovery trend": "performance_report",
    # film study
    "opponent":       "film_study",
    "scout":          "film_study",
    "film study":     "film_study",
    "tactical":       "film_study",
    # wearable sync
    "sync":           "wearable_sync",
    "wearable":       "wearable_sync",
    "hrv":            "wearable_sync",
    "garmin":         "wearable_sync",
    "apple watch":    "wearable_sync",
    # session log
    "log session":    "session_log",
    "log my session": "session_log",
    "record session": "session_log",
    # morning / evening / weekly
    "morning":        "morning_briefing",
    "brief":          "morning_briefing",
    "briefing":       "morning_briefing",
    "evening":        "evening_review",
    "review":         "evening_review",
    "weekly":         "weekly_report",
    "week report":    "weekly_report",
    # reflection
    "reflect":        "reflect",
    "learned":        "reflect",
    "status":         "status",
    "what should":    "video_analysis",
    "technically":    "video_analysis",
    # sport tasks (prompt-3)
    "training plan":  "create_training_plan",
    "weekly plan":    "create_training_plan",
    "programme":      "create_training_plan",
    "match report":   "match_report",
    "post match":     "match_report",
    "write report":   "match_report",
    "scouting":       "scouting_report",
    "opponent report": "scouting_report",
    "analyse them":   "scouting_report",
    "nutrition":      "nutrition_plan",
    "meal plan":      "nutrition_plan",
    "diet":           "nutrition_plan",
    "food":           "nutrition_plan",
    "instagram":      "social_media_post",
    "twitter":        "social_media_post",
    "social media":   "social_media_post",
    "email":          "draft_email",
    "draft":          "draft_email",
    "write to":       "draft_email",
    "message agent":  "draft_email",
    "dashboard":      "performance_dashboard",
    "overview":       "performance_dashboard",
    "how am i doing": "performance_dashboard",
    "predict":        "prediction_report",
    "prediction":     "prediction_report",
    "match preview":  "prediction_report",
    "odds":           "prediction_report",
}


# ---------------------------------------------------------------------------
# KDEAgent
# ---------------------------------------------------------------------------

class KDEAgent:
    """
    The complete digital AI coach and practitioner assistant.
    """

    def __init__(
        self,
        profile: SportsProProfile,
        config:  Optional[KDEConfig] = None,
    ) -> None:
        self._profile = profile
        self._config  = config or KDEConfig()

        # Expand paths
        db_path   = str(Path(self._config.db_path).expanduser())
        media_dir = str(Path(self._config.media_dir).expanduser())
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(media_dir).mkdir(parents=True, exist_ok=True)

        # Core layers
        self._assistant  = SportsProAssistant(db_path)
        self._hub        = DeviceHub(db_path)
        self._mp         = MediaProcessor(
            output_dir   = media_dir,
            ffmpeg_path  = self._config.ffmpeg_path,
        )
        self._va         = VisionAnalyzer(
            host  = self._config.ollama_host,
            model = self._config.ollama_model,
        )
        self._registry   = SnapshotRegistry(db_path)
        self._router     = MasterFulcrum(self._registry)
        self._injector   = LiveWeightInjector()
        self._optimizer  = GroundTruthOptimizer()

        # Register profile
        self._assistant.register(profile)

        # Daily workflow
        self._workflow = DailyWorkflow(
            assistant       = self._assistant,
            device_hub      = self._hub,
            media_processor = self._mp,
            vision_analyzer = self._va,
            profile_name    = profile.name,
        )

        # Executors
        self._executors = {
            "video_analysis":    VideoAnalysisExecutor(
                self._hub, self._mp, self._va, self._registry,
                sport=profile.sport, role=profile.role.value,
            ),
            "highlight_reel":    HighlightReelExecutor(
                self._hub, self._mp,
                output_dir=str(Path(media_dir) / "highlights"),
                registry=self._registry,
            ),
            "performance_report": PerformanceReportExecutor(
                self._hub, self._mp, self._registry,
                output_dir=str(Path(media_dir) / "reports"),
            ),
            "film_study":        FilmStudyExecutor(
                self._hub, self._mp, self._va, self._registry,
                sport=profile.sport,
            ),
            "wearable_sync":     WearableSyncExecutor(self._hub, self._registry),
            "session_log":       SessionLogExecutor(
                self._hub, self._mp, self._va, self._registry,
                sport=profile.sport, role=profile.role.value,
            ),
        }

        # Prediction platform + sport-task executors (prompt-3)
        self._platform = PredictionPlatform()
        _task_kwargs = dict(
            registry    = self._registry,
            platform    = self._platform,
            output_dir  = str(Path(media_dir) / "artifacts"),
            ollama_host = self._config.ollama_host,
            text_model  = self._config.text_model,
        )
        self._executors.update({
            "create_training_plan":  TrainingPlanTask(**_task_kwargs),
            "match_report":          MatchReportTask(**_task_kwargs),
            "scouting_report":       ScoutingReportTask(**_task_kwargs),
            "nutrition_plan":        NutritionPlanTask(**_task_kwargs),
            "social_media_post":     SocialMediaTask(**_task_kwargs),
            "draft_email":           EmailDraftTask(**_task_kwargs),
            "performance_dashboard": PerformanceDashboardTask(**_task_kwargs),
            "prediction_report":     PredictionReportTask(**_task_kwargs),
        })

        # Register intents in router
        self._register_router_intents()

        # Auto-watch
        if self._config.auto_watch:
            try:
                self._workflow.watch_for_sessions()
            except Exception as exc:
                logger.warning("Auto-watch failed to start: %s", exc)

    # ── class method setup ────────────────────────────────────────────────

    @classmethod
    def setup(
        cls,
        name:   str,
        role:   Role,
        sport:  str,
        team:   str        = "",
        config: Optional[KDEConfig] = None,
    ) -> "KDEAgent":
        """One-line setup: create profile, register, return agent."""
        profile = SportsProProfile(name=name, role=role, sport=sport, team=team)
        return cls(profile, config)

    # ── device management ─────────────────────────────────────────────────

    def add_device(
        self,
        name:        str,
        device_type: DeviceType,
        watch_path:  str,
        api_url:     str = "",
    ) -> str:
        import uuid as _uuid
        device = Device(
            device_id   = str(_uuid.uuid4()),
            name        = name,
            device_type = device_type,
            watch_path  = watch_path,
            api_url     = api_url,
            api_key     = "",
            enabled     = True,
            last_sync   = "",
        )
        return self._hub.register_device(device)

    def sync_devices(self) -> dict:
        """Sync all connected devices. Returns {device_name: files_ingested}."""
        result: dict = {}
        for device in self._hub.list_devices():
            if not device.enabled:
                continue
            try:
                path = Path(device.watch_path).expanduser()
                if path.exists():
                    files = self._hub.ingest_folder(str(path), device.device_id)
                    result[device.name] = len(files)
                else:
                    result[device.name] = 0
            except Exception as exc:
                logger.warning("Sync failed for %s: %s", device.name, exc)
                result[device.name] = -1
        return result

    # ── daily lifecycle ───────────────────────────────────────────────────

    def morning_briefing(
        self,
        hrv_ms:    Optional[float] = None,
        sleep_hrs: Optional[float] = None,
        soreness:  Optional[int]   = None,
        energy:    Optional[int]   = None,
    ) -> MorningBrief:
        """If wearable values provided, use them. Otherwise auto-sync devices."""
        from sports_pro import WearableReader
        reading = None
        if hrv_ms is not None:
            reading = WearableReader.manual(
                hrv_ms       = hrv_ms,
                sleep_hrs    = sleep_hrs or 7.5,
                soreness     = soreness or 3,
                energy       = energy or 7,
                baseline_hrv = 60.0,
            )
        return self._workflow.morning_briefing(manual_reading=reading)

    def log_session(
        self,
        rpe:          int,
        session_type: str = "training",
        notes:        str = "",
        video_folder: Optional[str] = None,
        gps_file:     Optional[str] = None,
    ) -> SessionLog:
        return self._workflow.log_session(
            session_type = session_type,
            rpe          = rpe,
            notes        = notes,
            video_folder = video_folder,
            gps_file     = gps_file,
        )

    def evening_review(
        self,
        day_rating: Optional[float] = None,
        notes:      str             = "",
    ) -> EveningReview:
        return self._workflow.evening_review(day_rating=day_rating, notes=notes)

    # ── natural language interface ────────────────────────────────────────

    def ask(self, prompt: str) -> TaskResult:
        """
        Route a natural-language prompt to the appropriate executor.
        1. Match against INTENT_MAP keywords
        2. If no match and text_model configured, ask Ollama to classify
        3. Dispatch to appropriate executor or workflow method
        4. Return TaskResult
        """
        t0 = time.perf_counter()
        prompt_lower = prompt.lower()

        # Step 1: keyword routing
        task_name = None
        for keyword, name in INTENT_MAP.items():
            if keyword in prompt_lower:
                task_name = name
                method    = "keyword"
                break

        # Step 2: LLM fallback
        if task_name is None and self._va.is_available():
            task_name, method = self._llm_classify(prompt)
        if task_name is None:
            task_name = "video_analysis"
            method    = "keyword"

        # Step 3: dispatch
        try:
            output = self._dispatch(task_name, prompt)
            success = True
        except Exception as exc:
            logger.exception("Dispatch failed for task '%s'", task_name)
            output  = str(exc)
            success = False

        elapsed = (time.perf_counter() - t0) * 1000
        return TaskResult(
            task       = task_name,
            method     = method,
            output     = output,
            success    = success,
            elapsed_ms = elapsed,
        )

    # ── analysis & reporting ──────────────────────────────────────────────

    def analyze_footage(
        self,
        path:       str,
        run_vision: bool = True,
    ) -> dict:
        """Analyse a specific video file."""
        p = Path(path).expanduser()
        if not p.exists():
            return {"error": f"File not found: {path}"}

        try:
            record = self._mp.probe(str(p))
            result: dict = {
                "path":         str(p),
                "duration_sec": record.duration_sec,
                "fps":          record.fps,
                "resolution":   f"{record.width}x{record.height}",
            }

            if run_vision and self._va.is_available():
                frames = self._mp.extract_frames(record, rate=0.2)
                fas    = []
                for frame in frames[:5]:
                    b64 = self._mp.frame_to_base64(frame)
                    if b64:
                        fa = self._va.analyze_frame(b64, self._profile.sport, self._profile.role.value)
                        fas.append(fa)
                if fas:
                    report = self._va.analyze_technique(frames, self._profile.sport, self._profile.role.value, self._mp)
                    result["technique_report"] = {
                        "overall_score": report.overall_score,
                        "key_findings":  report.key_findings,
                        "improvements":  report.improvements,
                        "strengths":     report.strengths,
                    }
            return result
        except Exception as exc:
            return {"error": str(exc)}

    def generate_report(
        self,
        report_type: str = "weekly",
        output_path: Optional[str] = None,
    ) -> str:
        if report_type == "weekly":
            report = self._workflow.generate_weekly_report()
        else:
            report = self._workflow.generate_weekly_report()

        if output_path:
            try:
                Path(output_path).expanduser().write_text(report, encoding="utf-8")
            except Exception as exc:
                logger.warning("Could not save report to %s: %s", output_path, exc)
        return report

    def reflect(self) -> dict:
        """Current learned state: fixed_fulcrum, drift, history summary."""
        return self._assistant.reflect(self._profile.name)

    def start_server(self, port: int = 8742, blocking: bool = False) -> str:
        """Start the local REST API. Returns the server URL."""
        from kde_server import KDEServer
        self._server = KDEServer(
            agent    = self,
            port     = port,
            platform = self._platform,
        )
        self._server.start(blocking=blocking)
        return self._server.url

    # ── status ────────────────────────────────────────────────────────────

    def status(self) -> dict:
        devices       = self._hub.list_devices()
        ffmpeg_ok     = shutil.which(self._config.ffmpeg_path) is not None
        ollama_ok     = self._va.is_available()
        reflect_data  = self._assistant.reflect(self._profile.name)
        history       = self._assistant.history(self._profile.name, days=30)

        plans_month   = len([h for h in history if h.get("type") == "plan"])
        sessions_month = len(self._workflow._sessions)
        artifacts     = 0
        try:
            tasks = self._registry.list_tasks()
            artifacts = len(tasks)
        except Exception:
            pass

        return {
            "profile":          self._profile.name,
            "role":             self._profile.role.value,
            "sport":            self._profile.sport,
            "devices":          [{"name": d.name, "enabled": d.enabled} for d in devices],
            "ollama_available": ollama_ok,
            "ffmpeg_available": ffmpeg_ok,
            "plans_this_month": plans_month,
            "sessions_this_month": sessions_month,
            "artifacts_stored": artifacts,
            "fixed_fulcrum":    reflect_data.get("fixed_fulcrum"),
            "fulcrum_trend":    reflect_data.get("fulcrum_trend"),
        }

    # ── private helpers ───────────────────────────────────────────────────

    def _dispatch(self, task_name: str, prompt: str) -> Any:
        if task_name == "morning_briefing":
            return self._workflow.morning_briefing()
        if task_name == "evening_review":
            return self._workflow.evening_review()
        if task_name == "weekly_report":
            return self._workflow.generate_weekly_report()
        if task_name == "reflect":
            return self.reflect()
        if task_name == "status":
            return self.status()

        executor = self._executors.get(task_name)
        if executor is None:
            return {"error": f"No executor for task '{task_name}'"}

        # Build a minimal EquilibriumResult for the executor context
        from ksa_lever import ThreeBarSystem, TiltDirection, EquilibriumResult, LeverState
        system = ThreeBarSystem.from_defaults()
        system.levers[0].set_weights(left=6.0, right=4.0)
        eq = system.simulate()

        from ksa_executor import ExecutionContext
        ctx = ExecutionContext(
            task_name   = task_name,
            version     = 1,
            result      = eq,
            working_dir = str(Path(self._config.media_dir).expanduser()),
            payload     = {"sport": self._profile.sport, "role": self._profile.role.value},
        )

        outcome = executor.primary(ctx)
        try:
            return json.loads(outcome.stdout) if outcome.stdout else outcome.stderr
        except (json.JSONDecodeError, ValueError):
            return outcome.stdout or outcome.stderr

    def _llm_classify(self, prompt: str) -> tuple[str, str]:
        """Ask Ollama to classify the intent. Returns (task_name, method)."""
        task_names = list(set(INTENT_MAP.values()))
        body = json.dumps({
            "model": self._config.text_model,
            "prompt": (
                f"Classify this request into exactly one of these tasks: "
                f"{', '.join(task_names)}.\n"
                f"Request: {prompt}\n"
                f"Reply with ONLY the task name, nothing else."
            ),
            "stream": False,
        }).encode()

        try:
            req = urllib.request.Request(
                f"{self._config.ollama_host}/api/generate",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                task = data.get("response", "").strip().lower()
                # validate
                if task in {t.replace(" ", "_") for t in task_names}:
                    return task, "llm"
                for name in task_names:
                    if name in task:
                        return name, "llm"
        except Exception as exc:
            logger.debug("LLM classify failed: %s", exc)

        return None, "keyword"

    def _register_router_intents(self) -> None:
        task_keywords: dict[str, list[str]] = {}
        for kw, task in INTENT_MAP.items():
            task_keywords.setdefault(task, []).append(kw)
        for task_name, keywords in task_keywords.items():
            try:
                self._router.register_intent(task_name, keywords=keywords)
            except Exception:
                pass
