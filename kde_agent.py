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
import re
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from artifact_store import Artifact, ArtifactStore
from daily_workflow import DailyWorkflow, EveningReview, MorningBrief, SessionLog
from device_hub import Device, DeviceHub, DeviceType
from digital_identity import CrystallisationEngine
from domain_configs import ALL_DOMAINS, DomainDecisionModel
from identity_bus import IdentityBus
from kde_profiles import UserProfile, UserRole, from_toml
from ksa_executor import (
    ExecutionContext,
    ExecutionOutcome,
    TaskExecutor,
    _ResourceSampler,
)
from ksa_fixes import GroundTruthOptimizer, LiveWeightInjector
from ksa_registry import PerformanceMetrics, SnapshotRegistry
from ksa_router import MasterFulcrum
from media_processor import MediaProcessor
from prediction_engine import PredictionPlatform
from sport_executor import (
    FilmStudyExecutor,
    HighlightReelExecutor,
    PerformanceReportExecutor,
    SessionLogExecutor,
    VideoAnalysisExecutor,
    WearableSyncExecutor,
)
from sport_tasks import (
    EmailDraftTask,
    MatchReportTask,
    NutritionPlanTask,
    PerformanceDashboardTask,
    PredictionReportTask,
    ScoutingReportTask,
    SocialMediaTask,
    TrainingPlanTask,
)
from sports_pro import Role, SportsProAssistant, SportsProProfile
from vision_analyzer import VisionAnalyzer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config & Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class KDEConfig:
    db_path:       str  = "~/.kde/kde.db"
    media_dir:     str  = "~/.kde/media"
    bus_db_path:   str  = "~/.prism/identity_bus.db"
    identity_db_path: str = "~/.prism/identity.db"
    artifact_db_path: str = "~/.prism/artifacts.db"
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


def _domain_outcome(
    ctx: ExecutionContext,
    action: str,
    rc: int,
    stdout: str,
    stderr: str,
    elapsed: float,
    sampler: _ResourceSampler,
) -> ExecutionOutcome:
    metrics = PerformanceMetrics(
        execution_time_ms=elapsed,
        cpu_peak_pct=sampler.cpu_peak,
        ram_peak_mb=sampler.ram_peak_mb,
        success=rc == 0,
        override_fired=ctx.result.override_active,
        notes=f"action={action}",
    )
    return ExecutionOutcome(
        task_name=ctx.task_name,
        version=ctx.version,
        action_taken=action,
        return_code=rc,
        stdout=stdout,
        stderr=stderr,
        metrics=metrics,
        elapsed_ms=elapsed,
    )


class DomainEvaluateExecutor(TaskExecutor):
    task_name = "domain_evaluate"

    def __init__(
        self,
        domain_models: dict[str, DomainDecisionModel],
        registry=None,
        mode: str = "evaluate",
    ) -> None:
        self.task_name = {
            "evaluate": "domain_evaluate",
            "compare": "domain_compare",
            "report": "domain_report",
        }.get(mode, "domain_evaluate")
        self._models = domain_models
        self._registry = registry
        self._mode = mode

    def _extract_domain(self, prompt: str) -> str | None:
        prompt_lower = prompt.lower()
        for domain in self._models:
            if domain.lower() in prompt_lower:
                return domain
        return next(iter(self._models), None)

    def _extract_profile(self, domain: str, prompt: str) -> str | None:
        prompt_lower = prompt.lower()
        for profile in self._models[domain].config.profiles:
            if profile.name.lower() in prompt_lower:
                return profile.name
        return self._models[domain].config.profiles[0].name if self._models[domain].config.profiles else None

    def _extract_factors(self, domain: str, prompt: str) -> dict[str, float]:
        factors = {factor.id: 0.5 for factor in self._models[domain].config.factors}
        prompt_lower = prompt.lower()
        for factor in self._models[domain].config.factors:
            match = re.search(rf"{re.escape(factor.id.lower())}\s*[:=]?\s*(0(?:\.\d+)?|1(?:\.0+)?)", prompt_lower)
            if match:
                factors[factor.id] = max(0.0, min(1.0, float(match.group(1))))
        return factors

    def primary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        prompt = str(ctx.payload.get("prompt", ""))
        try:
            domain = self._extract_domain(prompt)
            if not domain or domain not in self._models:
                elapsed = (time.perf_counter() - t0) * 1000
                sampler.stop()
                return _domain_outcome(ctx, "primary", 1, "", "No matching domain found", elapsed, sampler)

            model = self._models[domain]
            profile = self._extract_profile(domain, prompt)
            factors = self._extract_factors(domain, prompt)

            if self._mode == "compare":
                payload = {
                    "domain": domain,
                    "profiles": model.cross_profile_compare(factors),
                }
            elif self._mode == "report":
                payload = {
                    "domain": domain,
                    "profiles": [profile.name for profile in model.config.profiles],
                    "factors": [factor.id for factor in model.config.factors],
                    "guidance": "Provide labeled cases to /domain/validate for an accuracy report.",
                }
            else:
                beam = model.make_beam(profile, factors)
                diagnosis = beam.evaluate()
                payload = {
                    "domain": domain,
                    "profile": profile,
                    "recommended": diagnosis.primary_plank.name,
                    "confidence": diagnosis.activations[0].activation,
                    "fulcrum": diagnosis.fulcrum_position,
                    "options": [
                        {
                            "name": activation.plank.name,
                            "activation": activation.activation,
                            "position": activation.plank.position,
                        }
                        for activation in diagnosis.activations
                    ],
                }

            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _domain_outcome(ctx, "primary", 0, json.dumps(payload), "", elapsed, sampler)
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _domain_outcome(ctx, "primary", 1, "", str(exc), elapsed, sampler)

    def secondary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        data = {
            domain: [profile.name for profile in model.config.profiles]
            for domain, model in self._models.items()
        }
        elapsed = (time.perf_counter() - t0) * 1000
        sampler.stop()
        return _domain_outcome(ctx, "secondary", 0, json.dumps(data), "", elapsed, sampler)

    def safe(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        prompt = str(ctx.payload.get("prompt", ""))
        domain = self._extract_domain(prompt)
        profile = self._extract_profile(domain, prompt) if domain else None
        factors = self._extract_factors(domain, prompt) if domain else {}
        out = json.dumps({
            "would_evaluate": {
                "domain": domain,
                "profile": profile,
                "factor_values": factors,
            }
        })
        elapsed = (time.perf_counter() - t0) * 1000
        sampler.stop()
        return _domain_outcome(ctx, "safe", 0, out, "", elapsed, sampler)


# ---------------------------------------------------------------------------
# Intent routing map
# ---------------------------------------------------------------------------

# keyword fragments → task_name
INTENT_MAP: dict[str, str] = {
    # domain decisions
    "triage":         "domain_evaluate",
    "advise":         "domain_evaluate",
    "recommend":      "domain_evaluate",
    "portfolio":      "domain_evaluate",
    "legal advice":   "domain_evaluate",
    "hr decision":    "domain_evaluate",
    "supply chain":   "domain_evaluate",
    "climate":        "domain_evaluate",
    "compare profiles": "domain_compare",
    "who should":     "domain_compare",
    "which option":   "domain_compare",
    "best approach":  "domain_compare",
    "all profiles":   "domain_compare",
    "domain report":  "domain_report",
    "validation report": "domain_report",
    "accuracy":       "domain_report",
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
        capabilities: Optional[list[str]] = None,
        display_role: Optional[str] = None,
    ) -> None:
        self._profile = profile
        self._config  = config or KDEConfig()
        self._capabilities = set(capabilities or [
            "sports_pro", "daily_workflow", "device_hub", "moment_analyzer",
            "prediction_engine", "sport_tasks", "domain_configs",
        ])
        self._display_role = display_role or profile.role.value
        self._user_profile: Optional[UserProfile] = None

        # Expand paths
        db_path   = str(Path(self._config.db_path).expanduser())
        media_dir = str(Path(self._config.media_dir).expanduser())
        bus_db_path = str(Path(self._config.bus_db_path).expanduser())
        identity_db = str(Path(self._config.identity_db_path).expanduser())
        artifact_db = str(Path(self._config.artifact_db_path).expanduser())
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(media_dir).mkdir(parents=True, exist_ok=True)
        Path(bus_db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(identity_db).parent.mkdir(parents=True, exist_ok=True)
        Path(artifact_db).parent.mkdir(parents=True, exist_ok=True)

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
        self._bus        = IdentityBus(db_path=bus_db_path)
        self._crystal    = CrystallisationEngine(profile.name, self._bus, db_path=identity_db)
        self._artifacts  = ArtifactStore(db_path=artifact_db)

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
        self._platform = PredictionPlatform() if self._has_capability("prediction_engine", "sport_tasks") else None
        if self._platform is not None and self._has_capability("sport_tasks"):
            _task_kwargs = dict(
                registry=self._registry,
                platform=self._platform,
                output_dir=str(Path(media_dir) / "artifacts"),
                ollama_host=self._config.ollama_host,
                text_model=self._config.text_model,
            )
            self._executors.update({
                "create_training_plan": TrainingPlanTask(**_task_kwargs),
                "match_report": MatchReportTask(**_task_kwargs),
                "scouting_report": ScoutingReportTask(**_task_kwargs),
                "nutrition_plan": NutritionPlanTask(**_task_kwargs),
                "social_media_post": SocialMediaTask(**_task_kwargs),
                "draft_email": EmailDraftTask(**_task_kwargs),
                "performance_dashboard": PerformanceDashboardTask(**_task_kwargs),
                "prediction_report": PredictionReportTask(**_task_kwargs),
            })
        if self._has_capability("domain_configs"):
            domain_models = {
                domain: DomainDecisionModel(config)
                for domain, config in ALL_DOMAINS.items()
            }
            self._executors.update({
                "domain_evaluate": DomainEvaluateExecutor(
                    domain_models=domain_models,
                    registry=self._registry,
                    mode="evaluate",
                ),
                "domain_compare": DomainEvaluateExecutor(
                    domain_models=domain_models,
                    registry=self._registry,
                    mode="compare",
                ),
                "domain_report": DomainEvaluateExecutor(
                    domain_models=domain_models,
                    registry=self._registry,
                    mode="report",
                ),
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
        profile: Optional[UserProfile] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> "KDEAgent":
        """Create an agent from a UserProfile or legacy keyword arguments."""
        config = kwargs.pop("config", None)
        if profile is None:
            candidate = config_path
            if candidate is None:
                for option in (
                    os.environ.get("KDE_CONFIG"),
                    "~/.kde/config.toml",
                    "~/.kde/kde.toml",
                    "./kde_config.toml",
                ):
                    if option and Path(option).expanduser().exists():
                        candidate = option
                        break
            if candidate:
                try:
                    profile = from_toml(candidate)
                except Exception as exc:
                    logger.debug("Could not load UserProfile from %s: %s", candidate, exc)

        if profile is not None:
            sports_role_map = {
                UserRole.DEVELOPER: Role.ANALYST,
                UserRole.ATHLETE: Role.ATHLETE,
                UserRole.COACH: Role.COACH,
                UserRole.ANALYST: Role.ANALYST,
                UserRole.PHYSIO: Role.PHYSIOTHERAPIST,
                UserRole.AGENT: Role.AGENT,
                UserRole.UNIVERSAL: Role.ATHLETE,
            }
            if config is None:
                config = KDEConfig(
                    db_path=profile.db_path,
                    media_dir=profile.media_dir,
                    bus_db_path=getattr(profile, "bus_db_path", "~/.prism/identity_bus.db"),
                    identity_db_path=getattr(profile, "identity_db_path", "~/.prism/identity.db"),
                    artifact_db_path=getattr(profile, "artifact_db_path", "~/.prism/artifacts.db"),
                    ollama_host=profile.ollama_host,
                    ollama_model=profile.ollama_model,
                    text_model=profile.text_model,
                    ffmpeg_path=profile.ffmpeg_path,
                    poll_interval=profile.poll_interval,
                    auto_watch=profile.auto_watch,
                )
            sports_profile = SportsProProfile(
                name=profile.name,
                role=sports_role_map.get(profile.role, Role.ATHLETE),
                sport=profile.sport,
                team=profile.team,
            )
            agent = cls(
                sports_profile,
                config,
                capabilities=profile.capabilities,
                display_role=profile.role.value,
            )
            agent._user_profile = profile
            return agent

        name = kwargs.pop("name")
        role = kwargs.pop("role")
        sport = kwargs.pop("sport")
        team = kwargs.pop("team", "")
        sports_profile = SportsProProfile(name=name, role=role, sport=sport, team=team)
        return cls(sports_profile, config)

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
        brief = self._workflow.morning_briefing(manual_reading=reading)
        try:
            self._record_identity_artifact(
                domain="sport",
                fulcrum=float(brief.plan.fulcrum),
                outcome_rating=float(brief.plan.activation),
                artifact_type="plan",
                title=f"{self._profile.name} morning briefing",
                content={
                    "time": brief.time,
                    "plan": {
                        "primary_focus": brief.plan.primary_focus,
                        "activation": brief.plan.activation,
                        "fulcrum": brief.plan.fulcrum,
                        "tasks": [task.__dict__ for task in brief.plan.tasks],
                        "warnings": list(brief.plan.warnings),
                        "rationale": brief.plan.rationale,
                    },
                    "priority_tasks": list(brief.priority_tasks),
                    "alerts": list(brief.alerts),
                },
            )
        except Exception as exc:
            logger.debug("Identity recording skipped for morning briefing: %s", exc)
        return brief

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

    def identity(self) -> dict:
        identity = self._crystal.get_identity()
        return identity.to_card_data() if identity else {}

    def identity_domains(self) -> list[dict]:
        identity = self._crystal.get_identity()
        if identity is None:
            return []
        return [
            {
                "domain": domain,
                "fixed_fulcrum": profile.fixed_fulcrum,
                "variance": profile.variance,
                "n_observations": profile.n_observations,
                "crystallised": profile.crystallised,
                "confidence": profile.confidence,
                "last_updated": profile.last_updated,
            }
            for domain, profile in sorted(identity.domains.items())
        ]

    def observe_identity(self, domain: str, fulcrum: float, rating: float, context: Optional[dict] = None) -> dict:
        self._crystal.observe(domain, fulcrum, rating, context=context)
        identity = self._crystal.get_identity()
        return identity.to_card_data() if identity else {}

    def reset_identity_domain(self, domain: str) -> dict:
        self._crystal.reset_domain(domain)
        identity = self._crystal.get_identity()
        return identity.to_card_data() if identity else {}

    def recent_artifacts(self, domain: Optional[str] = None, n: int = 10) -> list[dict]:
        return [
            {
                "artifact_id": artifact.artifact_id,
                "user_name": artifact.user_name,
                "domain": artifact.domain,
                "artifact_type": artifact.artifact_type,
                "title": artifact.title,
                "content": artifact.content,
                "fulcrum_at_time": artifact.fulcrum_at_time,
                "identity_version": artifact.identity_version,
                "created_at": artifact.created_at,
                "rating": artifact.rating,
            }
            for artifact in self._artifacts.recent(domain=domain, n=n)
        ]

    def rate_artifact(self, artifact_id: str, rating: float) -> dict:
        self._artifacts.rate(artifact_id, rating)
        artifact = self._artifacts.get(artifact_id)
        return {
            "artifact_id": artifact_id,
            "rating": artifact.rating if artifact else max(0.0, min(1.0, float(rating))),
        }

    def start_server(self, port: int = 8742, blocking: bool = False) -> str:
        """Start the local REST API via ASGI server. Returns the server URL."""
        import threading

        from prism_asgi import serve
        from prism_state import _set_state
        _set_state(agent=self, platform=getattr(self, "_platform", None))
        url = f"http://127.0.0.1:{port}"
        if blocking:
            serve(host="127.0.0.1", port=port)
        else:
            t = threading.Thread(
                target=serve, kwargs={"host": "127.0.0.1", "port": port},
                daemon=True, name="asgi-server",
            )
            t.start()
            self._server_thread = t
        return url

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
            "role":             self._display_role,
            "sport":            self._profile.sport,
            "team":             self._profile.team,
            "capabilities":     sorted(self._capabilities),
            "devices":          [{"name": d.name, "enabled": d.enabled} for d in devices],
            "ollama_available": ollama_ok,
            "ffmpeg_available": ffmpeg_ok,
            "plans_this_month": plans_month,
            "sessions_this_month": sessions_month,
            "artifacts_stored": artifacts,
            "fixed_fulcrum":    reflect_data.get("fixed_fulcrum"),
            "fulcrum_trend":    reflect_data.get("fulcrum_trend"),
        }

    def _has_capability(self, *names: str) -> bool:
        return any(name in self._capabilities for name in names)

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
        from ksa_lever import ThreeBarSystem
        system = ThreeBarSystem.from_defaults()
        system.levers[0].set_weights(left=6.0, right=4.0)
        eq = system.simulate()

        ctx = ExecutionContext(
            task_name   = task_name,
            version     = 1,
            result      = eq,
            working_dir = str(Path(self._config.media_dir).expanduser()),
            payload     = {
                "sport": self._profile.sport,
                "role": self._profile.role.value,
                "prompt": prompt,
            },
        )

        outcome = executor.primary(ctx)
        try:
            payload = json.loads(outcome.stdout) if outcome.stdout else outcome.stderr
        except (json.JSONDecodeError, ValueError):
            payload = outcome.stdout or outcome.stderr
        is_domain_eval = task_name == "domain_evaluate" and isinstance(payload, dict)
        if is_domain_eval and "domain" in payload and "fulcrum" in payload:
            try:
                self._record_identity_artifact(
                    domain=str(payload["domain"]),
                    fulcrum=float(payload["fulcrum"]),
                    outcome_rating=float(payload.get("confidence", 0.5)),
                    artifact_type="domain",
                    title=f"{payload['domain']} domain evaluation",
                    content=payload,
                )
            except Exception as exc:
                logger.debug("Identity recording skipped for domain evaluation: %s", exc)
        return payload

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

    def _record_identity_artifact(
        self,
        domain: str,
        fulcrum: float,
        outcome_rating: float,
        artifact_type: str,
        title: str,
        content: dict,
        context: Optional[dict] = None,
    ) -> str:
        self._crystal.observe(domain, fulcrum, outcome_rating, context=context)
        identity = self._crystal.get_identity()
        version = identity.version if identity else 1
        artifact = Artifact(
            artifact_id="",
            user_name=self._profile.name,
            domain=domain,
            artifact_type=artifact_type,
            title=title,
            content=content,
            fulcrum_at_time=max(0.0, min(1.0, float(fulcrum))),
            identity_version=version,
            created_at=time.time(),
        )
        return self._artifacts.save(artifact)
