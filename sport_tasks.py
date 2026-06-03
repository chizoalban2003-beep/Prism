"""
sport_tasks.py
==============
KDE Sports Platform — Digital Task Executors

Eight concrete TaskExecutor subclasses that generate rich content via
Ollama (text generation) and the KDE prediction platform.

Each follows the TaskExecutor contract:
    primary()   — full execution (LLM + file saved)
    secondary() — reduced version (shorter output or no file)
    safe()      — describe-only, zero side effects

All LLM calls use Ollama via stdlib urllib.request only.
All file output is markdown or HTML (no external doc-conversion libs).

SECURITY: All output goes to the local ~/.kde directory.
          No external network calls are made beyond localhost Ollama.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Optional

from ksa_executor import (
    ExecutionContext,
    ExecutionOutcome,
    TaskExecutor,
    _ResourceSampler,
)
from ksa_registry import PerformanceMetrics, SnapshotRegistry

logger = logging.getLogger(__name__)

OLLAMA_HOST = "http://localhost:11434"
TEXT_MODEL  = "mistral"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ollama_text(
    prompt:     str,
    model:      str = TEXT_MODEL,
    host:       str = OLLAMA_HOST,
    timeout:    int = 30,
) -> str:
    """
    Call Ollama /api/generate for text generation.
    Returns the response string.
    Raises ConnectionError if Ollama is unavailable.
    """
    body = json.dumps({
        "model":  model,
        "prompt": prompt,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"{host}/api/generate",
        data    = body,
        headers = {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data.get("response", "").strip()
    except urllib.error.URLError as exc:
        raise ConnectionError(f"Ollama unavailable at {host}: {exc}") from exc


def _save_artifact(content: str, filename: str, output_dir: str) -> str:
    """Write content to output_dir/filename. Return full path."""
    path = Path(output_dir).expanduser() / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return str(path)


def _now_date() -> str:
    return date.today().isoformat()


def _make_outcome(
    ctx:     ExecutionContext,
    action:  str,
    rc:      int,
    stdout:  str,
    stderr:  str,
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
# TrainingPlanTask
# ---------------------------------------------------------------------------

class TrainingPlanTask(TaskExecutor):
    """
    task_name = "create_training_plan"

    primary:   Generate a full 7-day training plan as markdown via Ollama.
               Saved to output_dir/training_plan_{date}.md
    secondary: Generate a 3-day plan only.
    safe:      Return plan structure template without LLM content.
    """

    task_name = "create_training_plan"

    def __init__(
        self,
        registry:    Optional[SnapshotRegistry] = None,
        platform=None,
        output_dir:  str = "~/.kde/artifacts",
        ollama_host: str = OLLAMA_HOST,
        text_model:  str = TEXT_MODEL,
    ) -> None:
        self._registry    = registry
        self._platform    = platform
        self._output_dir  = output_dir
        self._ollama_host = ollama_host
        self._text_model  = text_model

    def _build_prompt(self, ctx: ExecutionContext, days: int = 7) -> str:
        p = ctx.payload
        sport        = p.get("sport",        "general")
        role         = p.get("role",         "athlete")
        fitness      = p.get("fitness_level", "moderate")
        season_phase = p.get("season_phase",  "in-season")
        days_to_match = p.get("days_to_match", 7)
        return (
            f"You are an elite {sport} {role} coach. "
            f"Create a detailed {days}-day training plan for a {fitness} athlete "
            f"in the {season_phase} phase. Match in {days_to_match} days. "
            f"Include for each day: session type, duration, intensity (RPE 1-10), "
            f"key focus, recovery notes. Format as markdown with a table for each day."
        )

    def primary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        try:
            content = _ollama_text(self._build_prompt(ctx, days=7),
                                   self._text_model, self._ollama_host)
            filename = f"training_plan_{_now_date()}.md"
            path     = _save_artifact(content, filename, self._output_dir)
            out = json.dumps({"path": path, "plan": content})
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "primary", 0, out, "", elapsed, sampler)
        except ConnectionError as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "primary", 1, "", str(exc), elapsed, sampler)
        except Exception as exc:
            logger.exception("TrainingPlanTask.primary failed")
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "primary", 1, "", str(exc), elapsed, sampler)

    def secondary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        try:
            content = _ollama_text(self._build_prompt(ctx, days=3),
                                   self._text_model, self._ollama_host)
            out = json.dumps({"plan": content, "days": 3})
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "secondary", 0, out, "", elapsed, sampler)
        except ConnectionError as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "secondary", 1, "", str(exc), elapsed, sampler)
        except Exception as exc:
            logger.exception("TrainingPlanTask.secondary failed")
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "secondary", 1, "", str(exc), elapsed, sampler)

    def safe(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        template = {
            "structure": ["Day 1: High-intensity training", "Day 2: Technical work",
                          "Day 3: Tactical session", "Day 4: Recovery",
                          "Day 5: Match prep", "Day 6: Active recovery", "Day 7: Rest"],
            "sections":  ["session_type", "duration", "intensity_rpe",
                          "key_focus", "recovery_notes"],
            "note":      "Run primary() to generate full LLM-powered plan",
        }
        out = json.dumps(template)
        elapsed = (time.perf_counter() - t0) * 1000
        sampler.stop()
        return _make_outcome(ctx, "safe", 0, out, "", elapsed, sampler)


# ---------------------------------------------------------------------------
# MatchReportTask
# ---------------------------------------------------------------------------

class MatchReportTask(TaskExecutor):
    """
    task_name = "match_report"

    primary:   Full post-match analysis report saved as markdown artifact.
    secondary: Executive summary only (3 paragraphs), not saved.
    safe:      List data sources available for the report.
    """

    task_name = "match_report"

    def __init__(
        self,
        registry:    Optional[SnapshotRegistry] = None,
        platform=None,
        output_dir:  str = "~/.kde/artifacts",
        ollama_host: str = OLLAMA_HOST,
        text_model:  str = TEXT_MODEL,
    ) -> None:
        self._registry    = registry
        self._platform    = platform
        self._output_dir  = output_dir
        self._ollama_host = ollama_host
        self._text_model  = text_model

    def _build_prompt(self, ctx: ExecutionContext, summary_only: bool = False) -> str:
        p = ctx.payload
        home   = p.get("home_team",      "Home Team")
        away   = p.get("away_team",      "Away Team")
        score  = p.get("score",          "Unknown")
        stats  = p.get("stats",          {})
        vision = p.get("vision_summary", "")

        sections = (
            "sections: ## Performance, ## Tactics, ## Improvements"
            if not summary_only else "3-paragraph summary"
        )
        return (
            f"You are a professional football analyst. Write a {sections} "
            f"post-match report for {home} vs {away} (score: {score}). "
            f"Stats: {json.dumps(stats)}. "
            f"Vision analysis notes: {vision}. "
            f"Be precise and professional."
        )

    def primary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        try:
            content = _ollama_text(self._build_prompt(ctx), self._text_model, self._ollama_host)
            filename = f"match_report_{_now_date()}.md"
            path     = _save_artifact(content, filename, self._output_dir)
            out = json.dumps({"path": path, "report": content})
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "primary", 0, out, "", elapsed, sampler)
        except ConnectionError as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "primary", 1, "", str(exc), elapsed, sampler)
        except Exception as exc:
            logger.exception("MatchReportTask.primary failed")
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "primary", 1, "", str(exc), elapsed, sampler)

    def secondary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        try:
            content = _ollama_text(self._build_prompt(ctx, summary_only=True),
                                   self._text_model, self._ollama_host)
            out = json.dumps({"summary": content})
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "secondary", 0, out, "", elapsed, sampler)
        except ConnectionError as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "secondary", 1, "", str(exc), elapsed, sampler)
        except Exception as exc:
            logger.exception("MatchReportTask.secondary failed")
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "secondary", 1, "", str(exc), elapsed, sampler)

    def safe(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        sources = {
            "data_sources": ["session_log", "match_prediction_vs_actual",
                             "performance_metrics", "vision_summary"],
            "sections":     ["Performance", "Tactics", "Improvements"],
            "note":         "Run primary() to generate full LLM-powered report",
        }
        out = json.dumps(sources)
        elapsed = (time.perf_counter() - t0) * 1000
        sampler.stop()
        return _make_outcome(ctx, "safe", 0, out, "", elapsed, sampler)


# ---------------------------------------------------------------------------
# ScoutingReportTask
# ---------------------------------------------------------------------------

class ScoutingReportTask(TaskExecutor):
    """
    task_name = "scouting_report"

    primary:   Full scouting report saved as markdown artifact.
    secondary: Key Players section only.
    safe:      Return report template structure.
    """

    task_name = "scouting_report"

    def __init__(
        self,
        registry:    Optional[SnapshotRegistry] = None,
        platform=None,
        output_dir:  str = "~/.kde/artifacts",
        ollama_host: str = OLLAMA_HOST,
        text_model:  str = TEXT_MODEL,
    ) -> None:
        self._registry    = registry
        self._platform    = platform
        self._output_dir  = output_dir
        self._ollama_host = ollama_host
        self._text_model  = text_model

    def _tactical_context(self, ctx: ExecutionContext) -> str:
        if self._platform is None:
            return ""
        try:
            home    = ctx.payload.get("home_team", "")
            away    = ctx.payload.get("opponent",  "")
            sport   = ctx.payload.get("sport",     "football")
            tp      = self._platform.tactical.predict(home, away, sport)
            return f"Tactical analysis: {tp.matchup_summary}"
        except Exception:
            return ""

    def _build_prompt(self, ctx: ExecutionContext, section_only: str = "") -> str:
        opponent = ctx.payload.get("opponent", "Opponent Team")
        sport    = ctx.payload.get("sport",    "football")
        context  = self._tactical_context(ctx)
        if section_only:
            return (
                f"Write the Key Players section of a scouting report for {opponent} "
                f"({sport}). Focus on top 5 players, their strengths and weaknesses."
            )
        return (
            f"Write a professional scouting report for {opponent} ({sport}). "
            f"{context}. "
            f"Include sections: ## Team Overview, ## Key Players, "
            f"## Tactical Tendencies, ## Set Piece Analysis, ## Recommended Approach. "
            f"Be analytical and professional."
        )

    def primary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        try:
            content  = _ollama_text(self._build_prompt(ctx), self._text_model, self._ollama_host)
            opponent = ctx.payload.get("opponent", "opponent").replace(" ", "_")
            filename = f"scouting_{opponent}_{_now_date()}.md"
            path     = _save_artifact(content, filename, self._output_dir)
            out = json.dumps({"path": path, "report": content})
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "primary", 0, out, "", elapsed, sampler)
        except ConnectionError as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "primary", 1, "", str(exc), elapsed, sampler)
        except Exception as exc:
            logger.exception("ScoutingReportTask.primary failed")
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "primary", 1, "", str(exc), elapsed, sampler)

    def secondary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        try:
            content = _ollama_text(self._build_prompt(ctx, section_only="key_players"),
                                   self._text_model, self._ollama_host)
            out = json.dumps({"key_players": content})
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "secondary", 0, out, "", elapsed, sampler)
        except ConnectionError as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "secondary", 1, "", str(exc), elapsed, sampler)
        except Exception as exc:
            logger.exception("ScoutingReportTask.secondary failed")
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "secondary", 1, "", str(exc), elapsed, sampler)

    def safe(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        template = {
            "sections": ["Team Overview", "Key Players", "Tactical Tendencies",
                         "Set Piece Analysis", "Recommended Approach"],
            "note":     "Run primary() to generate full LLM-powered scouting report",
        }
        out = json.dumps(template)
        elapsed = (time.perf_counter() - t0) * 1000
        sampler.stop()
        return _make_outcome(ctx, "safe", 0, out, "", elapsed, sampler)


# ---------------------------------------------------------------------------
# NutritionPlanTask
# ---------------------------------------------------------------------------

class NutritionPlanTask(TaskExecutor):
    """
    task_name = "nutrition_plan"

    primary:   Full daily meal plan with macros, timing, hydration.
    secondary: Pre/post training nutrition only.
    safe:      Return nutrition timing framework.
    """

    task_name = "nutrition_plan"

    def __init__(
        self,
        registry:    Optional[SnapshotRegistry] = None,
        platform=None,
        output_dir:  str = "~/.kde/artifacts",
        ollama_host: str = OLLAMA_HOST,
        text_model:  str = TEXT_MODEL,
    ) -> None:
        self._registry    = registry
        self._platform    = platform
        self._output_dir  = output_dir
        self._ollama_host = ollama_host
        self._text_model  = text_model

    def _build_prompt(self, ctx: ExecutionContext, pre_post_only: bool = False) -> str:
        p            = ctx.payload
        sport        = p.get("sport",          "football")
        body_weight  = p.get("body_weight_kg", 75)
        session_type = p.get("session_type",   "training")
        intensity    = p.get("rpe",            6)
        focus        = p.get("primary_focus",  "performance")

        if pre_post_only:
            return (
                f"Create pre-training and post-training nutrition recommendations "
                f"for a {sport} {session_type} session (RPE {intensity}) "
                f"for an athlete weighing {body_weight}kg. "
                f"Include timing, macros, and hydration."
            )
        return (
            f"Create a detailed daily nutrition plan for a {sport} athlete "
            f"({body_weight}kg) on a {session_type} day (RPE {intensity}). "
            f"Focus: {focus}. "
            f"Include: breakfast, mid-morning snack, lunch, pre-training, "
            f"post-training, dinner. For each: foods, macros (protein/carbs/fat), "
            f"calories, and timing. Also include hydration targets. "
            f"Format as markdown."
        )

    def primary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        try:
            content  = _ollama_text(self._build_prompt(ctx), self._text_model, self._ollama_host)
            filename = f"nutrition_plan_{_now_date()}.md"
            path     = _save_artifact(content, filename, self._output_dir)
            out = json.dumps({"path": path, "plan": content})
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "primary", 0, out, "", elapsed, sampler)
        except ConnectionError as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "primary", 1, "", str(exc), elapsed, sampler)
        except Exception as exc:
            logger.exception("NutritionPlanTask.primary failed")
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "primary", 1, "", str(exc), elapsed, sampler)

    def secondary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        try:
            content = _ollama_text(self._build_prompt(ctx, pre_post_only=True),
                                   self._text_model, self._ollama_host)
            out = json.dumps({"pre_post_nutrition": content})
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "secondary", 0, out, "", elapsed, sampler)
        except ConnectionError as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "secondary", 1, "", str(exc), elapsed, sampler)
        except Exception as exc:
            logger.exception("NutritionPlanTask.secondary failed")
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "secondary", 1, "", str(exc), elapsed, sampler)

    def safe(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        framework = {
            "timing_framework": {
                "wake":          "Hydration + light carbs",
                "pre_training":  "Carbs + moderate protein (90 min before)",
                "during":        "Electrolytes if >60 min",
                "post_training": "Protein + carbs (within 30 min)",
                "evening":       "Protein + vegetables + complex carbs",
            },
            "note": "Run primary() to generate personalised LLM-powered plan",
        }
        out = json.dumps(framework)
        elapsed = (time.perf_counter() - t0) * 1000
        sampler.stop()
        return _make_outcome(ctx, "safe", 0, out, "", elapsed, sampler)


# ---------------------------------------------------------------------------
# SocialMediaTask
# ---------------------------------------------------------------------------

class SocialMediaTask(TaskExecutor):
    """
    task_name = "social_media_post"

    primary:   Platform-specific social posts (Twitter ≤280 chars, Instagram, LinkedIn).
    secondary: Caption only, no hashtags.
    safe:      Return post template with placeholders.
    """

    task_name = "social_media_post"

    _MAX_TWITTER = 280

    def __init__(
        self,
        registry:    Optional[SnapshotRegistry] = None,
        platform=None,
        output_dir:  str = "~/.kde/artifacts",
        ollama_host: str = OLLAMA_HOST,
        text_model:  str = TEXT_MODEL,
    ) -> None:
        self._registry    = registry
        self._platform    = platform
        self._output_dir  = output_dir
        self._ollama_host = ollama_host
        self._text_model  = text_model

    def _build_prompt(
        self,
        ctx:          ExecutionContext,
        platform_name: str  = "twitter",
        hashtags:     bool = True,
    ) -> str:
        p         = ctx.payload
        session   = p.get("session_type", "training")
        sport     = p.get("sport",        "football")
        highlight = p.get("highlight",    "")
        tone      = p.get("tone",         "motivational")

        char_limit = ""
        if platform_name == "twitter":
            char_limit = " Keep it under 280 characters."
        elif platform_name == "instagram":
            char_limit = " Write an engaging Instagram caption (2-4 sentences)."
        elif platform_name == "linkedin":
            char_limit = " Write a professional LinkedIn post (3-5 sentences)."

        hashtag_inst = " Include 3-5 relevant hashtags." if hashtags else " No hashtags."
        return (
            f"Write a {tone} {platform_name} post about a {sport} {session} session. "
            f"Highlight: {highlight or 'great effort and progress'}."
            f"{char_limit}{hashtag_inst}"
        )

    def primary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        try:
            posts: dict = {}
            for plat in ["twitter", "instagram", "linkedin"]:
                content = _ollama_text(
                    self._build_prompt(ctx, platform_name=plat, hashtags=True),
                    self._text_model, self._ollama_host,
                )
                # Enforce Twitter char limit
                if plat == "twitter" and len(content) > self._MAX_TWITTER:
                    content = content[:self._MAX_TWITTER - 1].rsplit(" ", 1)[0] + "…"
                posts[plat] = content

            out = json.dumps(posts)
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "primary", 0, out, "", elapsed, sampler)
        except ConnectionError as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "primary", 1, "", str(exc), elapsed, sampler)
        except Exception as exc:
            logger.exception("SocialMediaTask.primary failed")
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "primary", 1, "", str(exc), elapsed, sampler)

    def secondary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        try:
            content = _ollama_text(
                self._build_prompt(ctx, platform_name="instagram", hashtags=False),
                self._text_model, self._ollama_host,
            )
            out = json.dumps({"caption": content})
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "secondary", 0, out, "", elapsed, sampler)
        except ConnectionError as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "secondary", 1, "", str(exc), elapsed, sampler)
        except Exception as exc:
            logger.exception("SocialMediaTask.secondary failed")
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "secondary", 1, "", str(exc), elapsed, sampler)

    def safe(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        template = {
            "twitter":   "Just completed a [session_type] session in [sport]. [highlight]. #[sport] #training",
            "instagram": (
                "Another day, another grind 💪 [highlight]."
                " Feeling [tone] after this [session_type]. #[sport]"
            ),
            "linkedin": (
                "Proud to share another milestone in my [sport] journey."
                " [highlight]. Grateful for the process."
            ),
            "note":      "Run primary() to generate LLM-powered posts",
        }
        out = json.dumps(template)
        elapsed = (time.perf_counter() - t0) * 1000
        sampler.stop()
        return _make_outcome(ctx, "safe", 0, out, "", elapsed, sampler)


# ---------------------------------------------------------------------------
# EmailDraftTask
# ---------------------------------------------------------------------------

class EmailDraftTask(TaskExecutor):
    """
    task_name = "draft_email"

    primary:   Draft professional email, saved to output_dir.
    secondary: Subject line + bullet points only.
    safe:      Return available email templates.
    """

    task_name = "draft_email"

    _TYPES = ["agent_update", "coach_request", "media_response",
              "recovery_report", "contract_enquiry"]

    def __init__(
        self,
        registry:    Optional[SnapshotRegistry] = None,
        platform=None,
        output_dir:  str = "~/.kde/drafts",
        ollama_host: str = OLLAMA_HOST,
        text_model:  str = TEXT_MODEL,
    ) -> None:
        self._registry    = registry
        self._platform    = platform
        self._output_dir  = output_dir
        self._ollama_host = ollama_host
        self._text_model  = text_model

    def _build_prompt(self, ctx: ExecutionContext, bullets_only: bool = False) -> str:
        p          = ctx.payload
        email_type = p.get("email_type",  "agent_update")
        recipient  = p.get("recipient",   "My Agent")
        context    = p.get("context",     "")
        sport      = p.get("sport",       "football")

        if bullets_only:
            return (
                f"Write a subject line and 3-5 bullet points for a {email_type} "
                f"email to {recipient} about {sport}. Context: {context}."
            )
        return (
            f"Write a professional {email_type} email to {recipient} "
            f"in the context of {sport}. Context: {context}. "
            f"Include: subject line, greeting, body paragraphs, and sign-off. "
            f"Tone: professional but warm."
        )

    def primary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        try:
            content    = _ollama_text(self._build_prompt(ctx), self._text_model, self._ollama_host)
            email_type = ctx.payload.get("email_type", "email")
            filename   = f"{email_type}_{_now_date()}.txt"
            path       = _save_artifact(content, filename, self._output_dir)
            out = json.dumps({"path": path, "draft": content})
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "primary", 0, out, "", elapsed, sampler)
        except ConnectionError as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "primary", 1, "", str(exc), elapsed, sampler)
        except Exception as exc:
            logger.exception("EmailDraftTask.primary failed")
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "primary", 1, "", str(exc), elapsed, sampler)

    def secondary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        try:
            content = _ollama_text(self._build_prompt(ctx, bullets_only=True),
                                   self._text_model, self._ollama_host)
            out = json.dumps({"outline": content})
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "secondary", 0, out, "", elapsed, sampler)
        except ConnectionError as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "secondary", 1, "", str(exc), elapsed, sampler)
        except Exception as exc:
            logger.exception("EmailDraftTask.secondary failed")
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "secondary", 1, "", str(exc), elapsed, sampler)

    def safe(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        out = json.dumps({"available_types": self._TYPES,
                          "note": "Run primary() with email_type in payload"})
        elapsed = (time.perf_counter() - t0) * 1000
        sampler.stop()
        return _make_outcome(ctx, "safe", 0, out, "", elapsed, sampler)


# ---------------------------------------------------------------------------
# PerformanceDashboardTask
# ---------------------------------------------------------------------------

class PerformanceDashboardTask(TaskExecutor):
    """
    task_name = "performance_dashboard"

    primary:   Self-contained HTML dashboard saved as artifact.
    secondary: Markdown text report only.
    safe:      List data available for dashboard.
    """

    task_name = "performance_dashboard"

    def __init__(
        self,
        registry:    Optional[SnapshotRegistry] = None,
        platform=None,
        output_dir:  str = "~/.kde/reports",
        ollama_host: str = OLLAMA_HOST,
        text_model:  str = TEXT_MODEL,
    ) -> None:
        self._registry    = registry
        self._platform    = platform
        self._output_dir  = output_dir
        self._ollama_host = ollama_host
        self._text_model  = text_model

    def _build_html(self, ctx: ExecutionContext) -> str:
        p       = ctx.payload
        name    = p.get("name",  "Athlete")
        sport   = p.get("sport", "Football")
        metrics = p.get("metrics", {})
        today   = _now_date()

        load_values = metrics.get("load_history", [5, 6, 7, 6, 8, 7, 5])
        bar_chart   = _text_bar_chart(load_values, label="Load (7 days)")

        rows = ""
        for k, v in metrics.items():
            if k != "load_history":
                rows += f"<tr><td>{k}</td><td>{v}</td></tr>\n"

        recovery_badge = "green" if metrics.get("recovery_score", 0.7) > 0.6 else "red"
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>KDE Performance Dashboard — {name}</title>
<style>
body{{font-family:sans-serif;background:#0d1117;color:#e6edf3;padding:2rem;}}
h1{{color:#58a6ff;}} h2{{color:#79c0ff;border-bottom:1px solid #30363d;padding-bottom:.5rem;}}
table{{border-collapse:collapse;width:100%;margin:1rem 0;}}
th,td{{border:1px solid #30363d;padding:.5rem 1rem;text-align:left;}}
th{{background:#161b22;}} pre{{background:#161b22;padding:1rem;border-radius:6px;}}
.badge{{display:inline-block;padding:.2rem .6rem;border-radius:4px;font-size:.85rem;}}
.green{{background:#1f6b2e;}} .yellow{{background:#5a4a00;}} .red{{background:#6e1f1f;}}
</style>
</head>
<body>
<h1>🏆 {name} — Performance Dashboard</h1>
<p>Sport: <strong>{sport}</strong> | Generated: <strong>{today}</strong></p>

<h2>📊 Load Chart</h2>
<pre>{bar_chart}</pre>

<h2>📈 Performance Metrics</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
{rows if rows else '<tr><td colspan="2">No metrics available</td></tr>'}
</table>

<h2>💤 Recovery Status</h2>
<p>Recovery score: <span class="badge {recovery_badge}">{metrics.get('recovery_score', 'N/A')}</span></p>

<footer style="margin-top:2rem;color:#484f58;font-size:.8rem;">
Generated by KDE Sports Agent • Local processing only
</footer>
</body>
</html>"""
        return html

    def primary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        try:
            html     = self._build_html(ctx)
            filename = f"dashboard_{_now_date()}.html"
            path     = _save_artifact(html, filename, self._output_dir)
            out = json.dumps({"path": path})
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "primary", 0, out, "", elapsed, sampler)
        except Exception as exc:
            logger.exception("PerformanceDashboardTask.primary failed")
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "primary", 1, "", str(exc), elapsed, sampler)

    def secondary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        try:
            p       = ctx.payload
            name    = p.get("name",  "Athlete")
            metrics = p.get("metrics", {})
            lines   = [f"# Performance Report — {name}", f"Date: {_now_date()}", ""]
            for k, v in metrics.items():
                lines.append(f"- **{k}**: {v}")
            content = "\n".join(lines)
            out = json.dumps({"report": content})
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "secondary", 0, out, "", elapsed, sampler)
        except Exception as exc:
            logger.exception("PerformanceDashboardTask.secondary failed")
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "secondary", 1, "", str(exc), elapsed, sampler)

    def safe(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        sources = {
            "data_sources": ["30-day load history", "session logs", "wearable trends",
                             "technique scores", "upcoming fixtures"],
            "note":         "Run primary() to generate full HTML dashboard",
        }
        out = json.dumps(sources)
        elapsed = (time.perf_counter() - t0) * 1000
        sampler.stop()
        return _make_outcome(ctx, "safe", 0, out, "", elapsed, sampler)


# ---------------------------------------------------------------------------
# PredictionReportTask
# ---------------------------------------------------------------------------

class PredictionReportTask(TaskExecutor):
    """
    task_name = "prediction_report"

    primary:   Full HTML prediction report (pre-match brief) saved as artifact.
    secondary: Match prediction card only (one-pager JSON).
    safe:      Describe which predictions would be included.
    """

    task_name = "prediction_report"

    def __init__(
        self,
        registry:    Optional[SnapshotRegistry] = None,
        platform=None,
        output_dir:  str = "~/.kde/reports",
        ollama_host: str = OLLAMA_HOST,
        text_model:  str = TEXT_MODEL,
    ) -> None:
        self._registry    = registry
        self._platform    = platform
        self._output_dir  = output_dir
        self._ollama_host = ollama_host
        self._text_model  = text_model

    def primary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        try:
            p         = ctx.payload
            home_team = p.get("home_team", "Home Team")
            away_team = p.get("away_team", "Away Team")
            sport     = p.get("sport",     "football")
            squad     = p.get("squad",     [])
            hf        = p.get("home_factors", {})
            af        = p.get("away_factors", {})

            if self._platform is None:
                from prediction_engine import PredictionPlatform
                platform = PredictionPlatform(self._registry)
            else:
                platform = self._platform

            brief = platform.pre_match_brief(home_team, away_team, sport, squad, hf, af)
            mp    = brief["match_prediction"]
            tp    = brief["tactical_analysis"]

            risk_rows = ""
            for rp in brief["squad_risk"]:
                badge = {"low": "green", "moderate": "yellow",
                         "high": "red", "critical": "red"}.get(rp.risk_level, "yellow")
                risk_rows += (
                    f"<tr><td>{rp.athlete_name}</td>"
                    f'<td><span class="badge {badge}">{rp.risk_level}</span></td>'
                    f"<td>{rp.recommendations[0] if rp.recommendations else 'Monitor'}</td></tr>\n"
                )

            html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Pre-Match Brief: {home_team} vs {away_team}</title>
<style>
body{{font-family:sans-serif;background:#0d1117;color:#e6edf3;padding:2rem;max-width:900px;margin:auto;}}
h1{{color:#58a6ff;}} h2{{color:#79c0ff;border-bottom:1px solid #30363d;padding-bottom:.5rem;}}
table{{border-collapse:collapse;width:100%;margin:1rem 0;}}
th,td{{border:1px solid #30363d;padding:.5rem 1rem;text-align:left;}}
th{{background:#161b22;}}
.badge{{display:inline-block;padding:.2rem .6rem;border-radius:4px;font-size:.85rem;}}
.green{{background:#1f6b2e;}} .yellow{{background:#5a4a00;}} .red{{background:#6e1f1f;}}
.prob-bar{{height:20px;background:#58a6ff;display:inline-block;}}
</style>
</head>
<body>
<h1>⚽ Pre-Match Brief</h1>
<h2>{home_team} vs {away_team}</h2>
<p>Sport: {sport} | Generated: {_now_date()}</p>

<h2>📊 Match Outlook</h2>
<table>
<tr><th>Outcome</th><th>Probability</th></tr>
<tr><td>{home_team} Win</td><td>{mp.p_home_win:.1%}</td></tr>
<tr><td>Draw</td><td>{mp.p_draw:.1%}</td></tr>
<tr><td>{away_team} Win</td><td>{mp.p_away_win:.1%}</td></tr>
</table>
<p><strong>Prediction:</strong> {mp.prediction} (confidence: {mp.confidence:.0%})</p>

<h2>🧠 Tactical Analysis</h2>
<p>{tp.matchup_summary}</p>
<ul>
<li><strong>Home advantage:</strong> {tp.home_advantage}</li>
<li><strong>Away advantage:</strong> {tp.away_advantage}</li>
</ul>

<h2>🏥 Squad Risk Overview</h2>
<table>
<tr><th>Player</th><th>Risk Level</th><th>Recommendation</th></tr>
{risk_rows if risk_rows else '<tr><td colspan="3">No squad data provided</td></tr>'}
</table>

<footer style="margin-top:2rem;color:#484f58;font-size:.8rem;">
Generated by KDE Sports Agent • Local processing only
</footer>
</body>
</html>"""

            filename = f"prediction_{home_team.replace(' ','_')}_vs_{away_team.replace(' ','_')}_{_now_date()}.html"
            path     = _save_artifact(html, filename, self._output_dir)
            out = json.dumps({
                "path":            path,
                "prediction":      mp.prediction,
                "p_home_win":      mp.p_home_win,
                "p_draw":          mp.p_draw,
                "p_away_win":      mp.p_away_win,
                "confidence":      mp.confidence,
                "matchup_summary": tp.matchup_summary,
            })
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "primary", 0, out, "", elapsed, sampler)

        except Exception as exc:
            logger.exception("PredictionReportTask.primary failed")
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "primary", 1, "", str(exc), elapsed, sampler)

    def secondary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        try:
            p = ctx.payload
            home_team = p.get("home_team", "Home Team")
            away_team = p.get("away_team", "Away Team")
            sport     = p.get("sport",     "football")
            hf        = p.get("home_factors", {})
            af        = p.get("away_factors", {})

            if self._platform is None:
                from prediction_engine import PredictionPlatform
                platform = PredictionPlatform(self._registry)
            else:
                platform = self._platform

            mp = platform.match.predict(
                home_team, away_team, sport,
                home_form  = hf.get("form", 0.5),
                away_form  = af.get("form", 0.5),
            )
            out = json.dumps({
                "match_card":  f"{home_team} vs {away_team}",
                "prediction":  mp.prediction,
                "p_home_win":  mp.p_home_win,
                "p_draw":      mp.p_draw,
                "p_away_win":  mp.p_away_win,
                "confidence":  mp.confidence,
            })
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "secondary", 0, out, "", elapsed, sampler)
        except Exception as exc:
            logger.exception("PredictionReportTask.secondary failed")
            elapsed = (time.perf_counter() - t0) * 1000
            sampler.stop()
            return _make_outcome(ctx, "secondary", 1, "", str(exc), elapsed, sampler)

    def safe(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        info = {
            "would_include": ["match_prediction", "squad_risk", "tactical_analysis",
                              "player_performance_forecasts", "transfer_intelligence"],
            "note":          "Run primary() to generate full HTML prediction report",
        }
        out = json.dumps(info)
        elapsed = (time.perf_counter() - t0) * 1000
        sampler.stop()
        return _make_outcome(ctx, "safe", 0, out, "", elapsed, sampler)


# ---------------------------------------------------------------------------
# Utility: text bar chart
# ---------------------------------------------------------------------------

def _text_bar_chart(values: list, label: str = "", max_width: int = 30) -> str:
    """Generate a simple text-art bar chart."""
    if not values:
        return "(no data)"
    max_v = max(values) or 1
    lines = [label] if label else []
    for i, v in enumerate(values):
        bar   = "█" * int((v / max_v) * max_width)
        lines.append(f"Day {i+1:2d} |{bar:<{max_width}}| {v:.1f}")
    return "\n".join(lines)
