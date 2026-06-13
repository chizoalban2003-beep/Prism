"""
prism_narrative.py
==================
Synthesises the living user model into readable reports.
Weekly narratives are stored back to memory so they can be recalled later.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from prism_calibration import PrismCalibration
    from prism_memory import PrismMemory
    from prism_outcome_tracker import OutcomeTracker
    from prism_persona import PrismPersona
    from prism_soul import PrismSoul

logger = logging.getLogger(__name__)


@dataclass
class NarrativeEntry:
    narrative_id: str
    period: str       # "weekly" | "monthly" | "snapshot"
    content: str
    generated_at: float


class PrismNarrative:
    """
    Generates human-readable synthesis of the living user model.
    Uses LLM when available; falls back to structured output.
    """

    def __init__(
        self,
        persona: PrismPersona,
        memory: Optional[PrismMemory] = None,
        outcome_tracker: Optional[OutcomeTracker] = None,
        calibration: Optional[PrismCalibration] = None,
        soul: Optional[PrismSoul] = None,
        llm_router=None,
    ):
        self._persona = persona
        self._memory = memory
        self._outcome_tracker = outcome_tracker
        self._calibration = calibration
        self._soul = soul
        self._router = llm_router

    # ── Public reports ────────────────────────────────────────────────────────

    def weekly(self) -> str:
        return self._build_period_report(days=7, label="weekly")

    def monthly(self) -> str:
        return self._build_period_report(days=30, label="monthly")

    def snapshot(self) -> str:
        """Current state — who PRISM thinks you are right now."""
        parts: list[str] = []

        persona_ctx = self._persona.build_context(max_chars=500)
        if persona_ctx:
            parts.append(persona_ctx)

        if self._soul is not None:
            try:
                soul_ctx = self._soul.compress_for_llm(400)
                if soul_ctx:
                    parts.append(f"[Soul — beliefs & values]\n{soul_ctx}")
            except Exception as exc:
                logger.debug("[narrative] soul compress failed: %s", exc)

        if self._outcome_tracker is not None:
            try:
                stats = self._outcome_tracker.stats(days=7)
                parts.append(
                    f"[Recent outcomes — 7 days]\n"
                    f"Completed: {stats.get('done', 0)}/{stats.get('total', 0)} "
                    f"(rate: {stats.get('completion_rate', 0):.0%})"
                )
            except Exception as exc:
                logger.debug("[narrative] outcome stats failed: %s", exc)

        return "\n\n".join(parts) if parts else "No profile data yet."

    def growth_report(self) -> str:
        """How much has PRISM learned — trait growth, confidence trends, outcome rates."""
        lines: list[str] = ["**What PRISM knows about you**\n"]

        growth_7 = self._persona.growth_since(days=7)
        growth_30 = self._persona.growth_since(days=30)
        lines.append(
            f"Last 7 days: {growth_7['new_traits']} new traits, "
            f"{growth_7['new_patterns']} new patterns, "
            f"avg confidence {growth_7['confidence_avg']:.0%}"
        )
        lines.append(
            f"Last 30 days: {growth_30['new_traits']} new traits, "
            f"{growth_30['new_patterns']} new patterns"
        )

        traits = self._persona.list_traits()
        if traits:
            high_conf = [t for t in traits if t.confidence >= 0.7]
            lines.append(
                f"\nHigh-confidence traits ({len(high_conf)}/{len(traits)}):"
            )
            for t in high_conf[:5]:
                lines.append(f"  • {t.name}: {t.value} ({int(t.confidence * 100)}%)")

        if self._outcome_tracker is not None:
            try:
                s7 = self._outcome_tracker.stats(days=7)
                s30 = self._outcome_tracker.stats(days=30)
                lines.append(
                    f"\nOutcome rates: {s7.get('completion_rate', 0):.0%} this week, "
                    f"{s30.get('completion_rate', 0):.0%} this month"
                )
            except Exception:
                pass

        if self._calibration is not None:
            try:
                events = self._calibration.history(n=20)
                if events:
                    from collections import Counter
                    dirs = Counter(e.direction for e in events)
                    lines.append(
                        f"\nCalibration history ({len(events)} events): "
                        + ", ".join(f"{v}× {k}" for k, v in dirs.most_common(3))
                    )
            except Exception:
                pass

        peaks = self._persona.peak_hours()
        if peaks:
            lines.append(
                f"\nPeak activity hours: {', '.join(f'{h}:00' for h in peaks)}"
            )

        return "\n".join(lines)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_period_report(self, days: int, label: str) -> str:
        data = self._gather_period_data(days)
        content = self._synthesise(data, days, label)
        self._store_to_memory(content, label)
        return content

    def _gather_period_data(self, days: int) -> dict:
        data: dict = {"days": days}

        data["persona_ctx"] = self._persona.build_context(max_chars=600)
        data["growth"] = self._persona.growth_since(days=days)

        if self._outcome_tracker is not None:
            try:
                data["outcomes"] = self._outcome_tracker.stats(days=days)
                data["recent_outcomes"] = [
                    {"goal": r.goal[:80], "outcome": r.outcome}
                    for r in self._outcome_tracker.recent(n=10)
                ]
            except Exception:
                pass

        if self._calibration is not None:
            try:
                since = time.time() - days * 86400
                events = [e for e in self._calibration.history(n=30) if e.timestamp >= since]
                data["calibration"] = [
                    f"{e.direction} ({e.domain})" for e in events[:10]
                ]
            except Exception:
                pass

        if self._soul is not None:
            try:
                data["soul_ctx"] = self._soul.compress_for_llm(300)
            except Exception:
                pass

        return data

    def _synthesise(self, data: dict, days: int, label: str) -> str:
        if self._router is not None:
            try:
                return self._llm_synthesise(data, days, label)
            except Exception as exc:
                logger.debug("[narrative] LLM synthesis failed: %s", exc)

        return self._structured_fallback(data, days, label)

    def _llm_synthesise(self, data: dict, days: int, label: str) -> str:
        outcomes = data.get("outcomes", {})
        calibration = data.get("calibration", [])
        growth = data.get("growth", {})

        prompt = (
            f"Write a {label} PRISM narrative — a concise personal reflection "
            f"(3–5 short paragraphs) about what you observed about this user over the last {days} days. "
            "Speak in first person as PRISM. Be specific about patterns and growth. "
            "Do not use bullet points — prose only.\n\n"
            f"Behavioural profile:\n{data.get('persona_ctx', '(none)')}\n\n"
            f"Outcome stats: completed {outcomes.get('done', 0)} tasks, "
            f"rate {outcomes.get('completion_rate', 0):.0%}\n"
            f"Calibration feedback: {', '.join(calibration) or 'none'}\n"
            f"Growth: {growth.get('new_traits', 0)} new traits learned, "
            f"{growth.get('new_patterns', 0)} new patterns\n"
            f"Soul context:\n{data.get('soul_ctx', '(none)')}\n\n"
            "Keep under 300 words. Make it feel like a genuine personal reflection."
        )

        raw, _ = self._router.call(prompt, min_capability=1, max_tokens=400)
        return raw.strip() if raw.strip() else self._structured_fallback(data, days, label)

    def _structured_fallback(self, data: dict, days: int, label: str) -> str:
        outcomes = data.get("outcomes", {})
        growth = data.get("growth", {})
        calibration = data.get("calibration", [])

        period_label = f"last {days} days"
        lines = [f"**PRISM {label.capitalize()} Narrative — {period_label}**\n"]

        total = outcomes.get("total", 0)
        done = outcomes.get("done", 0)
        rate = outcomes.get("completion_rate", 0)
        if total:
            lines.append(
                f"Over the {period_label}, PRISM tracked {total} tasks with a "
                f"{rate:.0%} completion rate ({done} completed)."
            )

        new_traits = growth.get("new_traits", 0)
        new_patterns = growth.get("new_patterns", 0)
        if new_traits or new_patterns:
            lines.append(
                f"PRISM learned {new_traits} new behavioural trait(s) and "
                f"identified {new_patterns} new pattern(s)."
            )

        if calibration:
            lines.append(f"Calibration signals: {', '.join(calibration[:5])}.")

        peaks = self._persona.peak_hours()
        if peaks:
            lines.append(
                f"Peak activity at {', '.join(f'{h}:00' for h in peaks[:2])}."
            )

        lines.append(f"\n{data.get('persona_ctx', '')}")
        return "\n".join(lines)

    def _store_to_memory(self, content: str, period: str) -> None:
        if self._memory is None or not content:
            return
        try:
            title = f"PRISM {period.capitalize()} Narrative — {_date_label()}"
            self._memory.ingest(
                content=content,
                source="prism_narrative",
                title=title,
                tags=[period, "narrative", "living_model"],
            )
        except Exception as exc:
            logger.debug("[narrative] memory store failed: %s", exc)


def _date_label() -> str:
    from datetime import date

    return date.today().isoformat()
