"""
prism_info_intents.py
=====================
Read-only 'info / stats' intent handlers, grouped out of PrismAgent._execute
to keep the agent module focused. These are pure read-only reports (identity
profile/narrative/growth, outcome stats, budget, weekly reflection, loaded
organs) with no side effects.

handle_info_intent(agent, intent, message, ctx) returns a PrismCard for a
handled intent, or None so the caller continues dispatching. Behaviour is
identical to the original inline blocks.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from prism_responses import PrismCard, text_card

logger = logging.getLogger(__name__)


def handle_info_intent(agent: Any, intent: str, message: str,
                       ctx: dict) -> Optional[PrismCard]:
    if intent == "my_profile":
        persona = getattr(agent, '_persona', None)
        soul = getattr(agent, '_soul', None)
        narrative = getattr(agent, '_narrative', None)
        if persona or soul:
            parts = []
            if persona:
                parts.append(persona.summary())
            if soul:
                parts.append("\n**Soul (beliefs & values):**\n" + soul.compress_for_llm(400))
            if narrative:
                try:
                    parts.append("\n**Current snapshot:**\n" + narrative.snapshot())
                except Exception:
                    pass
            return text_card("\n\n".join(parts), "Your crystallised profile")
        return text_card("Profile not yet initialised.", "Profile")

    if intent == "my_narrative":
        narrative = getattr(agent, '_narrative', None)
        if narrative:
            try:
                return text_card(narrative.weekly(), "Weekly narrative")
            except Exception as exc:
                return text_card(f"Could not generate narrative: {exc}", "Narrative")
        return text_card("Narrative engine not available.", "Narrative")

    if intent == "my_growth":
        persona = getattr(agent, '_persona', None)
        narrative = getattr(agent, '_narrative', None)
        if persona and narrative:
            try:
                report = narrative.growth_report()
                return text_card(report, "What PRISM knows about you")
            except Exception as exc:
                return text_card(f"Growth report failed: {exc}", "Growth")
        return text_card("Not enough data yet.", "Growth")


    # ── Outcome / learning stats ───────────────────────────────────────
    if intent == "outcome_stats":
        tracker = getattr(agent, '_outcome_tracker', None)
        if tracker is None:
            return text_card("OutcomeTracker not available.", "Error")
        stats = tracker.stats(days=30)
        lines5: list[str] = [
            "Chain outcomes (last 30 days):",
            f"  Total chains:     {stats['total']}",
            f"  Completed:        {stats['done']}",
            f"  Abandoned:        {stats['abandoned']}",
            f"  User-corrected:   {stats['user_corrected']}",
            f"  Completion rate:  {stats['completion_rate']:.0%}",
            f"  Avg steps/chain:  {stats['avg_steps']}",
            f"  Avg policy flags: {stats['avg_policy_flags']}",
        ]
        return text_card("\n".join(lines5), "Learning stats")

    # ── Budget snapshot — CEO governance dashboard ────────────────────
    if intent == "budget_status":
        b = getattr(agent, "_budget", None)
        if b is None:
            return text_card("Budget engine not available.", "Budget")
        snap = b.snapshot()
        bar_width = 24
        filled = int(snap["fraction_used"] * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        budget_lines: list[str] = [
            f"Daily LLM budget: ${snap['daily_limit_usd']:.2f}",
            f"Spent today:     ${snap['spent_today_usd']:.4f}",
            f"Remaining:       ${snap['remaining_usd']:.4f}",
            f"  [{bar}] {int(snap['fraction_used']*100)}%",
        ]
        if snap["monthly_limit_usd"]:
            budget_lines.append(
                f"Monthly budget:  ${snap['monthly_limit_usd']:.2f} "
                f"(spent ${snap['spent_this_month_usd']:.4f})"
            )
        budget_lines.append(
            "Local providers (Ollama): "
            + ("not counted" if snap["free_provider_bypass"] else "counted")
        )
        policy_label = "block" if snap["block_at_ceiling"] else "warn only"
        budget_lines.append(
            f"At ceiling: {policy_label}. Warn at {int(snap['warn_at_fraction']*100)}%."
        )
        return text_card("\n".join(budget_lines), "Budget")

    # ── Weekly reflection ─────────────────────────────────────────────
    if intent == "reflection":
        refl = getattr(agent, '_reflection', None)
        if refl is None:
            return text_card("Reflection engine not available.", "Error")
        try:
            summary = refl.summarise_for_chat()
            return text_card(summary, "Weekly reflection")
        except Exception as exc:
            return text_card(f"Reflection failed: {exc}", "Error")

    # ── Organ registry ────────────────────────────────────────────────
    if intent == "list_organs":
        organs = agent._organ_loader.list_organs() if hasattr(
            agent._organ_loader, 'list_organs') else []
        if not organs:
            return text_card(
                "No organs loaded yet. Organs are synthesized on demand when "
                "you ask me to do something I don't have a built-in handler for.",
                "Loaded organs")
        lines = "\n".join(f"• **{o}**" for o in organs[:20])
        return text_card(lines, f"Loaded organs ({len(organs)})")

    return None

