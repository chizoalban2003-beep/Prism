"""
prism_goal_intents.py
=====================
Goal & context-management intent handlers, grouped out of PrismAgent._execute
to keep the agent module focused: horizon goals (add/list/abandon) and
context switching/status.

handle_goal_intent(agent, intent, message, ctx) returns a PrismCard for a
handled intent, or None so the caller continues dispatching. Behaviour is
identical to the original inline blocks.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from prism_responses import PrismCard, text_card

logger = logging.getLogger(__name__)


def handle_goal_intent(agent: Any, intent: str, message: str,
                       ctx: dict) -> Optional[PrismCard]:
    # ── Horizon Planner ───────────────────────────────────────────────
    if intent == "horizon_add":
        if agent._horizon is None:
            return text_card("Horizon Planner is unavailable.", "Error")
        # Extract intent/condition from the message via LLM if available
        intent_text = message
        trigger     = message
        completion  = ""
        if agent._router:
            try:
                parse_prompt = (
                    f"Extract a horizon goal from this message.\n"
                    f"Message: {message}\n\n"
                    f"Return JSON with keys:\n"
                    f"  intent: the full goal (what to do)\n"
                    f"  trigger_condition: what condition makes this fire\n"
                    f"  completion_condition: what success looks like\n"
                    f"Example: {{\"intent\": \"book a flight to Lisbon\","
                    f" \"trigger_condition\": \"price drops below 300\","
                    f" \"completion_condition\": \"flight booked\"}}\n"
                    f"Return only valid JSON."
                )
                raw, _ = agent._router.call(
                    parse_prompt, min_capability=1, max_tokens=200, json_mode=True)
                import json as _j
                parsed = _j.loads(raw.strip().lstrip("```json").rstrip("```").strip())
                intent_text = parsed.get("intent", message)
                trigger     = parsed.get("trigger_condition", message)
                completion  = parsed.get("completion_condition", "")
            except Exception:
                pass
        gid = agent._horizon.add(
            intent=intent_text,
            trigger_condition=trigger,
            completion_condition=completion,
        )
        return text_card(
            f"Got it. I'll watch for: **{trigger}**\n"
            f"Goal: {intent_text}\n"
            f"Goal ID: `{gid}`\n\n"
            f"I'll check every session and act as soon as the condition is met. "
            f"Say *'show my horizon goals'* to see status.",
            "Horizon goal registered")

    if intent == "horizon_list":
        if agent._horizon is None:
            return text_card("Horizon Planner is unavailable.", "Error")
        goals = agent._horizon.list_goals()
        if not goals:
            return text_card(
                "No horizon goals registered yet.\n\n"
                "Say something like: *'watch for flight prices to drop below $300 "
                "and book for me'* to register one.",
                "No horizon goals")
        from prism_horizon import HorizonGoalStatus
        status_icon = {
            HorizonGoalStatus.WATCHING:   "👁",
            HorizonGoalStatus.TRIGGERED:  "⚡",
            HorizonGoalStatus.PAUSED:     "⏸",
            HorizonGoalStatus.COMPLETED:  "✅",
            HorizonGoalStatus.ABANDONED:  "🚫",
        }
        lines3: list[str] = []
        for g in goals[:10]:
            icon = status_icon.get(g.status, "•")
            lines3.append(
                f"{icon} **{g.intent[:60]}**\n"
                f"  Condition: {g.trigger_condition[:50]}\n"
                f"  Status: {g.status.value} | "
                f"Sessions checked: {g.session_count} | "
                f"Steps done: {len(g.completed_steps)} | "
                f"ID: `{g.goal_id}`"
            )
        return text_card("\n\n".join(lines3), f"Horizon goals ({len(goals)})")

    if intent == "horizon_abandon":
        if agent._horizon is None:
            return text_card("Horizon Planner is unavailable.", "Error")
        # Try to find a goal_id in the message, else abandon the most recent watching
        import re as _re
        gid_match = _re.search(r'\b([0-9a-f]{8})\b', message)
        if gid_match:
            gid = gid_match.group(1)
        else:
            watching = agent._horizon.list_goals()
            watching = [g for g in watching
                        if g.status.value in ("watching", "triggered", "paused")]
            if not watching:
                return text_card("No active horizon goals to abandon.", "Nothing to abandon")
            gid = watching[0].goal_id
        goal = agent._horizon.get(gid)
        if not goal:
            return text_card(f"No goal found with ID `{gid}`.", "Not found")
        agent._horizon.abandon(gid, reason="user requested via chat")
        return text_card(
            f"Stopped watching: **{goal.intent[:60]}**\n"
            f"Goal `{gid}` has been abandoned.",
            "Horizon goal abandoned")

    # ── Context switching ─────────────────────────────────────────────
    if intent in ("switch_context", "context_switch"):
        import re as _re
        cm = getattr(agent, '_context_manager', None)
        if cm is None:
            return text_card("Context manager not available.", "Error")
        m = _re.search(r"\b(work|personal|focus|default)\b", message, _re.IGNORECASE)
        if not m:
            profiles = [p.context_id for p in cm.list_profiles()]
            return text_card(
                f"Available contexts: {', '.join(profiles)}\n"
                "Say: 'switch to work' / 'switch to personal' / 'switch to focus'",
                "Context")
        target = m.group(1).lower()
        try:
            profile = cm.switch(target)
            cm.apply_to_policy(agent._policy)
            cm.inject_into_chain(agent._chain)
            return text_card(
                f"Switched to **{target}** context.\n{profile.description}",
                f"Context: {target}")
        except ValueError as exc:
            return text_card(str(exc), "Error")

    if intent == "context_status":
        cm = getattr(agent, '_context_manager', None)
        if cm is None:
            return text_card("Context manager not available.", "Error")
        profile = cm.active()
        lines4: list[str] = [f"Active context: **{profile.context_id}**",
                 f"{profile.description}",
                 f"Policy overrides: {profile.policy_overrides or 'none'}",
                 f"Organ priorities: {profile.organ_priorities or 'none'}"]
        return text_card("\n".join(lines4), "Context status")

    return None
