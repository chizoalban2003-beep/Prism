"""
prism_organ_dispatch.py
=======================
Organ execution gate, extracted from ``PrismAgent._execute`` to keep the agent
module focused and to give the security-critical path a single home.

``dispatch_organ(agent, intent, message, ctx)`` runs a loaded organ through the
three-layer gate:

    L1  ConstitutionGuard.check          — capability requirements
    L2  ORGAN_POLICY approval gate        — requires_approval → pending + card
        + L1 organ-session ceiling, L2 per-organ rate limit
    L3  BudManager.spawn/execute          — capability-scoped ctx, token lifecycle

Returns a ``PrismCard`` when the intent maps to a loaded organ (success,
approval, block, or error), or ``None`` when no organ matches the intent (the
caller then falls back to synthesis via ``_handle_unknown``). Behaviour is
identical to the original inline block — only the location changed.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

from prism_responses import PrismCard, approval_card, text_card

logger = logging.getLogger(__name__)


def dispatch_organ(agent: Any, intent: str, message: str,
                   ctx: dict) -> Optional[PrismCard]:
    organ_fn = agent._organ_loader.get(intent)
    if organ_fn is None:
        return None
    try:
        ctx.setdefault("organ_loader", agent._organ_loader)
        ctx.setdefault("policy_engine", agent._policy)
        ctx.setdefault("tasks", getattr(agent, "_tasks", None))
        ctx.setdefault("email", getattr(agent, "_email", None))
        ctx.setdefault("calendar", getattr(agent, "_calendar", None))
        ctx.setdefault("router", getattr(agent, "_router", None))
        ctx.setdefault("memory_graph", getattr(agent, "_memory_graph", None))
        _tw = dict(agent._config.get("twilio", {}))
        _tw.setdefault("account_sid", os.environ.get("TWILIO_ACCOUNT_SID", ""))
        _tw.setdefault("auth_token",  os.environ.get("TWILIO_AUTH_TOKEN", ""))
        _tw.setdefault("from_number", os.environ.get("TWILIO_FROM", ""))
        ctx.setdefault("twilio_config", _tw)
        ctx.setdefault("contacts", getattr(agent, "_contacts", None))

        # L1 Constitution check — validate organ capabilities against L1 rules
        if agent._constitution is not None:
            caps = agent._organ_loader.get_organ_capabilities(intent)
            ok, reason = agent._constitution.check(intent, caps)
            if not ok:
                logger.warning("[constitution] Blocked %s: %s", intent, reason)
                return text_card(
                    f"This action is restricted by PRISM's constitution.\n\n{reason}",
                    f"Blocked — {intent}",
                )

        # L2 Hard approval gate — block irreversible/requires_approval organs
        if not ctx.get(f"_approved_{intent}"):
            policy = agent._organ_loader.get_organ_policy(intent)
            if policy.get("requires_approval"):
                agent._pending_approval = {
                    "organ_intent":  intent,
                    "organ_message": message,
                    "organ_ctx":     dict(ctx),
                    "expires":       time.time() + 300,
                }
                action_desc = message[:200]
                risk_level = policy.get("risk_level", "medium")
                why_parts = []
                if policy.get("irreversible"):
                    why_parts.append("Action is irreversible once taken")
                caps = agent._organ_loader.get_organ_capabilities(intent)
                if caps:
                    why_parts.append("Uses capability: " + ", ".join(sorted(caps)))
                why_parts.append(f"Organ '{intent}' is policy-gated")
                prior = []
                try:
                    if agent._instructions is not None:
                        prior = agent._instructions.prior_denials_for(intent)
                except Exception:
                    prior = []
                if prior:
                    last = prior[0].text[:200]
                    why_parts.append(f"You denied this before: \"{last}\"")
                return approval_card(
                    task       = intent,
                    reason     = f"Run <strong>{intent}</strong> for: {action_desc}",
                    params     = {"organ_intent": intent, "organ_message": message},
                    risk_level = risk_level,
                    risk_why   = " · ".join(why_parts),
                )

        # L1 absolute ceiling — total organ executions per session.
        if agent._bud_mgr is not None and agent._bud_mgr.organ_budget_exceeded():
            logger.warning(
                "[constitution] organ session ceiling reached — blocking %s", intent)
            return text_card(
                "PRISM has reached its per-session organ execution limit "
                "(constitution L1). Start a new session to continue.",
                f"Blocked — {intent}",
            )

        # L2 per-organ rate limit — ORGAN_POLICY.max_per_session.
        _policy = agent._organ_loader.get_organ_policy(intent)
        _cap = _policy.get("max_per_session")
        if (_cap is not None and agent._bud_mgr is not None
                and agent._bud_mgr.session_intent_count(intent) >= int(_cap)):
            logger.warning(
                "[policy] %s exceeded max_per_session=%s — blocking", intent, _cap)
            return text_card(
                f"'{intent}' has hit its per-session limit ({_cap} run"
                f"{'s' if int(_cap) != 1 else ''}). It won't run again this session.",
                f"Rate limited — {intent}",
            )

        # Execute via BudManager (scoped context, token lifecycle)
        if agent._bud_mgr is not None:
            caps = agent._organ_loader.get_organ_capabilities(intent)
            handle = agent._bud_mgr.spawn(intent, message, ctx, caps)
            try:
                return agent._bud_mgr.execute(handle, organ_fn)
            except Exception as exc:
                return text_card(f"Organ '{intent}' failed: {exc}", intent)
        else:
            return organ_fn(intent, message, ctx)
    except Exception as exc:
        return text_card(f"Organ '{intent}' failed: {exc}", intent)
