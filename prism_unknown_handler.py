"""
prism_unknown_handler.py
========================
Managerial-PA fallback for unrecognised intents, extracted from
PrismAgent._handle_unknown to keep the agent module focused.

When no intent/organ matches, PRISM either reuses a cached autonomous tool,
returns an actionable setup card for an unconfigured service, or surfaces a
synthesis-approval card to build a new organ on demand. Behaviour is
identical to the original inline method.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from prism_responses import (
    PrismCard,
    setup_required_card,
    synthesis_approval_card,
    text_card,
)

logger = logging.getLogger(__name__)


def handle_unknown(agent: Any, intent: str, message: str, ctx: dict) -> PrismCard:
    """
    Managerial PA fallback: PRISM autonomously acquires the capability,
    executes the task, and reports back. Never returns instructions to the user.
    """
    # Check if autonomous engine has a cached tool for this
    if agent._autonomous.can_handle(message):
        task_id = agent._autonomous.execute_async(
            message, ctx, on_complete=None)
        return text_card(
            f"On it. I have a tool for this — working in the background.\n"
            f"Task ID: {task_id}\n"
            f"I'll notify you when done."
            + (" Check your phone." if agent._push.configured else ""),
            "Working on it")

    # No cached tool — synthesise and execute asynchronously
    router = getattr(agent, '_router', None)

    # Ask LLM whether this needs approval or is safe to do autonomously
    approval_needed = False
    capability_desc = ""
    if router:
        assess_prompt = (
            f"A personal assistant is about to autonomously handle: '{message}'\n"
            f"Assess:\n"
            f"1. What external service/capability is needed?\n"
            f"2. Does this require user approval before acting "
            f"(e.g. sending emails, making purchases, deleting data)? yes/no\n"
            f"Return JSON: {{\"capability\": \"...\", \"needs_approval\": true/false, "
            f"\"reason\": \"...\"}}"
        )
        raw, _ = router.call(assess_prompt, min_capability=1,
                              max_tokens=150, json_mode=True)
        try:
            import json as _j
            clean = raw.strip().lstrip("```json").rstrip("```").strip()
            assessment = _j.loads(clean)
            capability_desc  = assessment.get("capability", "")
            approval_needed  = assessment.get("needs_approval", False)
        except Exception:
            pass

    # Short-circuit: if the LLM-assessed capability (or the raw user
    # message) clearly needs a service whose [section] in prism_config.toml
    # is empty, return the actionable setup card instead of synthesising
    # a tool that will fail at runtime. The synthesis path stays available
    # for genuinely novel capabilities the user hasn't pre-configured for.
    _combined = (message + " " + (capability_desc or "")).lower()
    if (any(k in _combined for k in ("calendar", "schedule", "agenda", "appointment", "meeting"))
            and not getattr(agent._calendar, "configured", False)):
        return setup_required_card(
            service        = "Calendar",
            why            = (
                "PRISM can't fetch events or schedule anything until you connect "
                "a calendar. iCal URL is the simplest provider."
            ),
            config_section = "calendar",
            snippet        = (
                'provider = "ical_url"          # or "google" or "caldav"\n'
                'ical_url = "webcal://..."      # paste your private iCal feed URL\n'
                '# google_token = ""            # OAuth2 token  (provider="google")\n'
                '# caldav_url   = ""            # CalDAV server (provider="caldav")'
            ),
            steps = [
                "Google Calendar → Settings → 'Integrate calendar' → copy the Secret iCal address",
                "Paste that URL above as ical_url",
                "Restart PRISM: pkill -f prism_daemon && python3 -m prism_daemon &",
                "Ask 'what is on my calendar today?' again",
            ],
            docs_url = "https://support.google.com/calendar/answer/37648",
        )
    if (any(k in _combined for k in ("email", "inbox", "mailbox", "gmail"))
            and not getattr(agent._email, "configured", False)):
        return setup_required_card(
            service        = "Email",
            why            = (
                "PRISM needs IMAP credentials to read or send mail. For Gmail use "
                "an App Password (NOT your normal password) — 2FA must already be on."
            ),
            config_section = "email",
            snippet        = (
                'provider  = "gmail"\n'
                'address   = "you@gmail.com"\n'
                'imap_host = "imap.gmail.com"\n'
                'imap_port = 993\n'
                'password  = "xxxx xxxx xxxx xxxx"   # 16-char App Password\n'
                'max_fetch = 20'
            ),
            steps = [
                "Open https://myaccount.google.com/apppasswords",
                "Generate an App Password labeled 'PRISM' and copy the 16-char code",
                "Paste it above as password",
                "Restart PRISM: pkill -f prism_daemon && python3 -m prism_daemon &",
                "Ask 'check my emails' again",
            ],
            docs_url = "https://support.google.com/accounts/answer/185833",
        )

    # Self-extension: PRISM is about to synthesise a new organ for this
    # request. Instead of doing it silently, surface a synthesis approval
    # card disclosing what will be built, where it lands on disk, and
    # offering a free-text instructions slot so the user can shape the
    # synthesis prompt. On approval, /device/approve dispatches back to
    # handle_synthesis_approval() below. The legacy `approval_needed`
    # text-card path used to wait for a typed 'yes' — replaced because
    # synthesis_approval_card already carries the same disclosure plus
    # a structured Approve/Deny button pair and an optional instructions
    # field, so a single gate covers both 'risky external action' and
    # 'novel capability' cases from the user's POV.
    intent_slug = (
        intent
        if intent
        and intent not in ("general_chat", "novel_capability", "chat")
        and re.match(r"^[a-z_][a-z0-9_]*$", intent)
        else agent._slugify_intent(message)
    )
    risk_hint = (
        f"This may affect external systems via {capability_desc}. "
        if approval_needed and capability_desc
        else ""
    ) + (
        "Writes a Python file to ~/.prism/organs/ and may pip-install "
        "dependencies. AST-validated against unsafe operations before running."
    )
    risk_level = "high" if approval_needed else "medium"
    try:
        prior_synth = agent._instructions.prior_denials_for("_synthesize_organ") if agent._instructions else []
        prior_intent = agent._instructions.prior_denials_for(intent_slug) if agent._instructions else []
        prior = (prior_intent or []) + (prior_synth or [])
    except Exception:
        prior = []
    if prior:
        last = prior[0].text[:200]
        risk_hint = (risk_hint + " ").strip() + f" You denied this before: \"{last}\""
    return synthesis_approval_card(
        intent     = intent_slug,
        message    = message,
        capability = capability_desc,
        risk_hint  = risk_hint,
        risk_level = risk_level,
    )

