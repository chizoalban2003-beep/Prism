"""
prism_chat_context.py
=====================
Chat-prelude helpers extracted from ``PrismAgent.chat``.

Three small attachers that populate slots on the per-turn ``context`` dict
the tier dispatchers later read, and one short-circuit that returns a
``setup_required_card`` when the user is asking about a service whose
configuration block is empty.

The attachers mutate the passed-in ``context`` dict and return ``None``.
They are no-ops when their dependency is missing, so calls remain safe
during partial bootstrap. The short-circuit is pure: it inspects only
its arguments and returns either a ``PrismCard`` or ``None``.
"""
from __future__ import annotations

from typing import Any, Optional

from prism_responses import PrismCard, setup_required_card


def attach_perception(context: dict, perception: Any) -> None:
    if perception is None:
        return
    state = perception.current_context()
    context["perception"] = state.to_factor_updates()
    context["perception_summary"] = state.summary


def attach_memory_recall(context: dict, memory: Any, message: str) -> None:
    if memory is None or not message:
        return
    try:
        results = memory.search(message, top_n=3)
    except Exception:
        return
    if not results:
        return
    context["memory_context"] = [
        {"title": r.entry.title, "excerpt": r.excerpt,
         "source": r.entry.source, "score": round(r.score, 3)}
        for r in results
    ]


def attach_persona(context: dict, persona: Any) -> None:
    context["persona_context"] = (
        persona.build_context() if persona is not None else ""
    )


_CALENDAR_TRIGGERS = (
    "my calendar", "my schedule", "my agenda",
    "calendar today", "calendar tomorrow",
    "schedule today", "agenda today",
    "my meetings", "my appointments", "my events today",
)

_EMAIL_TRIGGERS = (
    "my email", "my emails", "my inbox", "my mailbox",
    "check my mail", "check my inbox",
    "any new emails", "unread emails",
)


def setup_required_short_circuit(
    message: str,
    calendar: Any,
    email: Any,
) -> Optional[PrismCard]:
    """Return a setup card when the user asks about an unconfigured service.

    Skips the four planning tiers downstream — they'd otherwise either fail
    at runtime or have the LLM apologise for missing access. Triggered only
    on unambiguous service references; anything else falls through to the
    normal pipeline.
    """
    msg_lw = (message or "").lower()
    if not msg_lw:
        return None

    if (not getattr(calendar, "configured", False)
            and any(k in msg_lw for k in _CALENDAR_TRIGGERS)):
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

    if (not getattr(email, "configured", False)
            and any(k in msg_lw for k in _EMAIL_TRIGGERS)):
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

    return None
