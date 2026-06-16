"""Bundled organ: calendar_write — create calendar events or find free slots."""
ORGAN_META = {
    "intent": "calendar_write",
    "description": "Create a calendar event or find the next free slot from a natural-language request",
    "version": "1.0",
    "capabilities": ["internet_write"],
}

ORGAN_POLICY = {
    "risk_level": "medium",
    "requires_approval": True,
    "irreversible": False,
    "max_per_session": 20,
}


def execute(intent: str, message: str, ctx: dict):
    from prism_responses import text_card, setup_required_card

    calendar = ctx.get("calendar")
    if calendar is None or not getattr(calendar, "configured", False):
        return setup_required_card(
            service        = "Calendar",
            why            = "PRISM needs read access to your calendar to schedule events, find free slots, or surface today's agenda. Pick one provider below — iCal URL is the simplest.",
            config_section = "calendar",
            snippet        = (
                'provider = "ical_url"          # or "google" or "caldav"\n'
                'ical_url = "webcal://..."      # paste your private iCal feed URL\n'
                '# google_token = ""            # OAuth2 access token (provider="google")\n'
                '# caldav_url   = ""            # CalDAV server URL    (provider="caldav")\n'
                '# username     = ""\n'
                '# password     = ""'
            ),
            steps = [
                "Google Calendar → Settings → 'Integrate calendar' → copy the Secret iCal address",
                "Paste that URL above as ical_url",
                "Restart PRISM: pkill -f prism_daemon && python3 -m prism_daemon &",
                "Ask 'what is on my calendar today?' again",
            ],
            docs_url = "https://support.google.com/calendar/answer/37648",
        )

    router = ctx.get("router")
    msg_lower = message.lower()

    # Free-slot query
    if "free slot" in msg_lower or "when am i free" in msg_lower or "available" in msg_lower:
        slot = calendar.find_free_slot()
        if slot:
            return text_card(
                f"Next free slot: {slot.strftime('%a %d %b at %H:%M')}", "Calendar"
            )
        return text_card("No free slots found in the next 48 hours.", "Calendar")

    # Event creation — delegate parsing to PrismCalendar
    parsed = calendar.parse_event_from_text(message, router)
    if parsed and parsed.get("start_iso"):
        from datetime import datetime as _dt
        try:
            start = _dt.fromisoformat(parsed["start_iso"])
        except ValueError as exc:
            return text_card(f"Could not parse event start time: {exc}", "Calendar")

        event = calendar.create_event(
            title         = parsed.get("title", "New Event"),
            start         = start,
            duration_mins = parsed.get("duration_mins", 60),
            location      = parsed.get("location", ""),
            attendees     = parsed.get("attendees", []),
        )
        if event:
            return text_card(f"Created: {event}", "Calendar")
        return text_card("Event parse succeeded but creation failed — check calendar auth.", "Calendar")

    return text_card(
        "Could not parse event details. "
        "Try: 'schedule a meeting with X on Friday at 2pm'",
        "Calendar",
    )
