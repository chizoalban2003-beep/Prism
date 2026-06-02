"""Bundled organ: meeting_brief — generate a pre-meeting brief from event details."""
ORGAN_META = {
    "intent": "meeting_brief",
    "description": "Generate a pre-meeting brief from calendar event details and attendee names",
    "version": "1.0",
}


def execute(intent: str, message: str, ctx: dict):
    import re

    from prism_responses import text_card

    _QUOTED_RE = re.compile(r'["\u2018\u2019\u201c\u201d]([^"\']+)["\u2018\u2019\u201c\u201d]')
    _ABOUT_RE  = re.compile(r'meeting (?:about|on|re:?)\s+([^,\.]+)', re.I)
    _WITH_RE   = re.compile(r'(?:with|and)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', re.I)
    _TIME_RE   = re.compile(
        r'\b(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\w*\s+)?'
        r'(?:\d{1,2}(?::\d{2})?\s*(?:am|pm)|'
        r'(?:today|tomorrow|monday|tuesday|wednesday|thursday|friday)'
        r'(?:\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm))?)\b',
        re.I,
    )

    try:
        title = (
            ctx.get("meeting_title")
            or (_QUOTED_RE.search(message) and _QUOTED_RE.search(message).group(1))
            or (_ABOUT_RE.search(message) and _ABOUT_RE.search(message).group(1).strip())
            or "Upcoming Meeting"
        )

        raw_attendees = ctx.get("attendees", "")
        if isinstance(raw_attendees, list):
            attendees = raw_attendees
        elif raw_attendees:
            attendees = [a.strip() for a in re.split(r"[,;]", raw_attendees) if a.strip()]
        else:
            attendees = [m.group(1) for m in _WITH_RE.finditer(message)]

        date_str = ctx.get("meeting_date", "")
        if not date_str:
            tm = _TIME_RE.search(message)
            date_str = tm.group(0) if tm else "Not specified"

        title_low = title.lower()
        if any(k in title_low for k in ("1:1", "one on one", "1-1", "check-in", "checkin")):
            meeting_type = "1:1"
        elif any(k in title_low for k in ("review", "retro", "retrospective")):
            meeting_type = "review"
        elif any(k in title_low for k in ("planning", "sprint", "roadmap", "kickoff")):
            meeting_type = "planning"
        else:
            meeting_type = "general"

        _CHECKLISTS = {
            "1:1": [
                "Review progress on previously agreed actions",
                "Prepare updates on current priorities",
                "Note any blockers or support needed",
                "Think about career or growth topics to raise",
            ],
            "review": [
                "Gather relevant metrics and data before the meeting",
                "Prepare a concise summary of achievements and gaps",
                "Document lessons learned and improvement ideas",
                "Bring specific examples to illustrate points",
                "Align on action items to track post-review",
            ],
            "planning": [
                "Review backlog or open items beforehand",
                "Draft a prioritised list of goals for the period",
                "Identify dependencies and risks early",
                "Confirm availability and capacity of attendees",
                "Prepare any required estimates or proposals",
            ],
            "general": [
                "Clarify the meeting objective in advance",
                "Share an agenda at least 24 hours before",
                "Prepare any materials or data referenced",
                "Identify the desired outcome or decision needed",
            ],
        }

        checklist_items = _CHECKLISTS[meeting_type]
        checklist = "\n".join(f"  [ ] {item}" for item in checklist_items)

        attendees_str = ", ".join(attendees) if attendees else "Not specified"

        result = (
            f"Pre-Meeting Brief: {title}\n"
            f"{'='*50}\n"
            f"Date/Time : {date_str}\n"
            f"Attendees : {attendees_str}\n"
            f"Type      : {meeting_type.upper()}\n\n"
            f"Meeting Context\n{'─'*30}\n"
            f"Purpose: {title}\n\n"
            f"Key Questions to Answer\n{'─'*30}\n"
            f"  • What is the desired outcome of this meeting?\n"
            f"  • What decisions need to be made?\n"
            f"  • What information must be shared or gathered?\n\n"
            f"Preparation Checklist\n{'─'*30}\n"
            f"{checklist}\n\n"
            f"Notes Template\n{'─'*30}\n"
            f"  Agenda items discussed:\n  -\n"
            f"  Decisions made:\n  -\n"
            f"  Action items (owner | due date):\n  -"
        )
    except Exception as exc:
        result = f"Error generating meeting brief: {exc}"

    return text_card(result, intent)
