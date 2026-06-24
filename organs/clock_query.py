"""Bundled organ: clock_query — report current local time, date, or weekday."""
import re

ORGAN_META = {
    "intent":      "clock_query",
    "description": "report the current local time, date, and weekday",
    "version":     "1.0",
    "capabilities": [],
    "inputs":  {},
    "outputs": {
        "iso":     "str",
        "date":    "str",
        "time":    "str",
        "weekday": "str",
    },
}

ORGAN_POLICY = {
    "risk_level":        "low",
    "requires_approval": False,
    "irreversible":      False,
    "max_per_session":   None,
}


_DATE_RE = re.compile(r"\bdate\b|\bday\b", re.IGNORECASE)
_TIME_RE = re.compile(r"\btime\b|\bhour\b|\bclock\b", re.IGNORECASE)


def execute(intent: str, message: str, ctx: dict):
    import datetime

    from prism_responses import text_card

    now      = datetime.datetime.now()
    iso      = now.isoformat(timespec="seconds")
    date_str = now.strftime("%A, %B %-d, %Y")
    time_str = now.strftime("%-I:%M %p")
    weekday  = now.strftime("%A")

    msg = message or ""
    wants_date = bool(_DATE_RE.search(msg))
    wants_time = bool(_TIME_RE.search(msg)) or not wants_date

    if wants_date and wants_time:
        body = f"It is {time_str} on {date_str}."
    elif wants_date:
        body = f"Today is {date_str}."
    else:
        body = f"It is {time_str}."

    card = text_card(body, "Time")
    card.card_data.update({
        "iso":     iso,
        "date":    date_str,
        "time":    time_str,
        "weekday": weekday,
    })
    return card
