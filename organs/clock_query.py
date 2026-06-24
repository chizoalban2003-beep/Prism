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

# Out-of-scope query patterns. The LLMClassifier occasionally routes
# countdowns, elapsed-time, and foreign-timezone queries to clock_query
# because they mention "time" or "days". We can only report local
# wall-clock state, so we decline these explicitly rather than
# silently answering the wrong question.
_COUNTDOWN_RE = re.compile(r"\b(?:until|till)\b", re.IGNORECASE)
_ELAPSED_RE = re.compile(r"\b(?:since|ago|how\s+long)\b", re.IGNORECASE)
_FOREIGN_TZ_RE = re.compile(
    r"\bin\s+(?:tokyo|paris|london|new\s+york|nyc|sydney|moscow|"
    r"berlin|madrid|rome|beijing|shanghai|hong\s*kong|singapore|"
    r"dubai|mumbai|delhi|bangalore|seoul|bangkok|jakarta|"
    r"san\s+francisco|los\s+angeles|chicago|toronto|vancouver|"
    r"mexico\s+city|sao\s+paulo|buenos\s+aires|cairo|lagos|"
    r"johannesburg|nairobi|istanbul|tehran|riyadh|karachi|"
    r"[a-z]+\s+time(?:zone)?)\b",
    re.IGNORECASE,
)


def execute(intent: str, message: str, ctx: dict):
    import datetime

    from prism_responses import text_card

    now      = datetime.datetime.now()
    iso      = now.isoformat(timespec="seconds")
    date_str = now.strftime("%A, %B %-d, %Y")
    time_str = now.strftime("%-I:%M %p")
    weekday  = now.strftime("%A")

    msg = message or ""

    # Refuse queries this organ can't actually serve. Returning local
    # time for a countdown or a foreign-timezone question is a lie
    # dressed up as an answer.
    if _COUNTDOWN_RE.search(msg):
        return text_card(
            f"I only report current local time/date. It is {time_str} on "
            f"{date_str}. I can't calculate countdowns — try a web search "
            "for date arithmetic.",
            "Time",
        )
    if _ELAPSED_RE.search(msg):
        return text_card(
            f"I only report current local time/date. It is {time_str} on "
            f"{date_str}. I can't calculate elapsed time — try a web search.",
            "Time",
        )
    if _FOREIGN_TZ_RE.search(msg):
        return text_card(
            f"I only know my local time ({time_str} on {date_str}). I "
            "can't convert to other timezones — try a web search like "
            "'time in tokyo right now'.",
            "Time",
        )

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
