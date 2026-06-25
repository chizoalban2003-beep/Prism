"""Bundled organ: reminder_set — set a reminder with sched/threading, saved to JSON."""
ORGAN_META = {
    "intent":      "reminder_set",
    "description": "set a reminder for a future time; fires to stdout and saves to JSON",
    "version":     "1.0",
    "capabilities": ["notifications"],
}

ORGAN_POLICY = {
    "risk_level":        "low",
    "requires_approval": False,
    "irreversible":      False,
    "max_per_session":   None,
}


def _parse_reminder(message: str) -> tuple:
    """Return (reminder_text, delay_seconds) parsed from message.

    Strategy: parse time information first, strip it from the message,
    then extract the action phrase from whatever remains. This avoids
    the previous order-dependence where ``remind me in 30 minutes to
    call mom`` leaked the time phrase into the reminder text.
    """
    import datetime
    import re

    delay = 0
    residual = message

    units = {
        'hour': 3600, 'hr': 3600, 'h': 3600,
        'minute': 60, 'min': 60, 'm': 60,
        'second': 1, 'sec': 1, 's': 1,
        'day': 86400,
    }
    duration_pattern = re.compile(
        r'(?:\b(?:in|after|for)\s+)?'
        r'(\d+(?:\.\d+)?)\s*'
        r'(days?|hours?|hrs?|minutes?|mins?|seconds?|secs?|[hms])\b',
        re.IGNORECASE,
    )
    for m in duration_pattern.finditer(message):
        val_str, unit = m.group(1), m.group(2)
        unit_lower = unit.lower()
        if unit_lower in ('s', 'm', 'h'):
            unit_key = unit_lower
        elif unit_lower.startswith('day'):
            unit_key = 'day'
        else:
            unit_key = unit_lower.rstrip('s')
        delay += int(float(val_str) * units.get(unit_key, 1))
    residual = duration_pattern.sub(' ', residual)

    # Absolute time — "at 14:30", "2:30pm", "at 10am", "3pm", "tomorrow at 9am".
    # The "at" prefix is now optional, so bare "3pm" parses too.
    if delay == 0:
        tomorrow = re.search(r'\btomorrow\b', message, re.IGNORECASE) is not None
        hhmm = re.search(
            r'(?:\bat\s+)?(\d{1,2}):(\d{2})\s*(am|pm)?\b',
            message, re.IGNORECASE,
        )
        bare = re.search(
            r'(?:\bat\s+)?(\d{1,2})\s*(am|pm)\b',
            message, re.IGNORECASE,
        )
        target = None
        if hhmm:
            hour = int(hhmm.group(1))
            minute = int(hhmm.group(2))
            ampm = (hhmm.group(3) or "").lower()
            if ampm == "pm" and hour < 12:
                hour += 12
            elif ampm == "am" and hour == 12:
                hour = 0
            now = datetime.datetime.now()
            target = datetime.datetime(now.year, now.month, now.day, hour, minute, 0)
        elif bare:
            hour = int(bare.group(1))
            ampm = bare.group(2).lower()
            if ampm == "pm" and hour < 12:
                hour += 12
            elif ampm == "am" and hour == 12:
                hour = 0
            now = datetime.datetime.now()
            target = datetime.datetime(now.year, now.month, now.day, hour, 0, 0)
        if target is not None:
            now = datetime.datetime.now()
            if tomorrow or target <= now:
                target += datetime.timedelta(days=1)
            delay = int((target - now).total_seconds())

    # Strip absolute-time phrases from residual so they don't leak into text.
    residual = re.sub(
        r'(?:\bat\s+)?\d{1,2}:\d{2}\s*(?:am|pm)?\b',
        ' ', residual, flags=re.IGNORECASE,
    )
    residual = re.sub(
        r'(?:\bat\s+)?\d{1,2}\s*(?:am|pm)\b',
        ' ', residual, flags=re.IGNORECASE,
    )
    residual = re.sub(r'\btomorrow\b', ' ', residual, flags=re.IGNORECASE)

    # Strip the lead-in: "remind me to/that/about/of", "set a reminder for",
    # "reminder:" etc. Whatever's left is the action.
    text = residual
    text = re.sub(
        r'\bset\s+(?:a\s+|an\s+|the\s+)?(?:reminder|alarm)\s+(?:for|to|about|that)?\b',
        ' ', text, flags=re.IGNORECASE,
    )
    text = re.sub(
        r'\breminder\s*[:\-]',
        ' ', text, flags=re.IGNORECASE,
    )
    text = re.sub(
        r'\bremind\s+(?:me\s+)?(?:to|that|about|of)?\b',
        ' ', text, flags=re.IGNORECASE,
    )
    # "wake me up at 7am" → strip "wake me up" so the reminder text is
    # the residue (often empty, in which case _DEFAULT_WAKE label fires).
    text = re.sub(
        r'\bwake\s+me(?:\s+up)?(?:\s+(?:at|in))?\b',
        ' ', text, flags=re.IGNORECASE,
    )
    text = re.sub(r'\s+', ' ', text).strip(' :,-.')
    # If a lone "to" survived from "...for 3pm to take medication"
    # (the lead-in only matched "set a reminder for"), drop it.
    text = re.sub(r'^(?:to|that|about|of)\s+', '', text, flags=re.IGNORECASE).strip()

    if not text:
        # When the user said "wake me up at 7am" or "set an alarm for 7am"
        # — pure time, no action — give a sensible default label.
        if re.search(r"\b(?:wake|alarm)\b", message, re.IGNORECASE):
            text = "wake up"
        else:
            text = message.strip()

    return text, delay


def _fire_reminder(text: str, reminder_id: str, reminders_file):
    import json
    from pathlib import Path

    print(f"\n[PRISM Reminder] {text}", flush=True)

    # Update status in JSON file
    try:
        path = Path(reminders_file)
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        for item in data:
            if item.get("id") == reminder_id:
                item["status"] = "fired"
                break
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def execute(intent: str, message: str, ctx: dict):
    import datetime
    import json
    import threading
    import time
    from pathlib import Path

    from prism_responses import text_card

    reminder_text, delay_seconds = _parse_reminder(message)

    if delay_seconds <= 0:
        return text_card(
            "Could not parse a time/duration from message.\n"
            "Examples:\n"
            "  'remind me to call John in 30 minutes'\n"
            "  'reminder: meeting in 2 hours'\n"
            "  'remind me at 14:30 to take medicine'",
            "Reminder",
        )

    reminders_dir = Path("~/.prism").expanduser()
    try:
        reminders_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return text_card(f"Could not create reminders directory: {exc}", "Reminder")

    reminders_file = reminders_dir / "reminders.json"
    now = datetime.datetime.now()
    fire_at = now + datetime.timedelta(seconds=delay_seconds)
    reminder_id = f"reminder_{int(time.time())}"

    # Load, append, save
    try:
        existing = (json.loads(reminders_file.read_text(encoding="utf-8"))
                    if reminders_file.exists() else [])
        existing.append({
            "id": reminder_id,
            "text": reminder_text,
            "set_at": now.isoformat(),
            "fire_at": fire_at.isoformat(),
            "status": "pending",
        })
        reminders_file.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    except Exception as exc:
        return text_card(f"Could not save reminder: {exc}", "Reminder")

    # Schedule via threading.Timer
    t = threading.Timer(
        delay_seconds,
        _fire_reminder,
        args=(reminder_text, reminder_id, str(reminders_file)),
    )
    t.daemon = True
    t.start()

    # Human-readable duration
    h, rem = divmod(delay_seconds, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    if s:
        parts.append(f"{s}s")
    duration_label = " ".join(parts)

    return text_card(
        f"Reminder set: '{reminder_text}'\n"
        f"Fires in {duration_label} at {fire_at.strftime('%H:%M:%S')}.\n"
        f"Saved to {reminders_file}",
        "Reminder",
    )
