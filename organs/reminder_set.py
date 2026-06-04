"""Bundled organ: reminder_set — set a reminder with sched/threading, saved to JSON."""
ORGAN_META = {
    "intent":      "reminder_set",
    "description": "set a reminder for a future time; fires to stdout and saves to JSON",
    "version":     "1.0",
}

ORGAN_POLICY = {
    "risk_level":        "low",
    "requires_approval": False,
    "irreversible":      False,
    "max_per_session":   None,
}


def _parse_reminder(message: str) -> tuple:
    """Return (reminder_text, delay_seconds) parsed from message."""
    import datetime
    import re

    text = ""
    delay = 0

    # Extract text after "remind me to/that" or "reminder:"
    m = re.search(
        r'remind\s+(?:me\s+)?(?:to|that|about)?\s*[:\s]?(.+?)(?:\s+(?:in|at|after)\s+|\s*$)',
        message, re.IGNORECASE | re.DOTALL,
    )
    if m:
        text = m.group(1).strip()

    # Parse duration
    units = {
        'hour': 3600, 'hr': 3600, 'h': 3600,
        'minute': 60, 'min': 60, 'm': 60,
        'second': 1, 'sec': 1, 's': 1,
        'day': 86400,
    }
    duration_pattern = re.compile(
        r'(\d+(?:\.\d+)?)\s*(days?|hours?|hrs?|minutes?|mins?|seconds?|secs?|[hms])',
        re.IGNORECASE,
    )
    for val_str, unit in duration_pattern.findall(message):
        unit_lower = unit.lower()
        if unit_lower in ('s', 'm', 'h'):
            unit_key = unit_lower
        elif unit_lower.startswith('day'):
            unit_key = 'day'
        else:
            unit_key = unit_lower.rstrip('s')
        delay += int(float(val_str) * units.get(unit_key, 1))

    # Parse absolute time — "at 14:30", "at 2:30pm", "at 10am", "tomorrow at 9am"
    if delay == 0:
        tomorrow = re.search(r'\btomorrow\b', message, re.IGNORECASE) is not None
        # Try "at H:MM [am/pm]"
        at_m = re.search(r'at\s+(\d{1,2}):(\d{2})\s*(am|pm)?', message, re.IGNORECASE)
        if at_m:
            hour = int(at_m.group(1))
            minute = int(at_m.group(2))
            ampm = (at_m.group(3) or "").lower()
            if ampm == "pm" and hour < 12:
                hour += 12
            elif ampm == "am" and hour == 12:
                hour = 0
            now = datetime.datetime.now()
            target = datetime.datetime(now.year, now.month, now.day, hour, minute, 0)
            if tomorrow:
                target += datetime.timedelta(days=1)
            elif target <= now:
                target += datetime.timedelta(days=1)
            delay = int((target - now).total_seconds())
        else:
            # Try "at Xam" or "at Xpm" (no minutes)
            at_m2 = re.search(r'at\s+(\d{1,2})\s*(am|pm)', message, re.IGNORECASE)
            if at_m2:
                hour = int(at_m2.group(1))
                ampm = at_m2.group(2).lower()
                if ampm == "pm" and hour < 12:
                    hour += 12
                elif ampm == "am" and hour == 12:
                    hour = 0
                now = datetime.datetime.now()
                target = datetime.datetime(now.year, now.month, now.day, hour, 0, 0)
                if tomorrow:
                    target += datetime.timedelta(days=1)
                elif target <= now:
                    target += datetime.timedelta(days=1)
                delay = int((target - now).total_seconds())

    if not text:
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
