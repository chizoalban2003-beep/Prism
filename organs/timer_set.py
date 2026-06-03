"""Bundled organ: timer_set — set a countdown timer using threading."""
ORGAN_META = {
    "intent":      "timer_set",
    "description": "set a countdown timer that fires after a specified duration",
    "version":     "1.0",
}

ORGAN_POLICY = {
    "risk_level":        "low",
    "requires_approval": False,
    "irreversible":      False,
    "max_per_session":   None,
}


def _parse_duration(message: str) -> tuple:
    """Return (seconds: int, label: str)."""
    import re
    total_secs = 0
    units = {
        'hour': 3600, 'hr': 3600, 'h': 3600,
        'minute': 60, 'min': 60, 'm': 60,
        'second': 1, 'sec': 1, 's': 1,
    }
    pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?|[hms])',
                         re.IGNORECASE)
    matches = pattern.findall(message)
    for val_str, unit in matches:
        val = float(val_str)
        unit_key = unit.lower().rstrip('s') if unit.lower() not in ('s', 'm', 'h') else unit.lower()
        mult = units.get(unit_key, 1)
        total_secs += int(val * mult)

    # Build human label
    parts = []
    remaining = total_secs
    if remaining >= 3600:
        h = remaining // 3600
        parts.append(f"{h}h")
        remaining %= 3600
    if remaining >= 60:
        m = remaining // 60
        parts.append(f"{m}m")
        remaining %= 60
    if remaining:
        parts.append(f"{remaining}s")
    label = " ".join(parts) if parts else "unknown"
    return total_secs, label


def _timer_callback(label: str, timers: dict, timer_id: str):
    print(f"\n[PRISM Timer] ⏰ Timer '{label}' has finished!", flush=True)
    timers.pop(timer_id, None)


def execute(intent: str, message: str, ctx: dict):
    import threading
    import time

    from prism_responses import text_card

    seconds, label = _parse_duration(message)
    if seconds <= 0:
        return text_card(
            "Could not parse a duration from message.\n"
            "Examples: 'set a timer for 5 minutes', 'timer 30 seconds', "
            "'2 hours 30 minutes'",
            "Timer",
        )

    # Store active timers in ctx so they can be inspected/cancelled
    if "timers" not in ctx:
        ctx["timers"] = {}
    timers = ctx["timers"]

    timer_id = f"timer_{int(time.time())}_{label}"
    t = threading.Timer(seconds, _timer_callback, args=(label, timers, timer_id))
    t.daemon = True
    t.start()
    timers[timer_id] = {"label": label, "seconds": seconds, "thread": t}

    active_count = len(timers)
    return text_card(
        f"Timer set for {label} ({seconds} seconds).\n"
        f"Active timers: {active_count}",
        "Timer",
    )
