"""Bundled organ: task_reminder — surface overdue/due-today tasks and add new reminders."""
ORGAN_META = {
    "intent": "task_reminder",
    "description": "Show overdue and due-today reminders, or add a new reminder with an optional due date",
    "version": "1.0",
}

ORGAN_POLICY = {
    "risk_level": "low",
    "requires_approval": False,
    "irreversible": False,
    "max_per_session": None,
}


def execute(intent: str, message: str, ctx: dict):
    import re
    from datetime import date, datetime, timedelta

    from prism_responses import text_card

    tasks_engine = ctx.get("tasks")

    # ── Add mode ─────────────────────────────────────────────────────────────
    ADD_RE = re.compile(
        r"\b(add|create|set|remind(er)?)\b.{0,80}?[\"']?(?P<title>[^\"'\n]{3,80})[\"']?",
        re.IGNORECASE,
    )
    DATE_RE = re.compile(
        r"\b(?:by|on|due|at)?\s*"
        r"(?P<date>\d{4}-\d{2}-\d{2}|today|tomorrow|"
        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        re.IGNORECASE,
    )

    msg_lower = message.lower()
    is_add = any(kw in msg_lower for kw in ("add", "create", "set reminder", "remind me", "new task"))

    if is_add and tasks_engine is not None:
        # Extract title: text after the trigger keyword
        title_match = re.search(
            r"(?:add|create|remind(?:er)?(?: me)?|set(?: a)? reminder(?: for)?)\s+(?:to\s+|that\s+)?(.+?)(?:\s+(?:by|on|due|at)\s+|$)",
            message, re.IGNORECASE,
        )
        title = title_match.group(1).strip() if title_match else message.strip()
        title = title[:120]

        # Extract due date
        due_str = ""
        dm = DATE_RE.search(message)
        if dm:
            raw = dm.group("date").lower()
            today = date.today()
            if raw == "today":
                due_str = today.isoformat()
            elif raw == "tomorrow":
                due_str = (today + timedelta(days=1)).isoformat()
            elif re.match(r"\d{4}-\d{2}-\d{2}", raw):
                due_str = raw
            else:
                days_ahead = {
                    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
                    "friday": 4, "saturday": 5, "sunday": 6,
                }
                target = days_ahead.get(raw)
                if target is not None:
                    diff = (target - today.weekday()) % 7 or 7
                    due_str = (today + timedelta(days=diff)).isoformat()

        try:
            task = tasks_engine.add(title=title, due_date=due_str)
            due_note = f" — due {due_str}" if due_str else ""
            return text_card(f"Reminder added: {task.title}{due_note}", intent)
        except Exception as exc:
            return text_card(f"Could not add reminder: {exc}", intent)

    # ── List mode ─────────────────────────────────────────────────────────────
    if tasks_engine is None:
        # Graceful degradation: read ~/.prism/reminders.json if present
        import json
        from pathlib import Path
        p = Path("~/.prism/reminders.json").expanduser()
        if not p.exists():
            return text_card(
                "No task engine available and no reminders file found at ~/.prism/reminders.json.", intent
            )
        try:
            data = json.loads(p.read_text())
        except Exception:
            return text_card("Could not read ~/.prism/reminders.json.", intent)
        items = data if isinstance(data, list) else data.get("tasks", [])
        lines = [f"  • {t.get('title', '?')}  [{t.get('due_date', 'no date')}]" for t in items[:20]]
        return text_card("Reminders (from file):\n" + "\n".join(lines) if lines else "No reminders.", intent)

    try:
        all_tasks = tasks_engine.list_tasks(done=False)
    except Exception as exc:
        return text_card(f"Could not fetch tasks: {exc}", intent)

    today_str = date.today().isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    overdue  = []
    due_today = []
    upcoming = []

    for t in all_tasks:
        d = (t.due_date or "").strip()
        if not d:
            upcoming.append(t)
            continue
        if d < today_str:
            overdue.append(t)
        elif d == today_str:
            due_today.append(t)
        else:
            upcoming.append(t)

    # Sort upcoming by due_date
    upcoming.sort(key=lambda t: t.due_date or "9999")

    lines: list[str] = []

    if overdue:
        lines.append(f"OVERDUE ({len(overdue)}):")
        for t in overdue[:5]:
            lines.append(f"  ⚠ {t.title}  [was due {t.due_date}]")

    if due_today:
        lines.append(f"\nDUE TODAY ({len(due_today)}):")
        for t in due_today:
            lines.append(f"  • {t.title}")

    if upcoming:
        lines.append(f"\nUPCOMING ({min(len(upcoming), 5)} shown):")
        for t in upcoming[:5]:
            due = f"  [{t.due_date}]" if t.due_date else ""
            lines.append(f"  · {t.title}{due}")

    if not lines:
        return text_card("No pending reminders.", intent)

    return text_card("\n".join(lines), intent)
