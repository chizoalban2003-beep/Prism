"""Bundled organ: health_summary — summarise weekly health metrics from a local JSON log."""
ORGAN_META = {
    "intent": "health_summary",
    "description": "Summarise health metrics from a local JSON log (steps, sleep, HRV, weight)",
    "version": "1.0",
    "capabilities": ["internet_read"],
}

ORGAN_POLICY = {
    "risk_level": "low",
    "requires_approval": False,
    "irreversible": False,
    "max_per_session": None,
}


def execute(intent: str, message: str, ctx: dict):
    import json
    from datetime import date, timedelta
    from pathlib import Path

    from prism_responses import text_card

    default_path = str(Path.home() / ".prism" / "health.json")
    log_path = Path(ctx.get("health_log", default_path))

    if not log_path.exists():
        msg = (
            "No health log found.\n\n"
            f"Expected location: {log_path}\n\n"
            "Create a JSON file with an array of daily entries:\n"
            '  [{"date": "2026-06-01", "steps": 8500, '
            '"sleep_hours": 7.5, "hrv": 55.0, "weight_kg": 75.2}, ...]\n\n'
            "All fields except 'date' are optional.\n"
            "You can also set a custom path via ctx['health_log']."
        )
        return text_card(msg, intent)

    try:
        entries = json.loads(log_path.read_text(encoding="utf-8"))

        today = date.today()
        cutoff_7 = today - timedelta(days=7)
        cutoff_14 = today - timedelta(days=14)

        week = [e for e in entries if e.get("date", "") >= str(cutoff_7)]
        prev_week = [
            e for e in entries
            if str(cutoff_14) <= e.get("date", "") < str(cutoff_7)
        ]

        if not week:
            return text_card(
                f"No entries found in the last 7 days (since {cutoff_7}).\n"
                f"Log has {len(entries)} total entries.",
                intent,
            )

        def avg(records, field):
            vals = [float(r[field]) for r in records if field in r and r[field] is not None]
            return sum(vals) / len(vals) if vals else None

        metrics = {
            "steps":       ("Steps/day",    avg(week, "steps")),
            "sleep_hours": ("Sleep (hrs)",  avg(week, "sleep_hours")),
            "hrv":         ("HRV (ms)",     avg(week, "hrv")),
            "weight_kg":   ("Weight (kg)",  avg(week, "weight_kg")),
        }

        prev_hrv = avg(prev_week, "hrv")

        flags = []
        steps_avg = metrics["steps"][1]
        sleep_avg = metrics["sleep_hours"][1]
        hrv_avg   = metrics["hrv"][1]

        if steps_avg is not None and steps_avg < 3000:
            flags.append(f"Low activity: avg {steps_avg:,.0f} steps/day (target >= 3,000)")
        if sleep_avg is not None and sleep_avg < 6:
            flags.append(f"Insufficient sleep: avg {sleep_avg:.1f} hrs/night (target >= 6)")
        if hrv_avg is not None and prev_hrv is not None and prev_hrv > 0:
            drop_pct = (prev_hrv - hrv_avg) / prev_hrv * 100
            if drop_pct > 20:
                flags.append(
                    f"HRV drop: {drop_pct:.0f}% below prior-week avg "
                    f"({hrv_avg:.1f} vs {prev_hrv:.1f} ms)"
                )

        lines = [
            f"Health Summary — Last 7 days ({cutoff_7} to {today})",
            "=" * 50,
            f"  Entries analysed: {len(week)}",
            "",
            "Averages",
            "─" * 30,
        ]
        for field, (label, val) in metrics.items():
            if val is not None:
                fmt = f"{val:,.0f}" if field == "steps" else f"{val:.1f}"
                lines.append(f"  {label:<18} {fmt}")

        if flags:
            lines += ["", "Flags / Concerns", "─" * 30]
            lines += [f"  ⚑ {f}" for f in flags]
        else:
            lines += ["", "  No concerns flagged — keep it up!"]

        result = "\n".join(lines)
    except Exception as exc:
        result = f"Error reading health log '{log_path}': {exc}"

    return text_card(result, intent)
