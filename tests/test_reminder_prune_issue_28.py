"""
tests/test_reminder_prune_issue_28.py
=====================================
Fired reminders must not accumulate forever in ~/.prism/reminders.json —
one live install had 189 fired entries. Both writers prune fired items
older than 30 days: the reminder_set organ (on append) and the proactive
poller (on fire). Pending reminders are never pruned, whatever their age.

Relies on conftest's hermetic HOME: ~/.prism here is a throwaway temp dir.
"""
from __future__ import annotations

import datetime
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_reminder_set_organ",
    Path(__file__).resolve().parent.parent / "organs" / "reminder_set.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _seed(items):
    f = Path("~/.prism/reminders.json").expanduser()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(items), encoding="utf-8")
    return f


def _entry(status, days_ago):
    ts = datetime.datetime.now() - datetime.timedelta(days=days_ago)
    return {
        "id": f"reminder_{status}_{days_ago}",
        "text": f"{status} {days_ago}d",
        "set_at": ts.isoformat(),
        "fire_at": ts.isoformat(),
        "status": status,
    }


class TestReminderSetPrunes:
    def test_old_fired_dropped_recent_and_pending_kept(self):
        f = _seed([
            _entry("fired", 90),    # pruned
            _entry("fired", 5),     # kept — inside 30-day window
            _entry("pending", 90),  # kept — pending never pruned
        ])
        card = _mod.execute("reminder_set", "remind me in 5 minutes to stretch", {})
        assert "Reminder set" in card.body
        ids = {i["id"] for i in json.loads(f.read_text())}
        assert "reminder_fired_90" not in ids
        assert "reminder_fired_5" in ids
        assert "reminder_pending_90" in ids
        assert len(ids) == 3  # two survivors + the new one

    def test_malformed_fire_at_is_kept_not_crashed(self):
        f = _seed([{"id": "r_bad", "text": "x", "status": "fired",
                    "fire_at": "not-a-date"}])
        _mod.execute("reminder_set", "remind me in 5 minutes to stretch", {})
        ids = {i["id"] for i in json.loads(f.read_text())}
        assert "r_bad" in ids
