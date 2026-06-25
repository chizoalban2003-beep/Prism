"""reminder_set text-extraction fixes for issue #28 bug 24.

Live tests revealed two bugs in ``_parse_reminder``:

1. **Order-dependent extraction**: ``remind me in 30 minutes to call
   mom`` stored ``"in 30 minutes to call mom"`` as the reminder
   text. The original regex captured everything between
   ``remind me`` and end-of-string, so when the time phrase came
   *before* the action, it leaked into the body.

2. **Bare ``Xpm/am`` not parsed**: ``set a reminder for 3pm to take
   medication`` rejected the input with "Could not parse a time".
   The absolute-time regex required an ``at`` prefix.

Fix: parse durations / absolute times first, strip them from the
message, then extract the action phrase from what remains. The
``at`` prefix is now optional.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_reminder_set_organ",
    Path(__file__).resolve().parent.parent / "organs" / "reminder_set.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_parse_reminder = _mod._parse_reminder


class TestTextOrderIndependence:
    def test_time_after_action(self):
        text, delay = _parse_reminder("remind me to buy milk in 2 hours")
        assert text == "buy milk"
        assert delay == 7200

    def test_time_before_action(self):
        text, delay = _parse_reminder("remind me in 30 minutes to call mom")
        assert text == "call mom"
        assert delay == 1800

    def test_time_before_action_hours(self):
        text, delay = _parse_reminder("remind me in 2 hours to take a walk")
        assert text == "take a walk"
        assert delay == 7200

    def test_reminder_colon_form(self):
        text, delay = _parse_reminder("reminder: meeting in 1 hour")
        assert text == "meeting"
        assert delay == 3600


class TestAbsoluteTimeWithoutAt:
    def test_bare_3pm(self):
        text, delay = _parse_reminder("set a reminder for 3pm to take medication")
        assert text == "take medication"
        assert delay > 0

    def test_bare_9am(self):
        # 9am — either today or tomorrow depending on now; just confirm parsed.
        text, delay = _parse_reminder("remind me at 9am to stand up")
        assert text == "stand up"
        assert delay > 0


class TestAlarmAndWake:
    """Alarms and wake-ups are reminders with a time; previously
    "set an alarm for 7am" routed to organ synthesis."""

    def test_set_an_alarm_for_7am(self):
        text, delay = _parse_reminder("set an alarm for 7am")
        assert text == "wake up"
        assert delay > 0

    def test_wake_me_up_at_7am(self):
        text, delay = _parse_reminder("wake me up at 7am")
        assert text == "wake up"
        assert delay > 0

    def test_wake_me_at_7am_with_action(self):
        text, delay = _parse_reminder("wake me at 6am to go running")
        assert text == "go running"
        assert delay > 0


class TestBackwardCompat:
    """The cases the original code already handled must keep working."""

    def test_at_hhmm(self):
        text, delay = _parse_reminder("remind me at 14:30 to take medicine")
        assert text == "take medicine"
        assert delay > 0

    def test_in_minutes(self):
        text, delay = _parse_reminder("remind me in 45 minutes to stretch")
        assert text == "stretch"
        assert delay == 2700

    def test_no_time_returns_zero_delay(self):
        # No time/duration — caller surfaces the "could not parse" error.
        _, delay = _parse_reminder("remind me to do something")
        assert delay == 0
