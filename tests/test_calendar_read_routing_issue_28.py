"""calendar_read routing fix for issue #28 bug 61.

Live probes::

  user: "any meetings today"          → universal_plan (hung)
  user: "any meetings"                → calendar setup card (general_chat)
  user: "what meetings do I have"     → general_chat
  user: "do I have meetings today"    → universal_plan
  user: "meetings today"              → universal_plan

A ``calendar_read`` intent already exists at line ~243 of
``prism_intents`` but it sits *after* ``universal_plan``, which claims
anything containing "today". Same trap that already required hoisting
``list_tasks`` and ``budget_status`` above the planner.

Fix:

1. Add a hoisted, widened ``calendar_read`` rule above
   ``universal_plan`` covering the natural phrasings the user actually
   types.
2. The original calendar_read rule below stays in place for the
   "what's on my calendar / show me my agenda" verb-noun pattern.
"""
from __future__ import annotations

from prism_intents import INTENTS
from prism_routing import route_intent


def _route(message: str) -> str:
    return route_intent(message, INTENTS, lambda _m: None)


class TestMeetingsTodayPhrases:
    """All four variants the user typed must reach calendar_read."""

    def test_any_meetings_today(self):
        assert _route("any meetings today") == "calendar_read"

    def test_do_i_have_meetings_today(self):
        assert _route("do I have meetings today") == "calendar_read"

    def test_meetings_today(self):
        assert _route("meetings today") == "calendar_read"

    def test_what_meetings_do_i_have(self):
        assert _route("what meetings do I have") == "calendar_read"


class TestAppointmentEventPhrases:

    def test_any_appointments_today(self):
        assert _route("any appointments today") == "calendar_read"

    def test_events_tomorrow(self):
        assert _route("events tomorrow") == "calendar_read"

    def test_whats_on_my_calendar(self):
        assert _route("what's on my calendar") == "calendar_read"


class TestNoUniversalPlanRegression:

    def test_plan_my_day_still_plans(self):
        assert _route("plan my day") == "universal_plan"

    def test_good_morning_still_plans(self):
        assert _route("good morning") == "universal_plan"

    def test_what_should_i_do_today_still_plans(self):
        # No meeting/calendar keyword — planner keeps it.
        assert _route("what should I do today") == "universal_plan"
