"""calendar_write + calendar_read agenda routing for issue #28 bug 62.

Probes (post-#28-61 calendar_read hoist):

  user: "schedule a meeting"          → universal_plan ("schedule")
  user: "schedule meeting with bob"   → universal_plan
  user: "book an appointment"         → browser_task ("book")
  user: "create an event"             → general_chat (no match)
  user: "set up a meeting"            → general_chat
  user: "find a free time"            → general_chat
  user: "show my agenda"              → general_chat (regex too tight)
  user: "show my schedule"            → universal_plan ("schedule")
  user: "check my schedule"           → universal_plan

calendar_write phrases are all stolen by intents above it.
"show my agenda" gets stolen by the bare "show" + noun pattern: the
existing calendar_read regex `(?:what's on my|check my|show) (?:cal-
endar|schedule|agenda)` doesn't allow `my` between the verb and noun.

Fix:

1. Hoist a widened calendar_write rule above universal_plan / browser_task
   to claim the natural verbs (schedule/book/create/add/set up/find).
2. Extend the calendar_read hoist from #28-61 to also claim
   "show/check my agenda" and "show/check my schedule".
"""
from __future__ import annotations

from prism_intents import INTENTS
from prism_routing import route_intent


def _route(message: str) -> str:
    return route_intent(message, INTENTS, lambda _m: None)


class TestCalendarWriteSchedule:

    def test_schedule_a_meeting(self):
        assert _route("schedule a meeting") == "calendar_write"

    def test_schedule_meeting_with_bob(self):
        assert _route("schedule meeting with bob") == "calendar_write"

    def test_schedule_the_appointment(self):
        assert _route("schedule the appointment") == "calendar_write"


class TestCalendarWriteBook:

    def test_book_an_appointment(self):
        assert _route("book an appointment") == "calendar_write"

    def test_book_a_meeting(self):
        assert _route("book a meeting") == "calendar_write"

    def test_book_an_event(self):
        assert _route("book an event") == "calendar_write"


class TestCalendarWriteCreate:

    def test_create_an_event(self):
        assert _route("create an event") == "calendar_write"

    def test_add_a_meeting(self):
        assert _route("add a meeting") == "calendar_write"

    def test_set_up_a_meeting(self):
        assert _route("set up a meeting") == "calendar_write"


class TestCalendarWriteFreeSlot:

    def test_find_a_free_time(self):
        assert _route("find a free time") == "calendar_write"

    def test_when_is_the_next_free_slot(self):
        assert _route("when is the next free slot") == "calendar_write"


class TestCalendarReadAgendaSchedule:

    def test_show_my_agenda(self):
        assert _route("show my agenda") == "calendar_read"

    def test_check_my_agenda(self):
        assert _route("check my agenda") == "calendar_read"

    def test_show_my_schedule(self):
        assert _route("show my schedule") == "calendar_read"

    def test_check_my_schedule(self):
        assert _route("check my schedule") == "calendar_read"


class TestNoRegression:
    """Existing intents must not lose ground."""

    def test_plan_my_day(self):
        assert _route("plan my day") == "universal_plan"

    def test_good_morning(self):
        # Changed in #28-79: bare greetings route to general_chat.
        assert _route("good morning") == "general_chat"

    def test_book_a_table(self):
        # "book" + non-calendar noun → browser_task still applicable.
        assert _route("book a table at the restaurant") == "browser_task"

    def test_open_browser(self):
        assert _route("open chrome") == "browser_task"

    def test_meetings_today_still_read(self):
        # The #28-61 hoist must keep claiming this.
        assert _route("any meetings today") == "calendar_read"

    def test_whats_on_my_calendar_still_read(self):
        assert _route("what's on my calendar") == "calendar_read"
