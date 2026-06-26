"""clock_query over-claim bug for issue #28 bug 60.

Live probe::

  user: "what time is sunset" → Time card ("It is 2:47 AM.")
  user: "what time is the meeting" → would also hit Time card

Both queries are about *named* events, not the wall clock, but the
``clock_query`` regex started with the loose alternative::

    what(?:'s| is)?\\s+(?:the\\s+)?time\\b

…which matches "what time" as long as it's followed by a word
boundary. So "what time IS sunset" hits on the "time" suffix.

Surgical fix: anchor the bare-time alternative so the message must
*end* after "time" (with optional punctuation). "what time is it" /
"what time do you have" are already covered by a separate, tighter
alternative — that one stays untouched.
"""
from __future__ import annotations

from prism_intents import INTENTS
from prism_routing import route_intent


def _route(message: str) -> str:
    return route_intent(message, INTENTS, lambda _m: None)


class TestClockQueryFalsePositives:
    """The headline bug — these used to wrongly hit clock_query."""

    def test_what_time_is_sunset_is_not_clock(self):
        assert _route("what time is sunset") != "clock_query"

    def test_what_time_is_sunrise_is_not_clock(self):
        assert _route("what time is sunrise") != "clock_query"

    def test_what_time_is_the_meeting_is_not_clock(self):
        assert _route("what time is the meeting") != "clock_query"

    def test_what_time_is_the_concert_is_not_clock(self):
        assert _route("what time is the concert") != "clock_query"

    def test_what_time_is_dinner_is_not_clock(self):
        assert _route("what time is dinner") != "clock_query"


class TestClockQueryStillClaimsGenuineTimeQueries:
    """No regression on the queries clock_query SHOULD claim."""

    def test_what_time_is_it(self):
        assert _route("what time is it") == "clock_query"

    def test_what_time_is_it_punct(self):
        assert _route("what time is it?") == "clock_query"

    def test_what_time(self):
        assert _route("what time") == "clock_query"

    def test_whats_the_time(self):
        assert _route("what's the time") == "clock_query"

    def test_what_is_the_time(self):
        assert _route("what is the time") == "clock_query"

    def test_whats_the_time_question(self):
        assert _route("what's the time?") == "clock_query"

    def test_current_time(self):
        assert _route("current time") == "clock_query"

    def test_time_now(self):
        assert _route("time now") == "clock_query"

    def test_what_day_is_it(self):
        assert _route("what day is it") == "clock_query"

    def test_what_is_the_date(self):
        assert _route("what is the date") == "clock_query"

    def test_what_time_do_you_have(self):
        assert _route("what time do you have") == "clock_query"
