"""clock_query intent + organ for issue #28 bug 19.

Live test: ``what time is it`` returned a "Build new organ?" approval
card — PRISM had no time-of-day capability. The fix adds a clock_query
intent above the LLM classifier and a clock_query organ that uses
Python's stdlib datetime.
"""
from __future__ import annotations

import importlib.util
from datetime import datetime
from unittest.mock import patch

from prism_intents import INTENTS
from prism_routing import route_intent


def _route(message: str) -> str:
    return route_intent(message, INTENTS, lambda _m: None)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, f"organs/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestRoutingClockQuery:
    def test_what_time_is_it(self):
        assert _route("what time is it") == "clock_query"

    def test_whats_the_time(self):
        assert _route("what's the time") == "clock_query"

    def test_current_time(self):
        assert _route("current time") == "clock_query"

    def test_whats_todays_date(self):
        assert _route("what's today's date") == "clock_query"

    def test_what_day_is_it(self):
        assert _route("what day is it") == "clock_query"

    def test_what_day_is_today(self):
        assert _route("what day is today") == "clock_query"


class TestNoOverreach:
    """Make sure clock_query doesn't grab timer / scheduling phrases."""

    def test_set_a_timer(self):
        assert _route("set a timer for 5 minutes") == "timer_set"

    def test_schedule_a_meeting(self):
        assert _route("schedule a meeting tomorrow at 3pm") != "clock_query"

    def test_what_is_the_eiffel_tower(self):
        assert _route("what is the eiffel tower") == "wikipedia_lookup"


class TestClockOrgan:
    organ = _load("clock_query")
    _fixed = datetime(2026, 6, 24, 14, 30, 0)

    def _call(self, message: str):
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = self._fixed
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            return self.organ.execute("clock_query", message, {})

    def test_time_query_mentions_time(self):
        card = self._call("what time is it")
        assert "2:30 PM" in card.body
        assert card.card_data["weekday"] == "Wednesday"

    def test_date_query_mentions_date(self):
        card = self._call("what's today's date")
        assert "June" in card.body and "2026" in card.body

    def test_combined_mentions_both(self):
        card = self._call("what's the date and time")
        assert "2:30 PM" in card.body and "June" in card.body

    def test_card_data_contains_iso(self):
        card = self._call("what time is it")
        assert card.card_data["iso"].startswith("2026-06-24")
