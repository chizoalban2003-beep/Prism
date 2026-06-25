"""clock_query date-phrasing fix for issue #28 bug 41.

Live test: ``what is the date today`` and ``what date is it`` fell
through to the LLM classifier — 30s+ hangs on a busy daemon — even
though the clock_query organ trivially returns today's date.

The original regex covered ``what is today's date`` (apostrophe-s
attached to today) but not the date-first phrasing ``what is the
date today``.

Fix: extend the alternation with:
* ``what (is) the date [today]``
* ``what date is (it|today)``
* ``current date``
* ``today's date`` (standalone)
"""
from __future__ import annotations

import re

from prism_intents import INTENTS


def _route(message: str) -> str:
    for pattern, intent in INTENTS:
        if re.search(pattern, message, re.IGNORECASE):
            return intent
    return "_NO_MATCH_"


class TestDatePhrasings:
    def test_what_is_the_date_today(self):
        # The reported bug.
        assert _route("what is the date today") == "clock_query"

    def test_what_date_is_it(self):
        # The reported bug.
        assert _route("what date is it") == "clock_query"

    def test_what_is_the_date(self):
        assert _route("what is the date") == "clock_query"

    def test_whats_the_date(self):
        assert _route("what's the date") == "clock_query"

    def test_whats_the_date_today(self):
        assert _route("what's the date today") == "clock_query"

    def test_what_date_is_today(self):
        assert _route("what date is today") == "clock_query"

    def test_current_date(self):
        assert _route("current date") == "clock_query"

    def test_todays_date(self):
        assert _route("today's date") == "clock_query"


class TestExistingPhrasingsStillWork:
    def test_what_is_todays_date(self):
        assert _route("what is today's date") == "clock_query"

    def test_what_time_is_it(self):
        assert _route("what time is it") == "clock_query"

    def test_whats_the_time(self):
        assert _route("what's the time") == "clock_query"

    def test_what_day_is_it(self):
        assert _route("what day is it") == "clock_query"

    def test_what_day_is_today(self):
        assert _route("what day is today") == "clock_query"

    def test_current_time(self):
        assert _route("current time") == "clock_query"
