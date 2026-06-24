"""clock_query refuse-misclassified-queries guard for issue #28 bug 25.

Live tests revealed clock_query returns the local time for queries
it can't actually serve:

  "how many days until christmas"  → "It is 11:13 PM" (wrong)
  "what time is it in tokyo"       → "It is 11:12 PM" (ignores tz)

These hit clock_query via the LLMClassifier fallback (the regex
doesn't match them), and the organ blindly answers with local time
data. We harden the organ to detect three out-of-scope categories
and return an explicit fallback card:

  - countdowns ("until", "till X")
  - elapsed ("since", "ago")
  - foreign timezones ("in tokyo", "in NYC time")

The card body must mention the limitation so the user (or a future
re-route) knows to search instead.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_clock_query_organ",
    Path(__file__).resolve().parent.parent / "organs" / "clock_query.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
execute = _mod.execute


class TestStillAnswersLegitTimeQueries:
    def test_what_time_is_it(self):
        card = execute("clock_query", "what time is it", {})
        assert "It is" in card.body
        # Should NOT have refused.
        assert "can't" not in card.body.lower()
        assert "cannot" not in card.body.lower()

    def test_whats_the_date(self):
        card = execute("clock_query", "what's the date today", {})
        assert "Today is" in card.body or "It is" in card.body

    def test_what_day_is_it(self):
        card = execute("clock_query", "what day is it", {})
        # "Today is …" lists weekday.
        body = card.body.lower()
        assert "today" in body or "it is" in body


class TestDeclinesCountdowns:
    def test_days_until_christmas(self):
        card = execute("clock_query", "how many days until christmas", {})
        body = card.body.lower()
        assert "current" in body or "only" in body or "can't" in body or "cannot" in body
        # Must NOT pretend it answered a countdown.
        assert "until christmas" not in card.body.lower() or "can" in body

    def test_hours_until_midnight(self):
        card = execute("clock_query", "how many hours till midnight", {})
        assert ("can't" in card.body.lower() or "only" in card.body.lower()
                or "cannot" in card.body.lower())


class TestDeclinesElapsed:
    def test_time_since_event(self):
        card = execute("clock_query", "how long since the meeting started", {})
        assert ("can't" in card.body.lower() or "only" in card.body.lower()
                or "cannot" in card.body.lower())


class TestDeclinesForeignTimezone:
    def test_time_in_tokyo(self):
        card = execute("clock_query", "what time is it in tokyo", {})
        body = card.body.lower()
        assert ("tokyo" in body or "timezone" in body or "only" in body
                or "local" in body), card.body

    def test_time_in_paris(self):
        card = execute("clock_query", "what time is it in paris", {})
        body = card.body.lower()
        assert ("paris" in body or "timezone" in body or "only" in body
                or "local" in body), card.body

    def test_time_in_new_york(self):
        card = execute("clock_query", "what is the time in new york", {})
        body = card.body.lower()
        # Should NOT just give local time silently.
        assert ("new york" in body or "timezone" in body or "only" in body
                or "local" in body), card.body
