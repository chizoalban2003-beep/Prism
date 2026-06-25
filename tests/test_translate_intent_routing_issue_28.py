"""Translate intent-routing fix for issue #28 bug 32.

Live test: ``translate good morning into french`` returned a
``plan`` card with "Planner LLM unavailable — model 'tinyllama' timed
out". The translate_text organ itself handles ``into`` fine — the bug
was in ``prism_intents.INTENTS`` ordering. ``universal_plan`` claims
any message containing "morning"/"today"/"plan"/etc., guarded by a
negative lookahead that excluded only ``\\bto (lang)\\b``. "good
morning **into** french" slipped past, "morning" matched, and the
planner stub fired instead of routing to translation.

Fix: extend the lookahead to ``(?:in)?to`` (so "into french" is also
excluded) and add a second lookahead that bails on any literal
"translate" verb regardless of preposition.
"""
from __future__ import annotations

import re

from prism_intents import INTENTS


def _route(message: str) -> str:
    for pattern, intent in INTENTS:
        if re.search(pattern, message, re.IGNORECASE):
            return intent
    return "_NO_MATCH_"


class TestTranslateInto:
    def test_translate_into_french(self):
        # The reported bug.
        assert _route("translate good morning into french") == "translate_text"

    def test_translate_into_spanish(self):
        assert _route("translate hello into spanish") == "translate_text"

    def test_translate_into_japanese(self):
        assert _route("translate this sentence into japanese") == "translate_text"

    def test_translate_morning_into_german(self):
        # "morning" alone would hit universal_plan; the translate guard
        # must override.
        assert _route("translate morning into german") == "translate_text"

    def test_translate_to_still_works(self):
        # Don't regress the original "to" shape.
        assert _route("translate hello to spanish") == "translate_text"

    def test_translate_to_french(self):
        assert _route("translate good morning to french") == "translate_text"


class TestPlanStillRoutes:
    """Make sure widening the negative lookahead didn't kill the planner."""

    def test_plain_morning(self):
        assert _route("good morning") == "universal_plan"

    def test_plan_my_day(self):
        assert _route("plan my day") == "universal_plan"

    def test_daily_schedule(self):
        assert _route("daily schedule") == "universal_plan"

    def test_morning_routine(self):
        assert _route("my morning routine") == "universal_plan"
