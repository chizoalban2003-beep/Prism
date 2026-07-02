"""
tests/test_smalltalk_routing_issue_28.py
========================================
Companion small-talk routing for issue #28: greetings and emotional
check-ins must reach general_chat, not the structured planner.

Live probe before the fix: "good morning" and "i feel stressed today"
both matched universal_plan's bare keywords (morning/today) and
dead-ended in a "Planner LLM unavailable" error card.
"""
from __future__ import annotations

import pytest

from prism_intents import INTENTS
from prism_routing import route_intent


def _route(message: str) -> str:
    # llm_fallback must not be consulted for these deterministic cases.
    return route_intent(message, INTENTS, lambda m: "LLM_FALLBACK_USED")


class TestGreetingsRouteToChat:
    @pytest.mark.parametrize("msg", [
        "good morning",
        "Good morning!",
        "good morning prism",
        "good evening",
        "good night",
        "hello",
        "hi",
        "hey there",
        "how are you",
        "how's it going",
        "how are you doing today",
        "what's up",
        "thank you",
        "thanks so much",
        "thanks!",
        "well done",
        "bye for now",
        "see you tomorrow",
    ])
    def test_routes_to_general_chat(self, msg):
        assert _route(msg) == "general_chat", msg


class TestFeelingsRouteToChat:
    @pytest.mark.parametrize("msg", [
        "i feel stressed today",
        "I'm feeling overwhelmed",
        "i am feeling a bit anxious",
        "feeling really tired",
        "i feel burnt out",
        "i'm feeling great",
    ])
    def test_routes_to_general_chat(self, msg):
        assert _route(msg) == "general_chat", msg


class TestPlannerStillClaimsPlans:
    @pytest.mark.parametrize("msg", [
        "plan my morning",
        "plan my day",
        "good morning, plan my day",   # greeting + request = request
        "what should i do today",
        "plan for today",
        "schedule my afternoon",
    ])
    def test_routes_to_universal_plan(self, msg):
        assert _route(msg) == "universal_plan", msg


class TestNoCollateralReroutes:
    def test_weather_still_wins(self):
        assert _route("hi there, what's the weather today") == "weather_check"

    def test_horizon_still_wins(self):
        assert _route("tell me when bitcoin drops below 40k") == "horizon_add"

    def test_news_still_wins(self):
        assert _route("today's headlines") == "news_headlines"

    def test_clock_still_wins(self):
        assert _route("what time is it") == "clock_query"

    def test_feel_like_doing_is_not_emotion(self):
        # "feel like <verb>ing" is an idiom, not an emotional check-in;
        # it must not be claimed by the feelings pattern.
        assert _route("i feel like ordering pizza") != "general_chat"
