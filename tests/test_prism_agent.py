from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from prism_agent import PrismAgent
from prism_responses import PrismCard


def _make_agent() -> PrismAgent:
    kde_agent = MagicMock()
    kde_agent.morning_briefing.return_value = SimpleNamespace(
        plan=SimpleNamespace(
            primary_focus="Recovery",
            activation=0.5,
            tasks=[],
            warnings=[],
            rationale="Recovery first",
        )
    )
    kde_agent.reflect.return_value = {
        "profile": "Tester",
        "fixed_fulcrum": 0.5,
        "fulcrum_trend": "balanced",
        "total_ratings": 3,
        "total_plans": 2,
    }
    kde_agent.ask.return_value = SimpleNamespace(output={"ok": True})
    return PrismAgent(kde_agent=kde_agent, ksa_agent=None)


def test_route_plan_intent():
    assert _make_agent()._route("plan my day") == "plan"


def test_route_medical():
    assert _make_agent()._route("triage chest pain elderly") == "domain_medical"


def test_route_identity():
    assert _make_agent()._route("my identity profile") == "identity"


def test_chat_never_raises():
    agent = _make_agent()
    for message in ["", "???", "random string", "run something maybe"]:
        card = agent.chat(message)
        assert isinstance(card, PrismCard)


def test_chat_returns_card():
    card = _make_agent().chat("plan my day")
    assert isinstance(card, PrismCard)
