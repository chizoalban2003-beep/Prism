from __future__ import annotations

from prism_agent import PrismAgent
from prism_responses import PrismCard


def test_route_plan():
    assert PrismAgent()._route("plan my day") == "universal_plan"


def test_route_medical():
    assert PrismAgent()._route("triage chest pain") == "domain_medical"


def test_route_weight_units_not_currency():
    # "pounds" is a currency word, but a metric/imperial unit on the other side
    # ("kg") means this is a weight conversion, not GBP.
    agent = PrismAgent()
    assert agent._route("convert 10 kg to pounds") == "unit_convert"
    assert agent._route("how many miles in 10 km") == "unit_convert"


def test_route_currency_still_works():
    agent = PrismAgent()
    assert agent._route("convert 100 usd to gbp") == "currency_convert"
    assert agent._route("exchange 50 dollars to euros") == "currency_convert"


def test_chat_never_raises(offline_llm):
    # offline_llm: these messages route to LLM-touching paths. In CI no
    # Ollama exists so the calls fail fast, but on a dev machine with a
    # slow local model each message is a real multi-minute generation —
    # this test timed out at 300s locally while "passing" in CI.
    for message in ["random", "???", "what is this", "search code"]:
        assert isinstance(PrismAgent().chat(message), PrismCard)


def test_chat_returns_card(offline_llm):
    assert isinstance(PrismAgent().chat("plan my day"), PrismCard)
