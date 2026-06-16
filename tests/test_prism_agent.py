from __future__ import annotations

import pytest

from prism_agent import PrismAgent
from prism_responses import PrismCard


def test_route_plan():
    assert PrismAgent()._route("plan my day") == "plan"


def test_route_medical():
    assert PrismAgent()._route("triage chest pain") == "domain_medical"


def test_chat_never_raises():
    for message in ["random", "???", "what is this", "search code"]:
        assert isinstance(PrismAgent().chat(message), PrismCard)


@pytest.mark.slow
@pytest.mark.timeout(120)
def test_chat_returns_card():
    assert isinstance(PrismAgent().chat("plan my day"), PrismCard)
