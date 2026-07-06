"""
tests/test_route_memo_issue_28.py
=================================
chat() routes the same message twice per turn — once in the tier
dispatcher, once to label the turn for the crystalliser. For a message
the regex table doesn't match, each _route() fell through to the LLM
classifier: ~1.1k prompt tokens for a 24-token label, TWICE per chat
message (observed live in the DeepSeek ledger as identical 1122-in/3-out
calls bracketing every real generation).

_route() now memoises the last (message, intent) pair, so the classifier
runs at most once per turn — and both consumers see the same answer even
when the classifier is nondeterministic.
"""
from __future__ import annotations

from prism_agent import PrismAgent


def test_classifier_called_at_most_once_per_chat_turn(offline_llm):
    agent = PrismAgent()
    calls: list[str] = []

    def counting_classifier(message: str):
        calls.append(message)
        return "general_chat"

    agent._llm_classify = counting_classifier
    # A message no regex intent matches — must fall through to the
    # classifier, but only once for the whole turn.
    agent.chat("qqq zzz florble wibble")
    assert len(calls) == 1, (
        f"classifier ran {len(calls)}x for one chat turn "
        f"(want exactly 1 — 0 means this test no longer exercises the "
        f"fall-through, 2+ means the double-routing is back): {calls}"
    )


def test_route_memo_returns_consistent_intent(offline_llm):
    agent = PrismAgent()
    answers = iter(["general_chat", "device_task"])  # nondeterministic stub
    agent._llm_classify = lambda m: next(answers)
    first = agent._route("qqq zzz florble wibble")
    second = agent._route("qqq zzz florble wibble")
    assert first == second == "general_chat"


def test_route_memo_does_not_leak_across_messages(offline_llm):
    agent = PrismAgent()
    assert agent._route("what's the weather in Berlin") == "weather_check"
    assert agent._route("remind me in 5 minutes to stretch") != "weather_check"
