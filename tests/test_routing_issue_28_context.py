"""Routing fix for issue #28 bug 6 — "current context" was hitting memory_recall.

Live-test surfaced that asking "what's my current context?" routed to
memory_recall, which then dumped historical chat assistant outputs from
PrismMemory back as if they were facts about the user. The user expected
a perception snapshot — what PRISM is sensing right now.

The fix has two pieces, both pinned here:

  1. A dedicated `current_context` intent placed before `memory_recall` so
     the catch-all "what's my X" frame can't steal the query.
  2. `context` added to the memory_recall negative lookahead as defence-in-
     depth — even if the new pattern is changed/removed, memory_recall
     won't grab "what's my context" by mistake.
"""
from __future__ import annotations

from prism_intents import INTENTS
from prism_routing import route_intent


def _route(message: str) -> str:
    return route_intent(message, INTENTS, lambda _m: None)


class TestCurrentContextIntent:
    def test_whats_my_current_context(self):
        assert _route("What's my current context?") == "current_context"

    def test_what_is_my_current_context(self):
        assert _route("what is my current context") == "current_context"

    def test_show_me_my_context(self):
        assert _route("show me my context") == "current_context"

    def test_describe_my_current_context(self):
        assert _route("describe my current context") == "current_context"

    def test_my_context_alone(self):
        assert _route("my context") == "current_context"

    def test_context_right_now(self):
        assert _route("context right now") == "current_context"


class TestMemoryRecallStillWorks:
    """The pre-existing memory_recall behaviour must not regress."""

    def test_favourite_colour_still_memory_recall(self):
        assert _route("what is my favourite colour") == "memory_recall"

    def test_partner_name_still_memory_recall(self):
        assert _route("what's my partner's name") == "memory_recall"


class TestContextLookaheadDefence:
    """Even with `current_context` removed, memory_recall must not grab
    "context" queries — the negative lookahead is the safety net."""

    def test_context_excluded_from_memory_recall(self):
        # Strip the current_context entry and confirm memory_recall doesn't
        # take the slack. The query falls through to a later intent or to
        # general_chat — anything but memory_recall is acceptable here.
        filtered = [(p, n) for p, n in INTENTS if n != "current_context"]
        result = route_intent("what's my context", filtered, lambda _m: None)
        assert result != "memory_recall"
