"""Routing fixes for issue #26 — memory recall + reasoning-question fallback.

Bug 1: "what is my favourite colour?" was hitting wikipedia_lookup because
no memory_recall intent existed and the wikipedia catch-all matched the
"what is" frame. We need a dedicated memory_recall intent that wins for
"what is/are my X" without stealing the existing "my_profile"/"my_growth"/
etc. routes.

Bug 2: The LLM classifier blindly trusted its output, so messages like
"explain database deadlocks" got routed to devices_list because the word
"deadlocks" overlapped a tool-intent keyword. We pre-filter reasoning
questions and force them to fall through to general_chat.
"""
from __future__ import annotations

from prism_intents import INTENTS
from prism_routing import _is_reasoning_question, route_intent


def _route(message: str, *, llm_label: str | None = None) -> str:
    """Run the real first-match sweep over the production INTENTS table.

    `llm_label` simulates what the classifier LLM would return if the regex
    sweep falls through — used to verify pre-filter behaviour.
    """
    def fallback(_msg: str) -> str | None:
        return llm_label
    return route_intent(message, INTENTS, fallback)


# ── memory_recall intent ─────────────────────────────────────────────────

class TestMemoryRecallIntent:
    def test_what_is_my_favourite_colour(self):
        assert _route("what is my favourite colour?") == "memory_recall"

    def test_whats_my_partner_name(self):
        assert _route("what's my partner's name") == "memory_recall"

    def test_do_you_remember_my_birthday(self):
        assert _route("do you remember my birthday") == "memory_recall"

    def test_recall_my_address(self):
        assert _route("recall my address") == "memory_recall"

    # Negative cases — these must keep their specific routes, not steal them.

    def test_my_profile_still_wins(self):
        assert _route("what do you know about me") == "my_profile"

    def test_my_growth_still_wins(self):
        assert _route("how much have you learned about me") == "my_growth"

    def test_calendar_read_still_wins(self):
        # "today" gets eaten by universal_plan upstream — drop it so we
        # exercise just the calendar phrasing.
        assert _route("what's on my calendar") == "calendar_read"

    def test_email_read_still_wins(self):
        assert _route("check my inbox") == "email_read"

    def test_show_policies_still_wins(self):
        # show_policies has its own "what's my budget/policy/limit" route
        # earlier in the table — memory_recall must not steal it.
        assert _route("what's my budget") == "show_policies"


# ── word-boundary regression: smart_home "lock" must not eat "deadlock" ──
# Discovered during live-test of v0.2.3: "explain database deadlocks" was
# routing to smart_home because the alternation `lock|unlock` had no word
# boundary, so any substring containing "lock" matched. The reasoning
# pre-filter only sits in front of the LLM classifier, so a regex match
# bypassed it. Fix is `\b(?:un)?lock\b` — covered here so it doesn't
# silently regress.

class TestSmartHomeWordBoundary:
    def test_explain_deadlocks_does_not_route_to_smart_home(self):
        assert _route("explain database deadlocks") != "smart_home"

    def test_blocking_does_not_route_to_smart_home(self):
        # "blocking" contains "lock" too — make sure the boundary holds.
        assert _route("what's blocking the deadlock") != "smart_home"

    def test_lock_the_door_still_routes_to_smart_home(self):
        assert _route("lock the front door") == "smart_home"

    def test_unlock_my_phone_still_routes_to_smart_home(self):
        assert _route("unlock my phone") == "smart_home"


# ── reasoning-question pre-filter ────────────────────────────────────────

class TestReasoningPreFilter:
    def test_explain_is_reasoning(self):
        assert _is_reasoning_question("explain database deadlocks")

    def test_why_does_is_reasoning(self):
        assert _is_reasoning_question("why does TCP slow start exist")

    def test_how_does_is_reasoning(self):
        assert _is_reasoning_question("how does TLS handshake work")

    def test_name_three_is_reasoning(self):
        assert _is_reasoning_question("name three sorting algorithms")

    def test_what_is_a_X_is_reasoning(self):
        assert _is_reasoning_question("what is a Merkle tree")

    def test_tell_me_about_is_reasoning(self):
        assert _is_reasoning_question("tell me about the Roman Empire")

    # Negative cases — these must NOT pre-filter to chat, because they
    # legitimately want a tool.

    def test_my_question_is_not_reasoning(self):
        # "what is my X" is a recall question, not an explanation question.
        # The pre-filter explicitly excludes the "my X" form so memory_recall
        # routing isn't suppressed.
        assert not _is_reasoning_question("what is my favourite colour")

    def test_command_is_not_reasoning(self):
        assert not _is_reasoning_question("send a push notification")

    def test_calendar_query_is_not_reasoning(self):
        assert not _is_reasoning_question("what's on my calendar")


# ── classifier integration: reasoning questions never reach a tool intent

def test_reasoning_question_falls_through_to_chat():
    """End-to-end: a classifier that mis-fires on the word 'deadlocks' must
    have its output discarded when the pre-filter recognises the message as
    a reasoning question. The intent table has no 'deadlocks' regex, so the
    only way to reach 'devices_list' was via the LLM — that path is now
    blocked, so we land on general_chat."""
    from prism_routing import LLMClassifier

    intents = [("^never_matches_anything_xyz$", "devices_list")]

    classifier = LLMClassifier(
        intents=intents,
        router=None,
        ollama_host="http://127.0.0.1:0",  # unreachable on purpose
        text_model="missing",
        get_organ_intents=lambda: {},
    )

    # The reasoning pre-filter must fire before any network call, so this
    # is fast and deterministic even with no LLM reachable.
    assert classifier.classify("explain database deadlocks") is None

    # And the whole route falls through to general_chat (the route_intent
    # default when llm_fallback returns None).
    routed = route_intent("explain database deadlocks", intents,
                          classifier.classify)
    assert routed == "general_chat"
