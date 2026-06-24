"""Memory-recall verb-gap fix for issue #28 bug 8 — "who is my X" / "when is my X" hit web search.

Live test: storing "remember that my partner is Sarah" and then asking
"who is my partner" routed to ``web_search`` (returned a film result),
because the memory_recall regex only listed ``what is/are``, ``tell me``,
``do you know/remember``, ``recall``. ``who is`` and ``when is`` slipped
through to the generic search catch-all.

Fix: extend the verb alternation in the memory_recall pattern to also
accept ``who is/are``, ``when is/was``, ``where is/was`` — the negative
lookahead after ``my`` still protects all the dedicated my_X routes
(profile, narrative, calendar, inbox, etc).
"""
from __future__ import annotations

from prism_intents import INTENTS
from prism_routing import route_intent


def _route(message: str) -> str:
    return route_intent(message, INTENTS, lambda _m: None)


class TestPersonalFactVerbCoverage:
    def test_who_is_my_partner(self):
        assert _route("who is my partner") == "memory_recall"

    def test_whos_my_partner(self):
        assert _route("who's my partner") == "memory_recall"

    def test_when_is_my_birthday(self):
        assert _route("when is my birthday") == "memory_recall"

    def test_whens_my_birthday(self):
        assert _route("when's my birthday") == "memory_recall"

    def test_where_is_my_office(self):
        assert _route("where is my office") == "memory_recall"


class TestNoRegressionOnExistingPaths:
    def test_what_is_my_favourite_colour_still_recall(self):
        assert _route("what is my favourite colour") == "memory_recall"

    def test_whats_my_partners_name_still_recall(self):
        assert _route("what's my partner's name") == "memory_recall"


class TestDedicatedRoutesStillWin:
    """The negative lookahead in memory_recall protects dedicated my_X
    routes — extending the verb set must not break that."""

    def test_when_is_my_meeting_excluded_by_lookahead(self):
        # "meetings?" is in the negative lookahead — so memory_recall declines
        # the match and the query falls through to a later intent.
        assert _route("when is my meeting") != "memory_recall"

    def test_who_is_my_calendar_excluded_by_lookahead(self):
        # "calendar" is in the negative lookahead — extending the verb set
        # must not weaken that protection.
        assert _route("who is my calendar") != "memory_recall"

    def test_what_is_my_inbox_excluded_by_lookahead(self):
        # Sanity: the pre-fix behaviour is preserved.
        assert _route("what is my inbox") != "memory_recall"


class TestGenericWhoIsStillWebSearch:
    """Generic ``who is X`` queries without ``my`` must still go to web search."""

    def test_who_is_the_prime_minister(self):
        # No "my" → memory_recall pattern doesn't match → falls through.
        assert _route("who is the prime minister") == "web_search"

    def test_when_did_the_war_end(self):
        assert _route("when did the war end") == "web_search"
