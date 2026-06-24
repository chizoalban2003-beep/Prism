"""Horizon list/add precedence fix for issue #28 bug 13 — "list horizon goals" registered a goal.

Live test: ``list my horizon goals`` returned ``Horizon goal registered:
'user asks to list horizon goals'``. The horizon_add pattern included
``horizon goal`` as a literal trigger, so any message mentioning the
phrase (including list/show/cancel queries) hit horizon_add before the
more specific horizon_list / horizon_abandon patterns later in the list.

Fix: hoist horizon_list and horizon_abandon above horizon_add. The
add-trigger phrases (``watch for``, ``notify me when X``) don't collide
with show/list/abandon phrasing, so add still wins for its own queries.
"""
from __future__ import annotations

from prism_intents import INTENTS
from prism_routing import route_intent


def _route(message: str) -> str:
    return route_intent(message, INTENTS, lambda _m: None)


class TestListWinsOverAdd:
    def test_list_my_horizon_goals(self):
        assert _route("list my horizon goals") == "horizon_list"

    def test_show_my_horizon_goals(self):
        assert _route("show my horizon goals") == "horizon_list"

    def test_what_are_you_watching(self):
        assert _route("what are you watching") == "horizon_list"

    def test_horizon_status(self):
        assert _route("horizon status") == "horizon_list"


class TestAbandonWinsOverAdd:
    def test_stop_watching_horizon(self):
        assert _route("stop watching that horizon goal") == "horizon_abandon"

    def test_cancel_that_horizon_goal(self):
        assert _route("cancel that horizon goal") == "horizon_abandon"


class TestAddStillWinsForRealAdds:
    def test_notify_me_when_bitcoin_drops(self):
        assert _route("notify me when bitcoin drops below 60000") == "horizon_add"

    def test_watch_for_email_from_sarah(self):
        assert _route("watch for an email from Sarah") == "horizon_add"

    def test_add_a_long_term_goal(self):
        # "long.?term goal" is in the add pattern — a fresh add phrasing
        # should still win.
        assert _route("set a long-term goal to learn Spanish") == "horizon_add"


class TestNoDuplicateEntries:
    """The patch hoisted list/abandon and removed the original later entries.
    Make sure they aren't both still present — a duplicate would mean the
    later one is dead code and routing audits would surface noise."""

    def test_horizon_list_appears_once(self):
        count = sum(1 for _, name in INTENTS if name == "horizon_list")
        assert count == 1

    def test_horizon_abandon_appears_once(self):
        count = sum(1 for _, name in INTENTS if name == "horizon_abandon")
        assert count == 1
