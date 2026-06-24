"""approvals_list intent for issue #28 bug 21.

Live test: ``what are my pending approvals`` returned ``Recalled — My
partner is Sarah.`` (memory_recall hijack), and ``show pending
approvals`` returned the todo list. There was no way to ask PRISM what
it was waiting on.

Fix: add an approvals_list intent above memory_recall and list_tasks,
and an info-intent handler that reports agent._pending_approval.
"""
from __future__ import annotations

from types import SimpleNamespace

from prism_info_intents import handle_info_intent
from prism_intents import INTENTS
from prism_routing import route_intent


def _route(message: str) -> str:
    return route_intent(message, INTENTS, lambda _m: None)


class TestRoutingApprovals:
    def test_what_are_my_pending_approvals(self):
        assert _route("what are my pending approvals") == "approvals_list"

    def test_show_pending_approvals(self):
        assert _route("show pending approvals") == "approvals_list"

    def test_list_approvals(self):
        assert _route("list approvals") == "approvals_list"

    def test_my_approvals(self):
        assert _route("my approvals") == "approvals_list"

    def test_any_pending_approvals(self):
        assert _route("any pending approvals") == "approvals_list"

    def test_approvals_bare(self):
        assert _route("approvals?") == "approvals_list"


class TestNoOverreach:
    def test_show_my_tasks_still_list_tasks(self):
        assert _route("show my tasks") == "list_tasks"

    def test_what_is_my_partner_still_memory_recall(self):
        assert _route("what is my partner's name") == "memory_recall"


class TestApprovalsHandler:
    def _agent(self, pending):
        return SimpleNamespace(_pending_approval=pending)

    def test_no_pending(self):
        card = handle_info_intent(self._agent(None), "approvals_list", "approvals?", {})
        assert card is not None
        assert "No pending" in card.body

    def test_one_pending(self):
        agent = self._agent({"task": "send email to Bob", "reason": "external comms"})
        card = handle_info_intent(agent, "approvals_list", "list approvals", {})
        assert card is not None
        assert "send email to Bob" in card.body
        assert "external comms" in card.body
