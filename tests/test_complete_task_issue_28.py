"""
tests/test_complete_task_issue_28.py
====================================
PrismTasks.complete() existed (with Todoist/GitHub/Linear sync) but no
intent ever routed to it — "remove task X" fell into list_tasks and just
echoed the task list back. New complete_task intent + handler: match the
named task among open tasks (substring first, then unambiguous token
overlap), complete it, or ask when ambiguous.

Relies on conftest's hermetic HOME: ~/.prism/tasks.db here is throwaway.
"""
from __future__ import annotations

from prism_intents import INTENTS
from prism_routing import route_intent


def _route(m):
    return route_intent(m, INTENTS, lambda _m: None)


class TestRouting:
    def test_remove_delete_complete_forms(self):
        for m in ("remove task Review PR", "delete task buy milk",
                  "complete task do laundry", "mark task done buy milk",
                  "task done: buy milk", "done with the laundry task"):
            assert _route(m) == "complete_task", m

    def test_add_and_list_unchanged(self):
        assert _route("add task: buy eggs") == "add_task"
        assert _route("list my tasks") == "list_tasks"
        assert _route("show my tasks") == "list_tasks"
        assert _route("remind me to stretch in 2 hours") == "reminder_set"


class TestHandler:
    def _agent(self):
        from prism_agent import PrismAgent
        return PrismAgent()

    def test_complete_by_exact_fragment(self, offline_llm):
        agent = self._agent()
        agent._task_mgr.add(title="buy milk")
        agent._task_mgr.add(title="do laundry")
        card = agent.chat("complete task buy milk")
        assert "Done: buy milk" in card.body
        titles = [t.title for t in agent._task_mgr.list_tasks(done=False)]
        assert "buy milk" not in titles
        assert "do laundry" in titles

    def test_ambiguous_asks_instead_of_guessing(self, offline_llm):
        agent = self._agent()
        agent._task_mgr.add(title="call the plumber")
        agent._task_mgr.add(title="call the dentist")
        before = len(agent._task_mgr.list_tasks(done=False))
        card = agent.chat("remove task call")
        assert "Several tasks match" in card.body
        assert len(agent._task_mgr.list_tasks(done=False)) == before

    def test_no_match_says_so(self, offline_llm):
        agent = self._agent()
        agent._task_mgr.add(title="water the plants")
        card = agent.chat("delete task quarterly report")
        assert "No open task matching" in card.body
