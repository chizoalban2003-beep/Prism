"""list_tasks routing fix for issue #28 bug 36.

Live test: ``what reminders do I have for today`` returned the
"Planner LLM unavailable — tinyllama timed out" card. Dropping the
``for today`` suffix was enough to surface the reminder list (the
LLM classifier picked it up). The intent regex never recognised
"reminders" as a list keyword and ``today`` was claimed by
universal_plan.

Fix:
* Hoist a specific list-tasks rule above universal_plan keyed on
  LIST verbs (list/show/view) + the noun, the bare "my reminders
  for today" form, and "what reminders do I have" interrogatives.
  Keep it narrow so "set a reminder for 3pm" still routes to
  reminder_set.
* Extend the broader list_tasks fallback below to include
  ``reminders`` and verb forms.
"""
from __future__ import annotations

import re

from prism_intents import INTENTS


def _route(message: str) -> str:
    for pattern, intent in INTENTS:
        if re.search(pattern, message, re.IGNORECASE):
            return intent
    return "_NO_MATCH_"


class TestReminderList:
    def test_what_reminders_for_today(self):
        # The reported bug.
        assert _route("what reminders do I have for today") == "list_tasks"

    def test_list_my_reminders(self):
        assert _route("list my reminders") == "list_tasks"

    def test_show_my_reminders(self):
        assert _route("show my reminders") == "list_tasks"

    def test_my_reminders_for_today(self):
        assert _route("my reminders for today") == "list_tasks"

    def test_show_my_todos_for_today(self):
        assert _route("show my todos for today") == "list_tasks"

    def test_list_my_tasks(self):
        assert _route("list my tasks") == "list_tasks"

    def test_what_tasks_are_pending(self):
        assert _route("what tasks are pending") == "list_tasks"

    def test_view_my_todos(self):
        assert _route("view my todos") == "list_tasks"


class TestReminderSetStillRoutes:
    """The hoist must not steal reminder/task creation forms."""

    def test_set_a_reminder(self):
        assert _route("set a reminder for 3pm to take medication") == "reminder_set"

    def test_remind_me_to(self):
        assert _route("remind me to call john in 30 minutes") == "reminder_set"

    def test_add_a_task(self):
        assert _route("add a task to fix the bug") == "add_task"

    def test_create_new_todo(self):
        assert _route("create a new todo") == "add_task"


class TestUniversalPlanStillRoutes:
    """Plan-like queries without reminder/task nouns still hit the planner."""

    def test_plan_my_day(self):
        assert _route("plan my day") == "universal_plan"

    def test_good_morning(self):
        # Changed in #28-79: bare greetings route to general_chat.
        assert _route("good morning") == "general_chat"

    def test_daily_schedule(self):
        assert _route("daily schedule") == "universal_plan"
