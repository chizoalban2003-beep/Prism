"""
tests/test_desktop_intent_guard_issue_28.py
===========================================
CRUD-prefix guard (#28-128): the desktop *action* intents (window_control,
input_control, computer_use) match content keywords that legitimately appear
inside CRUD free text — "add task try window control" must stay add_task.
The _NOT_CRUD guard blocks a match when the message starts with a CRUD verb,
while every direct desktop imperative still routes correctly.

Found live while seeding tasks through the running daemon.
"""
from __future__ import annotations

from prism_intents import INTENTS
from prism_routing import route_intent


def _route(m):
    return route_intent(m, INTENTS, lambda _m: "")


class TestCrudContentNotStolen:
    def test_add_task_with_desktop_keywords(self):
        assert _route("add task try window control on desktop") == "add_task"
        assert _route("add task review the computer-use flow") == "add_task"
        assert _route("add task press enter to submit the form") == "add_task"
        assert _route("create task about scrolling down the page") == "add_task"

    def test_note_reminder_complete_with_desktop_keywords(self):
        assert _route("add note about window management") in ("note_append", "add_task")
        assert _route("remind me to use the computer to email") == "reminder_set"
        assert _route("complete task close the window") == "complete_task"

    def test_plain_crud_still_works(self):
        assert _route("add task buy milk") == "add_task"
        assert _route("list my tasks") == "list_tasks"


class TestDesktopImperativesStillRoute:
    def test_window(self):
        for m in ("close the window", "list windows", "minimize",
                  "maximize the editor", "window control",
                  "bring Chrome to the front"):
            assert _route(m) == "window_control", m

    def test_input(self):
        for m in ("click", "type hello world", "press ctrl+c",
                  "scroll down", "move the mouse to 100 200"):
            assert _route(m) == "input_control", m

    def test_computer_use(self):
        for m in ("use the computer to open settings",
                  "computer use: fill the form",
                  "control my screen to click login"):
            assert _route(m) == "computer_use", m
