"""note_append widening for issue #28 bug 65.

Live probes: ``create a note``, ``new note``, ``start a note`` all
routed to ``general_chat``. The existing note_append regex listed only
``(append|add|write|save|take|jot down) (a )? note``, so user-natural
"create/new/start" verbs missed.

Fix: extend the verb list to include create|start|new|begin.
"""
from __future__ import annotations

from prism_intents import INTENTS
from prism_routing import route_intent


def _route(message: str) -> str:
    return route_intent(message, INTENTS, lambda _m: None)


class TestNoteCreateVariants:

    def test_create_a_note(self):
        assert _route("create a note") == "note_append"

    def test_new_note(self):
        assert _route("new note") == "note_append"

    def test_start_a_note(self):
        assert _route("start a note") == "note_append"

    def test_begin_a_note(self):
        assert _route("begin a note") == "note_append"


class TestExistingNotePathsUnchanged:

    def test_add_a_note(self):
        assert _route("add a note") == "note_append"

    def test_jot_down_a_note(self):
        assert _route("jot down a note") == "note_append"

    def test_write_a_note(self):
        assert _route("write a note") == "note_append"

    def test_list_my_notes(self):
        assert _route("list my notes") == "note_list"

    def test_show_my_notes(self):
        assert _route("show my notes") == "note_list"
