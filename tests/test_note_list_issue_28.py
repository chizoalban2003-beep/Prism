"""Note-list intent + organ for issue #28 bug 18.

Live test: ``list my notes`` returned ``Note saved to notes.md  Content:
list my notes`` — the query itself was being persisted as a new note,
because the regex router didn't match, the LLM classifier picked
note_append, and the append organ took the whole query as the body.

Fix: introduce a note_list intent above note_append, and a note_list
organ that reads ~/.prism/notes.md.
"""
from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path
from unittest.mock import patch

from prism_intents import INTENTS
from prism_routing import route_intent


def _route(message: str) -> str:
    return route_intent(message, INTENTS, lambda _m: None)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, f"organs/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestRoutingNoteList:
    def test_list_my_notes(self):
        assert _route("list my notes") == "note_list"

    def test_show_my_notes(self):
        assert _route("show my notes") == "note_list"

    def test_read_my_notes(self):
        assert _route("read my notes") == "note_list"

    def test_what_are_my_notes(self):
        assert _route("what are my notes") == "note_list"

    def test_show_notes(self):
        assert _route("show notes") == "note_list"


class TestRoutingNoteAppendStillWorks:
    def test_note_colon(self):
        assert _route("note: buy milk") == "note_append"

    def test_save_a_note(self):
        assert _route("save a note about Postgres") == "note_append"

    def test_jot_down_a_note(self):
        assert _route("jot down a note: review PR") == "note_append"


class TestNoteListOrgan:
    organ = _load("note_list")

    def test_lists_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            prism_dir = Path(tmp) / ".prism"
            prism_dir.mkdir()
            notes_file = prism_dir / "notes.md"
            notes_file.write_text(
                "\n## 2026-06-24 10:00:00\n\nbuy milk\n"
                "\n## 2026-06-24 11:00:00\n\nreview PR #123\n",
                encoding="utf-8",
            )
            with patch("pathlib.Path.expanduser", return_value=notes_file):
                card = self.organ.execute("note_list", "list my notes", {})
            assert "buy milk" in card.body
            assert "review PR" in card.body
            assert card.card_data["count"] == 2

    def test_no_file_returns_empty_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "no_such.md"
            with patch("pathlib.Path.expanduser", return_value=missing):
                card = self.organ.execute("note_list", "list my notes", {})
            assert "No notes yet" in card.body
            assert card.card_data["count"] == 0
