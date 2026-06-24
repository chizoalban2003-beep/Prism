"""Vision/perception vs. wikipedia precedence fix for issue #28 bug 15.

Live test: ``what is on my screen`` returned a Wikipedia article on
"Spell checker". The broad wikipedia_lookup fallback matches
``what (?:is|was) (?:a |an |the )?[A-Za-z]`` and was placed before
vision_query, so any "what is X" phrasing was hijacked.

Fix: hoist the specific device/perception organ intents
(screenshot_capture, vision_query, clipboard_read, file_read,
file_write) above the broad wikipedia_lookup catch-all. The catch-all
remains intact for actual encyclopaedic queries — they just don't grab
"what is on my screen" any more.
"""
from __future__ import annotations

from prism_intents import INTENTS
from prism_routing import route_intent


def _route(message: str) -> str:
    return route_intent(message, INTENTS, lambda _m: None)


class TestPerceptionWinsOverWikipedia:
    def test_what_is_on_my_screen(self):
        assert _route("what is on my screen") == "vision_query"

    def test_whats_on_my_screen(self):
        assert _route("what's on my screen") == "vision_query"

    def test_describe_my_screen(self):
        assert _route("describe my screen") == "vision_query"

    def test_what_do_you_see(self):
        assert _route("what do you see") == "vision_query"


class TestClipboardWinsOverWikipedia:
    def test_what_is_on_my_clipboard(self):
        assert _route("what is on my clipboard") == "clipboard_read"

    def test_read_my_clipboard(self):
        assert _route("read my clipboard") == "clipboard_read"


class TestScreenshotStillWorks:
    def test_take_a_screenshot(self):
        assert _route("take a screenshot") == "screenshot_capture"

    def test_grab_screenshot(self):
        assert _route("grab screenshot") == "screenshot_capture"


class TestFileReadStillWorks:
    def test_read_the_file(self):
        assert _route("read the file ~/notes.md") == "file_read"


class TestWikipediaCatchAllStillWorks:
    """The catch-all must still grab genuinely encyclopaedic phrasings."""

    def test_what_is_the_eiffel_tower(self):
        assert _route("what is the eiffel tower") == "wikipedia_lookup"

    def test_who_was_einstein(self):
        assert _route("who was Einstein") == "wikipedia_lookup"

    def test_tell_me_about_python_language(self):
        assert _route("tell me about Python the programming language") == "wikipedia_lookup"


class TestNoDuplicateEntries:
    """The hoist removed the original positions of the moved entries —
    make sure they aren't both present."""

    def test_vision_query_appears_once(self):
        assert sum(1 for _, n in INTENTS if n == "vision_query") == 1

    def test_screenshot_capture_appears_once(self):
        assert sum(1 for _, n in INTENTS if n == "screenshot_capture") == 1

    def test_clipboard_read_appears_once(self):
        assert sum(1 for _, n in INTENTS if n == "clipboard_read") == 1

    def test_file_read_appears_once(self):
        assert sum(1 for _, n in INTENTS if n == "file_read") == 1

    def test_file_write_appears_once(self):
        assert sum(1 for _, n in INTENTS if n == "file_write") == 1
