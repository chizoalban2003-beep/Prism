"""Wikipedia HTML-leak fix for issue #28 bug 12 — displaytitle markup landed in card.

Live test: ``tell me about the eiffel tower on wikipedia`` returned a
body that began ``Wikipedia — <span lang="en" dir="ltr"><span class="mw-
page-title-main">Eiffel Tower</span></span>``. The REST summary endpoint
returns the title twice: a plain ``"title"`` and an HTML-decorated
``"displaytitle"``. The organ used ``displaytitle`` preferentially, so
the markup landed in plaintext output.

Fix: prefer the plain ``"title"`` field. If only ``displaytitle`` is
available (older or non-English responses), strip tags before use.
"""
from __future__ import annotations

import importlib.util
import json
from unittest.mock import MagicMock, patch


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, f"organs/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _urlopen(body: bytes, status: int = 200):
    resp = MagicMock()
    resp.read.return_value = body
    resp.status = status
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestNoHtmlInWikipediaCard:
    organ = _load("wikipedia_lookup")

    def _body(self, payload: dict) -> str:
        with patch(
            "urllib.request.urlopen",
            return_value=_urlopen(json.dumps(payload).encode()),
        ):
            card = self.organ.execute(
                "wikipedia_lookup", "tell me about Eiffel Tower", {}
            )
        return card.body or ""

    def test_displaytitle_html_does_not_leak(self):
        # The exact shape that surfaced live.
        body = self._body({
            "title": "Eiffel Tower",
            "displaytitle": (
                '<span lang="en" dir="ltr">'
                '<span class="mw-page-title-main">Eiffel Tower</span></span>'
            ),
            "extract": "The Eiffel Tower is a lattice tower.",
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Eiffel_Tower"}},
        })
        assert "<span" not in body, f"HTML tag leaked into card body: {body!r}"
        assert "mw-page-title" not in body
        assert "Eiffel Tower" in body

    def test_plain_title_used_when_available(self):
        body = self._body({
            "title": "Eiffel Tower",
            "displaytitle": "<span class='x'>Eiffel Tower</span>",
            "extract": "A tower.",
            "content_urls": {"desktop": {"page": "u"}},
        })
        # Header line should be exactly "Wikipedia — Eiffel Tower"
        assert "Wikipedia — Eiffel Tower" in body

    def test_displaytitle_stripped_when_title_missing(self):
        # Some API variants omit "title". Strip tags from displaytitle.
        body = self._body({
            "displaytitle": '<span class="mw-page-title-main">Stripped Title</span>',
            "extract": "A description.",
            "content_urls": {"desktop": {"page": "u"}},
        })
        assert "<span" not in body
        assert "Stripped Title" in body

    def test_no_title_at_all_uses_topic(self):
        # Neither title nor displaytitle. Falls back to the user's topic.
        body = self._body({
            "extract": "A description.",
            "content_urls": {"desktop": {"page": "u"}},
        })
        # Topic extraction picks up "Eiffel Tower" from the prompt.
        assert "Eiffel Tower" in body
