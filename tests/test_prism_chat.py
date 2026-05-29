from __future__ import annotations

from prism_chat import get_chat_html


def test_html_non_empty():
    html = get_chat_html()
    assert isinstance(html, str)
    assert html


def test_no_cdn():
    html = get_chat_html().lower()
    assert "cdn.jsdelivr" not in html
    assert "unpkg" not in html
