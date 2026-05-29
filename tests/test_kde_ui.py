from __future__ import annotations

from kde_ui import get_ui_html


def test_get_ui_html_returns_string():
    html = get_ui_html()
    assert isinstance(html, str)
    assert html.strip()


def test_html_has_tabs():
    html = get_ui_html()
    assert "Morning" in html
    assert "Match" in html
    assert "Moment" in html


def test_html_has_api_constant():
    assert "127.0.0.1:8742" in get_ui_html()


def test_html_no_external_cdn():
    html = get_ui_html()
    assert "cdn.jsdelivr" not in html
    assert "unpkg.com" not in html
