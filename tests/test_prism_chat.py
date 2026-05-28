from __future__ import annotations

import json
import time
import urllib.request
from types import SimpleNamespace
from unittest.mock import MagicMock

from kde_server import KDEServer
from prism_chat import get_chat_html


def _make_agent() -> MagicMock:
    agent = MagicMock()
    agent.status.return_value = {"ok": True}
    agent.morning_briefing.return_value = SimpleNamespace(
        plan=SimpleNamespace(
            primary_focus="Recovery",
            activation=0.55,
            tasks=[],
            warnings=[],
            rationale="Recovery first",
        )
    )
    agent.ask.return_value = SimpleNamespace(output={"ok": True}, task="test", method="kw", success=True, elapsed_ms=1.0)
    agent.reflect.return_value = {"profile": "Tester", "fixed_fulcrum": 0.5, "total_ratings": 1, "total_plans": 1}
    agent._assistant = MagicMock()
    agent._assistant.history.return_value = []
    agent._hub = MagicMock()
    agent._hub.list_devices.return_value = []
    agent._profile = MagicMock()
    agent._profile.name = "Tester"
    return agent


def _start(port: int) -> KDEServer:
    server = KDEServer(agent=_make_agent(), port=port, platform=MagicMock())
    server.start(blocking=False)
    time.sleep(0.15)
    return server


def test_html_served():
    server = _start(19420)
    try:
        with urllib.request.urlopen("http://127.0.0.1:19420/", timeout=5) as response:
            body = response.read().decode("utf-8")
            assert response.status == 200
            assert "text/html" in response.headers.get("Content-Type", "")
            assert "PRISM" in body
    finally:
        server.stop()


def test_html_contains_prism():
    server = _start(19421)
    try:
        with urllib.request.urlopen("http://127.0.0.1:19421/", timeout=5) as response:
            body = response.read().decode("utf-8")
            assert "PRISM" in body
    finally:
        server.stop()


def test_post_chat_returns_card():
    server = _start(19422)
    try:
        request = urllib.request.Request(
            "http://127.0.0.1:19422/chat",
            data=json.dumps({"message": "plan my day"}).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            data = json.loads(response.read())
            assert response.status == 200
            assert "type" in data
    finally:
        server.stop()


def test_html_has_sidebar_nav():
    html = get_chat_html()
    assert "Medical" in html
    assert "Financial" in html
    assert "Legal" in html


def test_html_has_render_card():
    assert "function renderCard(card)" in get_chat_html()


def test_html_has_demo_fallback():
    assert "function demoFallback(msg)" in get_chat_html()


def test_html_no_external_deps():
    html = get_chat_html().lower()
    assert "cdn.jsdelivr" not in html
    assert "unpkg" not in html
    assert 'script src="http' not in html
    assert "script src='http" not in html


def test_html_css_dark_mode():
    assert "prefers-color-scheme" in get_chat_html()


def test_html_mobile_responsive():
    assert "@media" in get_chat_html()
