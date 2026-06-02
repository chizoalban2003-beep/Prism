"""Tests for prism_push.py — Gap Prompt 14a."""
from unittest.mock import patch

from prism_push import PrismPush


def test_not_configured_when_no_topic():
    """PrismPush() with no topic is not configured."""
    assert PrismPush().configured is False


def test_configured_when_topic_set():
    """PrismPush(topic='test') is configured."""
    assert PrismPush(topic="test").configured is True


def test_send_returns_false_unconfigured():
    """send() returns False when no topic is set."""
    assert PrismPush().send("title", "body") is False


def test_priority_map_has_urgent():
    """PRIORITY_MAP["urgent"] == 5."""
    assert PrismPush.PRIORITY_MAP["urgent"] == 5


def test_status_summary():
    """status_summary() contains 'topic' and 'server' keys."""
    s = PrismPush(topic="my-topic").status_summary()
    assert "topic" in s
    assert "server" in s
    assert "configured" in s


def test_send_returns_false_on_network_error():
    """send() returns False when the HTTP request fails."""
    push = PrismPush(topic="test-topic")
    with patch("urllib.request.urlopen") as mock_open:
        mock_open.side_effect = OSError("no network")
        result = push.send("title", "body")
    assert result is False


def test_alert_returns_false_unconfigured():
    """alert() returns False when not configured."""
    assert PrismPush().alert("hello") is False


def test_urgent_uses_priority_5():
    """urgent() sends with priority 5 (urgent)."""
    push = PrismPush(topic="test-topic")
    sent_headers = {}
    class FakeResponse:
        def read(self): return b""
        def __enter__(self): return self
        def __exit__(self, *a): pass
    def fake_urlopen(req, timeout=None):
        sent_headers.update(req.headers)
        return FakeResponse()
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        push.urgent("something urgent")
    assert sent_headers.get("Priority") == str(PrismPush.PRIORITY_MAP["urgent"])


def test_from_config_reads_push_section():
    """from_config reads the [push] section."""
    cfg = {"push": {"topic": "my-topic", "priority": "high"}}
    p = PrismPush.from_config(cfg)
    assert p._topic == "my-topic"
    assert p._priority == "high"
