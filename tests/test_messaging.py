"""Tests for prism_messaging.py — Gap Prompt 13b."""
import sys
from unittest.mock import patch
from prism_messaging import PrismMessaging, Message


def test_no_platforms_when_empty():
    """PrismMessaging() with no args should have no configured platforms on non-macOS."""
    if sys.platform != "darwin":
        assert PrismMessaging().configured_platforms == []


def test_telegram_platform_when_token():
    """PrismMessaging with a telegram_token should include 'telegram'."""
    m = PrismMessaging(telegram_token="mytoken")
    assert "telegram" in m.configured_platforms


def test_imessage_on_macos():
    """On darwin, 'imessage' should appear in configured_platforms."""
    if sys.platform == "darwin":
        assert "imessage" in PrismMessaging().configured_platforms


def test_send_to_self_returns_bool():
    """send_to_self() should return a bool (False when not configured)."""
    result = PrismMessaging().send_to_self("test message")
    assert isinstance(result, bool)


def test_status_summary_has_configured_platforms():
    """status_summary() should contain 'configured_platforms' key."""
    summary = PrismMessaging().status_summary()
    assert "configured_platforms" in summary


def test_from_config_empty():
    """from_config({}) should produce a PrismMessaging with default values."""
    m = PrismMessaging.from_config({})
    assert m._tg_token == ""
    assert m._tg_chat == ""


def test_whatsapp_platform_when_sid():
    """PrismMessaging with wa_sid set should include 'whatsapp' in platforms."""
    m = PrismMessaging(wa_sid="ACtest", wa_token="tok", wa_from="whatsapp:+1234")
    assert "whatsapp" in m.configured_platforms


def test_get_updates_returns_list():
    """get_updates() returns a list (empty when not configured)."""
    result = PrismMessaging().get_updates("telegram", n=5)
    assert isinstance(result, list)


def test_send_unknown_platform_returns_false():
    """send() with an unknown platform should return False."""
    m = PrismMessaging()
    assert m.send("unknownplatform", "recipient", "hello") is False


def test_configured_platforms_only_telegram():
    """On non-macOS with only telegram_token, only 'telegram' in platforms."""
    if sys.platform == "darwin":
        return  # iMessage would also appear
    m = PrismMessaging(telegram_token="tok")
    assert m.configured_platforms == ["telegram"]
