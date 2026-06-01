"""Tests for prism_smart_home.py — Gap Prompt 9b."""
from prism_smart_home import PrismSmartHome


def test_not_configured_empty():
    """PrismSmartHome() with no args should not be configured."""
    sh = PrismSmartHome()
    assert sh.configured is False


def test_configured_when_set():
    """PrismSmartHome with url + token should report configured."""
    sh = PrismSmartHome(ha_url="http://ha.local", token="tok")
    assert sh.configured is True


def test_status_unconfigured():
    """status_summary() on unconfigured instance returns configured=False."""
    sh = PrismSmartHome()
    summary = sh.status_summary()
    assert summary.get("configured") is False
