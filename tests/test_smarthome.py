"""Tests for prism_smart_home.py — Gap Prompt 9b."""
import logging

from prism_smart_home import HA_TOKEN_ENV, PrismSmartHome


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


def test_env_token_overrides_toml(monkeypatch):
    """PRISM_HA_TOKEN env var must take precedence over ha_token in toml."""
    monkeypatch.setenv(HA_TOKEN_ENV, "env-token-xyz")
    sh = PrismSmartHome.from_config({"smarthome": {"ha_token": "toml-token"}})
    assert sh._token == "env-token-xyz"


def test_toml_token_warns(monkeypatch, caplog):
    """Loading ha_token from toml emits a deprecation warning."""
    monkeypatch.delenv(HA_TOKEN_ENV, raising=False)
    with caplog.at_level(logging.WARNING, logger="prism_smart_home"):
        sh = PrismSmartHome.from_config({"smarthome": {"ha_token": "toml-token"}})
    assert sh._token == "toml-token"
    assert any(HA_TOKEN_ENV in r.message for r in caplog.records)


def test_no_warning_when_env_used(monkeypatch, caplog):
    """No warning when token comes from env even if toml also has one."""
    monkeypatch.setenv(HA_TOKEN_ENV, "env-token")
    with caplog.at_level(logging.WARNING, logger="prism_smart_home"):
        PrismSmartHome.from_config({"smarthome": {"ha_token": "toml-token"}})
    assert not any(HA_TOKEN_ENV in r.message for r in caplog.records)
