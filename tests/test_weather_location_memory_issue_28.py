"""
tests/test_weather_location_memory_issue_28.py
==============================================
Weather location memory (issue #28-83): a bare "what's the weather?"
must never silently default to London. The organ remembers the last
explicitly named city (settings store, section "user", key
"home_location") and asks when it has nothing to go on.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

import prism_settings_store
from organs import weather_check


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path):
    db = str(tmp_path / "settings.db")
    prism_settings_store.reset_settings_store(db)
    with patch.object(prism_settings_store, "get_settings_store",
                      lambda db_path="ignored": prism_settings_store.reset_settings_store(db)):
        yield
    prism_settings_store.reset_settings_store()


def _fetch_stub(url, timeout=6):
    raise OSError("offline test")


class TestNoSavedLocation:
    def test_bare_weather_asks_instead_of_london(self):
        card = weather_check.execute("weather_check", "what's the weather?", {})
        assert "Which city" in card.body
        assert "london" not in card.body.lower()

    def test_explicit_city_used_and_remembered(self):
        with patch("urllib.request.urlopen", _fetch_stub):
            card = weather_check.execute(
                "weather_check", "what's the weather in berlin", {})
        assert card.card_data.get("location") == "Berlin"
        store = prism_settings_store.get_settings_store()
        assert store.get_section("user").get("home_location") == "berlin"
        assert "from now on" in card.body


class TestSavedLocation:
    def test_bare_weather_uses_saved_city(self):
        store = prism_settings_store.get_settings_store()
        store.set_section("user", {"home_location": "lagos"})
        with patch("urllib.request.urlopen", _fetch_stub):
            card = weather_check.execute("weather_check", "weather?", {})
        assert card.card_data.get("location") == "Lagos"

    def test_new_explicit_city_updates_default(self):
        store = prism_settings_store.get_settings_store()
        store.set_section("user", {"home_location": "lagos"})
        with patch("urllib.request.urlopen", _fetch_stub):
            weather_check.execute("weather_check", "weather in tokyo", {})
        assert store.get_section("user").get("home_location") == "tokyo"

    def test_same_city_no_repeat_note(self):
        store = prism_settings_store.get_settings_store()
        store.set_section("user", {"home_location": "tokyo"})
        with patch("urllib.request.urlopen", _fetch_stub):
            card = weather_check.execute("weather_check", "weather in tokyo", {})
        assert "from now on" not in card.body


class TestCtxLocationPrecedence:
    def test_ctx_location_wins_and_does_not_overwrite_default(self):
        store = prism_settings_store.get_settings_store()
        store.set_section("user", {"home_location": "lagos"})
        with patch("urllib.request.urlopen", _fetch_stub):
            card = weather_check.execute("weather_check", "weather", {"location": "Oslo"})
        assert card.card_data.get("location") == "Oslo"
        assert store.get_section("user").get("home_location") == "lagos"
