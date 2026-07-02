"""Weather location extraction fix for issue #28 bug 17.

Live test: ``what's the weather`` returned ``Weather in Weather:
Thunderstorm, Rain And Small Hail/Snow Pallets With Thunderstorm`` with
a 41°C temp — wttr.in's fallback for a meaningless query. The organ was
taking the LAST word longer than two letters as the city, which for
"what's the weather" is "weather" itself.

Fix: drop query stopwords (what, the, weather, today, ...) before
choosing a city. Originally the empty case fell back to "London";
since #28-83 it asks which city to use (and remembers the answer), so
the invariant these tests protect is now "a stopword is never treated
as a city": stopword-only queries must not hit wttr.in at all.
"""
from __future__ import annotations

import importlib.util
import json
from unittest.mock import MagicMock, patch

import pytest

import prism_settings_store


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path):
    """Point the settings store at a temp DB — the organ persists the
    last explicitly named city (#28-83), and without isolation these
    tests would write into the developer's real ~/.prism/settings.db."""
    db = str(tmp_path / "settings.db")
    prism_settings_store.reset_settings_store(db)
    with patch.object(
            prism_settings_store, "get_settings_store",
            lambda db_path="ignored": prism_settings_store.reset_settings_store(db)):
        yield
    prism_settings_store.reset_settings_store()


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, f"organs/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _urlopen(body: bytes):
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


_FAKE_BODY = json.dumps({
    "current_condition": [{
        "weatherDesc": [{"value": "Clear"}],
        "temp_C": "18",
        "FeelsLikeC": "17",
        "humidity": "60",
        "windspeedKmph": "10",
    }]
}).encode()


class TestStopwordsStripped:
    organ = _load("weather_check")

    def _call(self, message: str, ctx: dict | None = None) -> tuple[str, dict]:
        # Capture every URL the organ probes during execute(). The wttr.in
        # request is the one we want to assert on, but other modules in
        # CI's import graph occasionally fire a stray Ollama-tags probe
        # via urllib.request.urlopen — using a list keeps the assertion
        # focused on the wttr.in call rather than whichever urlopen fired
        # last.
        captured: list[str] = []

        def fake_urlopen(url, timeout=6):
            captured.append(url)
            return _urlopen(_FAKE_BODY)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            card = self.organ.execute("weather_check", message, ctx or {})
        wttr = next((u for u in captured if "wttr.in" in u), "")
        return wttr, card.card_data

    def test_bare_weather_query_asks_for_city(self):
        url, _ = self._call("what's the weather")
        assert url == "", f"stopword query must not hit wttr.in: {url}"

    def test_weather_today_asks_for_city(self):
        url, _ = self._call("what is the weather today")
        assert url == "", f"stopword query must not hit wttr.in: {url}"

    def test_tomorrows_weather_asks_for_city(self):
        # "tomorrow's" — apostrophe gets stripped to "tomorrows" which
        # the initial stopword set missed, producing wttr.in lookup of
        # "Tomorrows" as a city name.
        url, _ = self._call("tomorrow's weather")
        assert url == "", f"stopword query must not hit wttr.in: {url}"

    def test_todays_weather_asks_for_city(self):
        url, _ = self._call("today's weather")
        assert url == "", f"stopword query must not hit wttr.in: {url}"

    def test_how_cold_is_it_asks_for_city(self):
        # "how cold is it" was treating "cold" as the city, querying
        # wttr.in for "Cold" and getting back nonsense.
        url, _ = self._call("how cold is it")
        assert url == "", f"stopword query must not hit wttr.in: {url}"

    def test_how_hot_is_it_asks_for_city(self):
        url, _ = self._call("how hot is it")
        assert url == "", f"stopword query must not hit wttr.in: {url}"

    def test_is_it_raining_asks_for_city(self):
        url, _ = self._call("is it raining")
        assert url == "", f"stopword query must not hit wttr.in: {url}"

    def test_how_is_the_weather_asks_for_city(self):
        url, _ = self._call("how is the weather")
        assert url == "", f"stopword query must not hit wttr.in: {url}"

    def test_real_city_survives(self):
        url, _ = self._call("what's the weather in Tokyo")
        assert "Tokyo" in url or "tokyo" in url

    def test_real_city_paris(self):
        url, _ = self._call("weather in Paris")
        assert "Paris" in url or "paris" in url

    def test_multiword_city(self):
        url, _ = self._call("weather in San Francisco")
        # Both "san" and "francisco" survive the stopword filter.
        assert "san" in url.lower() and "francisco" in url.lower()

    def test_ctx_location_wins(self):
        url, _ = self._call("what's the weather", ctx={"location": "Berlin"})
        assert "Berlin" in url

    def test_card_does_not_say_weather_in_weather(self):
        # The original symptom — body said "Weather in Weather".
        url, data = self._call("what's the weather")
        assert data.get("location", "").lower() != "weather"

    def test_will_it_rain_tomorrow_asks_for_city(self):
        # "will it rain tomorrow" was picking "will" as the city after
        # the routing fix in #28-34 stopped relying on the bare-rain
        # rule. The organ stopwords missed the future-tense auxiliary.
        url, _ = self._call("will it rain tomorrow")
        assert url == "", f"stopword query must not hit wttr.in: {url}"

    def test_will_it_snow_tonight_asks_for_city(self):
        url, _ = self._call("will it snow tonight")
        assert url == "", f"stopword query must not hit wttr.in: {url}"
