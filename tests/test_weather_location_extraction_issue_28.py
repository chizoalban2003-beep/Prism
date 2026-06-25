"""Weather location extraction fix for issue #28 bug 17.

Live test: ``what's the weather`` returned ``Weather in Weather:
Thunderstorm, Rain And Small Hail/Snow Pallets With Thunderstorm`` with
a 41°C temp — wttr.in's fallback for a meaningless query. The organ was
taking the LAST word longer than two letters as the city, which for
"what's the weather" is "weather" itself.

Fix: drop query stopwords (what, the, weather, today, ...) before
choosing a city. If nothing real remains, fall back to "London".
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
        captured = {}

        def fake_urlopen(url, timeout=6):
            captured["url"] = url
            return _urlopen(_FAKE_BODY)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            card = self.organ.execute("weather_check", message, ctx or {})
        return captured.get("url", ""), card.card_data

    def test_bare_weather_query_uses_default(self):
        url, _ = self._call("what's the weather")
        assert "London" in url, f"expected London fallback, got {url}"

    def test_weather_today_uses_default(self):
        url, _ = self._call("what is the weather today")
        assert "London" in url

    def test_tomorrows_weather_uses_default(self):
        # "tomorrow's" — apostrophe gets stripped to "tomorrows" which
        # the initial stopword set missed, producing wttr.in lookup of
        # "Tomorrows" as a city name.
        url, _ = self._call("tomorrow's weather")
        assert "London" in url, f"expected London fallback, got {url}"

    def test_todays_weather_uses_default(self):
        url, _ = self._call("today's weather")
        assert "London" in url

    def test_how_cold_is_it_uses_default(self):
        # "how cold is it" was treating "cold" as the city, querying
        # wttr.in for "Cold" and getting back nonsense.
        url, _ = self._call("how cold is it")
        assert "London" in url, f"expected London fallback, got {url}"

    def test_how_hot_is_it_uses_default(self):
        url, _ = self._call("how hot is it")
        assert "London" in url

    def test_is_it_raining_uses_default(self):
        url, _ = self._call("is it raining")
        assert "London" in url

    def test_how_is_the_weather_uses_default(self):
        url, _ = self._call("how is the weather")
        assert "London" in url

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
