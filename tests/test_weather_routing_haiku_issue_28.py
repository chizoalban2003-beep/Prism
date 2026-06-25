"""Weather intent-routing fix for issue #28 bug 34.

Live test: ``give me a haiku about rain`` was routed to
``weather_check``, then the city extraction lifted "haiku about" as
the city. The 500 came from wttr.in choking on the string.

Root cause: a redundant second weather entry sat near the bottom of
prism_intents.INTENTS (``weather|temperature|forecast|how (?:hot|cold)
|rain|sunny``) — bare ``rain`` and ``sunny`` with no word boundary
and no contextual prefix. "haiku about rain" matched ``rain``.

Fix: remove the redundant entry; extend the existing weather hoist
near the top to also catch ``will it rain tomorrow`` and similar
forms; teach the hoist to match ``-ing`` / ``-y`` morphology so
``is it raining`` / ``is it snowing`` still hit.
"""
from __future__ import annotations

import re

from prism_intents import INTENTS


def _route(message: str) -> str:
    for pattern, intent in INTENTS:
        if re.search(pattern, message, re.IGNORECASE):
            return intent
    return "_NO_MATCH_"


class TestNoFalsePositives:
    """Bare condition words inside non-weather phrases must NOT route to weather."""

    def test_haiku_about_rain(self):
        # The reported bug — was hitting weather_check.
        assert _route("give me a haiku about rain") != "weather_check"

    def test_poem_about_sunny_days(self):
        assert _route("write a poem about sunny days") != "weather_check"

    def test_rain_on_my_parade(self):
        assert _route("rain on my parade") != "weather_check"

    def test_let_it_rain(self):
        assert _route("let it rain") != "weather_check"

    def test_sunny_disposition(self):
        assert _route("she has a sunny disposition") != "weather_check"


class TestWeatherStillRoutes:
    """All the real weather queries the original entries handled must keep working."""

    def test_how_is_the_weather(self):
        assert _route("how is the weather") == "weather_check"

    def test_what_is_the_temperature(self):
        assert _route("what is the temperature") == "weather_check"

    def test_forecast_for_monday(self):
        assert _route("forecast for monday") == "weather_check"

    def test_will_it_rain_tomorrow(self):
        # This one previously matched via the bare-rain rule and would
        # have regressed if we'd just deleted that without extending
        # the hoist.
        assert _route("will it rain tomorrow") == "weather_check"

    def test_is_it_going_to_rain(self):
        assert _route("is it going to rain") == "weather_check"

    def test_is_it_raining(self):
        # -ing morphology — the new hoist needs to accept it.
        assert _route("is it raining") == "weather_check"

    def test_is_it_snowing(self):
        assert _route("is it snowing") == "weather_check"

    def test_how_cold_is_it(self):
        assert _route("how cold is it") == "weather_check"

    def test_will_it_snow_tonight(self):
        assert _route("will it snow tonight") == "weather_check"
