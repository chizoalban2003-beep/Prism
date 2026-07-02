"""Bundled organ: weather_check — fetches weather via wttr.in (no API key)."""
ORGAN_META = {
    "intent":      "weather_check",
    "description": "fetches current weather conditions for a city or location",
    "version":     "1.0",
    "capabilities": ["internet_read"],
    "inputs": {
        "location": "str",
    },
    "outputs": {
        "location":      "str",
        "temperature_c": "float",
        "feels_like_c":  "float",
        "humidity":      "int",
        "wind_kmh":      "float",
        "description":   "str",
    },
}

ORGAN_POLICY = {
    "risk_level":        "low",
    "requires_approval": False,
    "irreversible":      False,
    "max_per_session":   None,
}


def execute(intent: str, message: str, ctx: dict):
    import json as _json
    import urllib.parse
    import urllib.request

    from prism_responses import text_card

    # Extract location from message or ctx.
    # Strip query stopwords so "what's the weather" doesn't pick "weather"
    # as the city and get back wttr.in's fallback nonsense.
    _STOP = {
        "what", "whats", "how", "hows", "tell", "show", "get", "give",
        "the", "a", "an", "is", "are", "was", "in", "for", "at", "on",
        "today", "tonight", "tomorrow", "tomorrows", "yesterday",
        "yesterdays", "tonights", "todays", "now", "currently", "outside",
        "like", "right", "going", "to", "be", "and",
        # Future-tense auxiliaries: "will it rain tomorrow" was picking
        # "will" as the city.
        "will", "would", "shall", "can", "could", "may", "might",
        "weather", "forecast", "temperature", "temp", "climate",
        # Condition words — "how cold is it" was looking up city "Cold",
        # "is it raining" was looking up "Raining".
        "cold", "hot", "warm", "chilly", "windy", "sunny", "cloudy",
        "rain", "rainy", "raining", "snow", "snowy", "snowing",
        "fog", "foggy", "hail", "hailing", "storm", "storming", "stormy",
        "humid", "freezing", "scorching", "wet", "dry", "it",
        "me", "my", "your", "our", "us", "please", "city",
    }
    def _saved_location() -> str:
        try:
            from prism_settings_store import get_settings_store
            section = get_settings_store().get_section("user") or {}
            return str(section.get("home_location", "")).strip()
        except Exception:
            return ""

    def _save_location(value: str) -> bool:
        try:
            from prism_settings_store import get_settings_store
            store = get_settings_store()
            section = store.get_section("user") or {}
            section["home_location"] = value
            store.set_section("user", section)
            return True
        except Exception:
            return False

    remembered_note = ""
    if ctx.get("location"):
        city = str(ctx["location"]).strip()
    else:
        # Drop apostrophes so "what's" / "hows" collapse to "whats"/"hows".
        normalised = message.lower().replace("'", "").replace("\u2019", "")
        words = [w.strip(".,?!\"") for w in normalised.split()]
        candidates = [w for w in words if len(w) > 2 and w not in _STOP]
        city = " ".join(candidates)
        if city:
            # Remember the last explicitly named city so a bare "weather?"
            # never falls back to a hardcoded default again.
            previous = _saved_location()
            if previous.lower() != city.lower() and _save_location(city):
                remembered_note = (
                    f"\n\n(I'll use {city.title()} for plain \"weather?\" "
                    "questions from now on \u2014 name any other city to switch.)"
                )
        else:
            city = _saved_location()
            if not city:
                return text_card(
                    "Which city should I check? Ask e.g. \"weather in Berlin\" "
                    "\u2014 I'll remember it and answer plain \"weather?\" with "
                    "your city from then on.",
                    "Weather \u2014 where?",
                )

    url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
    structured: dict = {"location": city.title()}
    try:
        with urllib.request.urlopen(url, timeout=6) as resp:
            data    = _json.loads(resp.read())
        cur     = data["current_condition"][0]
        desc    = cur["weatherDesc"][0]["value"]
        temp_c  = cur["temp_C"]
        feels   = cur["FeelsLikeC"]
        humid   = cur["humidity"]
        wind    = cur["windspeedKmph"]
        result  = (
            f"Weather in {city.title()}: {desc}\n"
            f"Temperature: {temp_c}°C (feels like {feels}°C)\n"
            f"Humidity: {humid}%  |  Wind: {wind} km/h"
        )
        structured.update({
            "description":   desc,
            "temperature_c": float(temp_c),
            "feels_like_c":  float(feels),
            "humidity":      int(humid),
            "wind_kmh":      float(wind),
        })
    except Exception as exc:
        result = f"Could not fetch weather for '{city}': {exc}"
        structured["error"] = str(exc)

    card = text_card(result + remembered_note, "Weather")
    card.card_data.update(structured)
    return card
