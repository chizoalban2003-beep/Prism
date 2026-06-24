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
        "today", "tonight", "tomorrow", "now", "currently", "outside",
        "like", "right", "going", "to", "be", "and",
        "weather", "forecast", "temperature", "temp", "climate",
        "me", "my", "your", "our", "us", "please", "city",
    }
    if ctx.get("location"):
        city = str(ctx["location"]).strip() or "London"
    else:
        # Drop apostrophes so "what's" / "hows" collapse to "whats"/"hows".
        normalised = message.lower().replace("'", "").replace("\u2019", "")
        words = [w.strip(".,?!\"") for w in normalised.split()]
        candidates = [w for w in words if len(w) > 2 and w not in _STOP]
        city = " ".join(candidates) if candidates else "London"

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

    card = text_card(result, "Weather")
    card.card_data.update(structured)
    return card
