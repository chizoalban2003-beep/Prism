"""Bundled organ: weather_check — fetches weather via wttr.in (no API key)."""
ORGAN_META = {
    "intent":      "weather_check",
    "description": "fetches current weather conditions for a city or location",
    "version":     "1.0",
}


def execute(intent: str, message: str, ctx: dict):
    import json as _json
    import urllib.parse
    import urllib.request

    from prism_responses import text_card

    # Extract location from message or ctx
    location = ctx.get("location") or message.strip() or "London"
    words    = [w.strip(".,?!") for w in location.split() if len(w) > 2]
    city     = words[-1] if words else "London"

    url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
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
    except Exception as exc:
        result = f"Could not fetch weather for '{city}': {exc}"

    return text_card(result, "Weather")
