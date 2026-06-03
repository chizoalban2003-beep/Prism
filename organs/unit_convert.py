"""Bundled organ: unit_convert — convert between units of measurement."""
ORGAN_META = {
    "intent":      "unit_convert",
    "description": "convert between units: length, weight, temperature, volume, speed",
    "version":     "1.0",
}

ORGAN_POLICY = {
    "risk_level":        "low",
    "requires_approval": False,
    "irreversible":      False,
    "max_per_session":   None,
}

# All conversion factors relative to SI base unit per category
_UNITS = {
    # Length → metres
    "length": {
        "m": 1, "metre": 1, "meter": 1, "metres": 1, "meters": 1,
        "km": 1000, "kilometre": 1000, "kilometer": 1000,
        "cm": 0.01, "centimetre": 0.01, "centimeter": 0.01,
        "mm": 0.001, "millimetre": 0.001, "millimeter": 0.001,
        "mi": 1609.344, "mile": 1609.344, "miles": 1609.344,
        "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
        "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
        "yd": 0.9144, "yard": 0.9144, "yards": 0.9144,
        "nm": 1852, "nautical mile": 1852,
    },
    # Weight → kilograms
    "weight": {
        "kg": 1, "kilogram": 1, "kilograms": 1,
        "g": 0.001, "gram": 0.001, "grams": 0.001,
        "mg": 1e-6, "milligram": 1e-6, "milligrams": 1e-6,
        "lb": 0.45359237, "lbs": 0.45359237, "pound": 0.45359237, "pounds": 0.45359237,
        "oz": 0.028349523, "ounce": 0.028349523, "ounces": 0.028349523,
        "t": 1000, "tonne": 1000, "metric ton": 1000,
        "stone": 6.35029318, "st": 6.35029318,
    },
    # Volume → litres
    "volume": {
        "l": 1, "litre": 1, "liter": 1, "litres": 1, "liters": 1,
        "ml": 0.001, "millilitre": 0.001, "milliliter": 0.001,
        "cl": 0.01, "centilitre": 0.01, "centiliter": 0.01,
        "gal": 3.78541, "gallon": 3.78541, "gallons": 3.78541,
        "qt": 0.946353, "quart": 0.946353, "quarts": 0.946353,
        "pt": 0.473176, "pint": 0.473176, "pints": 0.473176,
        "fl oz": 0.0295735, "fluid ounce": 0.0295735,
        "cup": 0.236588, "cups": 0.236588,
        "tbsp": 0.0147868, "tablespoon": 0.0147868,
        "tsp": 0.00492892, "teaspoon": 0.00492892,
    },
    # Speed → m/s
    "speed": {
        "m/s": 1, "mps": 1,
        "km/h": 1 / 3.6, "kph": 1 / 3.6, "kmh": 1 / 3.6,
        "mph": 0.44704, "mi/h": 0.44704,
        "knot": 0.514444, "knots": 0.514444, "kn": 0.514444,
        "ft/s": 0.3048, "fps": 0.3048,
    },
}


def _find_unit(token: str):
    """Return (category, canonical_key, factor) or None."""
    t = token.lower().strip()
    for cat, units in _UNITS.items():
        if t in units:
            return cat, t, units[t]
    return None


def _convert_temperature(value: float, src: str, tgt: str) -> float:
    src, tgt = src.lower(), tgt.lower()
    # Convert src → Celsius
    if src in ("f", "fahrenheit", "°f"):
        celsius = (value - 32) * 5 / 9
    elif src in ("k", "kelvin"):
        celsius = value - 273.15
    else:
        celsius = value  # already Celsius
    # Celsius → tgt
    if tgt in ("f", "fahrenheit", "°f"):
        return celsius * 9 / 5 + 32
    elif tgt in ("k", "kelvin"):
        return celsius + 273.15
    return celsius


_TEMP_TOKENS = {"c", "celsius", "°c", "f", "fahrenheit", "°f", "k", "kelvin"}


def _parse(message: str):
    """Return (value, src_unit, tgt_unit) or None."""
    import re
    # "X unit1 to unit2" or "convert X unit1 to unit2" or "X unit1 in unit2"
    m = re.search(
        r'(\d+(?:\.\d+)?)\s+'
        r'([\w/°]+(?:\s+\w+)?)\s+'
        r'(?:to|in|into|as)\s+'
        r'([\w/°]+(?:\s+\w+)?)',
        message, re.IGNORECASE,
    )
    if m:
        return float(m.group(1)), m.group(2).strip(), m.group(3).strip()
    return None


def execute(intent: str, message: str, ctx: dict):
    from prism_responses import text_card

    parsed = _parse(message)
    if parsed is None:
        return text_card(
            "Could not parse conversion request.\n"
            "Examples:\n"
            "  '100 km to miles'\n"
            "  '72 fahrenheit to celsius'\n"
            "  '5 kg in pounds'",
            "Convert",
        )

    value, src_unit, tgt_unit = parsed
    src_l = src_unit.lower()
    tgt_l = tgt_unit.lower()

    # Temperature special case
    if src_l in _TEMP_TOKENS or tgt_l in _TEMP_TOKENS:
        try:
            result = _convert_temperature(value, src_l, tgt_l)
            return text_card(
                f"{value} {src_unit} = {result:.4g} {tgt_unit}", "Convert"
            )
        except Exception:
            return text_card(
                f"Unknown temperature units: {src_unit} or {tgt_unit}", "Convert"
            )

    src_info = _find_unit(src_unit)
    tgt_info = _find_unit(tgt_unit)

    if src_info is None:
        return text_card(f"Unknown source unit: '{src_unit}'", "Convert")
    if tgt_info is None:
        return text_card(f"Unknown target unit: '{tgt_unit}'", "Convert")

    src_cat, _, src_factor = src_info
    tgt_cat, _, tgt_factor = tgt_info

    if src_cat != tgt_cat:
        return text_card(
            f"Cannot convert between {src_cat} and {tgt_cat}.", "Convert"
        )

    # Convert src → SI → tgt
    si_value = value * src_factor
    result = si_value / tgt_factor

    return text_card(
        f"{value} {src_unit} = {result:.6g} {tgt_unit}", "Convert"
    )
