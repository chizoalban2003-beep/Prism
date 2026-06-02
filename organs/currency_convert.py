"""Bundled organ: currency_convert — live exchange rates via open.er-api.com."""
import re as _re

ORGAN_META = {
    "intent":      "currency_convert",
    "description": "converts an amount between currencies using live exchange rates",
    "version":     "1.0",
}


def execute(intent: str, message: str, ctx: dict):
    from prism_responses import text_card
    import urllib.request
    import json as _json

    # Parse "100 USD to EUR" or "convert 50 GBP to JPY"
    m = _re.search(
        r"(\d+(?:\.\d+)?)\s*([A-Za-z]{3})\s+(?:to|in)\s+([A-Za-z]{3})",
        message, _re.I,
    )
    if not m:
        return text_card(
            "Could not parse request. Try: '100 USD to EUR'", intent)

    amount  = float(m.group(1))
    src     = m.group(2).upper()
    dst     = m.group(3).upper()

    url = f"https://open.er-api.com/v6/latest/{src}"
    try:
        with urllib.request.urlopen(url, timeout=6) as resp:
            data = _json.loads(resp.read())
        rate       = data["rates"].get(dst)
        if rate is None:
            return text_card(f"Unknown currency: {dst}", intent)
        converted  = amount * rate
        result     = f"{amount:,.2f} {src} = {converted:,.2f} {dst}  (rate: {rate:.4f})"
    except Exception as exc:
        result = f"Currency conversion failed: {exc}"

    return text_card(result, "Currency")
