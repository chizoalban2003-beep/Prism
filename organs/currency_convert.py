"""Bundled organ: currency_convert — live exchange rates via open.er-api.com."""
import re as _re

ORGAN_META = {
    "intent":      "currency_convert",
    "description": "converts an amount between currencies using live exchange rates",
    "version":     "1.0",
    "capabilities": ["internet_read"],
    "inputs": {
        "amount": "float",
    },
    "outputs": {
        "amount":         "float",
        "converted":      "float",
        "src_currency":   "str",
        "dst_currency":   "str",
        "rate":           "float",
    },
}

ORGAN_POLICY = {
    "risk_level":        "low",
    "requires_approval": False,
    "irreversible":      False,
    "max_per_session":   None,
}

# Spelled-out currency names → ISO 4217 codes
_CURRENCY_NAMES: dict[str, str] = {
    "us dollar": "USD", "us dollars": "USD", "american dollar": "USD",
    "dollar": "USD", "dollars": "USD",
    "euro": "EUR", "euros": "EUR",
    "british pound": "GBP", "british pounds": "GBP", "pound sterling": "GBP",
    "pound": "GBP", "pounds": "GBP", "sterling": "GBP",
    "japanese yen": "JPY", "yen": "JPY",
    "chinese yuan": "CNY", "yuan": "CNY", "renminbi": "CNY",
    "swiss franc": "CHF", "franc": "CHF", "francs": "CHF",
    "canadian dollar": "CAD", "canadian dollars": "CAD",
    "australian dollar": "AUD", "australian dollars": "AUD",
    "hong kong dollar": "HKD",
    "new zealand dollar": "NZD",
    "swedish krona": "SEK", "krona": "SEK",
    "norwegian krone": "NOK", "danish krone": "DKK", "krone": "NOK",
    "indian rupee": "INR", "rupee": "INR", "rupees": "INR",
    "south korean won": "KRW", "won": "KRW",
    "singapore dollar": "SGD",
    "mexican peso": "MXN", "peso": "MXN", "pesos": "MXN",
    "brazilian real": "BRL", "real": "BRL",
    "russian ruble": "RUB", "ruble": "RUB", "rubles": "RUB",
    "turkish lira": "TRY", "lira": "TRY",
    "south african rand": "ZAR", "rand": "ZAR",
    "polish zloty": "PLN", "zloty": "PLN",
    "thai baht": "THB", "baht": "THB",
    "indonesian rupiah": "IDR", "rupiah": "IDR",
    "malaysian ringgit": "MYR", "ringgit": "MYR",
    "philippine peso": "PHP",
    "czech koruna": "CZK", "koruna": "CZK",
    "hungarian forint": "HUF", "forint": "HUF",
    "israeli shekel": "ILS", "shekel": "ILS", "shekels": "ILS",
    "uae dirham": "AED", "dirham": "AED",
    "saudi riyal": "SAR", "riyal": "SAR",
    "kuwaiti dinar": "KWD", "dinar": "KWD",
    "nigerian naira": "NGN", "naira": "NGN",
    "egyptian pound": "EGP",
    "pakistani rupee": "PKR",
    "bangladeshi taka": "BDT", "taka": "BDT",
    "vietnamese dong": "VND", "dong": "VND",
}


def _fmt_amount(v: float) -> str:
    # ``:,.4g`` (the original) collapses to scientific notation for any
    # value >= 1e4, so "50 GBP to JPY" rendered as "1.065e+04 JPY".
    # Fixed-point for ordinary amounts; fall back to 6 significant digits
    # only for sub-unit conversions where decimals matter.
    if abs(v) >= 1:
        return f"{v:,.2f}"
    return f"{v:.6g}"


def _resolve_currency(token: str) -> str:
    """Resolve a token to an ISO 4217 code. Returns the token uppercased if unknown."""
    t = token.strip().lower()
    if t in _CURRENCY_NAMES:
        return _CURRENCY_NAMES[t]
    # Try 3-letter ISO code
    if len(token) == 3 and token.isalpha():
        return token.upper()
    return token.upper()


def _parse(message: str):
    """
    Return (amount, src_iso, dst_iso) or None.
    Handles both ISO codes ("100 USD to EUR") and spelled-out names
    ("1500 Japanese Yen to British Pounds").
    """
    msg = message.lower()

    # Build a combined pattern: amount + (ISO | known name) + to/in/into + (ISO | known name)
    # Sort names longest-first to avoid partial matches
    name_pattern = "|".join(
        _re.escape(n) for n in sorted(_CURRENCY_NAMES, key=len, reverse=True)
    )
    iso_pattern = r"[a-z]{3}"
    currency_pat = f"(?:{name_pattern}|{iso_pattern})"

    m = _re.search(
        rf"(\d+(?:[\.,]\d+)?)\s+({currency_pat})\s+(?:to|in|into)\s+({currency_pat})",
        msg,
    )
    if m:
        amount = float(m.group(1).replace(",", ""))
        return amount, _resolve_currency(m.group(2)), _resolve_currency(m.group(3))

    # "convert X from SRC to DST"
    m = _re.search(
        rf"(\d+(?:[\.,]\d+)?)\s+(?:from\s+)?({currency_pat})\s+(?:to|in|into)\s+({currency_pat})",
        msg,
    )
    if m:
        amount = float(m.group(1).replace(",", ""))
        return amount, _resolve_currency(m.group(2)), _resolve_currency(m.group(3))

    return None


def execute(intent: str, message: str, ctx: dict):
    import json as _json
    import urllib.request

    from prism_responses import text_card

    parsed = _parse(message)
    if not parsed:
        return text_card(
            "Could not parse request.\n"
            "Examples:\n"
            "  '100 USD to EUR'\n"
            "  '1500 Japanese Yen to British Pounds'\n"
            "  '50 euros to dollars'",
            "Currency",
        )

    amount, src, dst = parsed
    url = f"https://open.er-api.com/v6/latest/{src}"
    structured: dict = {"amount": amount, "src_currency": src, "dst_currency": dst}
    try:
        with urllib.request.urlopen(url, timeout=6) as resp:
            data = _json.loads(resp.read())
        if data.get("result") == "error":
            return text_card(f"Unknown currency code: {src}", "Currency")
        rate = data["rates"].get(dst)
        if rate is None:
            return text_card(f"Unknown target currency: {dst}", "Currency")
        converted = amount * rate
        result = f"{amount:,.2f} {src} = {_fmt_amount(converted)} {dst}  (rate: {rate:.6g})"
        structured.update({"converted": float(converted), "rate": float(rate)})
    except Exception as exc:
        result = f"Currency conversion failed: {exc}"
        structured["error"] = str(exc)

    card = text_card(result, "Currency")
    card.card_data.update(structured)
    return card
