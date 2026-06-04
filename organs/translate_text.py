"""Bundled organ: translate_text — translate text using the MyMemory free API."""
ORGAN_META = {
    "intent":      "translate_text",
    "description": "translate text between languages using the MyMemory API",
    "version":     "1.0",
    "capabilities": ["internet_read"],
}

ORGAN_POLICY = {
    "risk_level":        "low",
    "requires_approval": False,
    "irreversible":      False,
    "max_per_session":   None,
}

_LANG_CODES = {
    "english": "en", "spanish": "es", "french": "fr", "german": "de",
    "italian": "it", "portuguese": "pt", "dutch": "nl", "russian": "ru",
    "japanese": "ja", "chinese": "zh", "korean": "ko", "arabic": "ar",
    "hindi": "hi", "turkish": "tr", "polish": "pl", "swedish": "sv",
    "norwegian": "no", "danish": "da", "finnish": "fi", "greek": "el",
    "czech": "cs", "romanian": "ro", "hungarian": "hu", "thai": "th",
    "vietnamese": "vi", "indonesian": "id", "malay": "ms", "hebrew": "he",
    "ukrainian": "uk", "catalan": "ca",
}


def _parse_translation_request(message: str):
    """Return (text, source_lang_code, target_lang_code)."""
    import re
    # Pattern: translate "text" from LANG to LANG
    m = re.search(
        r'translate\s+["\']?(.+?)["\']?\s+from\s+(\w+)\s+to\s+(\w+)',
        message, re.IGNORECASE,
    )
    if m:
        text = m.group(1).strip()
        src = _LANG_CODES.get(m.group(2).lower(), m.group(2).lower()[:2])
        tgt = _LANG_CODES.get(m.group(3).lower(), m.group(3).lower()[:2])
        return text, src, tgt

    # Pattern: translate "text" to LANG (auto-detect source)
    m = re.search(
        r'translate\s+["\']?(.+?)["\']?\s+(?:in)?to\s+(\w+)',
        message, re.IGNORECASE,
    )
    if m:
        text = m.group(1).strip()
        tgt = _LANG_CODES.get(m.group(2).lower(), m.group(2).lower()[:2])
        return text, "autodetect", tgt

    # Fallback: everything after "translate" is text, target defaults to English
    m = re.search(r'translate\s+(.+)', message, re.IGNORECASE)
    if m:
        return m.group(1).strip(), "autodetect", "en"

    return message.strip(), "autodetect", "en"


def execute(intent: str, message: str, ctx: dict):
    import json
    import urllib.parse
    import urllib.request

    from prism_responses import text_card

    text, src, tgt = _parse_translation_request(message)
    if not text:
        return text_card("No text to translate found in message.", "Translate")

    lang_pair = f"{src}|{tgt}"
    params = urllib.parse.urlencode({
        "q": text[:500],
        "langpair": lang_pair,
    })
    url = f"https://api.mymemory.translated.net/get?{params}"

    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "PRISM/1.0"}
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        return text_card(f"Translation failed: {exc}", "Translate")

    status = data.get("responseStatus", 0)
    if status != 200:
        msg = data.get("responseDetails", "Unknown error")
        return text_card(f"Translation API error ({status}): {msg}", "Translate")

    translated = data.get("responseData", {}).get("translatedText", "")
    match_quality = data.get("responseData", {}).get("match", 0)

    if not translated:
        return text_card("Translation returned empty result.", "Translate")

    result = (
        f"Original ({src}): {text}\n\n"
        f"Translated ({tgt}): {translated}\n\n"
        f"Match quality: {match_quality:.0%}" if isinstance(match_quality, float)
        else f"Original ({src}): {text}\n\nTranslated ({tgt}): {translated}"
    )
    return text_card(result, "Translate")
