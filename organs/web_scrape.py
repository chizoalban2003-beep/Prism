"""Bundled organ: web_scrape — fetch a URL and return its cleaned text content."""
ORGAN_META = {
    "intent":      "web_scrape",
    "description": "fetch and return readable text from a URL",
    "version":     "1.0",
    "capabilities": ["internet_read"],
}

ORGAN_POLICY = {
    "risk_level":        "low",
    "requires_approval": False,
    "irreversible":      False,
    "max_per_session":   None,
}

_MAX_CHARS = 3000


def _strip_html(html: str) -> str:
    import re
    # Remove script/style blocks entirely
    html = re.sub(r'<(script|style)[^>]*>.*?</\1>', ' ', html,
                  flags=re.IGNORECASE | re.DOTALL)
    # Remove all tags
    text = re.sub(r'<[^>]+>', ' ', html)
    # Decode common HTML entities
    import re as _re2
    entity_map = [
        ('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>'),
        ('&quot;', '"'), ('&#x27;', "'"), ('&nbsp;', ' '),
        ('&#39;', "'"),
    ]
    for esc, char in entity_map:
        text = _re2.sub(_re2.escape(esc), char, text)
    # Collapse whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _extract_url(message: str) -> str:
    import re
    m = re.search(r'https?://[^\s"\'<>]+', message)
    return m.group(0).rstrip('.,;') if m else ""


def execute(intent: str, message: str, ctx: dict):
    import urllib.request

    from prism_responses import text_card

    url = _extract_url(message)
    if not url:
        return text_card(
            "No URL found in message. Include a full URL starting with http:// or https://.",
            "Web Scrape",
        )

    # SSRF guard: the URL is attacker-influenced (it comes straight from the
    # user/LLM message), so refuse loopback, RFC1918, link-local and cloud
    # metadata targets before issuing the request.
    try:
        from prism_ssrf import is_safe_external_url
        if not is_safe_external_url(url, allow_private=False):
            return text_card(
                "Refusing to fetch that URL — it resolves to a private, loopback, "
                "or otherwise blocked address (SSRF protection).",
                "Web Scrape",
            )
    except ImportError:
        pass

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; PRISM/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "text" not in content_type and "html" not in content_type:
                return text_card(
                    f"URL returned non-text content ({content_type}). Only HTML/text pages "
                    "are supported.",
                    "Web Scrape",
                )
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return text_card(f"Failed to fetch URL: {exc}", "Web Scrape")

    text = _strip_html(raw)
    if len(text) > _MAX_CHARS:
        text = text[:_MAX_CHARS] + f"\n\n[...truncated at {_MAX_CHARS} chars]"

    if not text:
        return text_card("Page returned no readable text.", "Web Scrape")

    return text_card(f"Content from {url}:\n\n{text}", "Web Scrape")
