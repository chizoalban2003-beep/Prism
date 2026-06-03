"""Bundled organ: wikipedia_lookup — fetch a Wikipedia article summary."""
ORGAN_META = {
    "intent":      "wikipedia_lookup",
    "description": "look up a topic on Wikipedia and return its summary",
    "version":     "1.0",
}

ORGAN_POLICY = {
    "risk_level":        "low",
    "requires_approval": False,
    "irreversible":      False,
    "max_per_session":   None,
}


def _extract_topic(message: str) -> str:
    import re
    for pat in [
        r'(?:wikipedia|wiki|look\s*up|tell\s+me\s+about|what\s+is|who\s+is)'
        r'[:\s]+(.+)',
        r'(?:search|find)\s+(?:wikipedia\s+)?(?:for\s+)?(.+)',
    ]:
        m = re.search(pat, message, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip('?.')
    return message.strip().rstrip('?.')


def execute(intent: str, message: str, ctx: dict):
    import json
    import re
    import urllib.parse
    import urllib.request

    from prism_responses import text_card

    topic = _extract_topic(message)
    if not topic:
        return text_card("No topic found in message.", "Wikipedia")

    # Try exact title first, then search
    def fetch_summary(title: str):
        encoded = urllib.parse.quote(re.sub(r' ', '_', title))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "PRISM/1.0 (local AI assistant)"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.loads(resp.read())

    try:
        data = fetch_summary(topic)
    except Exception:
        # Fall back to search API
        try:
            search_url = (
                "https://en.wikipedia.org/w/api.php?action=query&list=search"
                f"&srsearch={urllib.parse.quote(topic)}&format=json&srlimit=1"
            )
            req = urllib.request.Request(
                search_url,
                headers={"User-Agent": "PRISM/1.0 (local AI assistant)"},
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                search_data = json.loads(resp.read())
            hits = search_data.get("query", {}).get("search", [])
            if not hits:
                return text_card(f"No Wikipedia article found for: {topic}", "Wikipedia")
            best_title = hits[0]["title"]
            data = fetch_summary(best_title)
        except Exception as exc:
            return text_card(f"Wikipedia lookup failed: {exc}", "Wikipedia")

    if data.get("type") == "disambiguation":
        desc = data.get("extract", "This is a disambiguation page.")
        return text_card(f"Wikipedia — {data.get('title', topic)}\n\n{desc}", "Wikipedia")

    title   = data.get("displaytitle") or data.get("title", topic)
    extract = data.get("extract", "No summary available.")
    page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")

    result = f"Wikipedia — {title}\n\n{extract}"
    if page_url:
        result += f"\n\nFull article: {page_url}"
    return text_card(result, "Wikipedia")
