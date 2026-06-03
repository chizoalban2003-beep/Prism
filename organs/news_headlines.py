"""Bundled organ: news_headlines — fetch top news headlines from BBC RSS feed."""
ORGAN_META = {
    "intent":      "news_headlines",
    "description": "fetch and display the latest news headlines from BBC RSS",
    "version":     "1.0",
}

ORGAN_POLICY = {
    "risk_level":        "low",
    "requires_approval": False,
    "irreversible":      False,
    "max_per_session":   None,
}

_FEEDS = {
    "world":       "https://feeds.bbci.co.uk/news/world/rss.xml",
    "technology":  "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "science":     "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "business":    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "health":      "https://feeds.bbci.co.uk/news/health/rss.xml",
    "sport":       "https://feeds.bbci.co.uk/sport/rss.xml",
    "default":     "https://feeds.bbci.co.uk/news/rss.xml",
}


def _pick_feed(message: str) -> tuple:
    """Return (feed_url, category_label)."""
    msg_lower = message.lower()
    for key in _FEEDS:
        if key != "default" and key in msg_lower:
            return _FEEDS[key], key.title()
    return _FEEDS["default"], "Top"


def execute(intent: str, message: str, ctx: dict):
    import urllib.request
    import xml.etree.ElementTree as ET

    from prism_responses import text_card

    feed_url, category = _pick_feed(message)
    req = urllib.request.Request(
        feed_url,
        headers={"User-Agent": "PRISM/1.0 (RSS reader)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            xml_bytes = resp.read()
    except Exception as exc:
        return text_card(f"Failed to fetch news feed: {exc}", "News")

    try:
        root = ET.fromstring(xml_bytes)
    except Exception as exc:
        return text_card(f"Failed to parse RSS feed: {exc}", "News")

    # RSS: channel > item > title, description, link, pubDate
    ns = {}
    channel = root.find("channel")
    if channel is None:
        # Atom feed fallback
        channel = root

    items = channel.findall("item") if channel is not None else []
    if not items:
        return text_card("No news items found in feed.", "News")

    headlines = []
    for item in items[:10]:
        title = (item.findtext("title") or "").strip()
        desc = (item.findtext("description") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()

        # Strip CDATA markers and HTML from description
        import re
        desc = re.sub(r'<[^>]+>', '', desc)[:120].strip()

        entry = f"• {title}"
        if pub:
            entry += f"  [{pub[:16]}]"
        if desc:
            entry += f"\n  {desc}"
        if link:
            entry += f"\n  {link}"
        headlines.append(entry)

    result = f"{category} News Headlines\n\n" + "\n\n".join(headlines)
    return text_card(result, "News")
