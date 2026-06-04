"""Bundled organ: web_search — search the web via DuckDuckGo Lite (no API key)."""
ORGAN_META = {
    "intent":      "web_search",
    "description": "search the web using DuckDuckGo and return top results",
    "version":     "1.0",
    "capabilities": ["internet_read"],
}

ORGAN_POLICY = {
    "risk_level":        "low",
    "requires_approval": False,
    "irreversible":      False,
    "max_per_session":   None,
}


def _extract_query(message: str) -> str:
    import re
    for pat in [
        r'search\s+(?:the\s+)?(?:web|internet|online)\s+for\s+(.+)',
        r'(?:search\s+for|look\s+up|find\s+(?:out|info|information)\s+(?:about|on)|google)[:\s]+(.+)',
        r'(?:what\s+is|who\s+is|where\s+is|when\s+(?:did|does|is)|how\s+(?:do|does|to))\s+(.+)',
        r'(?:research|find)\s+(.+)',
    ]:
        m = re.search(pat, message, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip('?.')
    return message.strip()


def _strip_tags(html: str) -> str:
    import re
    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'&#x27;', "'", text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def execute(intent: str, message: str, ctx: dict):
    import re
    import urllib.parse
    import urllib.request

    from prism_responses import text_card

    query = _extract_query(message)
    if not query:
        return text_card("No search query found in message.", "Web Search")

    url = "https://lite.duckduckgo.com/lite/"
    data = urllib.parse.urlencode({"q": query}).encode()
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; PRISM/1.0)",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return text_card(f"Search failed: {exc}", "Web Search")

    # Parse result links and snippets from DDG Lite HTML
    results = []
    # DDG Lite wraps results in <a class="result-link"> and <td class="result-snippet">
    link_pattern = re.compile(
        r"""href=["'](https?://[^"']+)["']\s+class=["']result-link["']>(.+?)</a>"""
        r"""|class=["']result-link["']\s+href=["'](https?://[^"']+)["']>(.+?)</a>""",
        re.IGNORECASE | re.DOTALL,
    )
    snippet_pattern = re.compile(
        r"""class=["']result-snippet["'][^>]*>(.+?)</td>""",
        re.IGNORECASE | re.DOTALL,
    )
    links = link_pattern.findall(html)
    snippets = [_strip_tags(s) for s in snippet_pattern.findall(html)]

    # Normalise: each match has 4 groups (two alternations); pick the non-empty pair
    links = [(g[0] or g[2], g[1] or g[3]) for g in links]
    for i, (href, title) in enumerate(links[:5]):
        title_clean = _strip_tags(title).strip()
        snippet = snippets[i] if i < len(snippets) else ""
        results.append(f"{i+1}. {title_clean}\n   {href}\n   {snippet}")

    if not results:
        return text_card(
            f"No results found for: {query}\n"
            "(DuckDuckGo Lite may have changed its HTML structure.)",
            "Web Search",
        )

    output = f"Search results for: {query}\n\n" + "\n\n".join(results)
    return text_card(output, "Web Search")
