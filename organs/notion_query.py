"""Bundled organ: notion_query — search a Notion workspace via the Search API."""
ORGAN_META = {
    "intent":      "notion_query",
    "description": "search a Notion workspace for pages and databases matching a query",
    "version":     "1.0",
    "capabilities": ["internet_read"],
    "inputs": {
        "query": "str",
    },
    "outputs": {
        "hits": "list[{title:str,url:str,modified:str,snippet:str}]",
    },
}

ORGAN_POLICY = {
    "risk_level":        "low",
    "requires_approval": False,
    "irreversible":      False,
    "max_per_session":   None,
}


def _get_token(ctx: dict) -> str:
    import re
    cfg = ctx.get("documents_config") or {}
    token = (cfg.get("notion_token") or "").strip()
    if token:
        return token
    try:
        from pathlib import Path
        env = Path("/proc/self/environ").read_text(errors="replace")
        m = re.search(r'NOTION_TOKEN=([^\x00]+)', env)
        if m:
            return m.group(1).strip()
    except Exception:
        pass
    return ""


def _extract_query(message: str) -> str:
    import re
    for pat in [
        r'(?:in\s+)?notion\s+(?:for\s+|search\s+(?:for\s+)?|query\s+(?:for\s+)?|about\s+)?(.+)',
        r'(?:search|find|query|look\s+up)\s+(?:in\s+)?notion\s+(?:for\s+)?(.+)',
        r'notion[:\s]+(.+)',
    ]:
        m = re.search(pat, message, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip('?.')
    return message.strip().rstrip('?.')


def _extract_title(page: dict) -> str:
    """Notion returns title under various property shapes; pick the first plain_text."""
    props = page.get("properties") or {}
    for prop in props.values():
        if not isinstance(prop, dict):
            continue
        if prop.get("type") != "title":
            continue
        chunks = prop.get("title") or []
        text = "".join(c.get("plain_text", "") for c in chunks if isinstance(c, dict))
        if text:
            return text
    # Databases have a top-level title array instead
    chunks = page.get("title") or []
    if isinstance(chunks, list):
        text = "".join(c.get("plain_text", "") for c in chunks if isinstance(c, dict))
        if text:
            return text
    return "(untitled)"


def execute(intent: str, message: str, ctx: dict):
    import json
    import urllib.request

    from prism_responses import text_card

    token = _get_token(ctx)
    if not token:
        return text_card(
            "Notion token not configured.\n"
            "Add documents_config={'notion_token': 'secret_...'} to ctx\n"
            "or set NOTION_TOKEN env var "
            "(create an integration at notion.so/my-integrations).",
            "Notion",
        )

    query = _extract_query(message)
    if not query:
        return text_card("No search query found in message.", "Notion")

    payload = json.dumps({"query": query, "page_size": 5}).encode("utf-8")
    req = urllib.request.Request(
        "https://api.notion.com/v1/search",
        data=payload,
        method="POST",
        headers={
            "Authorization":  f"Bearer {token}",
            "Notion-Version": "2022-06-28",
            "Content-Type":   "application/json",
            "User-Agent":     "PRISM/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        return text_card(f"Notion search failed: {exc}", "Notion")

    results = data.get("results") or []
    if not results:
        return text_card(f"No Notion pages match: {query}", "Notion")

    lines = [f"Notion — results for: {query}\n"]
    for i, page in enumerate(results[:5], 1):
        title    = _extract_title(page)
        url      = page.get("url") or ""
        modified = page.get("last_edited_time") or ""
        obj_kind = page.get("object") or ""
        snippet  = f"kind: {obj_kind}" if obj_kind else ""
        lines.append(f"{i}. {title}\n   {url}\n   modified: {modified}\n   {snippet}")
    return text_card("\n\n".join(lines), "Notion")
