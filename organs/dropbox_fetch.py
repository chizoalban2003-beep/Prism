"""Bundled organ: dropbox_fetch — search Dropbox for files via the search_v2 API."""
ORGAN_META = {
    "intent":      "dropbox_fetch",
    "description": "search Dropbox for files by name or content",
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
    token = (cfg.get("dropbox_token") or "").strip()
    if token:
        return token
    try:
        from pathlib import Path
        env = Path("/proc/self/environ").read_text(errors="replace")
        m = re.search(r'DROPBOX_TOKEN=([^\x00]+)', env)
        if m:
            return m.group(1).strip()
    except Exception:
        pass
    return ""


def _extract_query(message: str) -> str:
    import re
    for pat in [
        r'(?:in\s+)?(?:my\s+)?dropbox\s+(?:for\s+|search\s+(?:for\s+)?|fetch\s+)?(.+)',
        r'(?:search|find|fetch|look\s+up)\s+(?:in\s+)?(?:my\s+)?dropbox\s+(?:for\s+)?(.+)',
        r'dropbox[:\s]+(.+)',
    ]:
        m = re.search(pat, message, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip('?.')
    return message.strip().rstrip('?.')


def execute(intent: str, message: str, ctx: dict):
    import json
    import urllib.request

    from prism_responses import text_card

    token = _get_token(ctx)
    if not token:
        return text_card(
            "Dropbox token not configured.\n"
            "Add documents_config={'dropbox_token': '...'} to ctx\n"
            "or set DROPBOX_TOKEN env var "
            "(create an app at dropbox.com/developers/apps).",
            "Dropbox",
        )

    query = _extract_query(message)
    if not query:
        return text_card("No search query found in message.", "Dropbox")

    payload = json.dumps({
        "query":   query,
        "options": {"max_results": 5, "file_status": "active"},
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.dropboxapi.com/2/files/search_v2",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
            "User-Agent":    "PRISM/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        return text_card(f"Dropbox search failed: {exc}", "Dropbox")

    matches = data.get("matches") or []
    if not matches:
        return text_card(f"No Dropbox files match: {query}", "Dropbox")

    lines = [f"Dropbox — results for: {query}\n"]
    for i, match in enumerate(matches[:5], 1):
        meta = ((match.get("metadata") or {}).get("metadata")) or {}
        title    = meta.get("name") or "(untitled)"
        path     = meta.get("path_display") or meta.get("path_lower") or ""
        modified = meta.get("server_modified") or meta.get("client_modified") or ""
        size_b   = meta.get("size") or 0
        snippet  = f"path: {path}"
        if size_b:
            snippet += f" · {size_b} bytes"
        lines.append(f"{i}. {title}\n   {path}\n   modified: {modified}\n   {snippet}")
    return text_card("\n\n".join(lines), "Dropbox")
