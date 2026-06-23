"""Bundled organ: gdrive_search — search Google Drive for files."""
ORGAN_META = {
    "intent":      "gdrive_search",
    "description": "search Google Drive for files by name or full-text content",
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
    token = (cfg.get("gdrive_token") or "").strip()
    if token:
        return token
    try:
        from pathlib import Path
        env = Path("/proc/self/environ").read_text(errors="replace")
        m = re.search(r'GDRIVE_TOKEN=([^\x00]+)', env)
        if m:
            return m.group(1).strip()
    except Exception:
        pass
    return ""


def _extract_query(message: str) -> str:
    import re
    for pat in [
        r'(?:in\s+)?(?:my\s+)?(?:g(?:oogle)?\s*drive|drive)\s+(?:for\s+|search\s+(?:for\s+)?)?(.+)',
        r'(?:search|find|look\s+up)\s+(?:in\s+)?(?:my\s+)?(?:g(?:oogle)?\s*drive)\s+(?:for\s+)?(.+)',
        r'(?:gdrive|drive)[:\s]+(.+)',
    ]:
        m = re.search(pat, message, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip('?.')
    return message.strip().rstrip('?.')


def execute(intent: str, message: str, ctx: dict):
    import json
    import urllib.parse
    import urllib.request

    from prism_responses import text_card

    token = _get_token(ctx)
    if not token:
        return text_card(
            "Google Drive token not configured.\n"
            "Add documents_config={'gdrive_token': '...'} to ctx\n"
            "or set GDRIVE_TOKEN env var (OAuth2 access token).",
            "Drive",
        )

    query = _extract_query(message)
    if not query:
        return text_card("No search query found in message.", "Drive")

    safe_query = query.replace("'", "\\'")
    drive_q = f"name contains '{safe_query}' or fullText contains '{safe_query}'"
    params = {
        "q":        drive_q,
        "fields":   "files(id,name,modifiedTime,webViewLink,mimeType)",
        "pageSize": "5",
    }
    url = "https://www.googleapis.com/drive/v3/files?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent":    "PRISM/1.0",
            "Accept":        "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        return text_card(f"Drive search failed: {exc}", "Drive")

    files = data.get("files") or []
    if not files:
        return text_card(f"No Drive files match: {query}", "Drive")

    lines = [f"Google Drive — results for: {query}\n"]
    for i, f in enumerate(files[:5], 1):
        title    = f.get("name") or "(untitled)"
        link     = f.get("webViewLink") or ""
        modified = f.get("modifiedTime") or ""
        kind     = (f.get("mimeType") or "").rsplit(".", 1)[-1] or ""
        snippet  = f"type: {kind}" if kind else ""
        lines.append(f"{i}. {title}\n   {link}\n   modified: {modified}\n   {snippet}")
    return text_card("\n\n".join(lines), "Drive")
