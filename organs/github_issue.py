"""Bundled organ: github_issue — create or list GitHub issues via REST API."""
ORGAN_META = {
    "intent":      "github_issue",
    "description": "create a new GitHub issue or list open issues in a repo",
    "version":     "1.0",
}

ORGAN_POLICY = {
    "risk_level":        "medium",
    "requires_approval": True,
    "irreversible":      False,
    "max_per_session":   None,
}


def _get_config(ctx: dict) -> tuple:
    """Return (token, repo) where repo is 'owner/repo'."""
    import re
    cfg = ctx.get("github_config") or {}
    token = cfg.get("token", "").strip()
    repo = cfg.get("repo", "").strip()
    if not token:
        try:
            from pathlib import Path
            env = Path("/proc/self/environ").read_text(errors="replace")
            m = re.search(r'GITHUB_TOKEN=([^\x00]+)', env)
            if m:
                token = m.group(1).strip()
            m = re.search(r'GITHUB_REPO=([^\x00]+)', env)
            if m and not repo:
                repo = m.group(1).strip()
        except Exception:
            pass
    return token, repo


def _parse_action(message: str) -> str:
    msg = message.lower()
    if any(w in msg for w in ("create", "open", "new", "file", "add", "report")):
        return "create"
    return "list"


def _parse_issue_fields(message: str) -> tuple:
    """Return (title, body)."""
    import re
    m = re.search(r'title[:\s]+["\']?(.+?)["\']?(?:\n|body|description|$)',
                  message, re.IGNORECASE | re.DOTALL)
    title = m.group(1).strip() if m else ""

    m2 = re.search(r'(?:body|description)[:\s]+(.+)', message,
                   re.IGNORECASE | re.DOTALL)
    body = m2.group(1).strip() if m2 else ""

    if not title:
        # Use first sentence as title
        sentences = re.split(r'[.!?\n]', message.strip())
        title = sentences[0].strip()[:200]
    return title, body


def execute(intent: str, message: str, ctx: dict):
    import json
    import urllib.request

    from prism_responses import text_card

    token, repo = _get_config(ctx)
    if not token:
        return text_card(
            "GitHub token not configured.\n"
            "Add github_config={'token': 'ghp_...', 'repo': 'owner/repo'} to ctx\n"
            "or set GITHUB_TOKEN and GITHUB_REPO env vars.",
            "GitHub",
        )
    if not repo:
        return text_card(
            "GitHub repo not configured.\n"
            "Add github_config={'repo': 'owner/repo'} to ctx or set GITHUB_REPO.",
            "GitHub",
        )

    action = _parse_action(message)
    api_base = f"https://api.github.com/repos/{repo}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "PRISM/1.0",
    }

    if action == "list":
        req = urllib.request.Request(
            f"{api_base}/issues?state=open&per_page=10",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                issues = json.loads(resp.read())
        except Exception as exc:
            return text_card(f"Failed to list issues: {exc}", "GitHub")

        if not issues:
            return text_card(f"No open issues in {repo}.", "GitHub")

        lines = [f"Open issues in {repo}:\n"]
        for issue in issues[:10]:
            num = issue.get("number", "?")
            title = issue.get("title", "Untitled")
            url = issue.get("html_url", "")
            lines.append(f"#{num}: {title}\n  {url}")
        return text_card("\n".join(lines), "GitHub")

    # Create
    title, body = _parse_issue_fields(message)
    payload = json.dumps({"title": title, "body": body}).encode("utf-8")
    req = urllib.request.Request(
        f"{api_base}/issues",
        data=payload,
        method="POST",
        headers={**headers, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            issue = json.loads(resp.read())
    except Exception as exc:
        return text_card(f"Failed to create issue: {exc}", "GitHub")

    num = issue.get("number", "?")
    url = issue.get("html_url", "")
    return text_card(
        f"GitHub issue #{num} created in {repo}.\n"
        f"Title: {title}\n"
        f"URL: {url}",
        "GitHub",
    )
