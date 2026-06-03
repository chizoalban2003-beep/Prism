"""Bundled organ: discord_send — send a message to a Discord webhook."""
ORGAN_META = {
    "intent":      "discord_send",
    "description": "send a message to a Discord channel via webhook URL",
    "version":     "1.0",
}

ORGAN_POLICY = {
    "risk_level":        "high",
    "requires_approval": True,
    "irreversible":      True,
    "max_per_session":   20,
}

_ENV_KEY = "DISCORD_WEBHOOK_URL"


def _get_webhook(ctx: dict) -> str:
    import re
    webhook = ctx.get("discord_webhook", "").strip()
    if webhook:
        return webhook
    # Try env via pathlib trick (read from /proc/self/environ)
    try:
        from pathlib import Path
        env_text = Path("/proc/self/environ").read_text(errors="replace")
        m = re.search(rf'{_ENV_KEY}=([^\x00]+)', env_text)
        if m:
            return m.group(1).strip()
    except Exception:
        pass
    return ""


def _extract_message_text(message: str) -> str:
    import re
    for pat in [
        r'(?:send|post|discord)[:\s]+["\'](.+?)["\']',
        r'(?:send|post)\s+(?:to\s+discord)[:\s]+(.+)',
        r'(?:say|message|tell)[:\s]+(.+)',
    ]:
        m = re.search(pat, message, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
    return message.strip()


def execute(intent: str, message: str, ctx: dict):
    import json
    import urllib.request

    from prism_responses import text_card

    webhook_url = _get_webhook(ctx)
    if not webhook_url:
        return text_card(
            "Discord webhook URL not configured.\n"
            f"Set ctx['discord_webhook'] or the {_ENV_KEY} environment variable.",
            "Discord",
        )

    content = _extract_message_text(message)
    if not content:
        return text_card("No message content found.", "Discord")

    content = content[:2000]  # Discord message limit

    payload = json.dumps({"content": content}).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json", "User-Agent": "PRISM/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            status = resp.status
    except Exception as exc:
        return text_card(f"Failed to send Discord message: {exc}", "Discord")

    if status in (200, 204):
        return text_card(f"Discord message sent ({len(content)} chars).", "Discord")
    return text_card(f"Discord webhook returned HTTP {status}.", "Discord")
