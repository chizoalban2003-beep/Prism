"""Bundled organ: telegram_send — send a message via Telegram Bot API."""
ORGAN_META = {
    "intent":      "telegram_send",
    "description": "send a Telegram message using a bot token and chat ID",
    "version":     "1.0",
    "capabilities": ["internet_write"],
}

ORGAN_POLICY = {
    "risk_level":        "high",
    "requires_approval": True,
    "irreversible":      True,
    "max_per_session":   20,
}


def _get_config(ctx: dict) -> tuple:
    """Return (bot_token, chat_id) from ctx or /proc/self/environ."""
    import re
    cfg = ctx.get("telegram_config") or {}
    token = cfg.get("bot_token", "").strip()
    chat_id = str(cfg.get("chat_id", "")).strip()

    if not token or not chat_id:
        try:
            from pathlib import Path
            env = Path("/proc/self/environ").read_text(errors="replace")
            if not token:
                m = re.search(r'TELEGRAM_BOT_TOKEN=([^\x00]+)', env)
                if m:
                    token = m.group(1).strip()
            if not chat_id:
                m = re.search(r'TELEGRAM_CHAT_ID=([^\x00]+)', env)
                if m:
                    chat_id = m.group(1).strip()
        except Exception:
            pass
    return token, chat_id


def _extract_text(message: str) -> str:
    import re
    for pat in [
        r'(?:send|telegram|message)[:\s]+["\'](.+?)["\']',
        r'(?:send|message)\s+(?:to\s+telegram)[:\s]+(.+)',
        r'(?:say|tell)[:\s]+(.+)',
    ]:
        m = re.search(pat, message, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
    return message.strip()


def execute(intent: str, message: str, ctx: dict):
    import json
    import urllib.parse
    import urllib.request

    from prism_responses import text_card

    token, chat_id = _get_config(ctx)
    if not token:
        return text_card(
            "Telegram bot token not configured.\n"
            "Add telegram_config={'bot_token': '...', 'chat_id': '...'} to ctx\n"
            "or set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars.",
            "Telegram",
        )
    if not chat_id:
        return text_card(
            "Telegram chat_id not configured.\n"
            "Add telegram_config={'bot_token': '...', 'chat_id': '...'} to ctx\n"
            "or set TELEGRAM_CHAT_ID env var.",
            "Telegram",
        )

    text = _extract_text(message)
    if not text:
        return text_card("No message content found.", "Telegram")
    text = text[:4096]  # Telegram message limit

    api_url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({"chat_id": chat_id, "text": text}).encode("utf-8")
    req = urllib.request.Request(
        api_url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json", "User-Agent": "PRISM/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        return text_card(f"Failed to send Telegram message: {exc}", "Telegram")

    if data.get("ok"):
        msg_id = data.get("result", {}).get("message_id", "?")
        return text_card(
            f"Telegram message sent (ID: {msg_id}, {len(text)} chars).", "Telegram"
        )
    err = data.get("description", "Unknown error")
    return text_card(f"Telegram API error: {err}", "Telegram")
