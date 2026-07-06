"""Bundled organ: notify_desktop — raise a local notification, credential-free.

Unlike phone/SMS/email, this needs no third-party account. It fires a native
desktop popup when a notifier is installed and always records the alert to
PRISM's local inbox (~/.prism/notifications.jsonl), so "notify me when ..." /
"alert me: ..." works on any box — including headless ones, where the alert is
logged for the UI/proactive layer to surface.
"""
ORGAN_META = {
    "intent":      "notify_desktop",
    "description": "raise a local desktop notification / alert (no credentials "
                   "needed) and record it to PRISM's notification inbox",
    "version":     "1.0",
    "capabilities": ["system_ui"],
    "inputs":  {"title": "str", "body": "str"},
    "outputs": {"popup": "str", "logged": "bool"},
}

ORGAN_POLICY = {
    "risk_level":        "low",
    "requires_approval": False,
    "irreversible":      False,
    "max_per_session":   None,
}


def _parse(message: str) -> tuple[str, str]:
    """Return (title, body). Body is the text after the notify verb; title is a
    short lead-in or the default 'PRISM'."""
    import re

    m = (message or "").strip()
    body_match = re.search(
        r"(?:notify|alert|remind|tell|ping|pop\s*up)\s+"
        r"(?:me\b\s*)?(?:that\s+|about\s+|with\s+|:\s*)?(.+)",
        m, re.IGNORECASE)
    body = body_match.group(1).strip() if body_match else m
    # a bare pronoun left over from "notify me" is not a body
    body = re.sub(r"^(me|myself|us)\b\s*", "", body, flags=re.IGNORECASE).strip()
    body = body.strip("\"'").strip()
    # a leading "X:" becomes the title
    title = "PRISM"
    lead = re.match(r"([\w ]{1,24}):\s+(.+)", body)
    if lead:
        title, body = lead.group(1).strip(), lead.group(2).strip()
    return title or "PRISM", body


def execute(intent: str, message: str, ctx: dict):
    from prism_responses import text_card

    title, body = _parse(message)
    if not body:
        return text_card("What should I notify you about?", "Notify")

    try:
        from prism_local_notify import deliver
    except ImportError as exc:
        return text_card(f"Notification backend unavailable: {exc}", "Notify")

    report = deliver(title, body, source="notify_desktop")
    if report["popup"]:
        line = f"Notified (popup via {report['popup']}): {body}"
    elif report["logged"]:
        line = (f"Logged to your PRISM inbox: {body}\n"
                "(No desktop notifier installed — install libnotify-bin for "
                "native popups.)")
    else:
        line = f"Could not deliver the notification: {body}"
    card = text_card(line, "Notify")
    card.card_data.update({"title": title, "body": body,
                           "popup": report["popup"], "logged": report["logged"]})
    return card
