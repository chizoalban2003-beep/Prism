"""Bundled organ: email_send — compose and send an email via the configured SMTP account."""
ORGAN_META = {
    "intent": "email_send",
    "description": "Send an email to a recipient — extracts to/subject/body from the user's message",
    "version": "1.0",
    "capabilities": ["internet_write"],
}

ORGAN_POLICY = {
    "risk_level": "high",
    "requires_approval": True,
    "irreversible": True,
    "max_per_session": 5,
}


def _resolve_contact_email(name_or_address: str, contacts) -> str:
    """Return a confirmed email address — look up by name if no @ present."""
    if not name_or_address or "@" in name_or_address:
        return name_or_address
    if contacts is None:
        return name_or_address
    try:
        hits = contacts.search(name_or_address)
        for c in hits:
            if c.emails:
                return c.emails[0]
    except Exception:
        pass
    return name_or_address


def execute(intent: str, message: str, ctx: dict):
    import json as _j

    from prism_responses import setup_required_card, text_card

    email = ctx.get("email")
    if email is None or not getattr(email, "configured", False):
        return setup_required_card(
            service        = "Email",
            why            = (
                "PRISM needs IMAP credentials to read and send mail on your behalf. "
                "For Gmail you must use an App Password (NOT your normal password) — "
                "2FA must already be on."
            ),
            config_section = "email",
            snippet        = (
                'provider  = "gmail"\n'
                'address   = "you@gmail.com"\n'
                'imap_host = "imap.gmail.com"\n'
                'imap_port = 993\n'
                'smtp_host = "smtp.gmail.com"\n'
                'smtp_port = 587\n'
                'password  = "xxxx xxxx xxxx xxxx"   # 16-char App Password\n'
                'max_fetch = 20'
            ),
            steps = [
                "Open https://myaccount.google.com/apppasswords (Google account, 2FA on)",
                "Generate an App Password labeled 'PRISM' and copy the 16-char code",
                "Paste it above as password (with or without spaces, both work)",
                "Restart PRISM: pkill -f prism_daemon && python3 -m prism_daemon &",
                "Ask 'check my emails' again",
            ],
            docs_url = "https://support.google.com/accounts/answer/185833",
        )

    router = ctx.get("router")
    if router is None:
        return text_card(
            "No LLM router available to parse email details. "
            "Try: 'send email to name@example.com, subject: X, body: Y'",
            "Email",
        )

    prompt = (
        f"Extract email details from the user's request below.\n"
        f"Request: '{message}'\n"
        f"Return ONLY valid JSON with keys: to, subject, body. "
        f"'to' should be the recipient name or email address exactly as mentioned.\n"
        f"No extra text, no markdown fences.\n"
        f"Example: {{\"to\":\"alice@example.com\",\"subject\":\"Hello\",\"body\":\"Hi Alice...\"}}"
    )
    try:
        raw, _ = router.call(prompt, min_capability=1, max_tokens=400, json_mode=True)
        data = _j.loads(raw.strip().lstrip("```json").rstrip("```").strip())
        to      = data.get("to", "").strip()
        subject = data.get("subject", "(no subject)").strip()
        body    = data.get("body", "").strip()
    except Exception as exc:
        return text_card(
            f"Could not parse email details ({exc}). "
            "Try: 'send email to name@example.com about <topic>'",
            "Email",
        )

    if not to:
        return text_card("No recipient address found in your message.", "Email")

    # Resolve contact name → email address
    contacts = ctx.get("contacts")
    resolved = _resolve_contact_email(to, contacts)
    if resolved != to:
        to = resolved
    elif "@" not in to:
        return text_card(
            f"Could not find an email address for '{to}'. "
            "Add them to contacts or use their full email address.",
            "Email",
        )

    ok = email.send(to, subject, body)
    if ok:
        return text_card(f"Sent to {to} — \"{subject}\"", "Email")
    return text_card(f"Failed to send email to {to}. Check SMTP settings.", "Email")
