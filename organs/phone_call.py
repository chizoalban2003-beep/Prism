"""Bundled organ: phone_call — make an outbound voice call or send an SMS via Twilio."""
ORGAN_META = {
    "intent": "phone_call",
    "description": "Call or text a phone number — extracts number and message from the user's request",
    "version": "1.0",
}

ORGAN_POLICY = {
    "risk_level": "high",
    "requires_approval": True,
    "irreversible": True,
    "max_per_session": 3,
}

_E164_RE = None


def _phone_re():
    global _E164_RE
    if _E164_RE is None:
        import re
        _E164_RE = re.compile(r"\+?\d[\d\s\-().]{6,}\d")
    return _E164_RE


def _normalise(raw: str) -> str:
    """Strip spaces/dashes and ensure leading +."""
    import re
    digits = re.sub(r"[\s\-().]+", "", raw)
    if not digits.startswith("+"):
        digits = "+" + digits
    return digits


def execute(intent: str, message: str, ctx: dict):
    import os

    from prism_responses import text_card

    # ── Credential resolution (config > env) ────────────────────────────────
    twilio_cfg = ctx.get("twilio_config") or {}
    account_sid = (twilio_cfg.get("account_sid") or
                   os.environ.get("TWILIO_ACCOUNT_SID", "")).strip()
    auth_token  = (twilio_cfg.get("auth_token") or
                   os.environ.get("TWILIO_AUTH_TOKEN", "")).strip()
    from_number = (twilio_cfg.get("from_number") or
                   os.environ.get("TWILIO_FROM", "")).strip()

    if not (account_sid and auth_token and from_number):
        return text_card(
            "Phone calls require Twilio credentials.\n"
            "Add [twilio] account_sid, auth_token, from_number to prism_config.toml\n"
            "or set TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN / TWILIO_FROM env vars.",
            "Phone",
        )

    try:
        from twilio.rest import Client  # type: ignore[import]
    except ImportError:
        return text_card(
            "Twilio library not installed. Run: pip install twilio", "Phone"
        )

    # ── Detect mode: call vs SMS ─────────────────────────────────────────────
    msg_lower = message.lower()
    is_sms = any(w in msg_lower for w in ("text", "sms", "message", "whatsapp"))
    is_call = any(w in msg_lower for w in ("call", "ring", "phone", "dial"))
    if not is_sms and not is_call:
        is_call = True  # default to voice call

    # ── Extract phone number ─────────────────────────────────────────────────
    m = _phone_re().search(message)
    if not m:
        return text_card(
            "No phone number found. Include a number like +447700900000.", "Phone"
        )
    to_number = _normalise(m.group(0))

    client = Client(account_sid, auth_token)

    if is_sms:
        # Extract body: text after the number or after "say/message/tell"
        import re
        body_match = re.search(
            r"(?:say|tell (?:them|him|her)|message|body)[:\s]+(.+)", message, re.IGNORECASE
        )
        sms_body = body_match.group(1).strip() if body_match else message.strip()
        sms_body = sms_body[:1600]
        try:
            msg_obj = client.messages.create(
                body=sms_body, from_=from_number, to=to_number
            )
            return text_card(f"SMS sent to {to_number} (SID: {msg_obj.sid})", "Phone")
        except Exception as exc:
            return text_card(f"SMS failed: {exc}", "Phone")

    # Voice call via TwiML say
    import re
    say_match = re.search(
        r"(?:say|tell (?:them|him|her)|speak|message)[:\s]+(.+)", message, re.IGNORECASE
    )
    say_text = say_match.group(1).strip() if say_match else "Hello, this is PRISM."
    say_text = say_text[:500]
    twiml = (
        f"<Response><Say voice=\"alice\">{say_text}</Say><Pause length=\"1\"/></Response>"
    )
    try:
        call = client.calls.create(
            twiml=twiml,
            to=to_number,
            from_=from_number,
        )
        return text_card(f"Call initiated to {to_number} (SID: {call.sid})", "Phone")
    except Exception as exc:
        return text_card(f"Call failed: {exc}", "Phone")
