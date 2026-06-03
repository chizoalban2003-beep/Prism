"""Bundled organ: qr_generate — generate a QR code as ASCII art or PNG file."""
ORGAN_META = {
    "intent":      "qr_generate",
    "description": "generate a QR code for text or a URL, as ASCII art or PNG",
    "version":     "1.0",
}

ORGAN_POLICY = {
    "risk_level":        "low",
    "requires_approval": False,
    "irreversible":      False,
    "max_per_session":   None,
}


def _extract_data(message: str) -> str:
    import re
    for pat in [
        r'(?:qr|generate|create|make)\s+(?:a\s+)?(?:qr\s+code\s+)?(?:for\s+)?["\'](.+?)["\']',
        r'(?:qr|generate|create|make)\s+(?:a\s+)?(?:qr\s+code\s+)?(?:for\s+)?(.+)',
        r'(?:encode|qr)[:\s]+(.+)',
    ]:
        m = re.search(pat, message, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return message.strip()


def _ascii_qr(data: str) -> str:
    """Generate a minimal ASCII-art QR code using the qrcode library."""
    import qrcode  # type: ignore[import]
    import qrcode.constants  # type: ignore[import]
    qr = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=1,
        border=1,
    )
    qr.add_data(data)
    qr.make(fit=True)
    matrix = qr.get_matrix()
    lines = []
    for row in matrix:
        lines.append("".join("██" if cell else "  " for cell in row))
    return "\n".join(lines)


def _save_png(data: str) -> str:
    """Save QR code as PNG. Return file path string."""
    import datetime
    from pathlib import Path

    import qrcode  # type: ignore[import]

    out_dir = Path("~/.prism/qrcodes").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"qr_{stamp}.png"
    img = qrcode.make(data)
    img.save(str(out_path))
    return str(out_path)


def _fallback_ascii(data: str) -> str:
    """Very simple fallback: just a bordered box with the data."""
    border = "+" + "-" * 42 + "+"
    lines = [
        "QR code generation requires the 'qrcode' library.",
        "Install with: pip install qrcode[pil]",
        "",
        "Data to encode:",
        data[:100],
    ]
    padded = [f"| {ln:<40} |" for ln in lines]
    return "\n".join([border] + padded + [border])


def execute(intent: str, message: str, ctx: dict):
    from prism_responses import text_card

    data = _extract_data(message)
    if not data:
        return text_card("No data to encode found in message.", "QR Code")

    # Try PNG first if "save" or "file" requested, else ASCII
    want_png = any(w in message.lower() for w in ("save", "file", "png", "image"))

    try:
        if want_png:
            path = _save_png(data)
            return text_card(
                f"QR code saved to: {path}\nEncoded: {data[:80]}", "QR Code"
            )
        else:
            art = _ascii_qr(data)
            return text_card(
                f"QR code for: {data[:60]}\n\n{art}", "QR Code"
            )
    except ImportError:
        fallback = _fallback_ascii(data)
        return text_card(fallback, "QR Code")
    except Exception as exc:
        return text_card(f"QR code generation failed: {exc}", "QR Code")
