"""Bundled organ: clipboard_read — read the system clipboard contents."""
ORGAN_META = {
    "intent":      "clipboard_read",
    "description": "read and return the current system clipboard contents",
    "version":     "1.0",
}

ORGAN_POLICY = {
    "risk_level":        "low",
    "requires_approval": False,
    "irreversible":      False,
    "max_per_session":   None,
}

_MAX_CHARS = 2000


def _read_via_tkinter() -> str:
    """Attempt to read clipboard via tkinter (no subprocess needed)."""
    import tkinter as tk  # noqa: PLC0415
    root = tk.Tk()
    root.withdraw()
    try:
        content = root.clipboard_get()
    finally:
        root.destroy()
    return content


def execute(intent: str, message: str, ctx: dict):
    from prism_responses import text_card

    # Prefer an injected reader from ctx (allows subprocess in host process)
    reader = ctx.get("clipboard_reader")
    if callable(reader):
        try:
            content = reader()
            content = str(content)[:_MAX_CHARS]
            return text_card(
                f"Clipboard contents ({len(content)} chars):\n\n{content}",
                "Clipboard",
            )
        except Exception as exc:
            return text_card(f"Clipboard reader failed: {exc}", "Clipboard")

    # Try tkinter (available on most desktop systems)
    try:
        content = _read_via_tkinter()
        content = str(content)[:_MAX_CHARS]
        return text_card(
            f"Clipboard contents ({len(content)} chars):\n\n{content}",
            "Clipboard",
        )
    except ImportError:
        pass
    except Exception as exc:
        return text_card(f"Could not read clipboard via tkinter: {exc}", "Clipboard")

    return text_card(
        "Clipboard access is not available in this environment.\n\n"
        "To enable clipboard reading, provide a 'clipboard_reader' callable in ctx:\n"
        "  import subprocess\n"
        "  ctx['clipboard_reader'] = lambda: "
        "subprocess.check_output(['xclip', '-o', '-selection', 'clipboard'], text=True)",
        "Clipboard",
    )
