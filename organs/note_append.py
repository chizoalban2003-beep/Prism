"""Bundled organ: note_append — append a timestamped note to ~/.prism/notes.md."""
import re

ORGAN_META = {
    "intent":      "note_append",
    "description": "save a note to the PRISM notes file with a timestamp",
    "version":     "1.0",
    "capabilities": ["filesystem_write"],
}

ORGAN_POLICY = {
    "risk_level":        "low",
    "requires_approval": False,
    "irreversible":      False,
    "max_per_session":   None,
}


_TRIGGER_PHRASE = re.compile(
    r'^\s*(?:please\s+)?'
    r'(?:note|write|save|record|jot(?:\s+down)?|log|remember|remind\s+me|'
    r'add\s+(?:a\s+)?note|take\s+(?:a\s+)?note|make\s+(?:a\s+)?note|'
    r'log\s+(?:a\s+)?(?:decision|note|thought|idea)|'
    r'(?:save|record)\s+(?:a\s+)?(?:decision|note|thought|idea))'
    r'\s*[.!?]?\s*$',
    re.IGNORECASE,
)


def _extract_note(message: str) -> str:
    """Return the note body, or '' if the message is a bare trigger phrase.

    Distinguishes 'Log a decision' (request to start logging — no content)
    from 'Log: chose Postgres' (real content after the trigger).
    """
    if _TRIGGER_PHRASE.match(message or ''):
        return ''
    for pat in [
        r'(?:log|save|record)\s+(?:a\s+)?(?:decision|note|thought|idea)[:\s]+(.+)',
        r'(?:add|take|make)\s+(?:a\s+)?note[:\s]+(.+)',
        r'(?:remember|remind\s+me)[:\s]+(.+)',
        r'(?:note|write|save|record|jot(?:\s+down)?|log)[:\s]+(.+)',
    ]:
        m = re.search(pat, message, re.IGNORECASE | re.DOTALL)
        if m:
            body = m.group(1).strip()
            if body:
                return body
    return message.strip()


def execute(intent: str, message: str, ctx: dict):
    import datetime
    from pathlib import Path

    from prism_responses import text_card

    note_text = _extract_note(message)
    if not note_text:
        return text_card(
            "What would you like me to save?\n\n"
            "Try one of:\n"
            "  • Note: <your note>\n"
            "  • Log decision: <what you decided and why>\n"
            "  • Remember: <thing to remember>",
            "Note",
        )

    notes_dir = Path("~/.prism").expanduser()
    try:
        notes_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return text_card(f"Could not create notes directory: {exc}", "Note")

    notes_file = notes_dir / "notes.md"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"\n## {timestamp}\n\n{note_text}\n"

    try:
        with notes_file.open("a", encoding="utf-8") as f:
            f.write(entry)
    except Exception as exc:
        return text_card(f"Could not write note: {exc}", "Note")

    return text_card(
        f"Note saved to {notes_file}\n\n"
        f"Timestamp: {timestamp}\n"
        f"Content: {note_text}",
        "Note",
    )
