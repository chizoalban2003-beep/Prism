"""Bundled organ: note_append — append a timestamped note to ~/.prism/notes.md."""
ORGAN_META = {
    "intent":      "note_append",
    "description": "save a note to the PRISM notes file with a timestamp",
    "version":     "1.0",
}

ORGAN_POLICY = {
    "risk_level":        "low",
    "requires_approval": False,
    "irreversible":      False,
    "max_per_session":   None,
}


def _extract_note(message: str) -> str:
    import re
    for pat in [
        r'(?:note|write|save|record|jot(?:\s+down)?)[:\s]+(.+)',
        r'(?:remember|remind\s+me)[:\s]+(.+)',
        r'(?:add\s+(?:a\s+)?note)[:\s]+(.+)',
    ]:
        m = re.search(pat, message, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
    return message.strip()


def execute(intent: str, message: str, ctx: dict):
    import datetime
    from pathlib import Path

    from prism_responses import text_card

    note_text = _extract_note(message)
    if not note_text:
        return text_card("No note content found in message.", "Note")

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
