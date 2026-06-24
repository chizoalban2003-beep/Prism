"""Bundled organ: note_list — read the PRISM notes file and show recent entries."""
ORGAN_META = {
    "intent":      "note_list",
    "description": "list recent notes from the PRISM notes file",
    "version":     "1.0",
    "capabilities": ["filesystem_read"],
    "inputs":  {"limit": "int"},
    "outputs": {"notes": "list[str]", "count": "int", "path": "str"},
}

ORGAN_POLICY = {
    "risk_level":        "low",
    "requires_approval": False,
    "irreversible":      False,
    "max_per_session":   None,
}


def execute(intent: str, message: str, ctx: dict):
    from pathlib import Path

    from prism_responses import text_card

    notes_file = Path("~/.prism/notes.md").expanduser()
    if not notes_file.exists():
        card = text_card(
            "No notes yet. Save one with: Note: <your note>",
            "Notes",
        )
        card.card_data.update({"notes": [], "count": 0, "path": str(notes_file)})
        return card

    try:
        raw = notes_file.read_text(encoding="utf-8")
    except Exception as exc:
        return text_card(f"Could not read notes file: {exc}", "Notes")

    entries: list[str] = []
    current: list[str] = []
    for line in raw.splitlines():
        if line.startswith("## "):
            if current:
                entries.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        entries.append("\n".join(current).strip())
    entries = [e for e in entries if e]

    limit = int(ctx.get("limit") or 10)
    recent = entries[-limit:]
    if not recent:
        body = "No notes yet. Save one with: Note: <your note>"
    else:
        body = f"Recent notes ({len(recent)} of {len(entries)}):\n\n" + "\n\n---\n\n".join(recent)

    card = text_card(body, "Notes")
    card.card_data.update({
        "notes": recent,
        "count": len(entries),
        "path":  str(notes_file),
    })
    return card
