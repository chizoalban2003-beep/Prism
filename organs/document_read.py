"""Bundled organ: document_read — read and summarise a local text document."""
ORGAN_META = {
    "intent": "document_read",
    "description": "Read and summarise a local document (markdown, txt, or plain text)",
    "version": "1.0",
    "capabilities": ["filesystem_read"],
}

ORGAN_POLICY = {
    "risk_level": "low",
    "requires_approval": False,
    "irreversible": False,
    "max_per_session": None,
}


def execute(intent: str, message: str, ctx: dict):
    import re
    from pathlib import Path

    from prism_responses import text_card

    _PATH_RE = re.compile(
        r'(?:~|/home/\S+|/[^\s]+|\./\S+|\S+)'
        r'\.(?:md|txt|py|json)\b'
    )

    path_str = ctx.get("document_path") or ""
    if not path_str:
        m = _PATH_RE.search(message)
        if m:
            path_str = m.group(0).strip("'\"")

    if not path_str:
        return text_card(
            "Please specify a file path in your message.\n"
            "Examples:\n"
            "  • Read ~/documents/notes.md\n"
            "  • Open /home/user/report.txt\n"
            "  • Summarise ./README.md\n\n"
            "Supported types: .md  .txt  .py  .json\n"
            "You can also pass the path via ctx['document_path'].",
            intent,
        )

    try:
        file_path = Path(path_str).expanduser().resolve()

        if not file_path.exists():
            return text_card(f"File not found: {file_path}", intent)

        if not file_path.is_file():
            return text_card(f"Path is not a file: {file_path}", intent)

        content = file_path.read_text(encoding="utf-8", errors="replace")
        total_chars = len(content)
        truncated = False

        if total_chars > 4000:
            remaining = total_chars - 4000
            content = content[:4000] + f"\n\n[truncated — {remaining} chars remain]"
            truncated = True

        size_label = f"{total_chars:,} chars" + (" (truncated)" if truncated else "")
        header = f"File: {file_path.name}  |  Size: {size_label}\n{'─'*50}\n"
        result = header + content
    except Exception as exc:
        result = f"Error reading '{path_str}': {exc}"

    return text_card(result, intent)
