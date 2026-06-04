"""Bundled organ: file_read — read and return the contents of a file."""
ORGAN_META = {
    "intent":      "file_read",
    "description": "read a file from the filesystem and return its contents",
    "version":     "1.0",
    "capabilities": ["filesystem_read"],
}

ORGAN_POLICY = {
    "risk_level":        "low",
    "requires_approval": False,
    "irreversible":      False,
    "max_per_session":   None,
}

_MAX_CHARS = 2000


def _extract_path(message: str) -> str:
    import re
    # Quoted path first
    m = re.search(r'["\']([^"\']+\.\w+)["\']', message)
    if m:
        return m.group(1)
    # Unix-style absolute path
    m = re.search(r'((?:/|~/)[\w./~-]+\.\w+)', message)
    if m:
        return m.group(1)
    # Relative path with extension
    m = re.search(r'([\w./~-]+\.\w{1,6})', message)
    if m:
        return m.group(1)
    return ""


def execute(intent: str, message: str, ctx: dict):
    from pathlib import Path

    from prism_responses import text_card

    path_str = _extract_path(message)
    if not path_str:
        return text_card(
            "No file path found in message. Example: 'read /tmp/hello.txt'",
            "File Read",
        )

    target = Path(path_str).expanduser()

    if not target.exists():
        return text_card(f"File not found: {target}", "File Read")

    if not target.is_file():
        return text_card(f"Path is not a regular file: {target}", "File Read")

    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return text_card(f"Failed to read file: {exc}", "File Read")

    total_chars = len(content)
    truncated = total_chars > _MAX_CHARS
    display = content[:_MAX_CHARS]

    header = f"File: {target}  ({total_chars} chars)\n"
    if truncated:
        header += f"[Showing first {_MAX_CHARS} of {total_chars} characters]\n"
    header += "\n"

    return text_card(header + display, "File Read")
