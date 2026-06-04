"""Bundled organ: file_write — write content to a specified file path."""
ORGAN_META = {
    "intent":      "file_write",
    "description": "write or overwrite content to a file at a given path",
    "version":     "1.0",
    "capabilities": ["filesystem_write"],
}

ORGAN_POLICY = {
    "risk_level":        "medium",
    "requires_approval": True,
    "irreversible":      True,
    "max_per_session":   None,
}

_FORBIDDEN_PATHS = (
    "/etc/", "/usr/", "/bin/", "/sbin/", "/boot/", "/sys/", "/proc/",
    "/dev/", "/lib/", "/lib64/",
)


def _parse_message(message: str):
    """Return (path_str, content) extracted from message."""
    import re
    # Pattern: write "content" to /path/to/file  OR  write to /path: content
    m = re.search(
        r'(?:write|save|create|output)\s+(?:to\s+)?'
        r'(["\']?)(.+?)\1\s+(?:to|at|into)\s+([^\s:]+)',
        message, re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(3).strip(), m.group(2).strip()

    # Pattern: path: content  or  /path/file\ncontent
    m = re.search(
        r'(?:file|path)[:\s]+([^\n]+)\n(.*)',
        message, re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # Look for any path-like token
    m = re.search(r'(/[\w./~-]+\.\w+)', message)
    if m:
        path = m.group(1)
        content = re.sub(re.escape(path), "", message).strip()
        return path, content

    return None, None


def execute(intent: str, message: str, ctx: dict):
    from pathlib import Path

    from prism_responses import text_card

    path_str, content = _parse_message(message)
    if not path_str:
        return text_card(
            "Could not determine file path from message.\n"
            "Example: 'write Hello World to /tmp/hello.txt'",
            "File Write",
        )

    # Expand ~ and resolve
    target = Path(path_str).expanduser()

    # Safety: block writes to system paths
    abs_str = str(target.resolve()) if target.exists() else str(target)
    for forbidden in _FORBIDDEN_PATHS:
        if abs_str.startswith(forbidden):
            return text_card(
                f"Writing to {forbidden}* is not permitted for safety reasons.",
                "File Write",
            )

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content or "", encoding="utf-8")
    except Exception as exc:
        return text_card(f"Failed to write file: {exc}", "File Write")

    size = len((content or "").encode("utf-8"))
    return text_card(
        f"File written: {target}\nSize: {size} bytes",
        "File Write",
    )
