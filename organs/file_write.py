"""Bundled organ: file_write — write content to a specified file path."""
ORGAN_META = {
    "intent":      "file_write",
    "description": "write or overwrite content to a file at a given path",
    "version":     "1.1",
    "capabilities": ["filesystem_write"],
}

ORGAN_POLICY = {
    "risk_level":        "medium",
    "requires_approval": True,
    "irreversible":      True,
    "max_per_session":   None,
}

# Allow-list rather than deny-list: refuse anything that doesn't resolve into
# a well-known user-data root. Older versions used a deny-list which let
# /tmp/, /var/tmp/, the daemon cwd, and ~/.ssh through.
def _allowed_roots():
    from pathlib import Path
    return [
        Path.home() / "Documents",
        Path.home() / "Downloads",
        Path.home() / "Desktop",
        Path("/tmp/prism"),
    ]

# Names that must never be written even inside an allowed root — most are
# dotfiles whose presence in ~/Documents would still be wrong.
_FORBIDDEN_NAMES = frozenset({
    ".bashrc", ".bash_profile", ".profile", ".zshrc", ".zprofile",
    "authorized_keys", "known_hosts", "id_rsa", "id_ed25519",
    ".netrc", ".env", "credentials", "config",
})
_FORBIDDEN_SUFFIXES = (".service", ".desktop", ".sh", ".bashrc")


def _parse_message(message: str):
    """Return (path_str, content) extracted from message."""
    import re
    m = re.search(
        r'(?:write|save|create|output)\s+(?:to\s+)?'
        r'(["\']?)(.+?)\1\s+(?:to|at|into)\s+([^\s:]+)',
        message, re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(3).strip(), m.group(2).strip()

    m = re.search(
        r'(?:file|path)[:\s]+([^\n]+)\n(.*)',
        message, re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(1).strip(), m.group(2).strip()

    m = re.search(r'(/[\w./~-]+\.\w+)', message)
    if m:
        path = m.group(1)
        content = re.sub(re.escape(path), "", message).strip()
        return path, content

    return None, None


def _is_path_allowed(target):
    """Return True if *target* resolves under an allowed root and isn't a
    forbidden filename."""
    name = target.name.lower()
    if name in _FORBIDDEN_NAMES:
        return False
    if any(name.endswith(s) for s in _FORBIDDEN_SUFFIXES):
        return False
    try:
        resolved = target.resolve()
    except (OSError, RuntimeError):
        return False
    for root in _allowed_roots():
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def execute(intent: str, message: str, ctx: dict):
    from pathlib import Path

    from prism_responses import text_card

    path_str, content = _parse_message(message)
    if not path_str:
        return text_card(
            "Could not determine file path from message.\n"
            "Example: 'write Hello World to ~/Documents/hello.txt'",
            "File Write",
        )

    target = Path(path_str).expanduser()
    if not _is_path_allowed(target):
        roots = ", ".join(str(r) for r in _allowed_roots())
        return text_card(
            f"Refusing to write {target}: writes are only permitted under {roots}.",
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
