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


_PATH_PATTERN = (
    # Priority order:
    #   1. Quoted "any path with spaces/file.ext"
    #   2. Tilde-rooted, absolute, or dotted-relative path with extension
    #   3. Bare "filename.ext" after a "file"/"called"/"named" keyword
    r"""(?P<q1>['"])(?P<qp>[^'"]+?\.\w{1,8})(?P=q1)"""
    r"""|"""
    r"""(?P<p>(?:~|\.{1,2})?/[\w./-]+\.\w{1,8})"""
    r"""|"""
    r"""(?:\b(?:file|called|named)\s+)(?P<bare>[A-Za-z0-9_-]+\.\w{1,8})"""
)


def _clean_content(text: str) -> str:
    """Strip surrounding quotes and trailing terminal punctuation/parens."""
    text = text.strip()
    # Strip a single pair of matching outer quotes.
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    # Drop a stray trailing period that came from prose like
    # "...with the content: hello." — the period was punctuation, not content.
    if text.endswith(".") and not text.endswith(".."):
        text = text[:-1].rstrip()
    return text


def _parse_message(message: str):
    """Return ``(path_str, content)`` extracted from *message*.

    Strategy: locate the path token first (so ``write 'hello' to file
    ~/Documents/hello.txt`` doesn't mistake ``file`` for the path), then
    carve content out of the substring before/after it.
    """
    import re

    m = re.search(_PATH_PATTERN, message, re.IGNORECASE)
    if not m:
        return None, None

    path = (m.group("qp") or m.group("p") or m.group("bare") or "").strip()
    if not path:
        return None, None

    pre  = message[:m.start()].rstrip()
    post = message[m.end():].lstrip()

    # ── Content after path ───────────────────────────────────────────────
    # Catches "...path: hello", "...path with [the content] X",
    # "...path containing X", "...path and put 'X' in it",
    # plus a trailing quoted/leftover blob.
    post_patterns = [
        # path: content   |   path - content   |   path — content
        (r"^[\s]*[:\-—][\s]*(.+)$",                                  1),
        # path with [the] [content|text|body|words] [colon/space] X
        # The keyword-and-separator group is fully optional so a bare
        # "with Hello World" still captures "Hello World" (not just the
        # ": "-delimited form).
        (r"^\s*with\s+(?:the\s+)?"
         r"(?:(?:content|text|body|words?)\s*[:\s]+)?(.+)$",          1),
        (r"^\s*containing\s+(.+)$",                                  1),
        # and put 'X' in it  /  and put X
        (r"^\s*and\s+(?:put|place|write)\s+"
         r"(['\"]?)(.+?)\1(?:\s+in(?:to)?\s+it)?\s*$",                2),
    ]
    # Trailing phrases like "inside it", "in it", "in the file" are prose
    # padding the user added — strip them after content extraction.
    _trailing_noise = re.compile(
        r"\s+(?:inside|into|in)\s+(?:it|the\s+file|there)\s*\.?$",
        re.IGNORECASE,
    )
    for pat, group in post_patterns:
        mm = re.match(pat, post, re.IGNORECASE | re.DOTALL)
        if mm:
            raw = mm.group(group)
            raw = _trailing_noise.sub("", raw)
            content = _clean_content(raw)
            if content:
                return path, content

    # ── Content before path ──────────────────────────────────────────────
    # "write CONTENT to/at/into [the file] PATH" — the canonical phrasing.
    # Trailing space lets the optional "to/at/into" anchor.
    pre_re = re.compile(
        r"\b(?:write|save|create|output|put|store)\b\s+"
        r"(?:the\s+(?:following|content|text|file|data)\s+)?"
        r"(?:to\s+)?"
        r"(?P<q>['\"]?)(?P<body>.+?)(?P=q)\s+"
        r"(?:to|at|into|in)\s+"
        r"(?:the\s+file\s+|file\s+)?$",
        re.IGNORECASE | re.DOTALL,
    )
    mm = pre_re.search(pre + " ")
    if mm:
        content = _clean_content(mm.group("body"))
        # Reject content that's just a noise word like "a file", "the file",
        # which is what the old regex used to capture from
        # "create a file at PATH with the content: X".
        if content and content.lower() not in {
            "a file", "the file", "a new file", "the new file",
            "file", "this", "that",
        }:
            return path, content

    return path, ""


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
