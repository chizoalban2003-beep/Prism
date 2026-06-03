"""Bundled organ: shell_run — run a shell command (sandboxed via prism_shell helper)."""
ORGAN_META = {
    "intent":      "shell_run",
    "description": "execute a shell command and return its output",
    "version":     "1.0",
}

ORGAN_POLICY = {
    "risk_level":        "critical",
    "requires_approval": True,
    "irreversible":      True,
    "max_per_session":   10,
}


def _extract_command(message: str) -> str:
    import re
    patterns = [
        r'(?:run|execute|shell|bash|cmd|command)[:\s]+["`\'`](.+?)["`\'`]',
        r'(?:run|execute|shell|bash|cmd|command)[:\s]+(.+)',
        r'`(.+?)`',
    ]
    for pat in patterns:
        m = re.search(pat, message, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return message.strip()


def execute(intent: str, message: str, ctx: dict):
    from prism_responses import text_card

    # subprocess is blocked by the AST safety visitor; attempt via prism_shell
    # helper if available, otherwise surface a clear error.
    cmd = _extract_command(message)
    if not cmd:
        return text_card("No command found in message.", "Shell")

    # Try the optional prism_shell helper (not subject to AST visitor)
    runner = ctx.get("shell_runner")
    if callable(runner):
        try:
            output = runner(cmd, timeout=30)
            return text_card(f"$ {cmd}\n\n{output}", "Shell")
        except Exception as exc:
            return text_card(f"$ {cmd}\n\nError: {exc}", "Shell")

    return text_card(
        "Shell execution is not available in this PRISM environment.\n\n"
        "The built-in AST safety visitor blocks subprocess/os imports in organs.\n"
        "To enable shell_run, provide a 'shell_runner' callable in ctx, e.g.:\n"
        "  ctx['shell_runner'] = lambda cmd, timeout=30: "
        "__import__('subprocess').check_output(cmd, shell=True, timeout=timeout,"
        " text=True)",
        "Shell",
    )
