"""Bundled organ: window_control — list / focus / close / minimise / maximise
desktop windows.

This is the "run your desktop like a command centre" depth the assessment
called out as missing. It drives whichever window backend the box actually
has — ``wmctrl`` (X11), ``xdotool`` (X11), or ``kdotool`` (KDE Wayland) — and
consults :class:`DeviceCapabilityScanner` first so it degrades with an install
hint instead of a stack trace when no backend (or no display) is present.

Design: the argv builder and output parser are pure functions (unit-tested
without a live compositor); ``execute`` wires them to subprocess behind the
capability gate. ``close`` requires a named target — "close everything" is not
a thing this organ will do.
"""
from __future__ import annotations

ORGAN_META = {
    "intent":      "window_control",
    "description": "list, focus, close, minimise or maximise desktop windows "
                   "by title via wmctrl / xdotool / kdotool",
    "version":     "1.0",
    "capabilities": ["system_ui"],
    "inputs":  {"action": "str", "target": "str"},
    "outputs": {"action": "str", "target": "str", "backend": "str",
                "windows": "list[str]"},
}

ORGAN_POLICY = {
    "risk_level":        "low",
    "requires_approval": False,
    "irreversible":      False,
    "max_per_session":   None,
}

# action → verbs that select it (checked in this order; first match wins)
_ACTION_WORDS = [
    ("list",     ("list windows", "list all windows", "what windows",
                  "show windows", "open windows", "which windows")),
    ("close",    ("close", "quit window", "kill window")),
    ("minimize", ("minimise", "minimize", "hide window")),
    ("maximize", ("maximise", "maximize", "fullscreen")),
    ("focus",    ("focus", "activate", "switch to", "bring up",
                  "raise", "go to")),
]


def _parse_action(message: str) -> tuple[str, str]:
    """Return (action, target_title). action ∈ {list,focus,close,minimize,
    maximize}. target is the free-text window title after the verb, "" for
    list. Defaults to focus (the most common ask) when a target is named but
    no verb matched."""
    import re

    m = (message or "").strip()
    lw = m.lower()

    # "bring <target> to (the) front" / "bring up <target>" → focus target
    bring = re.match(
        r"\s*bring\s+(?:up\s+)?(.+?)\s+(?:to\s+(?:the\s+)?front|up)\s*$",
        m, flags=re.IGNORECASE)
    if not bring:
        bring = re.match(r"\s*bring\s+up\s+(.+?)\s*$", m, flags=re.IGNORECASE)
    if bring:
        tgt = re.sub(r"^(the|my|window|app|application)\s+", "",
                     bring.group(1).strip(), flags=re.IGNORECASE)
        tgt = re.sub(r"\s+(window|app|application)$", "", tgt,
                     flags=re.IGNORECASE).strip().strip("\"'")
        return "focus", tgt

    action = ""
    matched = ""
    for name, words in _ACTION_WORDS:
        for w in words:
            if w in lw:
                action, matched = name, w
                break
        if action:
            break

    if action == "list":
        return "list", ""

    # target = text after the matched verb, stripped of filler
    target = ""
    if matched:
        idx = lw.find(matched) + len(matched)
        target = m[idx:].strip()
    else:
        target = m
    target = re.sub(r"^(the|my|window|app|application)\s+", "", target,
                    flags=re.IGNORECASE).strip()
    target = re.sub(r"\s+(window|app|application)$", "", target,
                    flags=re.IGNORECASE).strip()
    target = target.strip("\"'").strip()

    if not action:
        action = "focus" if target else "list"
    return action, target


def _build_argv(backend: str, action: str, target: str) -> list[str] | None:
    """Map (backend, action, target) → an argv list, or None when the backend
    cannot express the action (caller falls back / reports honestly)."""
    if backend == "wmctrl":
        if action == "list":
            return ["wmctrl", "-l"]
        if not target:
            return None
        if action == "focus":
            return ["wmctrl", "-a", target]
        if action == "close":
            return ["wmctrl", "-c", target]
        if action == "maximize":
            return ["wmctrl", "-r", target, "-b",
                    "add,maximized_vert,maximized_horz"]
        if action == "minimize":
            # EWMH has no portable "minimise"; hidden state is the closest.
            return ["wmctrl", "-r", target, "-b", "add,hidden"]
        return None

    if backend in ("xdotool", "kdotool"):
        if action == "list":
            # list every window's name; parser splits lines
            return [backend, "search", "--name", ""]
        if not target:
            return None
        verb = {
            "focus":    "windowactivate",
            "close":    "windowclose",
            "minimize": "windowminimize",
        }.get(action)
        if verb:
            return [backend, "search", "--name", target, verb]
        # neither xdotool nor kdotool maximises directly
        return None

    return None


def _parse_window_list(backend: str, output: str) -> list[str]:
    """Extract human-readable window titles from a backend's list output."""
    lines = [ln.rstrip() for ln in (output or "").splitlines() if ln.strip()]
    if backend == "wmctrl":
        # "0x03000007  0 hostname Window Title Here" → title is field 4+
        titles = []
        for ln in lines:
            parts = ln.split(None, 3)
            titles.append(parts[3] if len(parts) >= 4 else ln)
        return titles
    # xdotool/kdotool search --name "" prints numeric window ids only
    return lines


def execute(intent: str, message: str, ctx: dict):
    # subprocess is AST-blocked inside organs; run_argv is the trusted bridge.
    from prism_device_executor import run_argv
    from prism_responses import text_card

    action, target = _parse_action(message)

    # ── capability gate: honest degradation ─────────────────────────────
    try:
        from prism_device_agent import _INSTALL_HINTS, DeviceCapabilityScanner
        caps = DeviceCapabilityScanner().scan()
    except Exception as exc:
        return text_card(f"Capability scan failed: {exc}", "Window control")

    if not caps.has_display:
        return text_card(
            "No graphical display detected — I can't control windows on a "
            "headless/SSH session. Window control works on a local desktop.",
            "Window control")

    backend = caps.best_tool("window_manage")
    if not backend:
        hint = _INSTALL_HINTS.get("window_manage", "")
        return text_card(
            "No window-management backend is installed on this "
            f"{caps.session_type or 'desktop'} session.\n\nInstall one:\n  {hint}",
            "Window control")

    argv = _build_argv(backend, action, target)
    if argv is None:
        if action in ("close", "focus", "minimize", "maximize") and not target:
            return text_card(
                f"Name the window to {action} — e.g. \"{action} Firefox\".",
                "Window control")
        return text_card(
            f"'{action}' isn't supported by {backend} on this session.",
            "Window control")

    res = run_argv(argv, timeout=8)

    if action == "list":
        windows = _parse_window_list(backend, res.output)
        body = ("Open windows:\n" + "\n".join(f"  • {w}" for w in windows)
                if windows else "No windows reported.")
        card = text_card(body, "Windows")
        card.card_data.update({"action": "list", "backend": backend,
                               "windows": windows})
        return card

    if not res.success:
        err = (res.error or res.output or "").strip()
        return text_card(
            f"Couldn't {action} '{target}' via {backend}: {err or 'no match'}",
            "Window control")

    card = text_card(f"{action.capitalize()}d '{target}' via {backend}.",
                     "Window control")
    card.card_data.update({"action": action, "target": target,
                           "backend": backend})
    return card
