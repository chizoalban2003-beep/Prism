"""Bundled organ: input_control — synthesise keyboard/mouse input.

The last piece of "run your desktop like a command centre": type text, click,
press key combos, move the cursor, and scroll — driving whichever input-
synthesis backend the box has (ydotool on Wayland, xdotool on X11, wtype for
typing). Consults the capability scanner and degrades with an install hint
when no backend (or no display) is present.

Security: synthetic input can do ANYTHING the user's own hands can (type
`rm -rf` into a terminal, click "Delete"), so this organ is high-risk and
approval-gated — every action surfaces an L2 approval card before it fires,
and the belt's taint rule denies it after any untrusted-source content, so a
poisoned web page can never make PRISM type or click on its own.

Design mirrors window_control: pure _parse_action / _build_argv (unit-tested
without a live display); execute() wires them to prism_device_executor.run_argv
(import subprocess is AST-blocked in organs) behind the capability gate.
"""
from __future__ import annotations

ORGAN_META = {
    "intent":      "input_control",
    "description": "synthesise keyboard/mouse input — type text, click, press "
                   "key combos, move the cursor, scroll (via ydotool/xdotool)",
    "version":     "1.0",
    "capabilities": ["system_ui", "input_synthesis"],
    "inputs":  {"action": "str", "arg": "str"},
    "outputs": {"action": "str", "arg": "str", "backend": "str"},
}

# High-risk + approval-gated: synthetic input is as powerful as the user.
ORGAN_POLICY = {
    "risk_level":        "high",
    "requires_approval": True,
    "irreversible":      True,
    "max_per_session":   None,
}

# Named-key normalisation for xdotool (X keysyms) — the clean, primary backend.
_KEYSYMS = {
    "enter": "Return", "return": "Return", "esc": "Escape", "escape": "Escape",
    "tab": "Tab", "space": "space", "backspace": "BackSpace",
    "del": "Delete", "delete": "Delete", "up": "Up", "down": "Down",
    "left": "Left", "right": "Right", "home": "Home", "end": "End",
    "pageup": "Prior", "pagedown": "Next",
}


def _normalise_combo(combo: str) -> str:
    """Turn 'ctrl+C' / 'press enter' fragments into an xdotool key spec."""
    parts = [p.strip().lower() for p in combo.replace(" ", "+").split("+") if p.strip()]
    out = []
    for p in parts:
        if p in ("ctrl", "control"):
            out.append("ctrl")
        elif p in ("alt", "meta", "super", "shift"):
            out.append(p)
        else:
            out.append(_KEYSYMS.get(p, p))
    return "+".join(out)


def _parse_action(message: str) -> tuple[str, str]:
    """Return (action, arg). action ∈ {type,click,key,move,scroll}."""
    import re

    m = (message or "").strip()
    lw = m.lower()

    # type "..." / type the text ...
    tm = re.search(r"\btype\b[:\s]+(.+)", m, re.IGNORECASE)
    if tm:
        text = tm.group(1).strip()
        text = re.sub(r"^(the\s+text|out|in)\s+", "", text, flags=re.IGNORECASE)
        return "type", text.strip().strip("\"'")

    # press / hit / key <combo>
    km = re.search(r"\b(?:press|hit|key(?:board)?)\b[:\s]+(.+)", m, re.IGNORECASE)
    if km:
        return "key", _normalise_combo(km.group(1).strip().strip("\"'"))

    # move (the) (mouse|cursor) to X Y   /   move to X Y
    mm = re.search(r"\bmove\b.*?(-?\d+)\s*[, ]\s*(-?\d+)", lw)
    if mm:
        return "move", f"{mm.group(1)} {mm.group(2)}"

    # scroll up/down [n]
    sm = re.search(r"\bscroll\b\s*(up|down)?\s*(\d+)?", lw)
    if sm and "scroll" in lw:
        direction = sm.group(1) or "down"
        count = sm.group(2) or "3"
        return "scroll", f"{direction} {count}"

    # click / left|right|middle|double click
    if "click" in lw:
        if "right" in lw:
            return "click", "right"
        if "middle" in lw:
            return "click", "middle"
        if "double" in lw:
            return "click", "double"
        return "click", "left"

    return "", ""


def _build_argv(backend: str, action: str, arg: str) -> list[str] | None:
    """Map (backend, action, arg) → argv, or None when unsupported."""
    if action == "type":
        if not arg:
            return None
        if backend == "xdotool":
            return ["xdotool", "type", "--clearmodifiers", arg]
        if backend == "ydotool":
            return ["ydotool", "type", arg]
        if backend == "wtype":
            return ["wtype", arg]
        return None

    if action == "key":
        if not arg:
            return None
        if backend == "xdotool":
            return ["xdotool", "key", arg]
        if backend == "wtype":
            # wtype presses a single named key with -k
            return ["wtype", "-k", arg.split("+")[-1]]
        # ydotool key needs raw numeric keycodes — not expressed here
        return None

    if action == "click":
        button = {"left": "1", "right": "3", "middle": "2",
                  "double": "1"}.get(arg, "1")
        if backend == "xdotool":
            argv = ["xdotool", "click"]
            if arg == "double":
                argv += ["--repeat", "2"]
            return argv + [button]
        if backend == "ydotool":
            yb = {"1": "0xC0", "3": "0xC1", "2": "0xC2"}[button]
            return ["ydotool", "click", yb]
        return None

    if action == "move":
        parts = arg.split()
        if len(parts) != 2:
            return None
        x, y = parts
        if backend == "xdotool":
            return ["xdotool", "mousemove", x, y]
        if backend == "ydotool":
            return ["ydotool", "mousemove", "-a", x, y]
        return None

    if action == "scroll":
        direction, _, count = arg.partition(" ")
        n = count.strip() or "3"
        if backend == "xdotool":
            btn = "4" if direction == "up" else "5"
            return ["xdotool", "click", "--repeat", n, btn]
        return None

    return None


def execute(intent: str, message: str, ctx: dict):
    from prism_device_executor import run_argv
    from prism_responses import text_card

    action, arg = _parse_action(message)
    if not action:
        return text_card(
            "Tell me what input to send — e.g. \"type hello\", \"click\", "
            "\"press ctrl+c\", \"move to 400 300\", or \"scroll down\".",
            "Input control")

    try:
        from prism_device_agent import _INSTALL_HINTS, DeviceCapabilityScanner
        caps = DeviceCapabilityScanner().scan()
    except Exception as exc:
        return text_card(f"Capability scan failed: {exc}", "Input control")

    if not caps.has_display:
        return text_card(
            "No graphical display — I can't synthesise input on a headless "
            "session. Input control works on a local desktop.", "Input control")

    backend = caps.best_tool("input_synth")
    if not backend:
        hint = _INSTALL_HINTS.get("input_synth", "")
        return text_card(
            "No input-synthesis backend is installed on this "
            f"{caps.session_type or 'desktop'} session.\n\nInstall one:\n  {hint}",
            "Input control")

    argv = _build_argv(backend, action, arg)
    if argv is None:
        return text_card(
            f"'{action}' isn't supported by {backend} on this session "
            f"(try xdotool for full keyboard/mouse control).", "Input control")

    res = run_argv(argv, timeout=8)
    if not res.success:
        err = (res.error or res.output or "").strip()
        extra = ""
        if backend == "ydotool" and "socket" in err.lower():
            extra = " (ydotool needs its daemon — start ydotoold)"
        return text_card(
            f"Input '{action}' failed via {backend}: {err or 'unknown'}{extra}",
            "Input control")

    detail = f" '{arg}'" if arg else ""
    card = text_card(f"Sent {action}{detail} via {backend}.", "Input control")
    card.card_data.update({"action": action, "arg": arg, "backend": backend})
    return card
