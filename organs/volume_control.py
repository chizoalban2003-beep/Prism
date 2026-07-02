"""Bundled organ: volume_control — adjust local audio output volume.

Linux uses ``pactl`` (PulseAudio / PipeWire-compat) with ``amixer`` as a
fallback. macOS uses ``osascript`` to set output volume. Windows uses
``nircmd.exe`` if present, else ``powershell`` via SendKeys.

Low-risk and reversible — no approval needed.
"""
ORGAN_META = {
    "intent":       "volume_control",
    "description":  "raise / lower / set / mute the local audio output volume",
    "version":      "1.0",
    "capabilities": ["system_ui"],
    "inputs":       {"action": "str", "amount": "int"},
    "outputs":      {"action": "str", "amount": "int", "command": "str"},
}

ORGAN_POLICY = {
    "risk_level":        "low",
    "requires_approval": False,
    "irreversible":      False,
    "max_per_session":   None,
}


def _parse(message: str) -> tuple[str, int]:
    """Return (action, amount). action ∈ {up, down, set, mute, unmute}."""
    import re
    m = message.lower()
    if "unmute" in m:
        return "unmute", 0
    if "mute" in m:
        return "mute", 0
    set_m = re.search(r"\b(?:set\s+volume\s+to|volume\s+to)\s+(\d+)", m)
    if set_m:
        return "set", int(set_m.group(1))
    bare_set_m = re.search(r"\bvolume\s+(\d+)", m)
    if bare_set_m:
        return "set", int(bare_set_m.group(1))
    if "up" in m.split() or "louder" in m or "increase" in m or "raise" in m:
        return "up", 5
    if "down" in m.split() or "quieter" in m or "softer" in m or \
       "decrease" in m or "lower" in m:
        return "down", 5
    return "up", 5


def _linux_pactl_cmd(action: str, amount: int) -> list[str]:
    sink = "@DEFAULT_SINK@"
    if action == "mute":
        return ["pactl", "set-sink-mute", sink, "1"]
    if action == "unmute":
        return ["pactl", "set-sink-mute", sink, "0"]
    if action == "set":
        return ["pactl", "set-sink-volume", sink, f"{amount}%"]
    sign = "+" if action == "up" else "-"
    return ["pactl", "set-sink-volume", sink, f"{sign}{amount}%"]


def _linux_amixer_cmd(action: str, amount: int) -> list[str]:
    if action == "mute":
        return ["amixer", "-D", "pulse", "sset", "Master", "mute"]
    if action == "unmute":
        return ["amixer", "-D", "pulse", "sset", "Master", "unmute"]
    if action == "set":
        return ["amixer", "-D", "pulse", "sset", "Master", f"{amount}%"]
    sign = "+" if action == "up" else "-"
    return ["amixer", "-D", "pulse", "sset", "Master", f"{amount}%{sign}"]


def _command_for(action: str, amount: int) -> list[str]:
    import sys

    from prism_device_executor import which

    if sys.platform == "darwin":
        if action == "mute":
            return ["osascript", "-e", "set volume with output muted"]
        if action == "unmute":
            return ["osascript", "-e", "set volume without output muted"]
        if action == "set":
            return ["osascript", "-e", f"set volume output volume {amount}"]
        delta = 7 if action == "up" else -7
        return [
            "osascript", "-e",
            f"set volume output volume ((output volume of (get volume settings)) + {delta})",
        ]
    if sys.platform.startswith("win"):
        if which("nircmd.exe") or which("nircmd"):
            nircmd = which("nircmd.exe") or which("nircmd")
            if action == "mute":
                return [nircmd, "mutesysvolume", "1"]
            if action == "unmute":
                return [nircmd, "mutesysvolume", "0"]
            if action == "set":
                step = max(0, min(100, amount)) * 655  # 0..65535
                return [nircmd, "setsysvolume", str(step)]
            delta = 6553 if action == "up" else -6553
            return [nircmd, "changesysvolume", str(delta)]
        return []
    # Linux
    if which("pactl"):
        return _linux_pactl_cmd(action, amount)
    if which("amixer"):
        return _linux_amixer_cmd(action, amount)
    return []


def execute(intent: str, message: str, ctx: dict):
    from prism_device_executor import run_argv
    from prism_responses import text_card

    action, amount = _parse(message)
    cmd = _command_for(action, amount)
    if not cmd:
        return text_card(
            "No volume-control tool found. Install pactl (PulseAudio) or "
            "amixer (ALSA) on Linux, or nircmd on Windows.",
            "Volume",
        )

    res = run_argv(cmd, timeout=3)
    if not res.success:
        card = text_card(
            f"Could not adjust volume: {res.error}\nCommand: {' '.join(cmd)}",
            "Volume",
        )
        card.card_data.update({
            "action":  action,
            "amount":  amount,
            "command": " ".join(cmd),
            "error":   res.error,
        })
        return card

    summary = {
        "up":     f"Volume up {amount}%.",
        "down":   f"Volume down {amount}%.",
        "set":    f"Volume set to {amount}%.",
        "mute":   "Audio muted.",
        "unmute": "Audio unmuted.",
    }.get(action, f"Volume {action}.")

    card = text_card(summary, "Volume")
    card.card_data.update({
        "action":  action,
        "amount":  amount,
        "command": " ".join(cmd),
    })
    return card
