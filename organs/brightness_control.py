"""Bundled organ: brightness_control — adjust local display brightness.

Linux uses ``brightnessctl`` (most modern distros / Wayland-friendly)
falling back to ``light``. macOS uses ``brightness`` (Homebrew package).
Windows uses PowerShell + WMI ``WmiMonitorBrightnessMethods``.

Low-risk, reversible — no approval needed.
"""
ORGAN_META = {
    "intent":       "brightness_control",
    "description":  "raise / lower / set the local display brightness",
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
    """Return (action, amount). action ∈ {up, down, set}."""
    import re
    m = message.lower()
    set_m = re.search(r"\b(?:set\s+(?:screen\s+)?brightness\s+to|"
                      r"(?:screen\s+)?brightness\s+to)\s+(\d+)", m)
    if set_m:
        return "set", int(set_m.group(1))
    if "down" in m.split() or "dim" in m or "dimmer" in m or \
       "decrease" in m or "lower" in m:
        return "down", 10
    if "up" in m.split() or "brighter" in m or "brighten" in m or \
       "increase" in m or "raise" in m:
        return "up", 10
    return "up", 10


def _command_for(action: str, amount: int) -> list[str]:
    import sys

    from prism_device_executor import which

    if sys.platform == "darwin":
        if which("brightness"):
            if action == "set":
                pct = max(0, min(100, amount)) / 100.0
                return ["brightness", str(pct)]
            delta = 0.1 if action == "up" else -0.1
            return ["bash", "-c",
                    f"brightness $(echo $(brightness -l | grep brightness | "
                    f"awk '{{print $4}}')+{delta} | bc)"]
        return []
    if sys.platform.startswith("win"):
        ps_set = (
            "(Get-WmiObject -Namespace root/wmi -Class "
            "WmiMonitorBrightnessMethods).WmiSetBrightness(1, {pct})"
        )
        if action == "set":
            return ["powershell", "-Command", ps_set.format(pct=amount)]
        # Up/down: read current + apply delta
        delta = 10 if action == "up" else -10
        return [
            "powershell", "-Command",
            f"$cur=(Get-WmiObject -Namespace root/wmi -Class "
            f"WmiMonitorBrightness).CurrentBrightness; "
            f"$new=[math]::Max(0,[math]::Min(100,$cur+{delta})); "
            f"(Get-WmiObject -Namespace root/wmi -Class "
            f"WmiMonitorBrightnessMethods).WmiSetBrightness(1,$new)",
        ]
    # Linux
    if which("brightnessctl"):
        if action == "set":
            return ["brightnessctl", "set", f"{amount}%"]
        sign = "+" if action == "up" else "-"
        return ["brightnessctl", "set", f"{amount}%{sign}"]
    if which("light"):
        if action == "set":
            return ["light", "-S", str(amount)]
        flag = "-A" if action == "up" else "-U"
        return ["light", flag, str(amount)]
    return []


def execute(intent: str, message: str, ctx: dict):
    from prism_device_executor import run_argv
    from prism_responses import text_card

    action, amount = _parse(message)
    cmd = _command_for(action, amount)
    if not cmd:
        return text_card(
            "No brightness-control tool found. Install brightnessctl or "
            "light on Linux, brightness (brew install brightness) on "
            "macOS, or rely on WMI on Windows.",
            "Brightness",
        )

    res = run_argv(cmd, timeout=3)
    if not res.success:
        card = text_card(
            f"Could not adjust brightness: {res.error}\nCommand: {' '.join(cmd)}",
            "Brightness",
        )
        card.card_data.update({
            "action":  action,
            "amount":  amount,
            "command": " ".join(cmd),
            "error":   res.error,
        })
        return card

    summary = {
        "up":   f"Brightness up {amount}%.",
        "down": f"Brightness down {amount}%.",
        "set":  f"Brightness set to {amount}%.",
    }.get(action, f"Brightness {action}.")

    card = text_card(summary, "Brightness")
    card.card_data.update({
        "action":  action,
        "amount":  amount,
        "command": " ".join(cmd),
    })
    return card
