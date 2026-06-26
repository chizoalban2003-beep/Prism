"""Bundled organ: bluetooth_control — toggle and query the local Bluetooth radio.

Linux uses ``bluetoothctl power on/off`` (BlueZ); falls back to
``rfkill`` for unblock + status. macOS uses ``blueutil`` (Homebrew).
Windows uses PowerShell ``Get-PnpDevice`` to enable/disable the radio.

Low-risk, reversible — no approval needed.
"""
ORGAN_META = {
    "intent":       "bluetooth_control",
    "description":  "turn the local Bluetooth radio on/off or query its state",
    "version":      "1.0",
    "capabilities": ["system_ui"],
    "inputs":       {"action": "str"},
    "outputs":      {"action": "str", "state": "str", "command": "str"},
}

ORGAN_POLICY = {
    "risk_level":        "low",
    "requires_approval": False,
    "irreversible":      False,
    "max_per_session":   None,
}


def _parse(message: str) -> str:
    """Return action ∈ {on, off, status}."""
    m = message.lower()
    if "status" in m or "is bluetooth" in m:
        return "status"
    if any(w in m for w in ("disable", "off", "turn off", "switch off")):
        return "off"
    return "on"


def _command_for(action: str) -> list[str]:
    import shutil
    import sys

    if sys.platform == "darwin":
        if shutil.which("blueutil"):
            if action == "status":
                return ["blueutil", "-p"]
            return ["blueutil", "-p", "1" if action == "on" else "0"]
        return []
    if sys.platform.startswith("win"):
        ps = {
            "on":     "Get-PnpDevice -Class Bluetooth | Enable-PnpDevice -Confirm:$false",
            "off":    "Get-PnpDevice -Class Bluetooth | Disable-PnpDevice -Confirm:$false",
            "status": "Get-PnpDevice -Class Bluetooth | Format-Table Name,Status",
        }.get(action, "")
        if ps:
            return ["powershell", "-Command", ps]
        return []
    # Linux
    if shutil.which("bluetoothctl"):
        if action == "status":
            return ["bluetoothctl", "show"]
        return ["bluetoothctl", "power", action]
    if shutil.which("rfkill"):
        if action == "status":
            return ["rfkill", "list", "bluetooth"]
        return ["rfkill", "unblock" if action == "on" else "block", "bluetooth"]
    return []


def execute(intent: str, message: str, ctx: dict):
    import subprocess

    from prism_responses import text_card

    action = _parse(message)
    cmd = _command_for(action)
    if not cmd:
        return text_card(
            "No Bluetooth tool found. Install bluez (bluetoothctl) or "
            "rfkill on Linux, blueutil on macOS, or rely on PnpDevice "
            "on Windows.",
            "Bluetooth",
        )

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=4)
        out = (result.stdout or "").strip() or (result.stderr or "").strip()
    except Exception as exc:
        card = text_card(
            f"Could not run bluetooth {action}: {exc}\nCommand: {' '.join(cmd)}",
            "Bluetooth",
        )
        card.card_data.update({
            "action":  action,
            "command": " ".join(cmd),
            "error":   str(exc),
        })
        return card

    if action == "status":
        body = out[:800] if out else "Bluetooth status unavailable."
        card = text_card(body, "Bluetooth")
    else:
        card = text_card(f"Bluetooth turned {action}.", "Bluetooth")
    card.card_data.update({
        "action":  action,
        "state":   out[:200],
        "command": " ".join(cmd),
    })
    return card
