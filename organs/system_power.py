"""Bundled organ: system_power — suspend / shutdown / restart / logout.

PRISM's bridge to local OS power state. Approval-gated because every
action interrupts the user's work. The verb in the original message
picks the action; on Linux this uses ``systemctl`` (suspend, hibernate,
poweroff, reboot) and ``loginctl terminate-user`` for logout.
"""
ORGAN_META = {
    "intent":       "system_power",
    "description":  "control local OS power state — suspend, shutdown, restart, logout",
    "version":      "1.0",
    "capabilities": ["system_ui"],
    "inputs":       {"action": "str"},
    "outputs":      {"action": "str", "platform": "str", "command": "str"},
}

ORGAN_POLICY = {
    "risk_level":        "high",
    "requires_approval": True,
    "irreversible":      False,
    "max_per_session":   None,
}


def _detect_action(message: str) -> str:
    m = message.lower()
    if "hibernate" in m:
        return "hibernate"
    if "log" in m and "out" in m:
        return "logout"
    if "sign" in m and "out" in m:
        return "logout"
    if "reboot" in m or "restart" in m:
        return "restart"
    if "shut" in m or "power off" in m or "turn off" in m:
        return "shutdown"
    if "sleep" in m or "suspend" in m:
        return "suspend"
    return "suspend"


def _command_for(action: str) -> list[str]:
    import sys

    from prism_device_executor import current_uid, which

    if sys.platform == "darwin":
        return {
            "suspend":   ["pmset", "sleepnow"],
            "hibernate": ["pmset", "sleepnow"],
            "shutdown":  ["osascript", "-e", 'tell app "System Events" to shut down'],
            "restart":   ["osascript", "-e", 'tell app "System Events" to restart'],
            "logout":    ["osascript", "-e", 'tell app "System Events" to log out'],
        }.get(action, [])
    if sys.platform.startswith("win"):
        return {
            "suspend":   ["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"],
            "hibernate": ["shutdown", "/h"],
            "shutdown":  ["shutdown", "/s", "/t", "0"],
            "restart":   ["shutdown", "/r", "/t", "0"],
            "logout":    ["shutdown", "/l"],
        }.get(action, [])
    # Linux / *nix — prefer systemctl/loginctl.
    if action == "logout" and which("loginctl"):
        return ["loginctl", "terminate-user", str(current_uid())]
    if which("systemctl"):
        return {
            "suspend":   ["systemctl", "suspend"],
            "hibernate": ["systemctl", "hibernate"],
            "shutdown":  ["systemctl", "poweroff"],
            "restart":   ["systemctl", "reboot"],
        }.get(action, [])
    return []


def execute(intent: str, message: str, ctx: dict):
    import sys

    from prism_device_executor import run_argv
    from prism_responses import text_card

    action = _detect_action(message)
    cmd = _command_for(action)
    if not cmd:
        return text_card(
            f"No system command available for action `{action}` on this platform "
            f"({sys.platform}). Install systemd (systemctl) on Linux or rely on "
            f"your desktop environment's power menu.",
            f"System {action}",
        )

    res = run_argv(cmd, timeout=5)
    if not res.success:
        card = text_card(
            f"Could not {action}: {res.error}\nCommand: {' '.join(cmd)}",
            f"System {action}",
        )
        card.card_data.update({
            "action":   action,
            "platform": sys.platform,
            "command":  " ".join(cmd),
            "error":    res.error,
        })
        return card

    card = text_card(
        f"System {action} issued via `{cmd[0]}`.",
        f"System {action}",
    )
    card.card_data.update({
        "action":   action,
        "platform": sys.platform,
        "command":  " ".join(cmd),
    })
    return card
