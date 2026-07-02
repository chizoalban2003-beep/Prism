"""Bundled organ: system_lock — lock the local desktop session.

PRISM's bridge to the local OS lock screen. On Linux this uses
``loginctl lock-session`` (works under systemd/logind on KDE, GNOME,
XFCE, sway, etc.). On macOS it falls back to the CGSession suspend
shortcut. On Windows it invokes ``rundll32 user32.dll,LockWorkStation``.

The lock action is low-risk and reversible — the user just unlocks again
— so this organ does NOT require explicit per-call approval.
"""
ORGAN_META = {
    "intent":       "system_lock",
    "description":  "lock the local desktop session (loginctl/CGSession/LockWorkStation)",
    "version":      "1.0",
    "capabilities": ["system_ui"],
    "inputs":       {},
    "outputs":      {"locked": "bool", "platform": "str", "command": "str"},
}

ORGAN_POLICY = {
    "risk_level":        "low",
    "requires_approval": False,
    "irreversible":      False,
    "max_per_session":   None,
}


def _lock_command() -> list[str]:
    import sys

    from prism_device_executor import which

    if sys.platform == "darwin":
        return [
            "/System/Library/CoreServices/Menu Extras/User.menu/Contents/Resources/CGSession",
            "-suspend",
        ]
    if sys.platform.startswith("win"):
        return ["rundll32.exe", "user32.dll,LockWorkStation"]
    # Linux / *nix — prefer loginctl (logind), then fall back to common
    # desktop-specific lockers if loginctl is missing.
    if which("loginctl"):
        return ["loginctl", "lock-session"]
    for cand in (
        ["xdg-screensaver", "lock"],
        ["gnome-screensaver-command", "--lock"],
        ["dm-tool", "lock"],
        ["xscreensaver-command", "-lock"],
        ["i3lock"],
        ["swaylock"],
    ):
        if which(cand[0]):
            return cand
    return []


def execute(intent: str, message: str, ctx: dict):
    import sys

    from prism_device_executor import run_argv
    from prism_responses import text_card

    cmd = _lock_command()
    if not cmd:
        return text_card(
            "No screen-lock command found on this system. Install loginctl "
            "(systemd) or one of: xdg-screensaver, gnome-screensaver, "
            "dm-tool, xscreensaver, i3lock, swaylock.",
            "Lock screen",
        )

    res = run_argv(cmd, timeout=5)
    if not res.success:
        card = text_card(
            f"Could not lock the screen: {res.error}\nCommand: {' '.join(cmd)}",
            "Lock screen",
        )
        card.card_data.update({
            "locked":   False,
            "platform": sys.platform,
            "command":  " ".join(cmd),
            "error":    res.error,
        })
        return card

    card = text_card(
        f"Screen locked via `{cmd[0]}`.",
        "Lock screen",
    )
    card.card_data.update({
        "locked":   True,
        "platform": sys.platform,
        "command":  " ".join(cmd),
    })
    return card
