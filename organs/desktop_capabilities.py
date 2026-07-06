"""Bundled organ: desktop_capabilities — what PRISM can actually control here.

``device_inventory`` answers "what hardware / CLI tools exist". This organ
answers the command-centre question the user actually asks: *"what can you
control on this machine right now?"* — windows, notifications, screenshots,
audio, brightness, clipboard, the lock screen.

It is honest by construction. A GUI-control backend is only reported as
available when BOTH a display is present AND the backend binary is installed;
otherwise the category is marked unavailable with a one-line install hint.
This turns "shallow desktop control that fails opaquely" into an explicit,
inspectable reach map — the same :class:`DeviceCapabilityScanner` the window
and notification organs consult before acting.
"""
ORGAN_META = {
    "intent":      "desktop_capabilities",
    "description": "Report which desktop/GUI actions (windows, notifications, "
                   "screenshots, audio, brightness, clipboard, lock) PRISM can "
                   "control on this machine, with install hints for the rest",
    "version":     "1.0",
    "capabilities": [],
    "inputs":  {},
    "outputs": {
        "session_type":    "str",
        "has_display":     "bool",
        "can_control_gui": "bool",
        "available":       "dict[str,str]",
    },
}

ORGAN_POLICY = {
    "risk_level":        "low",
    "requires_approval": False,
    "irreversible":      False,
    "max_per_session":   None,
}


def execute(intent: str, message: str, ctx: dict):
    from prism_responses import text_card

    try:
        from prism_device_agent import (
            _DESKTOP_CATEGORIES,
            DeviceCapabilityScanner,
        )
    except ImportError as exc:
        return text_card(f"Device scanner unavailable: {exc}.",
                         "Desktop capabilities")

    try:
        caps = DeviceCapabilityScanner().scan()
    except Exception as exc:
        return text_card(f"Could not scan device: {exc}.",
                         "Desktop capabilities")

    card = text_card(caps.desktop_report(), "Desktop capabilities")
    available = {
        cat: (caps.best_tool(cat) or "")
        for cat in _DESKTOP_CATEGORIES
        if caps.best_tool(cat) and caps.has_display
    }
    card.card_data.update({
        "session_type":    caps.session_type,
        "has_display":     caps.has_display,
        "can_control_gui": caps.can_control_gui(),
        "available":       available,
    })
    return card
