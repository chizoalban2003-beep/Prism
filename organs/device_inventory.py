"""Bundled organ: device_inventory — show what hardware and tools PRISM can reach.

PRISM's mission is to be a bridge between the user and their hardware
/ applications. A user can't trust the bridge if they can't ask "what's
on the other side" — yet "what hardware do I have" used to route to
five different unrelated organs (crystallised profile, status, agents,
learned tools, smart-home status) depending on phrasing.

Backed by :class:`DeviceCapabilityScanner` (already used by the
``/device/capabilities`` HTTP route). Surfaces:

* Platform identification (linux / darwin / windows)
* Browser availability (chrome, firefox, etc.)
* CLI tool inventory by category (git, ffmpeg, ripgrep, ...)
* Useful Python packages
"""
ORGAN_META = {
    "intent":      "device_inventory",
    "description": "List the hardware platform, browsers, CLI tools and Python packages PRISM can reach on this device",
    "version":     "1.0",
    "capabilities": [],
    "inputs":  {},
    "outputs": {
        "platform":     "str",
        "has_browser":  "bool",
        "cli_tools":    "dict[str,list[str]]",
        "py_packages":  "list[str]",
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
        from prism_device_agent import DeviceCapabilityScanner
    except ImportError as exc:
        return text_card(
            f"Device scanner unavailable: {exc}.",
            "Device inventory",
        )

    try:
        caps = DeviceCapabilityScanner().scan()
    except Exception as exc:
        return text_card(
            f"Could not scan device: {exc}.",
            "Device inventory",
        )

    lines = [
        f"**Platform:** {caps.platform}",
        f"**Browser available:** {'yes' if caps.has_browser else 'no'}",
        "",
    ]
    if caps.cli_tools:
        lines.append("**CLI tools by category:**")
        for category in sorted(caps.cli_tools):
            tools = caps.cli_tools[category]
            if tools:
                lines.append(f"  • {category}: {', '.join(tools)}")
        lines.append("")
    if caps.py_packages:
        # Show installed Python packages PRISM looks for (subset).
        lines.append(f"**Python packages:** {', '.join(caps.py_packages)}")

    body = "\n".join(lines).rstrip()
    card = text_card(body, "Device inventory")
    card.card_data.update({
        "platform":    caps.platform,
        "has_browser": caps.has_browser,
        "cli_tools":   dict(caps.cli_tools),
        "py_packages": list(caps.py_packages),
    })
    return card
