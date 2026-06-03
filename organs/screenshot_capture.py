"""Bundled organ: screenshot_capture — capture a screenshot using mss."""
ORGAN_META = {
    "intent":      "screenshot_capture",
    "description": "capture a screenshot and save it to ~/.prism/screenshots/",
    "version":     "1.0",
}

ORGAN_POLICY = {
    "risk_level":        "low",
    "requires_approval": False,
    "irreversible":      False,
    "max_per_session":   None,
}


def _parse_monitor(message: str) -> int:
    """Return monitor index (0 = all monitors combined, 1+ = specific)."""
    import re
    m = re.search(r'monitor\s+(\d+)', message, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 1  # default: primary monitor


def execute(intent: str, message: str, ctx: dict):
    import datetime
    from pathlib import Path

    from prism_responses import text_card

    try:
        import mss  # type: ignore[import]
        import mss.tools  # type: ignore[import]
    except ImportError:
        return text_card(
            "mss library not installed. Run: pip install mss\n"
            "Then try again.",
            "Screenshot",
        )

    screenshots_dir = Path("~/.prism/screenshots").expanduser()
    try:
        screenshots_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return text_card(f"Could not create screenshots directory: {exc}", "Screenshot")

    monitor_idx = _parse_monitor(message)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{timestamp}_mon{monitor_idx}.png"
    out_path = screenshots_dir / filename

    try:
        with mss.mss() as sct:
            monitors = sct.monitors
            if monitor_idx >= len(monitors):
                return text_card(
                    f"Monitor {monitor_idx} not found. "
                    f"Available monitors: 0–{len(monitors)-1}",
                    "Screenshot",
                )
            sct_img = sct.grab(monitors[monitor_idx])
            mss.tools.to_png(sct_img.rgb, sct_img.size, output=str(out_path))
    except Exception as exc:
        return text_card(f"Screenshot failed: {exc}", "Screenshot")

    size_kb = out_path.stat().st_size // 1024 if out_path.exists() else 0
    return text_card(
        f"Screenshot saved: {out_path}\n"
        f"Size: {size_kb} KB  |  Monitor: {monitor_idx}\n"
        f"Timestamp: {timestamp}",
        "Screenshot",
    )
