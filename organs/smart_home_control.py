"""Bundled organ: smart_home_control — control devices via Home Assistant REST API."""
ORGAN_META = {
    "intent":      "smart_home_control",
    "description": "control smart home devices via Home Assistant API (turn on/off, set state)",
    "version":     "1.0",
}

ORGAN_POLICY = {
    "risk_level":        "medium",
    "requires_approval": True,
    "irreversible":      False,
    "max_per_session":   None,
}


def _get_config(ctx: dict) -> tuple:
    """Return (ha_url, ha_token)."""
    import re
    cfg = ctx.get("home_assistant_config") or {}
    url = cfg.get("url", "").strip().rstrip("/")
    token = cfg.get("token", "").strip()

    if not url or not token:
        try:
            from pathlib import Path
            env = Path("/proc/self/environ").read_text(errors="replace")
            if not url:
                m = re.search(r'HA_URL=([^\x00]+)', env)
                if m:
                    url = m.group(1).strip().rstrip("/")
            if not token:
                m = re.search(r'HA_TOKEN=([^\x00]+)', env)
                if m:
                    token = m.group(1).strip()
        except Exception:
            pass
    return url, token


def _parse_action(message: str) -> tuple:
    """Return (action, entity_hint, extra) — action in {turn_on, turn_off, toggle, status}."""
    import re
    msg = message.lower()

    # Brightness
    bright_m = re.search(r'brightness\s+(?:to\s+)?(\d+)', msg)
    brightness = int(bright_m.group(1)) if bright_m else None

    # Entity hint
    entity_m = re.search(
        r'(?:the\s+|my\s+)?(?:entity\s+id\s+)?'
        r'([\w.]+\.[\w]+|light\b|switch\b|fan\b|cover\b|climate\b|sensor\b)',
        message, re.IGNORECASE,
    )
    entity = entity_m.group(1).strip().lower() if entity_m else ""

    if any(w in msg for w in ("turn on", "switch on", "enable", "start")):
        return "turn_on", entity, {"brightness_pct": brightness}
    if any(w in msg for w in ("turn off", "switch off", "disable", "stop")):
        return "turn_off", entity, {}
    if "toggle" in msg:
        return "toggle", entity, {}
    return "status", entity, {}


def execute(intent: str, message: str, ctx: dict):
    import json
    import urllib.request

    from prism_responses import text_card

    ha_url, ha_token = _get_config(ctx)
    if not ha_url:
        return text_card(
            "Home Assistant URL not configured.\n"
            "Add home_assistant_config={'url': 'http://homeassistant.local:8123', "
            "'token': '...'} to ctx\nor set HA_URL and HA_TOKEN env vars.",
            "Smart Home",
        )
    if not ha_token:
        return text_card(
            "Home Assistant token not configured.\n"
            "Set HA_TOKEN env var or home_assistant_config['token'].",
            "Smart Home",
        )

    action, entity_hint, extra = _parse_action(message)
    headers = {
        "Authorization": f"Bearer {ha_token}",
        "Content-Type": "application/json",
        "User-Agent": "PRISM/1.0",
    }

    if action == "status":
        # List entity states
        endpoint = f"{ha_url}/api/states"
        if entity_hint and "." in entity_hint:
            endpoint = f"{ha_url}/api/states/{entity_hint}"
        req = urllib.request.Request(endpoint, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read())
        except Exception as exc:
            return text_card(f"Home Assistant request failed: {exc}", "Smart Home")

        if isinstance(data, list):
            lines = [f"Home Assistant — {len(data)} entities"]
            for item in data[:15]:
                eid = item.get("entity_id", "?")
                state = item.get("state", "?")
                lines.append(f"  {eid}: {state}")
            if len(data) > 15:
                lines.append(f"  ... and {len(data)-15} more")
            return text_card("\n".join(lines), "Smart Home")

        eid = data.get("entity_id", entity_hint)
        state = data.get("state", "unknown")
        attrs = data.get("attributes", {})
        friendly = attrs.get("friendly_name", eid)
        return text_card(
            f"Entity: {friendly} ({eid})\nState: {state}",
            "Smart Home",
        )

    # Service call: turn_on, turn_off, toggle
    domain = "homeassistant"
    if "." in entity_hint:
        domain = entity_hint.split(".")[0]

    service_url = f"{ha_url}/api/services/{domain}/{action}"
    payload_data: dict = {}
    if entity_hint and "." in entity_hint:
        payload_data["entity_id"] = entity_hint
    if action == "turn_on" and extra.get("brightness_pct") is not None:
        payload_data["brightness_pct"] = extra["brightness_pct"]

    if not payload_data.get("entity_id") and not entity_hint:
        return text_card(
            "Could not determine which device to control.\n"
            "Example: 'turn on light.living_room' or 'turn off switch.kitchen'",
            "Smart Home",
        )

    payload = json.dumps(payload_data).encode("utf-8")
    req = urllib.request.Request(
        service_url, data=payload, method="POST", headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            result = json.loads(resp.read())
    except Exception as exc:
        return text_card(f"Home Assistant service call failed: {exc}", "Smart Home")

    changed = [s.get("entity_id", "?") for s in result] if isinstance(result, list) else []
    import re as _re2
    action_label = _re2.sub("_", " ", action)
    if changed:
        return text_card(
            f"Home Assistant: {action_label} called.\n"
            f"Affected: {', '.join(changed)}",
            "Smart Home",
        )
    return text_card(f"Home Assistant: {action_label} called.", "Smart Home")
