"""
kde_config.py
=============
KDE Sports Agent — TOML / JSON Config Loader

Loads KDEAgent configuration from a TOML or JSON file and optionally
auto-registers devices declared in [[devices]] sections.

Config file search order (first found wins):
    1. Explicit path passed to load_config()
    2. $KDE_CONFIG environment variable
    3. ~/.kde/kde.toml
    4. ~/.kde/kde.json
    5. ~/.kde/config.toml
    6. ./prism_config.toml
    7. ./prism_config.json
    8. ./kde_config.toml
    9. ./kde_config.json

TOML schema:
    [agent]
    name="Marcus"; role="athlete"; sport="Football"; team="City FC"
    db_path="~/.kde/kde.db"; media_dir="~/.kde/media"
    ollama_host="http://localhost:11434"; ollama_model="llava"
    text_model="mistral"; ffmpeg_path="ffmpeg"
    auto_watch=true; poll_interval=30

    [[devices]]
    name="GoPro Hero 12"; type="gopro"
    watch_path="~/GoPro/DCIM"; api_url="http://10.5.5.9:8080"

    [[devices]]
    name="Apple Watch"; type="apple_watch"
    watch_path="~/Downloads/apple_health_export"

    [[devices]]
    name="Garmin Forerunner"; type="garmin"
    watch_path="~/Garmin/Activities"
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from device_hub import DeviceType
from kde_agent import KDEAgent, KDEConfig
from kde_profiles import from_toml
from sports_pro import Role

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Search paths
# ---------------------------------------------------------------------------

_SEARCH_PATHS = [
    Path("~/.kde/config.toml"),
    Path("~/.kde/kde.toml"),
    Path("~/.kde/kde.json"),
    Path("./prism_config.toml"),
    Path("./prism_config.json"),
    Path("./kde_config.toml"),
    Path("./kde_config.json"),
]

# Device type aliases
_DEVICE_TYPE_MAP: dict[str, DeviceType] = {
    "gopro":        DeviceType.GOPRO,
    "gopro_hero":   DeviceType.GOPRO,
    "phone":        DeviceType.PHONE_CAMERA,
    "phone_camera": DeviceType.PHONE_CAMERA,
    "drone":        DeviceType.DRONE,
    "whoop":        DeviceType.WEARABLE_WHOOP,
    "garmin":       DeviceType.WEARABLE_GARMIN,
    "apple_watch":  DeviceType.WEARABLE_APPLE,
    "apple":        DeviceType.WEARABLE_APPLE,
    "oura":         DeviceType.WEARABLE_OURA,
    "oura_ring":    DeviceType.WEARABLE_OURA,
    "gps":          DeviceType.GPS_TRACKER,
    "gps_tracker":  DeviceType.GPS_TRACKER,
    "hrm":          DeviceType.HRM,
    "heart_rate":   DeviceType.HRM,
    "csv":          DeviceType.TRACKING_CSV,
    "manual":       DeviceType.MANUAL,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(path: str = None) -> KDEConfig:
    """
    Load KDEConfig from a TOML or JSON file.

    Parameters
    ----------
    path : explicit path to config file (optional)

    Returns a KDEConfig populated from the file, merged with defaults.
    """
    raw = _find_and_parse(path)
    return _build_kde_config(raw.get("agent", raw))


def build_agent_from_config(path: str = None) -> KDEAgent:
    """
    Load a KDEConfig, create a KDEAgent, and auto-register any [[devices]].

    Parameters
    ----------
    path : explicit path to config file (optional)
    """
    raw    = _find_and_parse(path)
    agent_section = raw.get("agent", raw)

    # Build KDEConfig
    cfg = _build_kde_config(agent_section)

    if "user" in raw:
        agent = KDEAgent.setup(config_path=path, config=cfg) if path is None else KDEAgent.setup(profile=from_toml(path), config=cfg)
    else:
        name = agent_section.get("name", "KDE User")
        role = _parse_role(agent_section.get("role", "athlete"))
        sport = agent_section.get("sport", "general")
        team = agent_section.get("team", "")
        agent = KDEAgent.setup(name=name, role=role, sport=sport, team=team, config=cfg)

    # Register devices
    devices_section = raw.get("devices", [])
    for dev_cfg in devices_section:
        _register_device(agent, dev_cfg)

    return agent


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _find_and_parse(explicit_path: str = None) -> dict:
    """Locate and parse the first available config file."""
    candidates: list[Path] = []

    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())

    env_path = os.environ.get("KDE_CONFIG")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.extend(p.expanduser() for p in _SEARCH_PATHS)

    for p in candidates:
        if p.exists():
            logger.info("Loading config from: %s", p)
            return _parse_file(p)

    logger.warning("No config file found — using defaults")
    return {}


def _parse_file(path: Path) -> dict:
    suffix = path.suffix.lower()
    text   = path.read_text(encoding="utf-8")

    if suffix == ".toml":
        return _parse_toml(text)
    if suffix == ".json":
        return json.loads(text)

    # Try TOML first, then JSON
    try:
        return _parse_toml(text)
    except Exception:
        return json.loads(text)


def _parse_toml(text: str) -> dict:
    """Parse TOML using stdlib tomllib (Python 3.11+) or third-party tomli."""
    try:
        import tomllib  # Python 3.11+
        return tomllib.loads(text)
    except ImportError:
        pass
    try:
        import tomli  # pip install tomli
        return tomli.loads(text)
    except ImportError:
        pass
    # Minimal hand-rolled TOML parser for simple key=value configs
    return _simple_toml_parse(text)


def _simple_toml_parse(text: str) -> dict:
    """
    Very limited TOML parser: handles [sections], [[array-sections]], key=value.
    Sufficient for the kde_config schema. No nested tables beyond depth-1.
    """
    result: dict = {}
    current_section: dict = result
    array_key: str = ""

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        # Array section header [[key]]
        if line.startswith("[[") and line.endswith("]]"):
            array_key = line[2:-2].strip()
            if array_key not in result:
                result[array_key] = []
            new_item: dict = {}
            result[array_key].append(new_item)
            current_section = new_item
            continue

        # Regular section header [key]
        if line.startswith("[") and line.endswith("]"):
            section_name = line[1:-1].strip()
            if section_name not in result:
                result[section_name] = {}
            current_section = result[section_name]
            continue

        # Key = value
        if "=" in line:
            key, _, raw_val = line.partition("=")
            key = key.strip()
            raw_val = raw_val.strip()

            # Handle inline comments
            if "#" in raw_val and not raw_val.startswith('"') and not raw_val.startswith("'"):
                raw_val = raw_val.partition("#")[0].strip()

            current_section[key] = _parse_toml_value(raw_val)

    return result


def _parse_toml_value(raw: str) -> Any:
    """Parse a scalar TOML value."""
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    if raw.startswith("'") and raw.endswith("'"):
        return raw[1:-1]
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _build_kde_config(section: dict) -> KDEConfig:
    """Build a KDEConfig from a parsed dict."""
    return KDEConfig(
        db_path       = _expand(section.get("db_path",       "~/.kde/kde.db")),
        media_dir     = _expand(section.get("media_dir",     "~/.kde/media")),
        ollama_host   = section.get("ollama_host",   "http://localhost:11434"),
        ollama_model  = section.get("ollama_model",  "llava"),
        text_model    = section.get("text_model",    "mistral"),
        ffmpeg_path   = section.get("ffmpeg_path",   "ffmpeg"),
        poll_interval = int(section.get("poll_interval", 30)),
        auto_watch    = bool(section.get("auto_watch",    True)),
    )


def _parse_role(role_str: str) -> Role:
    role_str = role_str.lower().strip()
    for role in Role:
        if role.value == role_str or role.name.lower() == role_str:
            return role
    logger.warning("Unknown role '%s', defaulting to ATHLETE", role_str)
    return Role.ATHLETE


def _register_device(agent: KDEAgent, dev_cfg: dict) -> None:
    name       = dev_cfg.get("name", "Unknown Device")
    type_str   = dev_cfg.get("type", "manual").lower()
    watch_path = _expand(dev_cfg.get("watch_path", ""))
    api_url    = dev_cfg.get("api_url", "")

    device_type = _DEVICE_TYPE_MAP.get(type_str, DeviceType.MANUAL)
    try:
        device_id = agent.add_device(
            name        = name,
            device_type = device_type,
            watch_path  = watch_path,
            api_url     = api_url,
        )
        logger.info("Registered device '%s' → %s", name, device_id)
    except Exception as exc:
        logger.warning("Failed to register device '%s': %s", name, exc)


def _expand(path: str) -> str:
    return str(Path(path).expanduser()) if path else path
