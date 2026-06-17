"""
prism_settings_store.py
=======================
SQLite-backed user-settings store at ~/.prism/settings.db.

Replaces direct TOML editing for integrations: PRISM reads this DB at
agent-init time and OVERLAYS its values on top of prism_config.toml, so
the user can configure email/calendar/smarthome/etc. by submitting a
setup-form card from the chat surface — no file editing required.

Schema:
    settings(section TEXT, key TEXT, value TEXT, updated_at REAL,
             PRIMARY KEY(section, key))

Values are stored as JSON-serialised strings so we round-trip ints,
booleans, lists, and nested dicts cleanly.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Schema registry ─────────────────────────────────────────────────────────
# Per-section field metadata so the setup_form_card builder knows what to
# render. Each field: name, label, kind (text|password|number|select|bool|url),
# placeholder, default, required, choices (for select), help.
#
# When adding a new integration, append its schema here AND make sure the
# integration's `from_config(...)` classmethod consumes the same key names.

SETTINGS_SCHEMA: dict[str, dict] = {
    "email": {
        "label": "Email (IMAP/SMTP)",
        "why": "PRISM needs IMAP/SMTP credentials to read and send mail on your behalf. For Gmail, generate an App Password — your normal password will NOT work and 2FA must be on.",
        "docs_url": "https://support.google.com/accounts/answer/185833",
        "fields": [
            {"name": "provider",  "label": "Provider",       "kind": "select", "choices": ["gmail", "imap"], "default": "gmail", "required": True},
            {"name": "address",   "label": "Email address",  "kind": "text",     "placeholder": "you@gmail.com", "required": True},
            {"name": "password",  "label": "App password",   "kind": "password", "placeholder": "16-char App Password", "required": True, "secret": True},
            {"name": "imap_host", "label": "IMAP host",      "kind": "text",     "default": "imap.gmail.com"},
            {"name": "imap_port", "label": "IMAP port",      "kind": "number",   "default": 993},
            {"name": "smtp_host", "label": "SMTP host",      "kind": "text",     "default": "smtp.gmail.com"},
            {"name": "smtp_port", "label": "SMTP port",      "kind": "number",   "default": 587},
            {"name": "max_fetch", "label": "Inbox preview size", "kind": "number", "default": 20},
        ],
    },
    "calendar": {
        "label": "Calendar",
        "why": "PRISM reads your calendar to plan around real commitments. Easiest path: paste an iCal feed URL from your calendar provider (Apple/Google support a private webcal:// link).",
        "docs_url": "",
        "fields": [
            {"name": "provider",   "label": "Provider",  "kind": "select", "choices": ["", "ical_url", "caldav", "google"], "default": "ical_url", "required": True},
            {"name": "ical_url",   "label": "iCal URL",  "kind": "url",   "placeholder": "webcal://… or https://….ics"},
            {"name": "caldav_url", "label": "CalDAV URL","kind": "url",   "placeholder": "https://server/dav"},
            {"name": "username",   "label": "Username",  "kind": "text"},
            {"name": "password",   "label": "Password",  "kind": "password", "secret": True},
            {"name": "google_token", "label": "Google OAuth2 token", "kind": "password", "secret": True},
        ],
    },
    "smarthome": {
        "label": "Smart home (Home Assistant)",
        "why": "PRISM uses a Home Assistant long-lived access token to turn lights/devices on/off on your behalf.",
        "docs_url": "https://www.home-assistant.io/docs/authentication/#your-account-profile",
        "fields": [
            {"name": "ha_url",   "label": "Home Assistant URL", "kind": "url", "placeholder": "http://homeassistant.local:8123", "required": True},
            {"name": "ha_token", "label": "Long-lived token",   "kind": "password", "required": True, "secret": True},
        ],
    },
    "search": {
        "label": "Web search",
        "why": "PRISM uses a search backend (Brave or SerpAPI) for live web lookups. Without a key, web_search falls back to scraped DuckDuckGo, which is unreliable.",
        "docs_url": "https://brave.com/search/api/",
        "fields": [
            {"name": "provider",      "label": "Provider",       "kind": "select", "choices": ["auto", "brave", "serpapi"], "default": "auto"},
            {"name": "brave_api_key", "label": "Brave API key",  "kind": "password", "secret": True},
            {"name": "serp_api_key",  "label": "SerpAPI key",    "kind": "password", "secret": True},
        ],
    },
    "push": {
        "label": "Push notifications (ntfy)",
        "why": "PRISM uses an ntfy topic to push alerts to your phone. Free at ntfy.sh — pick any topic name, subscribe to it from the ntfy app, and paste the topic here.",
        "docs_url": "https://docs.ntfy.sh/subscribe/phone/",
        "fields": [
            {"name": "topic",    "label": "ntfy topic",  "kind": "text", "placeholder": "prism-yourname-7421", "required": True},
            {"name": "server",   "label": "ntfy server", "kind": "url",  "default": "https://ntfy.sh"},
            {"name": "priority", "label": "Priority",    "kind": "select", "choices": ["default", "low", "high", "max"], "default": "default"},
        ],
    },
    "twilio": {
        "label": "Phone & SMS (Twilio)",
        "why": "PRISM uses Twilio to place phone calls and send SMS on your behalf. Get credentials at console.twilio.com.",
        "docs_url": "https://www.twilio.com/console",
        "fields": [
            {"name": "account_sid", "label": "Account SID", "kind": "text",     "placeholder": "ACxxxxxxxxxxxxxxxx", "required": True},
            {"name": "auth_token",  "label": "Auth token",  "kind": "password", "required": True, "secret": True},
            {"name": "from_number", "label": "From number", "kind": "text",     "placeholder": "+14155552671", "required": True},
        ],
    },
    "tasks": {
        "label": "Task providers (Todoist / GitHub / Linear)",
        "why": "Optional integrations for task tracking. Leave blank to use PRISM's local task DB only.",
        "docs_url": "",
        "fields": [
            {"name": "provider",       "label": "Default provider", "kind": "select", "choices": ["auto", "local", "todoist", "github", "linear"], "default": "auto"},
            {"name": "todoist_token",  "label": "Todoist token",    "kind": "password", "secret": True},
            {"name": "github_token",   "label": "GitHub token",     "kind": "password", "secret": True},
            {"name": "github_repo",    "label": "GitHub repo",      "kind": "text", "placeholder": "owner/repo"},
            {"name": "linear_api_key", "label": "Linear API key",   "kind": "password", "secret": True},
        ],
    },
}


# ── Store ─────────────────────────────────────────────────────────────────


class SettingsStore:
    def __init__(self, db_path: str = "~/.prism/settings.db") -> None:
        resolved = Path(db_path).expanduser()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(resolved)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    section    TEXT NOT NULL,
                    key        TEXT NOT NULL,
                    value      TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (section, key)
                )
            """)
            conn.commit()

    def get_section(self, section: str) -> dict[str, Any]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key, value FROM settings WHERE section = ?",
                (section,),
            ).fetchall()
        out: dict[str, Any] = {}
        for r in rows:
            try:
                out[r["key"]] = json.loads(r["value"])
            except Exception:
                out[r["key"]] = r["value"]
        return out

    def set_section(self, section: str, values: dict[str, Any]) -> None:
        now = time.time()
        with self._lock:
            with self._connect() as conn:
                for k, v in values.items():
                    conn.execute(
                        """
                        INSERT INTO settings (section, key, value, updated_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(section, key)
                        DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                        """,
                        (section, k, json.dumps(v), now),
                    )
                conn.commit()

    def clear_section(self, section: str) -> int:
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM settings WHERE section = ?", (section,)
                )
                conn.commit()
                return cur.rowcount

    def overlay_on_toml(self, toml_config: dict) -> dict:
        """
        Return a new config dict with settings.db values overlaid on top of
        the TOML config. DB wins on conflict; missing sections fall through.
        """
        merged = dict(toml_config or {})
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT section, key, value FROM settings"
            ).fetchall()
        for r in rows:
            section = r["section"]
            key = r["key"]
            try:
                value = json.loads(r["value"])
            except Exception:
                value = r["value"]
            existing = merged.get(section)
            if isinstance(existing, dict):
                existing[key] = value
            else:
                merged[section] = {key: value}
        return merged


# ── Module-level singleton ─────────────────────────────────────────────────

_store: Optional[SettingsStore] = None
_store_lock = threading.Lock()


def get_settings_store(db_path: str = "~/.prism/settings.db") -> SettingsStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = SettingsStore(db_path)
    return _store


def reset_settings_store(db_path: str = "~/.prism/settings.db") -> SettingsStore:
    global _store
    with _store_lock:
        _store = SettingsStore(db_path)
    return _store


# ── Validation helpers ─────────────────────────────────────────────────────


def schema_for(section: str) -> Optional[dict]:
    return SETTINGS_SCHEMA.get(section)


def validate_values(section: str, values: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    """
    Coerce and validate user-submitted values against the section schema.
    Returns (ok, error_message, coerced_values).
    """
    schema = SETTINGS_SCHEMA.get(section)
    if schema is None:
        return False, f"Unknown settings section: {section}", {}

    coerced: dict[str, Any] = {}
    for field in schema["fields"]:
        name = field["name"]
        kind = field.get("kind", "text")
        raw = values.get(name)
        # Empty input → use default if available, else skip if not required
        if raw in (None, ""):
            if "default" in field:
                coerced[name] = field["default"]
            elif field.get("required"):
                return False, f"'{field.get('label', name)}' is required", {}
            else:
                continue
            continue

        if kind == "number":
            try:
                coerced[name] = int(raw) if str(raw).isdigit() else float(raw)
            except Exception:
                return False, f"'{field.get('label', name)}' must be a number", {}
        elif kind == "bool":
            coerced[name] = str(raw).lower() in ("true", "yes", "1", "on")
        elif kind == "select":
            choices = field.get("choices", [])
            if raw not in choices:
                return False, f"'{field.get('label', name)}' must be one of: {', '.join(choices)}", {}
            coerced[name] = raw
        else:
            coerced[name] = str(raw)

    return True, "", coerced
