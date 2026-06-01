"""
prism_smart_home.py
===================
Home Assistant bridge — hardware control.

Connects to a local Home Assistant instance via its REST API and allows
PRISM to read sensor states and control devices (lights, switches, etc.).

All communication is local. Never connects to Home Assistant Cloud.
"""
from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SmartHomeDevice:
    entity_id:  str
    state:      str
    attributes: dict = field(default_factory=dict)
    friendly_name: str = ""

    @classmethod
    def from_ha(cls, data: dict) -> "SmartHomeDevice":
        attrs = data.get("attributes", {})
        return cls(
            entity_id     = data.get("entity_id", ""),
            state         = data.get("state", "unknown"),
            attributes    = attrs,
            friendly_name = attrs.get("friendly_name", data.get("entity_id", "")),
        )


@dataclass
class SmartHomeResult:
    success: bool
    entity_id: str
    state: str = ""
    error: str = ""

    def __bool__(self) -> bool:
        return self.success


@dataclass
class HAEntity:
    entity_id:    str
    state:        str
    attributes:   dict
    friendly_name: str = ""


class PrismSmartHome:
    """
    Bridge to a local Home Assistant instance.

    Usage:
        sh = PrismSmartHome(ha_url="http://homeassistant.local:8123",
                            token="<long-lived-access-token>")
        devices = sh.list_devices()
        sh.turn_on("light.living_room")
        sh.turn_off("switch.fan")
        sh.set_state("input_boolean.sleep_mode", "on")
    """

    def __init__(
        self,
        ha_url: str = "http://homeassistant.local:8123",
        token:  str = "",
        timeout: int = 5,
    ):
        self._url     = ha_url.rstrip("/")
        self._token   = token
        self._timeout = timeout

    @classmethod
    def from_config(cls, config: dict) -> "PrismSmartHome":
        sh = config.get("smarthome", {})
        return cls(
            ha_url = sh.get("ha_url", "http://homeassistant.local:8123"),
            token  = sh.get("ha_token", ""),
        )

    @property
    def configured(self) -> bool:
        """True when a non-empty token has been supplied."""
        return bool(self._url and self._token)

    # ── Public API ────────────────────────────────────────────────────────

    def list_devices(self, domain: str = "") -> list[SmartHomeDevice]:
        """Return all entity states, optionally filtered by domain prefix."""
        try:
            data = self._get("/api/states")
            devices = [SmartHomeDevice.from_ha(d) for d in data]
            if domain:
                devices = [d for d in devices
                           if d.entity_id.startswith(domain + ".")]
            return devices
        except Exception as exc:
            logger.debug("SmartHome list_devices error: %s", exc)
            return []

    def get_state(self, entity_id: str) -> Optional[SmartHomeDevice]:
        """Return the current state of one entity."""
        try:
            data = self._get(f"/api/states/{entity_id}")
            return SmartHomeDevice.from_ha(data)
        except Exception as exc:
            logger.debug("SmartHome get_state(%s) error: %s", entity_id, exc)
            return None

    def turn_on(self, entity_id: str, **kwargs) -> SmartHomeResult:
        """Call homeassistant.turn_on service."""
        return self._call_service("homeassistant", "turn_on",
                                  {"entity_id": entity_id, **kwargs})

    def turn_off(self, entity_id: str) -> SmartHomeResult:
        """Call homeassistant.turn_off service."""
        return self._call_service("homeassistant", "turn_off",
                                  {"entity_id": entity_id})

    def toggle(self, entity_id: str) -> SmartHomeResult:
        """Call homeassistant.toggle service."""
        return self._call_service("homeassistant", "toggle",
                                  {"entity_id": entity_id})

    def set_state(self, entity_id: str, state: str,
                  attributes: dict = None) -> SmartHomeResult:
        """Directly set an entity state (works for input_boolean, etc.)."""
        body: dict = {"state": state}
        if attributes:
            body["attributes"] = attributes
        try:
            self._post(f"/api/states/{entity_id}", body)
            return SmartHomeResult(success=True, entity_id=entity_id, state=state)
        except Exception as exc:
            return SmartHomeResult(success=False, entity_id=entity_id,
                                   error=str(exc))

    @property
    def available(self) -> bool:
        """Return True if Home Assistant is reachable."""
        try:
            self._get("/api/")
            return True
        except Exception:
            return False

    # ── New HA-style API ──────────────────────────────────────────────────

    def get_states(self) -> list[HAEntity]:
        """Return all entity states as HAEntity objects."""
        if not self.configured:
            return []
        try:
            data = self._get("/api/states")
            return [
                HAEntity(
                    entity_id     = e.get("entity_id", ""),
                    state         = e.get("state", ""),
                    attributes    = e.get("attributes", {}),
                    friendly_name = e.get("attributes", {}).get("friendly_name", ""),
                )
                for e in (data or [])
            ]
        except Exception as exc:
            logger.debug("SmartHome get_states error: %s", exc)
            return []

    def call_service(
        self,
        domain:    str,
        service:   str,
        entity_id: str = "",
        **kwargs,
    ) -> bool:
        """Call any Home Assistant service. Returns True on success."""
        payload: dict = {}
        if entity_id:
            payload["entity_id"] = entity_id
        payload.update(kwargs)
        result = self._call_service(domain, service, payload)
        return result.success

    def set_temperature(self, entity_id: str, temperature: float) -> bool:
        return self.call_service(
            "climate", "set_temperature", entity_id, temperature=temperature)

    def lock(self, entity_id: str) -> bool:
        return self.call_service("lock", "lock", entity_id)

    def unlock(self, entity_id: str) -> bool:
        return self.call_service("lock", "unlock", entity_id)

    def find_entity(self, name: str) -> Optional[HAEntity]:
        """Find an entity by friendly name (case-insensitive substring match)."""
        name_lower = name.lower()
        for entity in self.get_states():
            if (name_lower in entity.friendly_name.lower()
                    or name_lower in entity.entity_id.lower()):
                return entity
        return None

    def status_summary(self) -> dict:
        """Return a concise overview of connected entities."""
        if not self.configured:
            return {"configured": False,
                    "message": "Add ha_url and ha_token to prism_config.toml"}
        entities = self.get_states()
        on_count = sum(
            1 for e in entities
            if e.state in ("on", "open", "unlocked", "home")
        )
        return {
            "configured":     True,
            "total_entities": len(entities),
            "on_count":       on_count,
            "domains":        list({e.entity_id.split(".")[0] for e in entities}),
        }

    # ── Internal ──────────────────────────────────────────────────────────

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _get(self, path: str):
        req = urllib.request.Request(
            f"{self._url}{path}", headers=self._headers())
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read())

    def _post(self, path: str, body: dict):
        payload = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self._url}{path}", data=payload,
            headers=self._headers(), method="POST")
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read())

    def _call_service(self, domain: str, service: str,
                      data: dict) -> SmartHomeResult:
        entity_id = data.get("entity_id", "")
        try:
            self._post(f"/api/services/{domain}/{service}", data)
            return SmartHomeResult(success=True, entity_id=entity_id)
        except Exception as exc:
            return SmartHomeResult(success=False, entity_id=entity_id,
                                   error=str(exc))
