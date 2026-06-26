"""bluetooth_control routing for issue #28 bug 73.

Live probes::

  user: "turn on bluetooth"   → smart_home (Smart Home Status, 0 devices)
  user: "bluetooth on"        → general_chat (TIMEOUT on LLM)
  user: "enable bluetooth"    → general_chat
  user: "disable bluetooth"   → general_chat

Bluetooth radio control is a fundamental local hardware action.
smart_home's `turn (?:on|off)` claims "turn on bluetooth" but treats it
as an IoT device, not the local Bluetooth radio.

Fix: dedicated ``bluetooth_control`` intent + organ. Hoisted above
smart_home, scoped to the literal noun "bluetooth".
"""
from __future__ import annotations

from prism_intents import INTENTS
from prism_routing import route_intent


def _route(message: str) -> str:
    return route_intent(message, INTENTS, lambda _m: None)


class TestBluetoothToggle:

    def test_bluetooth_on(self):
        assert _route("bluetooth on") == "bluetooth_control"

    def test_bluetooth_off(self):
        assert _route("bluetooth off") == "bluetooth_control"

    def test_turn_on_bluetooth(self):
        assert _route("turn on bluetooth") == "bluetooth_control"

    def test_turn_off_bluetooth(self):
        assert _route("turn off bluetooth") == "bluetooth_control"

    def test_enable_bluetooth(self):
        assert _route("enable bluetooth") == "bluetooth_control"

    def test_disable_bluetooth(self):
        assert _route("disable bluetooth") == "bluetooth_control"

    def test_bluetooth_status(self):
        assert _route("bluetooth status") == "bluetooth_control"

    def test_is_bluetooth_on(self):
        assert _route("is bluetooth on") == "bluetooth_control"


class TestNoOverclaim:

    def test_smart_home_still_works(self):
        assert _route("smart home") == "smart_home"

    def test_turn_on_lights_still_smart_home(self):
        assert _route("turn on the lights") in {"smart_home", "smart_home_control"}

    def test_lock_screen_still_system_lock(self):
        assert _route("lock screen") == "system_lock"

    def test_volume_up_still_volume_control(self):
        assert _route("volume up") == "volume_control"
