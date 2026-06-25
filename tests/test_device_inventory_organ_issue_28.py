"""device_inventory organ + intent routing for issue #28 bug 49.

Live test: PRISM is supposed to be a bridge between the user and their
hardware / applications. Asking "what hardware do I have" routed to
five different unrelated organs depending on phrasing:

  "what hardware do i have"        → Your crystallised profile
  "list my devices"                → Status (KDE/KSA)
  "show device capabilities"       → Status (KDE/KSA)
  "what is on this computer"       → Smart Home Status
  "what can you access"            → Agents inventory (LLMs, not hardware)

None of them returned actual platform / browser / CLI / package info,
yet the ``/device/capabilities`` HTTP endpoint already exposes exactly
that data via ``DeviceCapabilityScanner``.

Fix:
* New bundled organ ``device_inventory`` wrapping ``DeviceCapabilityScanner``.
* New intent regex declared early so it beats the broader matchers.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from prism_intents import INTENTS

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "_device_inventory_organ",
    _PROJECT_ROOT / "organs" / "device_inventory.py",
)
_organ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_organ)


def _route(text: str) -> str:
    lowered = text.lower()
    for pattern, intent in INTENTS:
        if re.search(pattern, lowered):
            return intent
    return ""


class TestIntentRouting:
    """The reported phrasings must all land on device_inventory."""

    def test_what_hardware_do_i_have(self):
        assert _route("what hardware do i have") == "device_inventory"

    def test_list_my_devices(self):
        assert _route("list my devices") == "device_inventory"

    def test_show_device_capabilities(self):
        assert _route("show device capabilities") == "device_inventory"

    def test_what_is_on_this_computer(self):
        assert _route("what is on this computer") == "device_inventory"

    def test_which_cli_tools_can_you_use(self):
        assert _route("which CLI tools can you use") == "device_inventory"

    def test_which_browsers_are_available(self):
        assert _route("which browsers are available") == "device_inventory"

    def test_hardware_inventory(self):
        assert _route("hardware inventory") == "device_inventory"


class TestDoesNotOverreach:
    """The widened regex must not steal from existing intents."""

    def test_status_alone_still_status(self):
        # The bare-word status intent must still win for plain "status".
        assert _route("status") == "status"

    def test_smart_home_lights_unaffected(self):
        # "turn on the lights" still routes to the (earlier-declared)
        # smart_home intent — confirming device_inventory didn't steal it.
        assert _route("turn on the lights") == "smart_home"

    def test_what_is_python_remains_wiki(self):
        assert _route("what is python") == "wikipedia_lookup"

    def test_what_tools_can_you_run_keeps_list_tools(self):
        # The "learned tools" registry is a different concept from device
        # hardware — that intent should still claim its phrasings.
        assert _route("what tools can you run") == "list_tools"


class TestOrganExecutesEndToEnd:
    """The organ must build a real card with the scanner's actual data."""

    def test_execute_returns_card_with_platform(self):
        card = _organ.execute("device_inventory", "what hardware do i have", {})
        assert card.title == "Device inventory"
        # Body must contain the platform line at minimum.
        assert "Platform:" in card.body
        # card_data structured fields must be populated.
        d = card.card_data
        assert "platform" in d
        assert "has_browser" in d
        assert isinstance(d.get("cli_tools"), dict)

    def test_execute_handles_scanner_import_failure(self, monkeypatch):
        # Force the import path to fail and confirm the organ degrades
        # gracefully rather than raising.
        import sys
        monkeypatch.setitem(sys.modules, "prism_device_agent", None)
        card = _organ.execute("device_inventory", "what hardware do i have", {})
        # The card should still come back — just an explanatory body.
        assert card.title == "Device inventory"
        assert "unavailable" in card.body.lower() or "could not" in card.body.lower()
