"""hardware_status intent for issue #28 bug 58.

Live probes (hardware-state queries the user asked):

  user: "battery level"     → Time card "It is 1:55 AM."
  user: "free disk space"   → Device inventory (lists tar/find tools)
  user: "is wifi connected" → Status card ("KDE: offline. KSA: active")
  user: "cpu usage"         → general_chat (no LLM fallback gives real number)
  user: "memory free"       → general_chat

Per the user's CEO directive — "limit users interaction with their
hardware to permissions, instructions, notifications, budgets and
policies. acting as a bridge for the user to share assets,
communication, automation between applications and the hardware
components" — these hardware-state observations should hit a single,
read-only ``hardware_status`` intent that returns real telemetry
(psutil + shutil), no approval flow, no LLM hop.

Fix:

1. prism_intents.INTENTS: add hardware_status regex covering battery,
   disk, memory/ram, cpu/load, wifi/network, uptime, system/hardware
   status. Placed before device_task so reads bypass approval.
2. prism_pa_intents.handle_pa_intent: hardware_status branch calls a
   focused renderer that picks the relevant subset of metrics based on
   keywords in the message — so "battery level" gives a one-line
   battery card, "system status" gives the full readout.
"""
from __future__ import annotations

from unittest import mock

from prism_intents import INTENTS
from prism_pa_intents import _hardware_status_card
from prism_routing import route_intent


def _route(message: str) -> str:
    return route_intent(message, INTENTS, lambda _m: None)


# ---------------------------------------------------------------------
# 1. Routing
# ---------------------------------------------------------------------

class TestRoutingBattery:
    def test_battery_level(self):
        assert _route("battery level") == "hardware_status"

    def test_battery_status(self):
        assert _route("battery status") == "hardware_status"

    def test_battery_percent(self):
        assert _route("battery percentage") == "hardware_status"


class TestRoutingDisk:
    def test_free_disk_space(self):
        assert _route("free disk space") == "hardware_status"

    def test_disk_usage(self):
        assert _route("disk usage") == "hardware_status"

    def test_how_much_disk(self):
        assert _route("how much disk") == "hardware_status"


class TestRoutingMemoryCpu:
    def test_memory_free(self):
        assert _route("memory free") == "hardware_status"

    def test_ram_usage(self):
        assert _route("ram usage") == "hardware_status"

    def test_cpu_usage(self):
        assert _route("cpu usage") == "hardware_status"

    def test_system_load(self):
        assert _route("system load") == "hardware_status"

    def test_load_average(self):
        assert _route("load average") == "hardware_status"


class TestRoutingNetwork:
    def test_wifi_connected(self):
        assert _route("wifi connected") == "hardware_status"

    def test_network_status(self):
        assert _route("network status") == "hardware_status"

    def test_internet_working(self):
        assert _route("internet working") == "hardware_status"


class TestRoutingUmbrella:
    def test_hardware_status(self):
        assert _route("hardware status") == "hardware_status"

    def test_system_status(self):
        # "status" alone has its own intent; "system status" is hardware.
        assert _route("system status") == "hardware_status"

    def test_uptime(self):
        assert _route("uptime") == "hardware_status"


class TestNoRegression:
    """Existing intents must not lose their phrasings."""

    def test_bare_status(self):
        assert _route("status") == "status"

    def test_git_status_is_shell_not_daemon_status(self):
        # The Developer chip sends the literal text "Git status" — it used
        # to hit the generic \bstatus\b pattern and answer with the
        # daemon-connectivity card instead of anything git-related.
        assert _route("git status") == "shell_run"
        assert _route("git diff") == "shell_run"

    def test_what_day_is_it(self):
        assert _route("what day is it") == "clock_query"

    def test_show_my_budget(self):
        assert _route("show my budget") == "budget_status"


# ---------------------------------------------------------------------
# 2. Handler — picks the relevant subset based on keywords
# ---------------------------------------------------------------------

class TestHandlerKeywordSelection:

    def test_battery_only_renders_battery_line(self):
        card = _hardware_status_card("battery level")
        assert card.title == "Battery"
        assert "• Battery" in card.body
        assert "• Disk" not in card.body
        assert "• Memory" not in card.body

    def test_disk_only_renders_disk_line(self):
        card = _hardware_status_card("free disk space")
        assert card.title == "Disk"
        assert "• Disk /" in card.body
        assert "GB free" in card.body

    def test_uptime_only_renders_uptime(self):
        card = _hardware_status_card("uptime")
        assert card.title == "Uptime"
        assert "• Uptime:" in card.body

    def test_umbrella_renders_all_sections(self):
        card = _hardware_status_card("system status")
        assert card.title == "System status"
        # All major sections present.
        for sec in ("Disk", "Memory", "CPU"):
            assert f"• {sec}" in card.body


class TestHandlerErrorIsolation:
    """Each section is wrapped — one psutil failure must not blank the
    whole card."""

    def test_battery_disabled_falls_back_to_message(self):
        with mock.patch("psutil.sensors_battery", return_value=None):
            card = _hardware_status_card("battery")
        assert "no battery" in card.body.lower() or "desktop" in card.body.lower()

    def test_individual_failure_isolated(self):
        # If battery raises, disk readings must still render in an
        # umbrella call.
        def _boom():
            raise OSError("permission denied")
        with mock.patch("psutil.sensors_battery", side_effect=_boom):
            card = _hardware_status_card("system status")
        assert "Battery: unavailable" in card.body
        assert "• Disk" in card.body

    def test_memory_line_actually_renders(self):
        # Regression: psutil's virtual_memory().percent is a FLOAT, and
        # "█" * float raises TypeError — the umbrella test above passed
        # because "• Memory" is also a prefix of the fallback line
        # "• Memory: unavailable (TypeError)". Pin real GB output.
        card = _hardware_status_card("memory usage")
        assert "Memory: unavailable" not in card.body
        assert "GB free of" in card.body
