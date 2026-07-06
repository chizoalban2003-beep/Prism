"""
tests/test_desktop_caps_issue_28.py
===================================
Desktop capability introspection (command-centre honesty): the scanner is
session/display aware, can_control_gui() requires BOTH a display and an
installed backend, the report degrades with install hints, and the
desktop_capabilities intent routes distinctly from device_inventory.
"""
from __future__ import annotations

from prism_device_agent import (
    _DESKTOP_CATEGORIES,
    CapabilityMap,
    DeviceCapabilityScanner,
)
from prism_intents import INTENTS
from prism_routing import route_intent


def _route(m):
    return route_intent(m, INTENTS, lambda _m: "")


def _cap(**kw):
    base = dict(cli_tools={}, py_packages=[], platform="linux",
                has_browser=False)
    base.update(kw)
    return CapabilityMap(**base)


class TestCanControlGui:
    def test_display_and_backend_required(self):
        # display present + a window backend installed → GUI controllable
        c = _cap(has_display=True, cli_tools={"window_manage": ["wmctrl"]})
        assert c.can_control_gui() is True

    def test_backend_without_display_is_false(self):
        # installed binary is useless on a headless session
        c = _cap(has_display=False, cli_tools={"window_manage": ["wmctrl"]})
        assert c.can_control_gui() is False

    def test_display_without_any_backend_is_false(self):
        c = _cap(has_display=True, cli_tools={})
        assert c.can_control_gui() is False


class TestReport:
    def test_headless_warns_plainly(self):
        c = _cap(has_display=False, session_type="",
                 cli_tools={"window_manage": ["wmctrl"]})
        report = c.desktop_report()
        assert "No graphical display" in report
        # an installed-but-unusable backend is marked, not claimed available
        assert "installed, but no display" in report

    def test_missing_backend_shows_install_hint(self):
        c = _cap(has_display=True, session_type="wayland", cli_tools={})
        report = c.desktop_report()
        assert "✗ window_manage" in report
        assert "apt install" in report  # a hint is present

    def test_available_backend_marked(self):
        c = _cap(has_display=True, session_type="x11",
                 cli_tools={"notify": ["notify-send"]})
        report = c.desktop_report()
        assert "✓ notify → notify-send" in report

    def test_every_desktop_category_appears(self):
        c = _cap(has_display=True)
        report = c.desktop_report()
        for cat in _DESKTOP_CATEGORIES:
            assert cat in report


class TestScanner:
    def test_scan_populates_session_fields(self, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", "KDE")
        cm = DeviceCapabilityScanner().scan()
        assert cm.session_type == "wayland"
        assert cm.has_display is True
        assert cm.desktop_env == "KDE"

    def test_headless_linux_has_no_display(self, monkeypatch):
        for var in ("WAYLAND_DISPLAY", "DISPLAY", "XDG_SESSION_TYPE"):
            monkeypatch.delenv(var, raising=False)
        cm = DeviceCapabilityScanner().scan()
        # only meaningful on linux; darwin/win always report a display
        if cm.platform not in ("darwin", "win32"):
            assert cm.has_display is False


class TestRouting:
    def test_control_phrasing_hits_desktop_capabilities(self):
        for m in ("what can you control",
                  "can you control my desktop",
                  "can you control my windows",
                  "desktop control capabilities",
                  "what can prism control"):
            assert _route(m) == "desktop_capabilities", m

    def test_hardware_phrasing_still_device_inventory(self):
        assert _route("what hardware do I have") == "device_inventory"
        assert _route("list my devices") == "device_inventory"
