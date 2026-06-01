"""
tests/test_device_scanner.py
============================
Tests for prism_device_scanner.py
"""
from __future__ import annotations

import shutil

from prism_device_scanner import CapabilityMap, DeviceCapabilityScanner


def test_scan_returns_capabilitymap():
    caps = DeviceCapabilityScanner().scan()
    assert isinstance(caps, CapabilityMap)
    assert isinstance(caps.cli_tools, dict)
    assert isinstance(caps.py_packages, list)
    assert isinstance(caps.platform, str)
    assert isinstance(caps.has_browser, bool)


def test_capabilitymap_has_platform():
    caps = DeviceCapabilityScanner().scan()
    assert caps.platform in ("darwin", "win32", "cygwin") or caps.platform.startswith("linux")


def test_summary_non_empty():
    caps = DeviceCapabilityScanner().scan()
    s = caps.summary()
    assert isinstance(s, str)
    assert len(s) > 0


def test_can_do_installed():
    if not shutil.which("git"):
        return  # skip — git not in PATH
    caps = DeviceCapabilityScanner().scan()
    assert caps.can_do("code") is True
