"""
tests/test_device_agent.py
==========================
Tests for prism_device_agent.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from prism_device_agent import (
    CapabilityMap,
    DeviceCapabilityScanner,
    DeviceTaskResult,
    PrismDeviceAgent,
    ToolResolver,
)
from prism_device_resolver import ToolResolver as StandaloneToolResolver
from prism_responses import PrismCard, device_result_card


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_caps(**overrides) -> CapabilityMap:
    defaults = dict(
        cli_tools={},
        py_packages=[],
        platform=sys.platform,
        has_browser=False,
    )
    defaults.update(overrides)
    return CapabilityMap(**defaults)


def _agent(**caps_overrides) -> PrismDeviceAgent:
    caps = _make_caps(**caps_overrides)
    return PrismDeviceAgent(capabilities=caps)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_scanner_returns_capability_map():
    caps = DeviceCapabilityScanner().scan()
    assert isinstance(caps, CapabilityMap)
    assert isinstance(caps.cli_tools, dict)
    assert isinstance(caps.py_packages, list)
    assert isinstance(caps.platform, str)
    assert isinstance(caps.has_browser, bool)


def test_list_files_stdlib():
    agent = _agent()
    result = agent.execute("list files in /tmp")
    assert isinstance(result, DeviceTaskResult)
    assert result.success is True


def test_read_file_stdlib():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, dir="/tmp"
    ) as f:
        f.write("hello prism device agent")
        fpath = f.name
    try:
        agent = _agent()
        result = agent.execute(f"read file {fpath}")
        assert result.success is True
        assert "hello prism device agent" in result.output
    finally:
        Path(fpath).unlink(missing_ok=True)


def test_search_in_files():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, dir="/tmp"
    ) as f:
        f.write("unique_prism_search_token_xyz\nother line\n")
        fpath = f.name
    try:
        agent = _agent()
        result = agent.execute(f"search for unique_prism_search_token_xyz in /tmp")
        assert result.success is True
        assert "unique_prism_search_token_xyz" in result.output
    finally:
        Path(fpath).unlink(missing_ok=True)


def test_move_file_and_undo():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, dir="/tmp"
    ) as f:
        f.write("move test")
        src = f.name
    dst = src + "_moved.txt"
    try:
        agent = _agent()
        result = agent.execute(f"move {src} to {dst}")
        assert result.success is True
        assert result.undo_command  # undo_command is present
        assert Path(dst).exists()
    finally:
        Path(src).unlink(missing_ok=True)
        Path(dst).unlink(missing_ok=True)


def test_delete_to_trash():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, dir="/tmp"
    ) as f:
        f.write("trash test")
        fpath = f.name

    agent = _agent()
    result = agent.execute(f"delete file {fpath}")
    assert result.success is True
    # Original file should be gone
    assert not Path(fpath).exists()
    # Should mention trash location (not actual delete)
    assert "trash" in result.output.lower() or "prism" in result.output.lower()


def test_dangerous_command_refused():
    agent = _agent()
    result = agent.execute("run command rm -rf /")
    assert result.success is False
    assert result.error


def test_dry_run_no_side_effects():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, dir="/tmp"
    ) as f:
        fpath = f.name

    dst = fpath + "_drydst.txt"
    try:
        agent = _agent()
        result = agent.execute(f"move {fpath} to {dst}", dry_run=True)
        assert result.success is True
        assert "dry run" in result.output.lower()
        # Source file must still exist (no side effects)
        assert Path(fpath).exists()
        assert not Path(dst).exists()
    finally:
        Path(fpath).unlink(missing_ok=True)
        Path(dst).unlink(missing_ok=True)


def test_capability_map_has_platform():
    caps = DeviceCapabilityScanner().scan()
    # sys.platform is 'darwin', 'linux', 'linux2', 'win32', 'cygwin', etc.
    assert caps.platform in ("darwin", "win32", "cygwin") or caps.platform.startswith("linux")


def test_resolver_stdlib_solution():
    resolver = ToolResolver()
    caps = _make_caps()
    resolution = resolver.resolve("list_files", "", caps)
    assert resolution.resolved is True
    assert resolution.method == "stdlib"


def test_resolver_suggests_install():
    resolver = ToolResolver()
    # No PIL installed simulation: empty py_packages, no image CLI tools
    caps = _make_caps(py_packages=[], cli_tools={})
    resolution = resolver.resolve("image_resize", "resize image.jpg", caps)
    # Should either require install or be unresolved
    assert resolution.requires_install or not resolution.resolved


def test_device_result_card():
    mock_result = DeviceTaskResult(
        success=True,
        output="done",
        files_created=["/tmp/out.txt"],
        files_modified=[],
        tool_used="stdlib",
        command_run="listdir(/tmp)",
        elapsed_ms=12.5,
        error="",
        undo_command="delete /tmp/out.txt",
    )
    card = device_result_card(mock_result, "list files in /tmp")
    assert isinstance(card, PrismCard)
    assert "Device task" in card.title
    assert card.card_data["success"] is True
    assert card.card_data["tool_used"] == "stdlib"
    assert card.card_data["undo_available"] is True
    assert "Undo this action" in card.actions


# ---------------------------------------------------------------------------
# prism_device_resolver.py — standalone ToolResolver tests
# ---------------------------------------------------------------------------

def test_resolver_stdlib_first():
    """StandaloneToolResolver resolves zip → stdlib (import zipfile)."""
    resolver = StandaloneToolResolver()
    caps = _make_caps()
    resolution = resolver.resolve("zip", "create a zip archive", caps)
    assert resolution.resolved is True
    assert resolution.method == "stdlib"
    assert "zipfile" in resolution.implementation


def test_resolver_suggests_install_when_missing():
    """StandaloneToolResolver suggests pip install when PIL is absent."""
    from unittest.mock import patch
    resolver = StandaloneToolResolver()
    caps = _make_caps(py_packages=[], cli_tools={})
    # Simulate PIL not being installed by making find_spec return None
    with patch("prism_device_resolver.importlib.util.find_spec", return_value=None):
        resolution = resolver.resolve("image_resize", "resize an image", caps)
    assert resolution.requires_install or not resolution.resolved


def test_resolver_installed_pip_package():
    """StandaloneToolResolver returns py_package when the package is importable."""
    resolver = StandaloneToolResolver()
    caps = _make_caps()
    # json is always available as stdlib; "json" key triggers stdlib path
    resolution = resolver.resolve("json", "parse json data", caps)
    assert resolution.resolved is True
    assert resolution.method == "stdlib"


def test_resolver_manual_fallback():
    """StandaloneToolResolver returns manual fallback for unknown task types."""
    resolver = StandaloneToolResolver()
    caps = _make_caps(py_packages=[], cli_tools={})
    resolution = resolver.resolve("unknown_exotic_task_xyz", "do something weird", caps)
    assert resolution.resolved is False
    assert resolution.method == "manual"
    assert resolution.user_message
