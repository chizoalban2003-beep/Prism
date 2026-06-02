"""
tests/test_device_executor.py
=============================
Tests for prism_device_executor.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from prism_device_executor import BuiltinTasks, DeviceTaskResult, SafeSubprocess


def test_list_files_stdlib():
    result = BuiltinTasks.list_files("/tmp")
    assert isinstance(result, DeviceTaskResult)
    assert result.success is True


def test_read_write_roundtrip():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, dir="/tmp"
    ) as f:
        fpath = f.name

    try:
        content = "prism_executor_roundtrip_test_content"
        write_result = BuiltinTasks.write_file(fpath, content)
        assert write_result.success is True

        read_result = BuiltinTasks.read_file(fpath)
        assert read_result.success is True
        assert content in read_result.output
    finally:
        Path(fpath).unlink(missing_ok=True)


def test_trash_moves_file():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, dir="/tmp"
    ) as f:
        f.write("trash me")
        fpath = f.name

    result = BuiltinTasks.trash_file(fpath)
    assert result.success is True
    # Original file should no longer exist
    assert not Path(fpath).exists()
    # Output should mention the trash location
    assert ".prism" in result.output or "trash" in result.output.lower()


def test_safe_subprocess_shlex():
    """SafeSubprocess.run must NOT pass shell=True to subprocess.run."""
    import subprocess as sp

    captured_calls: list[dict] = []
    original_run = sp.run

    def mock_run(args, **kwargs):
        captured_calls.append(kwargs)
        return original_run(args, **kwargs)

    with patch("prism_device_executor.subprocess.run", side_effect=mock_run):
        SafeSubprocess().run("echo hello")

    assert captured_calls, "subprocess.run was never called"
    for call_kwargs in captured_calls:
        assert call_kwargs.get("shell", False) is False, \
            "subprocess.run must not use shell=True"


def test_subprocess_timeout():
    result = SafeSubprocess().run("sleep 60", timeout=1)
    assert result.success is False
    assert "Timed out" in result.error or "timeout" in result.error.lower()


def test_file_not_found_error():
    result = BuiltinTasks.read_file("/tmp/prism_nonexistent_file_xyz_123456.txt")
    assert result.success is False
    assert result.error
