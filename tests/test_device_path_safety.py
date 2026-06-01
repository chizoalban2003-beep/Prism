"""
Tests for PrismDeviceAgent._safe_path path sanitisation.
"""

import os
import pytest
from pathlib import Path

from prism_device_agent import PrismDeviceAgent


class TestSafePathRejectsTraversal:
    def test_safe_path_rejects_traversal(self):
        with pytest.raises(ValueError, match="traversal"):
            PrismDeviceAgent._safe_path("../../etc/passwd")

    def test_safe_path_rejects_traversal_in_longer_path(self):
        with pytest.raises(ValueError, match="traversal"):
            PrismDeviceAgent._safe_path("~/Documents/../../etc/shadow")


class TestSafePathRejectsEmpty:
    def test_safe_path_rejects_empty(self):
        with pytest.raises(ValueError, match="[Ee]mpty"):
            PrismDeviceAgent._safe_path("")


class TestSafePathExpandsHome:
    def test_safe_path_expands_home(self):
        result = PrismDeviceAgent._safe_path("~/Downloads")
        expected = os.path.realpath(os.path.expanduser("~/Downloads"))
        assert result == expected
        assert not result.startswith("~")
        assert os.path.isabs(result)


class TestSafePathOutsideRootRejected:
    def test_safe_path_outside_root_rejected(self):
        home = str(Path.home())
        with pytest.raises(ValueError, match="outside allowed"):
            PrismDeviceAgent._safe_path("/tmp/evil_file", allowed_roots=[home])

    def test_safe_path_system_dir_rejected(self):
        home = str(Path.home())
        with pytest.raises(ValueError, match="outside allowed"):
            PrismDeviceAgent._safe_path("/etc/passwd", allowed_roots=[home])


class TestSafePathValidHomePath:
    def test_safe_path_valid_home_path(self):
        home = str(Path.home())
        result = PrismDeviceAgent._safe_path(
            "~/Documents/file.txt", allowed_roots=[home]
        )
        assert os.path.isabs(result)
        assert result.startswith(os.path.realpath(home))
