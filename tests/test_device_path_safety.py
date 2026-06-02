"""
Tests for PrismDeviceAgent._safe_path path sanitisation.
"""

import os

import pytest

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
    def test_safe_path_expands_home(self, tmp_path):
        # Create a subdirectory to test expansion into
        sub = tmp_path / "Downloads"
        sub.mkdir()
        allowed = str(tmp_path)
        result = PrismDeviceAgent._safe_path(str(sub), allowed_roots=[allowed])
        assert os.path.isabs(result)
        assert result.startswith(os.path.realpath(allowed))

    def test_safe_path_tilde_is_expanded(self):
        """_safe_path always expands ~ to an absolute path."""
        result = PrismDeviceAgent._safe_path("~/Downloads")
        assert not result.startswith("~")
        assert os.path.isabs(result)


class TestSafePathOutsideRootRejected:
    def test_safe_path_outside_root_rejected(self, tmp_path):
        allowed = str(tmp_path)
        with pytest.raises(ValueError, match="outside allowed"):
            PrismDeviceAgent._safe_path("/tmp/evil_file", allowed_roots=[allowed])

    def test_safe_path_system_dir_rejected(self, tmp_path):
        allowed = str(tmp_path)
        with pytest.raises(ValueError, match="outside allowed"):
            PrismDeviceAgent._safe_path("/etc/passwd", allowed_roots=[allowed])


class TestSafePathValidHomePath:
    def test_safe_path_valid_home_path(self, tmp_path):
        allowed = str(tmp_path)
        target = str(tmp_path / "Documents" / "file.txt")
        result = PrismDeviceAgent._safe_path(target, allowed_roots=[allowed])
        assert os.path.isabs(result)
        assert result.startswith(os.path.realpath(allowed))
