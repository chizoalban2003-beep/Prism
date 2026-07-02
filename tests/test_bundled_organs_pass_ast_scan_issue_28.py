"""
tests/test_bundled_organs_pass_ast_scan_issue_28.py
===================================================
Regression gate for issue #28: every bundled organ must pass the organ
loader's AST safety scan, and every bundled organ file must actually
register when OrganLoader loads the real organs/ directory.

Background: system_lock/system_power/volume_control/brightness_control/
bluetooth_control shipped with direct `import subprocess` / `shutil` /
`os`, which the AST visitor blocks — so they passed their routing tests
in CI while the daemon silently skipped them at load time. Organs that
need PATH lookup or subprocess must go through the prism_device_executor
bridge (`which`, `run_argv`, `current_uid`) instead.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from prism_organ_loader import BUNDLED_DIR, OrganLoader, _is_safe

BUNDLED_FILES = sorted(
    p for p in BUNDLED_DIR.glob("*.py") if not p.name.startswith("_")
)


@pytest.mark.parametrize("path", BUNDLED_FILES, ids=lambda p: p.stem)
def test_bundled_organ_passes_ast_scan(path: Path):
    safe, reason = _is_safe(path.read_text())
    assert safe, (
        f"{path.name} would be skipped by the organ loader: {reason}. "
        "Use the prism_device_executor bridge (which/run_argv/current_uid) "
        "instead of importing os/subprocess/shutil directly."
    )


def test_every_bundled_organ_file_registers():
    with tempfile.TemporaryDirectory() as d:
        loader = OrganLoader(user_dir=Path(d))
        registered = set(loader.list_organs())
    expected = {p.stem for p in BUNDLED_FILES}
    missing = {
        stem for stem in expected
        if stem not in registered
        # An organ may declare an intent different from its filename;
        # count it as loaded if its declared intent registered.
        and not any(stem in r for r in registered)
    }
    assert not missing, f"Bundled organs failed to load: {sorted(missing)}"


def test_hardware_bridge_organs_registered():
    with tempfile.TemporaryDirectory() as d:
        loader = OrganLoader(user_dir=Path(d))
        registered = set(loader.list_organs())
    for intent in (
        "system_lock", "system_power", "volume_control",
        "brightness_control", "bluetooth_control",
    ):
        assert intent in registered, f"{intent} did not register"
