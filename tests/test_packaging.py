"""
tests/test_packaging.py
=======================
Guards against the v0.2.0 regression where the daemon imported root-level
modules that were absent from `pyproject.toml`'s `py-modules` list. Pure
`pip install` then failed with `ModuleNotFoundError` at daemon boot, while
editable / run-from-clone workflows masked the gap.

Every top-level `*.py` in the repo root (excluding tests, organs, and the
short list of known non-module scripts) must appear in `py-modules`. If you
add a new root module, list it in `pyproject.toml`.
"""
from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent

NON_MODULE_FILES: set[str] = {
    "conftest",
}


def _listed_modules() -> set[str]:
    text = (REPO_ROOT / "pyproject.toml").read_text()
    data = tomllib.loads(text)
    return set(data["tool"]["setuptools"]["py-modules"])


def _root_modules() -> set[str]:
    return {
        p.stem for p in REPO_ROOT.glob("*.py")
        if p.stem not in NON_MODULE_FILES
    }


def test_pyproject_lists_every_root_module():
    listed = _listed_modules()
    on_disk = _root_modules()
    missing = on_disk - listed
    assert not missing, (
        "These root-level modules exist on disk but are missing from "
        "pyproject.toml's `py-modules`, so `pip install` won't ship them:\n  "
        + "\n  ".join(sorted(missing))
    )


def test_pyproject_does_not_list_phantom_modules():
    listed = _listed_modules()
    on_disk = _root_modules()
    phantom = listed - on_disk
    assert not phantom, (
        "pyproject.toml lists modules that don't exist in the repo root:\n  "
        + "\n  ".join(sorted(phantom))
    )


def test_pyproject_is_valid_toml():
    text = (REPO_ROOT / "pyproject.toml").read_text()
    data = tomllib.loads(text)
    assert data["project"]["name"] == "prism-platform"
    assert re.match(r"^\d+\.\d+\.\d+", data["project"]["version"])
