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


def test_organs_package_is_declared():
    """v0.2.1 only declared py-modules (root files), so the `organs/`
    directory — 40+ bundled organs including the document-store organs and
    `agents_inventory` — was silently dropped from the wheel. Guard against
    repeating that.
    """
    text = (REPO_ROOT / "pyproject.toml").read_text()
    data = tomllib.loads(text)
    packages = data["tool"]["setuptools"].get("packages", [])
    assert "organs" in packages, (
        "pyproject.toml [tool.setuptools] must declare `packages = ['organs', ...]` "
        "or the bundled organs are excluded from the built wheel."
    )
    assert (REPO_ROOT / "organs" / "__init__.py").exists(), (
        "organs/__init__.py is required for setuptools to treat the "
        "directory as a package and copy it into the wheel."
    )


def test_organ_files_have_required_metadata():
    """Every bundled organ must expose ORGAN_META and an execute() callable
    — or the loader silently skips it, which masks packaging regressions.
    """
    organ_dir = REPO_ROOT / "organs"
    organ_files = sorted(p for p in organ_dir.glob("*.py") if p.stem != "__init__")
    assert organ_files, "organs/ directory is empty — something is very wrong"
    bad: list[str] = []
    for p in organ_files:
        src = p.read_text()
        if "ORGAN_META" not in src or "def execute" not in src:
            bad.append(p.name)
    assert not bad, f"organs missing ORGAN_META or execute(): {bad}"


def test_llm_router_from_config_default_path():
    """v0.2.0/0.2.1 shipped `from_config(config_path='~/.prism/config.toml')`
    while the bootstrap loader read `~/.prism/prism_config.toml`. The two
    defaults silently disagreed, so the daemon-level router saw no
    user config and `/agents` under-reported LLM providers.
    """
    src = (REPO_ROOT / "prism_llm_router.py").read_text()
    assert '"~/.prism/prism_config.toml"' in src, (
        "LLMRouter.from_config default path must match "
        "prism_agent_bootstrap.load_toml_config (~/.prism/prism_config.toml). "
        "A stale default silently strands user config."
    )
