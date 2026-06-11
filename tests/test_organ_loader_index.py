"""
Tests for prism_organ_loader — index.json manifest and bytecode compilation.
"""
from __future__ import annotations

import json
from pathlib import Path

from prism_organ_loader import OrganLoader

# ---------------------------------------------------------------------------
# Minimal valid organ source
# ---------------------------------------------------------------------------

def _organ_src(intent: str, version: str = "1.0", output: str = "ok") -> str:
    return f'''\
ORGAN_META = {{
    "intent":      "{intent}",
    "description": "test organ {intent}",
    "version":     "{version}",
}}

def execute(intent: str, message: str, ctx: dict):
    return {{"output": "{output}", "intent": intent}}
'''


_ORGAN_SRC    = _organ_src("test_op")
_ORGAN_SRC_V2 = _organ_src("test_op", version="2.0", output="v2")


def _write_organ(user_dir: Path, intent: str, src: str | None = None) -> Path:
    path = user_dir / f"{intent}.py"
    path.write_text(src if src is not None else _organ_src(intent))
    return path


def _loader(tmp_path: Path) -> OrganLoader:
    user_dir = tmp_path / "organs"
    user_dir.mkdir()
    return OrganLoader(bundled_dir=tmp_path / "bundled", user_dir=user_dir)


# ---------------------------------------------------------------------------
# index.json creation
# ---------------------------------------------------------------------------

class TestIndexCreation:
    def test_index_file_written_after_load(self, tmp_path):
        user_dir = tmp_path / "organs"
        user_dir.mkdir()
        _write_organ(user_dir, "test_op")
        OrganLoader(bundled_dir=tmp_path / "bundled", user_dir=user_dir)
        assert (user_dir / "index.json").exists()

    def test_index_has_version(self, tmp_path):
        user_dir = tmp_path / "organs"
        user_dir.mkdir()
        _write_organ(user_dir, "test_op")
        OrganLoader(bundled_dir=tmp_path / "bundled", user_dir=user_dir)
        data = json.loads((user_dir / "index.json").read_text())
        assert data["version"] == 1

    def test_index_has_entries(self, tmp_path):
        user_dir = tmp_path / "organs"
        user_dir.mkdir()
        _write_organ(user_dir, "test_op")
        OrganLoader(bundled_dir=tmp_path / "bundled", user_dir=user_dir)
        data = json.loads((user_dir / "index.json").read_text())
        assert "test_op" in data["entries"]

    def test_index_entry_has_required_fields(self, tmp_path):
        user_dir = tmp_path / "organs"
        user_dir.mkdir()
        _write_organ(user_dir, "test_op")
        OrganLoader(bundled_dir=tmp_path / "bundled", user_dir=user_dir)
        entry = json.loads((user_dir / "index.json").read_text())["entries"]["test_op"]
        for field in ("path", "version", "description", "hash", "compiled",
                      "safe", "source", "created_at"):
            assert field in entry, f"missing field: {field}"

    def test_index_entry_hash_is_nonempty(self, tmp_path):
        user_dir = tmp_path / "organs"
        user_dir.mkdir()
        _write_organ(user_dir, "test_op")
        OrganLoader(bundled_dir=tmp_path / "bundled", user_dir=user_dir)
        entry = json.loads((user_dir / "index.json").read_text())["entries"]["test_op"]
        assert len(entry["hash"]) == 64  # SHA-256 hex

    def test_index_entry_safe_true(self, tmp_path):
        user_dir = tmp_path / "organs"
        user_dir.mkdir()
        _write_organ(user_dir, "test_op")
        OrganLoader(bundled_dir=tmp_path / "bundled", user_dir=user_dir)
        entry = json.loads((user_dir / "index.json").read_text())["entries"]["test_op"]
        assert entry["safe"] is True

    def test_no_index_when_no_user_organs(self, tmp_path):
        _loader(tmp_path)
        index_path = (tmp_path / "organs") / "index.json"
        # No user organs → no entries → index not written (or written with empty entries)
        if index_path.exists():
            data = json.loads(index_path.read_text())
            assert data.get("entries", {}) == {}

    def test_multiple_organs_all_indexed(self, tmp_path):
        user_dir = tmp_path / "organs"
        user_dir.mkdir()
        for i in range(3):
            _write_organ(user_dir, f"op_{i}")
        OrganLoader(bundled_dir=tmp_path / "bundled", user_dir=user_dir)
        data = json.loads((user_dir / "index.json").read_text())
        assert len(data["entries"]) == 3


# ---------------------------------------------------------------------------
# Hash-based cache (skip re-scan on unchanged files)
# ---------------------------------------------------------------------------

class TestHashCache:
    def test_second_load_uses_cached_hash(self, tmp_path):
        user_dir = tmp_path / "organs"
        user_dir.mkdir()
        _write_organ(user_dir, "test_op")
        # First load — builds index
        OrganLoader(bundled_dir=tmp_path / "bundled", user_dir=user_dir)
        # Second load — should trust cached entry (no error, organ still loaded)
        loader2 = OrganLoader(bundled_dir=tmp_path / "bundled", user_dir=user_dir)
        assert loader2.get("test_op") is not None

    def test_changed_file_re_validated(self, tmp_path):
        user_dir = tmp_path / "organs"
        user_dir.mkdir()
        _write_organ(user_dir, "test_op")
        OrganLoader(bundled_dir=tmp_path / "bundled", user_dir=user_dir)
        # Overwrite with new content → hash mismatch → full re-scan
        _write_organ(user_dir, "test_op", _ORGAN_SRC_V2)
        loader2 = OrganLoader(bundled_dir=tmp_path / "bundled", user_dir=user_dir)
        # Organ should still load (v2 is also safe)
        assert loader2.get("test_op") is not None

    def test_index_hash_updated_after_file_change(self, tmp_path):
        user_dir = tmp_path / "organs"
        user_dir.mkdir()
        _write_organ(user_dir, "test_op")
        OrganLoader(bundled_dir=tmp_path / "bundled", user_dir=user_dir)
        old_hash = json.loads((user_dir / "index.json").read_text())["entries"]["test_op"]["hash"]
        _write_organ(user_dir, "test_op", _ORGAN_SRC_V2)
        OrganLoader(bundled_dir=tmp_path / "bundled", user_dir=user_dir)
        new_hash = json.loads((user_dir / "index.json").read_text())["entries"]["test_op"]["hash"]
        assert old_hash != new_hash

    def test_unsafe_organ_not_in_index(self, tmp_path):
        user_dir = tmp_path / "organs"
        user_dir.mkdir()
        bad_src = "import os\ndef execute(i,m,c): return os.system('rm -rf /')\nORGAN_META={}"
        (user_dir / "bad_op.py").write_text(bad_src)
        OrganLoader(bundled_dir=tmp_path / "bundled", user_dir=user_dir)
        if (user_dir / "index.json").exists():
            data = json.loads((user_dir / "index.json").read_text())
            assert "bad_op" not in data.get("entries", {})


# ---------------------------------------------------------------------------
# Bytecode compilation
# ---------------------------------------------------------------------------

class TestBytecodeCompilation:
    def test_compile_organ_returns_bool(self, tmp_path):
        path = tmp_path / "test_op.py"
        path.write_text(_ORGAN_SRC)
        from prism_organ_loader import OrganLoader as OL
        result = OL._compile_organ(path)
        assert isinstance(result, bool)

    def test_compile_valid_organ_succeeds(self, tmp_path):
        path = tmp_path / "test_op.py"
        path.write_text(_ORGAN_SRC)
        from prism_organ_loader import OrganLoader as OL
        assert OL._compile_organ(path) is True

    def test_compile_invalid_syntax_returns_false(self, tmp_path):
        path = tmp_path / "bad.py"
        path.write_text("def execute(: invalid syntax")
        from prism_organ_loader import OrganLoader as OL
        assert OL._compile_organ(path) is False

    def test_pyc_is_current_after_compile(self, tmp_path):
        path = tmp_path / "test_op.py"
        path.write_text(_ORGAN_SRC)
        from prism_organ_loader import OrganLoader as OL
        OL._compile_organ(path)
        assert OL._pyc_is_current(path) is True

    def test_pyc_not_current_before_compile(self, tmp_path):
        path = tmp_path / "test_op.py"
        path.write_text(_ORGAN_SRC)
        from prism_organ_loader import OrganLoader as OL
        assert OL._pyc_is_current(path) is False


# ---------------------------------------------------------------------------
# index_status()
# ---------------------------------------------------------------------------

class TestIndexStatus:
    def test_index_status_returns_dict(self, tmp_path):
        loader = _loader(tmp_path)
        status = loader.index_status()
        assert isinstance(status, dict)

    def test_index_status_has_required_keys(self, tmp_path):
        loader = _loader(tmp_path)
        status = loader.index_status()
        for key in ("version", "entry_count", "compiled_count", "entries"):
            assert key in status

    def test_entry_count_matches_user_organs(self, tmp_path):
        user_dir = tmp_path / "organs"
        user_dir.mkdir()
        for i in range(2):
            _write_organ(user_dir, f"op_{i}")
        loader = OrganLoader(bundled_dir=tmp_path / "bundled", user_dir=user_dir)
        assert loader.index_status()["entry_count"] == 2

    def test_entries_is_copy(self, tmp_path):
        user_dir = tmp_path / "organs"
        user_dir.mkdir()
        _write_organ(user_dir, "test_op")
        loader = OrganLoader(bundled_dir=tmp_path / "bundled", user_dir=user_dir)
        status = loader.index_status()
        status["entries"].clear()
        # Internal state must be unaffected
        assert loader.index_status()["entry_count"] == 1


# ---------------------------------------------------------------------------
# delete_user_organ — cleans up index
# ---------------------------------------------------------------------------

class TestDeleteCleansIndex:
    def test_delete_removes_from_index(self, tmp_path):
        user_dir = tmp_path / "organs"
        user_dir.mkdir()
        _write_organ(user_dir, "test_op")
        loader = OrganLoader(bundled_dir=tmp_path / "bundled", user_dir=user_dir)
        loader.delete_user_organ("test_op")
        assert loader.index_status()["entry_count"] == 0

    def test_delete_updates_index_json(self, tmp_path):
        user_dir = tmp_path / "organs"
        user_dir.mkdir()
        _write_organ(user_dir, "test_op")
        loader = OrganLoader(bundled_dir=tmp_path / "bundled", user_dir=user_dir)
        loader.delete_user_organ("test_op")
        if (user_dir / "index.json").exists():
            data = json.loads((user_dir / "index.json").read_text())
            assert "test_op" not in data.get("entries", {})


# ---------------------------------------------------------------------------
# reload() resets index
# ---------------------------------------------------------------------------

class TestReloadResetsIndex:
    def test_reload_repopulates_index(self, tmp_path):
        user_dir = tmp_path / "organs"
        user_dir.mkdir()
        _write_organ(user_dir, "test_op")
        loader = OrganLoader(bundled_dir=tmp_path / "bundled", user_dir=user_dir)
        count_before = loader.reload()
        assert loader.index_status()["entry_count"] == 1
        assert count_before >= 1
