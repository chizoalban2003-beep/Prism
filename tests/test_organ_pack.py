"""Tests for prism_organ_pack — the portable Organ-Pack share format."""
from __future__ import annotations

import tempfile
import textwrap
from pathlib import Path

import pytest

import prism_organ_pack as pack_mod
from prism_organ_loader import OrganLoader

GREET_ORGAN = textwrap.dedent("""
    ORGAN_META = {
        "intent":      "greet",
        "description": "say hello",
        "version":     "1.0",
        "capabilities": [],
    }

    ORGAN_POLICY = {
        "risk_level":        "low",
        "requires_approval": False,
        "irreversible":      False,
        "max_per_session":   None,
    }

    def execute(intent, message, ctx):
        from prism_responses import text_card
        return text_card("hello", "Greet")
""").strip()

FAREWELL_ORGAN = textwrap.dedent("""
    ORGAN_META = {
        "intent":      "farewell",
        "description": "say goodbye",
        "version":     "2.0",
        "capabilities": [],
    }

    def execute(intent, message, ctx):
        from prism_responses import text_card
        return text_card("bye", "Farewell")
""").strip()


def _empty_loader() -> OrganLoader:
    d = tempfile.mkdtemp()
    bundled = Path(d) / "bundled"
    user = Path(d) / "user"
    bundled.mkdir()
    user.mkdir()
    return OrganLoader(bundled_dir=bundled, user_dir=user)


def _loader_with_organs() -> OrganLoader:
    loader = _empty_loader()
    assert loader.install_bundle("greet", GREET_ORGAN)
    assert loader.install_bundle("farewell", FAREWELL_ORGAN)
    return loader


# ── build / export ────────────────────────────────────────────────────────────

class TestBuildPack:
    def test_build_pack_basic(self):
        loader = _loader_with_organs()
        pack = pack_mod.build_pack(loader, ["greet"], name="greetings")
        assert pack["format"] == pack_mod.PACK_FORMAT
        assert pack["name"] == "greetings"
        assert len(pack["organs"]) == 1
        assert pack["organs"][0]["intent"] == "greet"
        assert "code" in pack["organs"][0]
        assert pack["sha256"]

    def test_build_pack_multiple(self):
        loader = _loader_with_organs()
        pack = pack_mod.build_pack(
            loader, ["greet", "farewell"], name="convo", author="alice"
        )
        intents = {o["intent"] for o in pack["organs"]}
        assert intents == {"greet", "farewell"}
        assert pack["author"] == "alice"

    def test_build_pack_dedupes(self):
        loader = _loader_with_organs()
        pack = pack_mod.build_pack(loader, ["greet", "greet"], name="x")
        assert len(pack["organs"]) == 1

    def test_build_pack_requires_name(self):
        loader = _loader_with_organs()
        with pytest.raises(ValueError):
            pack_mod.build_pack(loader, ["greet"], name="")

    def test_build_pack_requires_intents(self):
        loader = _loader_with_organs()
        with pytest.raises(ValueError):
            pack_mod.build_pack(loader, [], name="x")

    def test_build_pack_unknown_organ_raises(self):
        loader = _loader_with_organs()
        with pytest.raises(ValueError):
            pack_mod.build_pack(loader, ["does_not_exist"], name="x")

    def test_summary_has_no_code(self):
        loader = _loader_with_organs()
        pack = pack_mod.build_pack(loader, ["greet"], name="x")
        summary = pack_mod.pack_summary(pack)
        assert summary["organ_count"] == 1
        assert "code" not in summary["organs"][0]


# ── verify ─────────────────────────────────────────────────────────────────────

class TestVerifyPack:
    def test_valid_pack_verifies(self):
        loader = _loader_with_organs()
        pack = pack_mod.build_pack(loader, ["greet"], name="x")
        ok, reason = pack_mod.verify_pack(pack)
        assert ok, reason

    def test_tampered_code_fails(self):
        loader = _loader_with_organs()
        pack = pack_mod.build_pack(loader, ["greet"], name="x")
        pack["organs"][0]["code"] += "\n# sneaky change"
        ok, reason = pack_mod.verify_pack(pack)
        assert not ok
        assert "sha256" in reason.lower()

    def test_wrong_format_fails(self):
        loader = _loader_with_organs()
        pack = pack_mod.build_pack(loader, ["greet"], name="x")
        pack["format"] = "bogus/v9"
        ok, reason = pack_mod.verify_pack(pack)
        assert not ok

    def test_empty_organs_fails(self):
        ok, _ = pack_mod.verify_pack({"format": pack_mod.PACK_FORMAT, "organs": []})
        assert not ok

    def test_tampered_pack_digest_fails(self):
        loader = _loader_with_organs()
        pack = pack_mod.build_pack(loader, ["greet", "farewell"], name="x")
        pack["sha256"] = "0" * 64
        ok, reason = pack_mod.verify_pack(pack)
        assert not ok
        assert "pack sha256" in reason.lower()


# ── round-trip import ───────────────────────────────────────────────────────────

class TestImportPack:
    def test_export_import_roundtrip(self):
        src = _loader_with_organs()
        pack = pack_mod.build_pack(src, ["greet", "farewell"], name="convo")

        dst = _empty_loader()
        assert dst.get("greet") is None
        report = pack_mod.import_pack(dst, pack)
        assert report["ok"]
        assert set(report["installed"]) == {"greet", "farewell"}
        assert dst.get("greet") is not None
        assert dst.get("farewell") is not None

    def test_import_skips_existing(self):
        src = _loader_with_organs()
        pack = pack_mod.build_pack(src, ["greet"], name="x")
        dst = _loader_with_organs()  # already has greet
        report = pack_mod.import_pack(dst, pack)
        assert report["installed"] == []
        assert any(s["intent"] == "greet" for s in report["skipped"])

    def test_import_overwrite(self):
        src = _loader_with_organs()
        pack = pack_mod.build_pack(src, ["greet"], name="x")
        dst = _loader_with_organs()
        report = pack_mod.import_pack(dst, pack, overwrite=True)
        assert "greet" in report["installed"]

    def test_import_rejects_tampered_pack(self):
        src = _loader_with_organs()
        pack = pack_mod.build_pack(src, ["greet"], name="x")
        pack["organs"][0]["code"] += "\n# tamper"
        dst = _empty_loader()
        report = pack_mod.import_pack(dst, pack)
        assert not report["ok"]
        assert dst.get("greet") is None

    def test_import_blocks_unsafe_code(self):
        # A pack whose code is internally consistent (valid sha256) but unsafe
        # must still be rejected by the loader's strict AST scan.
        unsafe = textwrap.dedent("""
            ORGAN_META = {"intent": "evil", "description": "x", "version": "1.0"}
            def execute(intent, message, ctx):
                import os
                os.system("echo pwned")
                return None
        """).strip()
        pack = {
            "format": pack_mod.PACK_FORMAT,
            "name": "evil-pack",
            "organs": [{
                "intent": "evil",
                "description": "x",
                "version": "1.0",
                "capabilities": [],
                "code": unsafe,
                "sha256": pack_mod._sha256(unsafe),
            }],
        }
        pack["sha256"] = pack_mod._pack_digest(pack["organs"])
        ok, _ = pack_mod.verify_pack(pack)
        assert ok  # hashes are consistent
        dst = _empty_loader()
        report = pack_mod.import_pack(dst, pack)
        assert not report["ok"]
        assert any(f["intent"] == "evil" for f in report["failed"])
        assert dst.get("evil") is None


# ── serialisation ──────────────────────────────────────────────────────────────

class TestSerialisation:
    def test_dumps_loads_roundtrip(self):
        loader = _loader_with_organs()
        pack = pack_mod.build_pack(loader, ["greet"], name="x")
        text = pack_mod.dumps(pack)
        restored = pack_mod.loads(text)
        assert restored == pack

    def test_write_read_file(self, tmp_path):
        loader = _loader_with_organs()
        pack = pack_mod.build_pack(loader, ["greet"], name="x")
        path = pack_mod.write_pack(pack, tmp_path / "x.organpack.json")
        assert path.exists()
        restored = pack_mod.read_pack(path)
        assert restored["name"] == "x"

    def test_loads_rejects_bad_json(self):
        with pytest.raises(ValueError):
            pack_mod.loads("{not json")


# ── HTTP routes ────────────────────────────────────────────────────────────────

class TestPackRoutes:
    def _client(self, loader):
        import types

        from fastapi.testclient import TestClient

        from prism_asgi import app
        from prism_state import _set_state
        agent = types.SimpleNamespace(_organ_loader=loader)
        _set_state(agent=agent, organ_loader=loader)
        return TestClient(app, raise_server_exceptions=False)

    def test_export_route(self):
        client = self._client(_loader_with_organs())
        r = client.post(
            "/organs/pack/export",
            json={"intents": ["greet"], "name": "greetings"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["format"] == pack_mod.PACK_FORMAT
        assert body["organs"][0]["intent"] == "greet"

    def test_export_preview_has_no_code(self):
        client = self._client(_loader_with_organs())
        r = client.post(
            "/organs/pack/export",
            json={"intents": ["greet"], "name": "x", "preview": True},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["organ_count"] == 1
        assert "code" not in body["organs"][0]

    def test_export_requires_intents(self):
        client = self._client(_loader_with_organs())
        r = client.post("/organs/pack/export", json={"name": "x"})
        assert r.status_code == 400

    def test_import_route_roundtrip(self):
        pack = pack_mod.build_pack(_loader_with_organs(), ["greet"], name="x")
        client = self._client(_empty_loader())
        r = client.post("/organs/pack/import", json=pack)
        assert r.status_code == 200
        body = r.json()
        assert body["ok"]
        assert "greet" in body["installed"]

    def test_import_route_rejects_tampered(self):
        pack = pack_mod.build_pack(_loader_with_organs(), ["greet"], name="x")
        pack["organs"][0]["code"] += "\n# tamper"
        client = self._client(_empty_loader())
        r = client.post("/organs/pack/import", json=pack)
        assert r.status_code == 400
