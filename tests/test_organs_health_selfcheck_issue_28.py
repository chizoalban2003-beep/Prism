"""
tests/test_organs_health_selfcheck_issue_28.py
==============================================
GET /organs/health + OrganLoader.health_report() — the runtime
self-check that catches "organ file ships, tests pass, daemon silently
skips it" for ANY skip reason (unsafe AST, import error, missing
execute()), not just the AST case the CI gate covers.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import prism_state
from prism_organ_loader import OrganLoader
from prism_routes_infra import router as infra_router

VALID_ORGAN = textwrap.dedent("""
    ORGAN_META = {"intent": "good_organ", "description": "ok", "version": "1.0"}

    def execute(intent, message, ctx):
        from prism_responses import text_card
        return text_card("ok", intent)
""").strip()

UNSAFE_ORGAN = textwrap.dedent("""
    ORGAN_META = {"intent": "bad_organ", "description": "bad", "version": "1.0"}

    def execute(intent, message, ctx):
        import subprocess
        subprocess.run(["true"])
""").strip()

NO_EXECUTE_ORGAN = textwrap.dedent("""
    ORGAN_META = {"intent": "no_fn", "description": "no fn", "version": "1.0"}
""").strip()

BROKEN_IMPORT_ORGAN = textwrap.dedent("""
    import module_that_does_not_exist_anywhere

    def execute(intent, message, ctx):
        return None
""").strip()


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    prism_state._state.clear()


def _loader(tmp_path: Path, files: dict[str, str]) -> OrganLoader:
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    for name, code in files.items():
        (bundled / name).write_text(code)
    return OrganLoader(bundled_dir=bundled, user_dir=tmp_path / "user")


class TestHealthReport:
    def test_all_healthy(self, tmp_path):
        loader = _loader(tmp_path, {"good_organ.py": VALID_ORGAN})
        report = loader.health_report()
        assert report["ok"] is True
        assert report["directories"]["bundled"]["files"] == 1
        assert report["directories"]["bundled"]["registered"] == 1
        assert report["directories"]["bundled"]["missing"] == []

    def test_unsafe_file_reported_with_reason(self, tmp_path):
        loader = _loader(tmp_path, {
            "good_organ.py": VALID_ORGAN,
            "bad_organ.py":  UNSAFE_ORGAN,
        })
        report = loader.health_report()
        assert report["ok"] is False
        missing = report["directories"]["bundled"]["missing"]
        assert len(missing) == 1
        assert missing[0]["file"] == "bad_organ.py"
        assert missing[0]["reason"].startswith("unsafe:")

    def test_no_execute_and_import_error_reported(self, tmp_path):
        loader = _loader(tmp_path, {
            "no_fn.py":  NO_EXECUTE_ORGAN,
            "broken.py": BROKEN_IMPORT_ORGAN,
        })
        report = loader.health_report()
        assert report["ok"] is False
        reasons = {
            m["file"]: m["reason"]
            for m in report["directories"]["bundled"]["missing"]
        }
        assert reasons["no_fn.py"] == "no execute() function"
        assert reasons["broken.py"].startswith("import error:")

    def test_reload_clears_stale_skips(self, tmp_path):
        bundled = tmp_path / "bundled"
        bundled.mkdir()
        bad = bundled / "bad_organ.py"
        bad.write_text(UNSAFE_ORGAN)
        loader = OrganLoader(bundled_dir=bundled, user_dir=tmp_path / "user")
        assert loader.health_report()["ok"] is False
        bad.write_text(VALID_ORGAN.replace("good_organ", "bad_organ"))
        loader.reload()
        assert loader.health_report()["ok"] is True


class TestHealthRoute:
    def _client(self, loader) -> TestClient:
        app = FastAPI()
        app.include_router(infra_router)
        prism_state._state.clear()
        agent = MagicMock()
        agent._organ_loader = loader
        prism_state._state["agent"] = agent
        return TestClient(app)

    def test_healthy_200(self, tmp_path):
        client = self._client(_loader(tmp_path, {"good_organ.py": VALID_ORGAN}))
        r = client.get("/organs/health")
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_unhealthy_503_with_details(self, tmp_path):
        client = self._client(_loader(tmp_path, {"bad_organ.py": UNSAFE_ORGAN}))
        r = client.get("/organs/health")
        assert r.status_code == 503
        body = r.json()
        assert body["ok"] is False
        assert body["directories"]["bundled"]["missing"][0]["file"] == "bad_organ.py"

    def test_no_loader_503(self):
        app = FastAPI()
        app.include_router(infra_router)
        prism_state._state.clear()
        client = TestClient(app)
        r = client.get("/organs/health")
        assert r.status_code == 503

    def test_not_shadowed_by_dynamic_route(self, tmp_path):
        """/organs/health must hit the static route, not /organs/{name}."""
        client = self._client(_loader(tmp_path, {"good_organ.py": VALID_ORGAN}))
        r = client.get("/organs/health")
        assert "directories" in r.json(), (
            "response has no 'directories' — the dynamic /organs/{name} "
            "route matched first"
        )
