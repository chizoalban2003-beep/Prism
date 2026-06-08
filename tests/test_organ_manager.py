"""
tests/test_organ_manager.py
============================
Comprehensive tests for the organ plugin manager:
  - OrganLoader unit tests (disable/enable, organ_details, reload, delete_user_organ)
  - HTTP route tests via FastAPI TestClient
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prism_asgi import app
from prism_organ_loader import OrganLoader
from prism_state import _set_state

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BUNDLED_DIR = Path(__file__).parent.parent / "organs"
# Pick a known bundled organ that should always exist
KNOWN_ORGAN = "currency_convert"


@pytest.fixture()
def loader(tmp_path):
    return OrganLoader(user_dir=tmp_path)


@pytest.fixture()
def client(loader):
    agent = MagicMock()
    agent._organ_loader = loader
    _set_state(agent=agent, organ_loader=None)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def client_no_loader():
    agent = MagicMock()
    agent._organ_loader = None
    _set_state(agent=agent, organ_loader=None)
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# OrganLoader unit tests
# ---------------------------------------------------------------------------

class TestOrganLoaderUnit:

    def test_known_intents_loaded(self, loader):
        intents = loader.list_organs()
        assert len(intents) > 0

    def test_known_organ_in_bundled(self, loader):
        assert KNOWN_ORGAN in loader.list_organs()

    def test_bundled_source(self, loader):
        assert loader._organ_sources.get(KNOWN_ORGAN) == "bundled"

    def test_get_returns_callable(self, loader):
        fn = loader.get(KNOWN_ORGAN)
        assert callable(fn)

    def test_get_returns_none_for_disabled(self, loader):
        loader.disable(KNOWN_ORGAN)
        assert loader.get(KNOWN_ORGAN) is None
        loader.enable(KNOWN_ORGAN)  # restore

    def test_disable_returns_true_for_known(self, loader):
        result = loader.disable(KNOWN_ORGAN)
        assert result is True
        loader.enable(KNOWN_ORGAN)

    def test_disable_returns_false_for_unknown(self, loader):
        result = loader.disable("nonexistent_organ_xyz")
        assert result is False

    def test_enable_returns_true_when_was_disabled(self, loader):
        loader.disable(KNOWN_ORGAN)
        result = loader.enable(KNOWN_ORGAN)
        assert result is True

    def test_enable_returns_false_when_not_disabled(self, loader):
        result = loader.enable(KNOWN_ORGAN)
        assert result is False

    def test_is_enabled_reflects_disable_enable_cycle(self, loader):
        assert loader.is_enabled(KNOWN_ORGAN) is True
        loader.disable(KNOWN_ORGAN)
        assert loader.is_enabled(KNOWN_ORGAN) is False
        loader.enable(KNOWN_ORGAN)
        assert loader.is_enabled(KNOWN_ORGAN) is True

    def test_organ_details_returns_none_for_unknown(self, loader):
        result = loader.organ_details("nonexistent_xyz")
        assert result is None

    def test_organ_details_returns_dict_for_known(self, loader):
        details = loader.organ_details(KNOWN_ORGAN)
        assert details is not None
        assert details["intent"] == KNOWN_ORGAN
        assert "description" in details
        assert "version" in details
        assert "source" in details
        assert "enabled" in details
        assert "risk_level" in details
        assert "requires_approval" in details
        assert "irreversible" in details
        assert "capabilities" in details

    def test_organ_details_enabled_reflects_state(self, loader):
        loader.disable(KNOWN_ORGAN)
        details = loader.organ_details(KNOWN_ORGAN)
        assert details["enabled"] is False
        loader.enable(KNOWN_ORGAN)
        details = loader.organ_details(KNOWN_ORGAN)
        assert details["enabled"] is True

    def test_list_organ_details_sorted(self, loader):
        items = loader.list_organ_details()
        intents = [o["intent"] for o in items]
        assert intents == sorted(intents)

    def test_reload_returns_positive_count(self, loader):
        count = loader.reload()
        assert count > 0

    def test_reload_preserves_disabled(self, loader):
        loader.disable(KNOWN_ORGAN)
        loader.reload()
        assert KNOWN_ORGAN in loader._disabled
        loader.enable(KNOWN_ORGAN)

    def test_delete_user_organ_returns_false_for_bundled(self, loader):
        result = loader.delete_user_organ(KNOWN_ORGAN)
        assert result is False

    def test_delete_user_organ_works(self, loader, tmp_path):
        # Create a synthetic user organ
        user_organ_path = tmp_path / "test_user_organ.py"
        user_organ_path.write_text(
            'ORGAN_META = {"intent": "test_user_organ", "description": "test"}\n'
            'ORGAN_POLICY = {}\n'
            'def execute(intent, message, ctx):\n'
            '    from prism_responses import text_card\n'
            '    return text_card("ok", intent)\n'
        )
        # Reload to pick it up
        loader.reload()
        assert "test_user_organ" in loader.list_organs()
        assert loader._organ_sources["test_user_organ"] == "user"

        result = loader.delete_user_organ("test_user_organ")
        assert result is True
        assert "test_user_organ" not in loader.list_organs()
        assert not user_organ_path.exists()

    def test_get_returns_none_for_unknown(self, loader):
        assert loader.get("does_not_exist_at_all") is None


# ---------------------------------------------------------------------------
# HTTP route tests
# ---------------------------------------------------------------------------

class TestOrgansRoutes:

    # GET /organs
    def test_list_organs_returns_list(self, client):
        resp = client.get("/organs")
        assert resp.status_code == 200
        data = resp.json()
        assert "organs" in data
        assert "count" in data
        assert data["count"] > 0

    def test_list_organs_fields(self, client):
        resp = client.get("/organs")
        assert resp.status_code == 200
        item = resp.json()["organs"][0]
        for field in ("intent", "description", "version", "source", "enabled", "risk_level"):
            assert field in item, f"Missing field: {field}"

    def test_list_organs_source_filter(self, client):
        resp = client.get("/organs?source=bundled")
        assert resp.status_code == 200
        data = resp.json()
        assert all(o["source"] == "bundled" for o in data["organs"])

    def test_list_organs_source_user_filter_empty(self, client):
        resp = client.get("/organs?source=user")
        assert resp.status_code == 200
        data = resp.json()
        # fresh loader with tmp user dir has no user organs
        assert data["count"] == 0

    def test_list_organs_enabled_only_filter(self, client, loader):
        loader.disable(KNOWN_ORGAN)
        resp = client.get("/organs?enabled_only=true")
        assert resp.status_code == 200
        data = resp.json()
        intents = [o["intent"] for o in data["organs"]]
        assert KNOWN_ORGAN not in intents
        loader.enable(KNOWN_ORGAN)

    def test_list_organs_no_loader_returns_empty(self, client_no_loader):
        resp = client_no_loader.get("/organs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["organs"] == []
        assert "note" in data

    # GET /organs/{name}
    def test_get_known_organ(self, client):
        resp = client.get(f"/organs/{KNOWN_ORGAN}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["intent"] == KNOWN_ORGAN

    def test_get_unknown_organ_returns_404(self, client):
        resp = client.get("/organs/nonexistent_xyz")
        assert resp.status_code == 404

    def test_get_organ_no_loader_returns_503(self, client_no_loader):
        resp = client_no_loader.get(f"/organs/{KNOWN_ORGAN}")
        assert resp.status_code == 503

    # POST /organs/{name}/disable
    def test_disable_organ(self, client, loader):
        resp = client.post(f"/organs/{KNOWN_ORGAN}/disable")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["enabled"] is False
        loader.enable(KNOWN_ORGAN)

    def test_disable_organ_unknown_returns_404(self, client):
        resp = client.post("/organs/nonexistent_xyz/disable")
        assert resp.status_code == 404

    def test_disable_organ_no_loader_returns_503(self, client_no_loader):
        resp = client_no_loader.post(f"/organs/{KNOWN_ORGAN}/disable")
        assert resp.status_code == 503

    # After disable, GET /organs with enabled_only excludes it
    def test_disabled_excluded_from_enabled_only(self, client, loader):
        loader.disable(KNOWN_ORGAN)
        resp = client.get("/organs?enabled_only=true")
        assert resp.status_code == 200
        intents = [o["intent"] for o in resp.json()["organs"]]
        assert KNOWN_ORGAN not in intents
        loader.enable(KNOWN_ORGAN)

    # After disable, GET /organs/{name} shows enabled=False
    def test_disabled_organ_details_shows_false(self, client, loader):
        loader.disable(KNOWN_ORGAN)
        resp = client.get(f"/organs/{KNOWN_ORGAN}")
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False
        loader.enable(KNOWN_ORGAN)

    # POST /organs/{name}/enable
    def test_enable_organ(self, client, loader):
        loader.disable(KNOWN_ORGAN)
        resp = client.post(f"/organs/{KNOWN_ORGAN}/enable")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["enabled"] is True

    def test_enable_organ_unknown_returns_404(self, client):
        resp = client.post("/organs/nonexistent_xyz/enable")
        assert resp.status_code == 404

    def test_enable_organ_no_loader_returns_503(self, client_no_loader):
        resp = client_no_loader.post(f"/organs/{KNOWN_ORGAN}/enable")
        assert resp.status_code == 503

    # POST /organs/reload
    def test_reload(self, client):
        resp = client.post("/organs/reload")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["loaded"] > 0

    def test_reload_no_loader_returns_503(self, client_no_loader):
        resp = client_no_loader.post("/organs/reload")
        assert resp.status_code == 503

    # DELETE /organs/{name}
    def test_delete_bundled_organ_returns_403(self, client):
        resp = client.delete(f"/organs/{KNOWN_ORGAN}")
        assert resp.status_code == 403

    def test_delete_unknown_organ_returns_404(self, client):
        resp = client.delete("/organs/nonexistent_xyz")
        assert resp.status_code == 404

    def test_delete_organ_no_loader_returns_503(self, client_no_loader):
        resp = client_no_loader.delete(f"/organs/{KNOWN_ORGAN}")
        assert resp.status_code == 503

    # POST /organs/synthesize
    def test_synthesize_missing_intent_returns_400(self, client):
        resp = client.post("/organs/synthesize", json={"message": "do something"})
        assert resp.status_code == 400
        assert "'intent' is required" in resp.json()["error"]

    def test_synthesize_no_loader_returns_503(self, client_no_loader):
        resp = client_no_loader.post("/organs/synthesize", json={"intent": "test"})
        assert resp.status_code == 503

    def test_synthesize_no_llm_router_returns_503(self, client):
        # loader has no LLM router, so synthesize returns False → 503
        resp = client.post("/organs/synthesize", json={"intent": "foo_bar_baz", "message": "test"})
        assert resp.status_code == 503
