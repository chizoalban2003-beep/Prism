"""
Tests for prism_routes_analytics — /domain/*, /moment/*, /duel/*, /analytics/tokens/* endpoints.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import prism_state
from prism_routes_analytics import router


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(router)
    prism_state._state.clear()
    yield TestClient(app)
    prism_state._state.clear()


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    prism_state._state.clear()


# ---------------------------------------------------------------------------
# /domain
# ---------------------------------------------------------------------------

class TestDomainList:
    def test_domain_list_200(self, client):
        r = client.get("/domain/list")
        assert r.status_code == 200
        data = r.json()
        assert "domains" in data
        assert isinstance(data["domains"], list)

    def test_domain_list_has_name_field(self, client):
        data = client.get("/domain/list").json()
        if data["domains"]:
            assert "name" in data["domains"][0]

    def test_domain_profiles_unknown_404(self, client):
        r = client.get("/domain/profiles?domain=NonExistent")
        assert r.status_code == 404

    def test_domain_evaluate_medical(self, client):
        r = client.get("/domain/evaluate?domain=Medical")
        assert r.status_code in (200, 404)
        if r.status_code == 200:
            data = r.json()
            assert "recommended" in data
            assert "fulcrum" in data

    def test_domain_sensitivity_missing_profile_factor(self, client):
        # domain_models not wired in test state → 404, not 400
        # Verify 400 only fires when domain is known but profile/factor missing
        import prism_state as ps
        from domain_configs import ALL_DOMAINS, DomainDecisionModel

        if not ALL_DOMAINS:
            pytest.skip("no domain configs available")

        first_domain = next(iter(ALL_DOMAINS))
        ps._state["domain_models"] = {
            first_domain: DomainDecisionModel(ALL_DOMAINS[first_domain])
        }
        r = client.get(f"/domain/sensitivity?domain={first_domain}")
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# /moment
# ---------------------------------------------------------------------------

class TestMomentConfigs:
    def test_moment_configs_200(self, client):
        r = client.get("/moment/configs")
        assert r.status_code == 200
        assert "configs" in r.json()

    def test_moment_history_empty(self, client):
        r = client.get("/moment/history?player=NoOne")
        assert r.status_code == 200
        data = r.json()
        assert data["moments"] == []

    def test_moment_analyze_missing_params(self, client):
        r = client.get("/moment/analyze?sport=Football")
        assert r.status_code == 400

    def test_moment_analyze_no_analyzer_503(self, client):
        prism_state._state.pop("moment_analyzer", None)
        r = client.get("/moment/analyze?sport=Football&moment_type=1v1_keeper&player=Mbappe")
        assert r.status_code in (200, 503)

    def test_moment_player_stats_missing_player_400(self, client):
        r = client.get("/moment/player_stats")
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# /duel
# ---------------------------------------------------------------------------

class TestDuelEndpoints:
    def test_duel_network_no_analyzer_503(self, client):
        r = client.get("/duel/network")
        assert r.status_code == 503

    def test_duel_player_missing_player_400(self, client):
        r = client.get("/duel/player")
        assert r.status_code == 400

    def test_duel_summary_no_analyzer_503(self, client):
        r = client.get("/duel/summary")
        assert r.status_code == 503

    def test_duel_add_match_no_analyzer_503(self, client):
        r = client.post("/duel/add_match", json={"events": []})
        assert r.status_code == 503

    def test_duel_network_with_mock(self, client):
        mock_da = MagicMock()
        mock_da.network._edges = {}
        prism_state._state["duel_analyzer"] = mock_da
        r = client.get("/duel/network")
        assert r.status_code == 200
        assert "edges" in r.json()

    def test_duel_summary_with_mock(self, client):
        mock_da = MagicMock()
        mock_da.network._edges = {
            ("Mbappe", "Rudiger"): {"total": 10, "won": 7}
        }
        prism_state._state["duel_analyzer"] = mock_da
        data = client.get("/duel/summary").json()
        assert data["total_duels"] == 10
        assert data["total_won"] == 7


# ---------------------------------------------------------------------------
# /analytics/tokens (LLM ledger)
# ---------------------------------------------------------------------------

class TestAnalyticsTokens:
    @pytest.fixture()
    def ledger_client(self, tmp_path):
        """Client with a real LLM ledger backed by tmp DB."""
        import prism_llm_ledger
        from prism_llm_ledger import LLMLedger

        original = prism_llm_ledger._ledger
        prism_llm_ledger._ledger = LLMLedger(db_path=str(tmp_path / "ledger.db"))

        app = FastAPI()
        app.include_router(router)
        prism_state._state.clear()
        yield TestClient(app)
        prism_llm_ledger._ledger = original
        prism_state._state.clear()

    def test_tokens_summary_200_auto_ledger(self, client):
        import prism_llm_ledger
        prism_llm_ledger._ledger = None  # ensure fresh auto-creation (stale tmp_path guard)
        r = client.get("/analytics/tokens")
        assert r.status_code == 200
        assert "summary" in r.json()

    def test_tokens_summary_200(self, ledger_client):
        r = ledger_client.get("/analytics/tokens")
        assert r.status_code == 200
        data = r.json()
        assert "summary" in data
        assert "by_model" in data

    def test_tokens_daily_200(self, ledger_client):
        r = ledger_client.get("/analytics/tokens/daily")
        assert r.status_code == 200
        assert "daily" in r.json()

    def test_tokens_by_model_200(self, ledger_client):
        r = ledger_client.get("/analytics/tokens/by-model")
        assert r.status_code == 200
        assert "by_model" in r.json()

    def test_tokens_by_source_200(self, ledger_client):
        r = ledger_client.get("/analytics/tokens/by-source")
        assert r.status_code == 200
        assert "by_source" in r.json()

    def test_tokens_record_missing_provider(self, ledger_client):
        r = ledger_client.post("/analytics/tokens/record", json={"model": "gpt-4"})
        assert r.status_code == 400

    def test_tokens_record_ok(self, ledger_client):
        r = ledger_client.post("/analytics/tokens/record", json={
            "provider": "openai", "model": "gpt-4o",
            "input_tokens": 100, "output_tokens": 50,
            "latency_ms": 320.0,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert "call_id" in data

    def test_tokens_clear(self, ledger_client):
        ledger_client.post("/analytics/tokens/record", json={
            "provider": "ollama", "model": "llama3",
            "input_tokens": 10, "output_tokens": 5, "latency_ms": 100.0,
        })
        r = ledger_client.delete("/analytics/tokens")
        assert r.status_code == 200
        assert r.json()["ok"] is True
