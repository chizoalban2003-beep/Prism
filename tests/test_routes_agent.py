"""
Tests for prism_routes_agent — /status, /plan, /reflect, /history,
/artifacts, /identity, /context, /outcomes/stats, /reflection.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import prism_state
from prism_routes_agent import router


def _app_no_agent():
    """App where _get_agent() returns None — tests 503 fallbacks."""
    app = FastAPI()
    app.include_router(router)
    prism_state._state.clear()
    return TestClient(app)


def _app_with_agent():
    """App with a minimal mock agent wired into state."""
    app = FastAPI()
    app.include_router(router)

    agent = MagicMock()
    agent.status.return_value = {"status": "ok", "horizon_goals": 2}
    agent.reflect.return_value = {"insight": "test"}
    agent._assistant.history.return_value = [{"role": "user", "content": "hi"}]
    agent.recent_artifacts.return_value = []
    agent.identity.return_value = {"domains": {}, "soul": ""}
    agent.identity_domains.return_value = ["sport", "health"]
    agent.observe_identity.return_value = {"domain": "sport", "score": 0.6}
    agent.reset_identity_domain.return_value = {"reset": True}
    agent._profile = MagicMock()
    agent._profile.name = "Test"
    agent._horizon = None
    agent._outcome_tracker = None
    agent._reflection = None
    agent._context_manager = None

    prism_state._state.clear()
    prism_state._state["agent"] = agent
    return TestClient(app), agent


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    prism_state._state.clear()


# ---------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------

class TestAgentStatus:
    def test_status_503_no_agent(self):
        r = _app_no_agent().get("/status")
        assert r.status_code == 503

    def test_status_200_with_agent(self):
        client, _ = _app_with_agent()
        r = client.get("/status")
        # Ollama check may fail (not running in CI) — we still expect 200
        assert r.status_code == 200

    def test_status_has_ollama_key(self):
        client, _ = _app_with_agent()
        data = client.get("/status").json()
        assert "ollama" in data


# ---------------------------------------------------------------------------
# /reflect
# ---------------------------------------------------------------------------

class TestAgentReflect:
    def test_reflect_503_no_agent(self):
        r = _app_no_agent().get("/reflect")
        assert r.status_code == 503

    def test_reflect_200_with_agent(self):
        client, _ = _app_with_agent()
        r = client.get("/reflect")
        assert r.status_code == 200

    def test_reflect_returns_dict(self):
        client, _ = _app_with_agent()
        data = client.get("/reflect").json()
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# /artifacts
# ---------------------------------------------------------------------------

class TestAgentArtifacts:
    def test_artifacts_503_no_agent(self):
        r = _app_no_agent().get("/artifacts")
        assert r.status_code == 503

    def test_artifacts_200(self):
        client, _ = _app_with_agent()
        r = client.get("/artifacts")
        assert r.status_code == 200
        assert "artifacts" in r.json()

    def test_artifacts_rate_missing_id(self):
        client, _ = _app_with_agent()
        r = client.post("/artifacts/rate", json={"rating": 0.8})
        assert r.status_code == 400

    def test_artifacts_rate_ok(self):
        client, agent = _app_with_agent()
        agent.rate_artifact.return_value = {"rated": True}
        r = client.post("/artifacts/rate", json={"artifact_id": "abc", "rating": 0.9})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# /identity
# ---------------------------------------------------------------------------

class TestAgentIdentity:
    def test_identity_503_no_agent(self):
        r = _app_no_agent().get("/identity")
        assert r.status_code == 503

    def test_identity_200(self):
        client, _ = _app_with_agent()
        r = client.get("/identity")
        assert r.status_code == 200

    def test_identity_domains_200(self):
        client, _ = _app_with_agent()
        data = client.get("/identity/domains").json()
        assert "domains" in data
        assert isinstance(data["domains"], list)

    def test_identity_observe_missing_domain(self):
        client, _ = _app_with_agent()
        r = client.post("/identity/observe", json={"fulcrum": 0.5})
        assert r.status_code == 400

    def test_identity_observe_ok(self):
        client, _ = _app_with_agent()
        r = client.post("/identity/observe", json={"domain": "sport", "rating": 0.7})
        assert r.status_code == 200

    def test_identity_reset_missing_domain(self):
        client, _ = _app_with_agent()
        r = client.post("/identity/reset", json={})
        assert r.status_code == 400

    def test_identity_reset_ok(self):
        client, _ = _app_with_agent()
        r = client.post("/identity/reset", json={"domain": "sport"})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# /context
# ---------------------------------------------------------------------------

class TestAgentContext:
    def test_context_no_agent(self):
        r = _app_no_agent().get("/context")
        assert r.status_code == 200
        data = r.json()
        assert data["active"] == "default"

    def test_context_with_agent_no_cm(self):
        client, _ = _app_with_agent()
        data = client.get("/context").json()
        assert "active" in data


# ---------------------------------------------------------------------------
# /outcomes/stats
# ---------------------------------------------------------------------------

class TestOutcomesStats:
    def test_outcomes_no_tracker(self):
        client, _ = _app_with_agent()
        data = client.get("/outcomes/stats").json()
        assert data["available"] is False

    def test_outcomes_with_tracker(self):
        client, agent = _app_with_agent()
        mock_tracker = MagicMock()
        mock_tracker.stats.return_value = {"done": 5, "total": 7, "completion_rate": 0.71}
        agent._outcome_tracker = mock_tracker
        data = client.get("/outcomes/stats").json()
        assert data["available"] is True
        assert "done" in data


# ---------------------------------------------------------------------------
# /reflection
# ---------------------------------------------------------------------------

class TestReflectionEndpoint:
    def test_reflection_no_agent(self):
        r = _app_no_agent().get("/reflection")
        assert r.status_code == 200
        assert r.json()["available"] is False

    def test_reflection_with_reflection(self):
        client, agent = _app_with_agent()
        report = MagicMock()
        report.summary = "Patterns detected."
        report.patterns = ["stays active in morning"]
        report.belief_proposals = []
        report.unresolved_goals = []
        report.applied = 0
        report.ran_at = 1234567890.0
        agent._reflection = MagicMock()
        agent._reflection.run.return_value = report
        data = client.get("/reflection").json()
        assert data["available"] is True
        assert "summary" in data
