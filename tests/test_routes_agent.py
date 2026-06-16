"""
Tests for prism_routes_agent — /status, /plan, /context,
/outcomes/stats, /reflection.
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
