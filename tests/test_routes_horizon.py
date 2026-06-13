"""
Tests for prism_routes_horizon — /horizon/* and /push/status endpoints.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import prism_state
from prism_routes_horizon import router


def _app_no_horizon():
    app = FastAPI()
    app.include_router(router)
    prism_state._state.clear()
    return TestClient(app)


@pytest.fixture()
def client(tmp_path):
    """App with a real HorizonPlanner backed by a temp DB."""
    from prism_horizon import HorizonPlanner

    app = FastAPI()
    app.include_router(router)

    h = HorizonPlanner(db_path=str(tmp_path / "horizon.db"))
    agent = MagicMock()
    agent._horizon = h
    agent._push = None

    prism_state._state.clear()
    prism_state._state["agent"] = agent
    yield TestClient(app)
    prism_state._state.clear()


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    prism_state._state.clear()


# ---------------------------------------------------------------------------
# /horizon/goals
# ---------------------------------------------------------------------------

class TestHorizonGoals:
    def test_goals_no_horizon(self):
        data = _app_no_horizon().get("/horizon/goals").json()
        assert data["goals"] == []
        assert data["total"] == 0

    def test_goals_empty(self, client):
        data = client.get("/horizon/goals").json()
        assert data["total"] == 0

    def test_goals_after_create(self, client):
        client.post("/horizon/goal", json={
            "intent": "Learn piano",
            "trigger_condition": "practice session starts",
        })
        data = client.get("/horizon/goals").json()
        assert data["total"] == 1
        assert data["goals"][0]["intent"] == "Learn piano"


# ---------------------------------------------------------------------------
# /horizon/status
# ---------------------------------------------------------------------------

class TestHorizonStatus:
    def test_status_no_horizon(self):
        data = _app_no_horizon().get("/horizon/status").json()
        assert data["available"] is False

    def test_status_200(self, client):
        data = client.get("/horizon/status").json()
        assert data["available"] is True
        assert "total_goals" in data or "goals" in str(data)


# ---------------------------------------------------------------------------
# POST /horizon/goal
# ---------------------------------------------------------------------------

class TestHorizonGoalCreate:
    def test_create_no_horizon_503(self):
        r = _app_no_horizon().post("/horizon/goal", json={
            "intent": "x", "trigger_condition": "y"
        })
        assert r.status_code == 503

    def test_create_missing_intent_400(self, client):
        r = client.post("/horizon/goal", json={"trigger_condition": "y"})
        assert r.status_code == 400

    def test_create_missing_trigger_400(self, client):
        r = client.post("/horizon/goal", json={"intent": "learn"})
        assert r.status_code == 400

    def test_create_returns_goal_id(self, client):
        r = client.post("/horizon/goal", json={
            "intent": "Read 12 books",
            "trigger_condition": "book session starts",
            "completion_condition": "12 books read",
        })
        assert r.status_code == 200
        data = r.json()
        assert "goal_id" in data
        assert data["status"] == "watching"

    def test_create_with_expiry(self, client):
        r = client.post("/horizon/goal", json={
            "intent": "Run 5k",
            "trigger_condition": "morning run",
            "expires_in_days": 30,
        })
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# POST /horizon/goal/{id}/complete + /abandon + /context
# ---------------------------------------------------------------------------

class TestHorizonGoalLifecycle:
    def _create_goal(self, client) -> str:
        r = client.post("/horizon/goal", json={
            "intent": "Write novel",
            "trigger_condition": "writing session",
        })
        return r.json()["goal_id"]

    def test_complete_goal(self, client):
        gid = self._create_goal(client)
        r = client.post(f"/horizon/goal/{gid}/complete", json={"notes": "Done!"})
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_abandon_goal(self, client):
        gid = self._create_goal(client)
        r = client.post(f"/horizon/goal/{gid}/abandon", json={"reason": "changed mind"})
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_update_context(self, client):
        gid = self._create_goal(client)
        r = client.post(f"/horizon/goal/{gid}/context", json={
            "chapters_written": 3, "mood": "inspired"
        })
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert set(data["context_keys"]) == {"chapters_written", "mood"}

    def test_complete_nonexistent_goal(self, client):
        r = client.post("/horizon/goal/ghost123/complete", json={})
        assert r.status_code == 200
        assert r.json()["ok"] is False


# ---------------------------------------------------------------------------
# /push/status
# ---------------------------------------------------------------------------

class TestPushStatus:
    def test_push_no_agent(self):
        r = _app_no_horizon().get("/push/status")
        assert r.status_code == 200
        assert r.json()["configured"] is False
