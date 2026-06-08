"""
test_proactive_triggers_api.py
==============================
Tests for the proactive trigger management API:
  GET    /proactive/triggers
  POST   /proactive/triggers
  DELETE /proactive/triggers/{trigger_id}
  POST   /proactive/triggers/{trigger_id}/pause
  POST   /proactive/triggers/{trigger_id}/resume
  POST   /proactive/deliver/{trigger_id}
"""
from __future__ import annotations

import tempfile
import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prism_asgi import app
from prism_state import _set_state


def _make_proactive(tmp_path):
    from prism_proactive import PrismProactive
    db = str(tmp_path / "proactive.db")
    return PrismProactive(db_path=db)


@pytest.fixture()
def tmp_path():
    with tempfile.TemporaryDirectory() as d:
        from pathlib import Path
        yield Path(d)


@pytest.fixture()
def client(tmp_path):
    p = _make_proactive(tmp_path)
    agent = MagicMock()
    agent._proactive = p
    _set_state(agent=agent, proactive=p)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def client_no_proactive():
    agent = MagicMock()
    agent._proactive = None  # prevent MagicMock auto-attr from looking initialised
    _set_state(agent=agent, proactive=None)  # explicitly clear proactive from shared state
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# GET /proactive/triggers
# ---------------------------------------------------------------------------

def test_list_triggers_empty(client):
    r = client.get("/proactive/triggers")
    assert r.status_code == 200
    data = r.json()
    assert "triggers" in data
    assert "scheduled" in data
    assert data["count"] == 0


def test_list_triggers_no_proactive(client_no_proactive):
    r = client_no_proactive.get("/proactive/triggers")
    assert r.status_code == 200
    data = r.json()
    assert data["triggers"] == []
    assert "note" in data


# ---------------------------------------------------------------------------
# POST /proactive/triggers — scheduled
# ---------------------------------------------------------------------------

def test_create_scheduled_trigger_fire_at(client):
    fire_at = time.time() + 3600
    r = client.post("/proactive/triggers", json={
        "type": "scheduled",
        "message": "Test reminder",
        "fire_at": fire_at,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["type"] == "scheduled"
    assert "trigger_id" in data


def test_create_scheduled_trigger_in_seconds(client):
    r = client.post("/proactive/triggers", json={
        "type": "scheduled",
        "message": "Reminder in 5 minutes",
        "in_seconds": 300,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["fire_at"] > time.time()


def test_create_scheduled_trigger_custom_id(client):
    r = client.post("/proactive/triggers", json={
        "type": "scheduled",
        "message": "Custom ID reminder",
        "in_seconds": 60,
        "trigger_id": "my_custom_reminder",
    })
    assert r.status_code == 200
    assert r.json()["trigger_id"] == "my_custom_reminder"


def test_create_scheduled_trigger_missing_message(client):
    r = client.post("/proactive/triggers", json={
        "type": "scheduled",
        "fire_at": time.time() + 100,
    })
    assert r.status_code == 400


def test_create_scheduled_trigger_missing_time(client):
    r = client.post("/proactive/triggers", json={
        "type": "scheduled",
        "message": "No time provided",
    })
    assert r.status_code == 400


def test_scheduled_trigger_appears_in_list(client):
    client.post("/proactive/triggers", json={
        "type": "scheduled",
        "message": "List test",
        "in_seconds": 100,
        "trigger_id": "list_test_id",
    })
    r = client.get("/proactive/triggers")
    data = r.json()
    ids = [s["trigger_id"] for s in data["scheduled"]]
    assert "list_test_id" in ids


# ---------------------------------------------------------------------------
# POST /proactive/triggers — condition
# ---------------------------------------------------------------------------

def test_create_condition_trigger(client):
    r = client.post("/proactive/triggers", json={
        "type": "condition",
        "trigger_id": "custom_cond",
        "name": "Custom condition",
        "check_every": 30,
        "cooldown": 300,
        "condition_attr": "some_flag",
        "message": "Flag was set!",
    })
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["type"] == "condition"
    assert data["trigger_id"] == "custom_cond"


def test_condition_trigger_appears_in_list(client):
    client.post("/proactive/triggers", json={
        "type": "condition",
        "trigger_id": "cond_list_test",
        "name": "List test condition",
        "check_every": 60,
        "message": "test",
    })
    r = client.get("/proactive/triggers")
    ids = [t["trigger_id"] for t in r.json()["triggers"]]
    assert "cond_list_test" in ids


def test_create_unknown_type(client):
    r = client.post("/proactive/triggers", json={"type": "unknown"})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# DELETE /proactive/triggers/{trigger_id}
# ---------------------------------------------------------------------------

def test_delete_scheduled_trigger(client):
    client.post("/proactive/triggers", json={
        "type": "scheduled",
        "message": "Delete me",
        "in_seconds": 60,
        "trigger_id": "to_delete",
    })
    r = client.delete("/proactive/triggers/to_delete")
    assert r.status_code == 200
    assert r.json()["ok"] is True

    r2 = client.get("/proactive/triggers")
    ids = [s["trigger_id"] for s in r2.json()["scheduled"]]
    assert "to_delete" not in ids


def test_delete_condition_trigger(client):
    client.post("/proactive/triggers", json={
        "type": "condition",
        "trigger_id": "del_cond",
        "name": "Delete condition",
        "check_every": 60,
        "message": "bye",
    })
    r = client.delete("/proactive/triggers/del_cond")
    assert r.status_code == 200


def test_delete_nonexistent_trigger(client):
    r = client.delete("/proactive/triggers/ghost_id")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# POST /proactive/triggers/{trigger_id}/pause  &  /resume
# ---------------------------------------------------------------------------

def test_pause_and_resume_trigger(client):
    client.post("/proactive/triggers", json={
        "type": "condition",
        "trigger_id": "pausable",
        "name": "Pausable trigger",
        "check_every": 60,
        "message": "test",
        "enabled": True,
    })

    r = client.post("/proactive/triggers/pausable/pause")
    assert r.status_code == 200
    assert r.json()["enabled"] is False

    r2 = client.get("/proactive/triggers")
    t = next(t for t in r2.json()["triggers"] if t["trigger_id"] == "pausable")
    assert t["enabled"] is False

    r3 = client.post("/proactive/triggers/pausable/resume")
    assert r3.status_code == 200
    assert r3.json()["enabled"] is True


def test_pause_nonexistent_trigger(client):
    r = client.post("/proactive/triggers/ghost/pause")
    assert r.status_code == 404


def test_resume_nonexistent_trigger(client):
    r = client.post("/proactive/triggers/ghost/resume")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# POST /proactive/deliver/{trigger_id}
# ---------------------------------------------------------------------------

def test_deliver_event(client):
    r = client.post("/proactive/deliver/some_event_id")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert r.json()["trigger_id"] == "some_event_id"


# ---------------------------------------------------------------------------
# 503 when proactive not configured
# ---------------------------------------------------------------------------

def test_503_on_create_no_proactive(client_no_proactive):
    r = client_no_proactive.post("/proactive/triggers", json={
        "type": "scheduled", "message": "test", "in_seconds": 60,
    })
    assert r.status_code == 503


def test_503_on_delete_no_proactive(client_no_proactive):
    r = client_no_proactive.delete("/proactive/triggers/any")
    assert r.status_code == 503


def test_503_on_pause_no_proactive(client_no_proactive):
    r = client_no_proactive.post("/proactive/triggers/any/pause")
    assert r.status_code == 503


def test_503_on_deliver_no_proactive(client_no_proactive):
    r = client_no_proactive.post("/proactive/deliver/any")
    assert r.status_code == 503
