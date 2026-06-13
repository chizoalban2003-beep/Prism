"""
Tests for prism_routes_sessions — full CRUD session lifecycle.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import prism_state
from prism_routes_sessions import router


@pytest.fixture()
def client(tmp_path):
    """App with a real SessionManager backed by a temp SQLite DB."""
    from prism_session_manager import SessionManager

    app = FastAPI()
    app.include_router(router)

    sm = SessionManager(db_path=str(tmp_path / "sessions.db"))
    # Patch get_session_manager so the route helper finds our instance
    import prism_session_manager
    original = prism_session_manager._manager
    prism_session_manager._manager = sm

    prism_state._state.clear()
    yield TestClient(app)

    prism_session_manager._manager = original
    prism_state._state.clear()


# ---------------------------------------------------------------------------
# GET /sessions  (no manager → 503 path tested separately)
# ---------------------------------------------------------------------------

class TestListSessions:
    def test_list_empty(self, client):
        data = client.get("/sessions").json()
        assert data["sessions"] == []
        assert data["total"] == 0

    def test_list_after_create(self, client):
        client.post("/sessions", json={"name": "Work"})
        data = client.get("/sessions").json()
        assert data["total"] == 1


# ---------------------------------------------------------------------------
# POST /sessions
# ---------------------------------------------------------------------------

class TestCreateSession:
    def test_create_200(self, client):
        r = client.post("/sessions", json={"name": "Research"})
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "Research"
        assert "session_id" in data

    def test_create_missing_name_400(self, client):
        r = client.post("/sessions", json={"description": "no name"})
        assert r.status_code == 400

    def test_create_sets_description(self, client):
        r = client.post("/sessions", json={"name": "S", "description": "desc"})
        assert r.json()["description"] == "desc"

    def test_create_sets_tags(self, client):
        r = client.post("/sessions", json={"name": "S", "tags": ["ai", "code"]})
        assert r.json()["tags"] == ["ai", "code"]


# ---------------------------------------------------------------------------
# GET /sessions/active
# ---------------------------------------------------------------------------

class TestActiveSession:
    def test_active_none_initially(self, client):
        data = client.get("/sessions/active").json()
        assert data["active_session_id"] is None

    def test_set_and_get_active(self, client):
        sid = client.post("/sessions", json={"name": "Active"}).json()["session_id"]
        client.post("/sessions/active", json={"session_id": sid})
        data = client.get("/sessions/active").json()
        assert data["active_session_id"] == sid
        assert data["session"]["name"] == "Active"

    def test_set_active_missing_id_400(self, client):
        r = client.post("/sessions/active", json={})
        assert r.status_code == 400

    def test_set_active_not_found_404(self, client):
        r = client.post("/sessions/active", json={"session_id": "nonexistent"})
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /sessions/{id} + PATCH + DELETE
# ---------------------------------------------------------------------------

class TestSessionCRUD:
    def test_get_not_found_404(self, client):
        r = client.get("/sessions/doesnotexist")
        assert r.status_code == 404

    def test_get_existing(self, client):
        sid = client.post("/sessions", json={"name": "X"}).json()["session_id"]
        data = client.get(f"/sessions/{sid}").json()
        assert data["session_id"] == sid

    def test_patch_name(self, client):
        sid = client.post("/sessions", json={"name": "Old"}).json()["session_id"]
        data = client.patch(f"/sessions/{sid}", json={"name": "New"}).json()
        assert data["name"] == "New"

    def test_patch_not_found_404(self, client):
        r = client.patch("/sessions/ghost", json={"name": "X"})
        assert r.status_code == 404

    def test_delete_200(self, client):
        sid = client.post("/sessions", json={"name": "Del"}).json()["session_id"]
        r = client.delete(f"/sessions/{sid}")
        assert r.status_code == 200
        assert r.json()["deleted"] is True

    def test_delete_clears_active(self, client):
        sid = client.post("/sessions", json={"name": "D"}).json()["session_id"]
        client.post("/sessions/active", json={"session_id": sid})
        client.delete(f"/sessions/{sid}")
        data = client.get("/sessions/active").json()
        assert data["active_session_id"] is None

    def test_delete_not_found_404(self, client):
        r = client.delete("/sessions/ghost")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

class TestSessionMessages:
    def test_add_message_200(self, client):
        sid = client.post("/sessions", json={"name": "Chat"}).json()["session_id"]
        r = client.post(f"/sessions/{sid}/messages",
                        json={"role": "user", "content": "Hello"})
        assert r.status_code == 200
        data = r.json()
        assert data["role"] == "user"
        assert data["content"] == "Hello"

    def test_add_message_missing_content_400(self, client):
        sid = client.post("/sessions", json={"name": "C"}).json()["session_id"]
        r = client.post(f"/sessions/{sid}/messages", json={"role": "user"})
        assert r.status_code == 400

    def test_add_message_not_found_404(self, client):
        r = client.post("/sessions/ghost/messages",
                        json={"role": "user", "content": "hi"})
        assert r.status_code == 404

    def test_get_history_200(self, client):
        sid = client.post("/sessions", json={"name": "H"}).json()["session_id"]
        client.post(f"/sessions/{sid}/messages", json={"content": "msg1"})
        client.post(f"/sessions/{sid}/messages", json={"content": "msg2"})
        data = client.get(f"/sessions/{sid}/history").json()
        assert len(data["messages"]) == 2

    def test_clear_history(self, client):
        sid = client.post("/sessions", json={"name": "C"}).json()["session_id"]
        client.post(f"/sessions/{sid}/messages", json={"content": "x"})
        r = client.delete(f"/sessions/{sid}/history")
        assert r.status_code == 200
        assert r.json()["deleted"] == 1
        history = client.get(f"/sessions/{sid}/history").json()
        assert history["messages"] == []
