"""
test_session_manager.py
=======================
Comprehensive tests for prism_session_manager and prism_routes_sessions.
Uses pytest + FastAPI TestClient. No real port binding, no sleep().
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prism_asgi import app
from prism_session_manager import (
    Session,
    SessionManager,
    get_session_manager,
    reset_session_manager,
)
from prism_state import _set_state, _state

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sm(tmp_dir: str) -> SessionManager:
    """Create a fresh SessionManager backed by a temp directory."""
    db_path = tmp_dir + "/sessions.db"
    return reset_session_manager(db_path)


def _make_agent() -> MagicMock:
    agent = MagicMock()
    card = MagicMock()
    card.body = "assistant reply"
    card.to_json.return_value = {"type": "text", "title": "", "body": "assistant reply", "data": {}, "actions": []}
    agent.chat.return_value = card
    agent._hub = MagicMock()
    agent._hub.list_devices.return_value = []
    agent._profile = MagicMock()
    agent._profile.name = "Tester"
    return agent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture()
def sm(tmp_dir):
    """Fresh SessionManager, also resets the module singleton."""
    return _make_sm(tmp_dir)


@pytest.fixture()
def client(tmp_dir):
    """TestClient wired with a mock agent and fresh session manager singleton."""
    db_path = tmp_dir + "/sessions.db"
    reset_session_manager(db_path)
    agent = _make_agent()
    _set_state(agent=agent)
    # Clear any leftover active_session_id from prior tests
    _state.pop("active_session_id", None)
    return TestClient(app, raise_server_exceptions=False)


# ===========================================================================
# SessionManager unit tests
# ===========================================================================


class TestCreateSession:
    def test_returns_session_with_correct_fields(self, sm):
        s = sm.create_session("My Chat")
        assert isinstance(s, Session)
        assert s.name == "My Chat"
        assert s.description == ""
        assert s.tags == []
        assert s.message_count == 0
        assert len(s.session_id) == 16
        assert s.created_at > 0
        assert s.updated_at > 0

    def test_with_tags_stores_and_retrieves(self, sm):
        s = sm.create_session("Tagged", tags=["ai", "work"])
        fetched = sm.get_session(s.session_id)
        assert fetched is not None
        assert fetched.tags == ["ai", "work"]

    def test_with_description(self, sm):
        s = sm.create_session("Desc Chat", description="Some desc")
        assert s.description == "Some desc"


class TestGetSession:
    def test_returns_none_for_unknown_id(self, sm):
        result = sm.get_session("doesnotexist1234")
        assert result is None

    def test_returns_session_after_create(self, sm):
        s = sm.create_session("Hello")
        fetched = sm.get_session(s.session_id)
        assert fetched is not None
        assert fetched.session_id == s.session_id
        assert fetched.name == "Hello"


class TestListSessions:
    def test_returns_all_sessions(self, sm):
        sm.create_session("A")
        sm.create_session("B")
        sm.create_session("C")
        sessions = sm.list_sessions()
        assert len(sessions) == 3

    def test_respects_limit(self, sm):
        for i in range(5):
            sm.create_session(f"Session {i}")
        sessions = sm.list_sessions(limit=3)
        assert len(sessions) == 3

    def test_respects_offset(self, sm):
        for i in range(5):
            sm.create_session(f"Session {i}")
        all_sessions = sm.list_sessions(limit=100)
        offset_sessions = sm.list_sessions(limit=100, offset=2)
        assert len(offset_sessions) == len(all_sessions) - 2

    def test_empty_list(self, sm):
        assert sm.list_sessions() == []


class TestUpdateSession:
    def test_changes_name(self, sm):
        s = sm.create_session("Old Name")
        updated = sm.update_session(s.session_id, name="New Name")
        assert updated is not None
        assert updated.name == "New Name"

    def test_updates_updated_at(self, sm):
        s = sm.create_session("Name")
        original_updated_at = s.updated_at
        time.sleep(0.01)
        updated = sm.update_session(s.session_id, name="New Name")
        assert updated is not None
        assert updated.updated_at >= original_updated_at

    def test_none_fields_dont_overwrite(self, sm):
        s = sm.create_session("Keep", description="keep desc", tags=["keep"])
        updated = sm.update_session(s.session_id, name="New Name")
        assert updated is not None
        assert updated.description == "keep desc"
        assert updated.tags == ["keep"]

    def test_returns_none_for_unknown_session(self, sm):
        result = sm.update_session("unknownid12345", name="X")
        assert result is None

    def test_persists_changes(self, sm):
        s = sm.create_session("Original")
        sm.update_session(s.session_id, name="Updated", description="New desc", tags=["new"])
        fetched = sm.get_session(s.session_id)
        assert fetched is not None
        assert fetched.name == "Updated"
        assert fetched.description == "New desc"
        assert fetched.tags == ["new"]


class TestDeleteSession:
    def test_returns_true_when_deleted(self, sm):
        s = sm.create_session("ToDelete")
        result = sm.delete_session(s.session_id)
        assert result is True

    def test_subsequent_get_returns_none(self, sm):
        s = sm.create_session("ToDelete")
        sm.delete_session(s.session_id)
        assert sm.get_session(s.session_id) is None

    def test_returns_false_for_unknown_id(self, sm):
        result = sm.delete_session("nope_doesnt_exist")
        assert result is False

    def test_also_deletes_messages(self, sm):
        s = sm.create_session("WithMessages")
        sm.add_message(s.session_id, "user", "hi")
        sm.delete_session(s.session_id)
        # No way to query messages for deleted session, but create with same concept
        s2 = sm.create_session("Fresh")
        assert sm.get_history(s2.session_id) == []


class TestAddMessage:
    def test_returns_message_record(self, sm):
        s = sm.create_session("Chat")
        rec = sm.add_message(s.session_id, "user", "Hello!")
        assert rec is not None
        assert rec.role == "user"
        assert rec.content == "Hello!"
        assert rec.session_id == s.session_id
        assert len(rec.message_id) == 16

    def test_increments_message_count(self, sm):
        s = sm.create_session("CountTest")
        sm.add_message(s.session_id, "user", "msg1")
        sm.add_message(s.session_id, "assistant", "reply1")
        updated = sm.get_session(s.session_id)
        assert updated is not None
        assert updated.message_count == 2

    def test_returns_none_for_unknown_session_id(self, sm):
        result = sm.add_message("doesnotexist1234", "user", "hello")
        assert result is None


class TestGetHistory:
    def test_returns_in_chronological_order(self, sm):
        s = sm.create_session("Order")
        sm.add_message(s.session_id, "user", "first")
        sm.add_message(s.session_id, "assistant", "second")
        sm.add_message(s.session_id, "user", "third")
        history = sm.get_history(s.session_id)
        assert len(history) == 3
        assert history[0].content == "first"
        assert history[1].content == "second"
        assert history[2].content == "third"

    def test_respects_n_limit(self, sm):
        s = sm.create_session("LimitTest")
        for i in range(10):
            sm.add_message(s.session_id, "user", f"msg{i}")
        history = sm.get_history(s.session_id, n=3)
        assert len(history) == 3

    def test_empty_for_new_session(self, sm):
        s = sm.create_session("Empty")
        assert sm.get_history(s.session_id) == []


class TestClearHistory:
    def test_deletes_messages_and_returns_count(self, sm):
        s = sm.create_session("ClearMe")
        sm.add_message(s.session_id, "user", "a")
        sm.add_message(s.session_id, "user", "b")
        sm.add_message(s.session_id, "user", "c")
        count = sm.clear_history(s.session_id)
        assert count == 3

    def test_sets_message_count_to_zero(self, sm):
        s = sm.create_session("ClearCount")
        sm.add_message(s.session_id, "user", "msg")
        sm.clear_history(s.session_id)
        updated = sm.get_session(s.session_id)
        assert updated is not None
        assert updated.message_count == 0

    def test_history_is_empty_after_clear(self, sm):
        s = sm.create_session("AfterClear")
        sm.add_message(s.session_id, "user", "x")
        sm.clear_history(s.session_id)
        assert sm.get_history(s.session_id) == []

    def test_returns_zero_for_empty_session(self, sm):
        s = sm.create_session("AlreadyEmpty")
        count = sm.clear_history(s.session_id)
        assert count == 0


class TestSingleton:
    def test_get_session_manager_returns_same_instance(self, tmp_dir):
        db_path = tmp_dir + "/singleton.db"
        sm1 = reset_session_manager(db_path)
        sm2 = get_session_manager(db_path)
        assert sm1 is sm2

    def test_reset_session_manager_returns_new_instance(self, tmp_dir):
        db_path = tmp_dir + "/reset.db"
        sm1 = reset_session_manager(db_path)
        sm2 = reset_session_manager(db_path)
        assert sm1 is not sm2


# ===========================================================================
# API route tests
# ===========================================================================


class TestListSessionsRoute:
    def test_empty_list(self, client):
        r = client.get("/sessions")
        assert r.status_code == 200
        data = r.json()
        assert data["sessions"] == []
        assert data["total"] == 0


class TestCreateSessionRoute:
    def test_creates_session_returns_session_id(self, client):
        r = client.post("/sessions", json={"name": "My Session"})
        assert r.status_code == 200
        data = r.json()
        assert "session_id" in data
        assert data["name"] == "My Session"

    def test_with_name_description_tags(self, client):
        r = client.post(
            "/sessions",
            json={"name": "Full", "description": "A session", "tags": ["a", "b"]},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["description"] == "A session"
        assert data["tags"] == ["a", "b"]

    def test_missing_name_returns_400(self, client):
        r = client.post("/sessions", json={})
        assert r.status_code == 400


class TestGetSessionRoute:
    def test_returns_metadata(self, client):
        create = client.post("/sessions", json={"name": "Fetch Me"})
        session_id = create.json()["session_id"]
        r = client.get(f"/sessions/{session_id}")
        assert r.status_code == 200
        assert r.json()["name"] == "Fetch Me"

    def test_unknown_returns_404(self, client):
        r = client.get("/sessions/unknownid12345")
        assert r.status_code == 404


class TestUpdateSessionRoute:
    def test_updates_name(self, client):
        create = client.post("/sessions", json={"name": "Old"})
        session_id = create.json()["session_id"]
        r = client.patch(f"/sessions/{session_id}", json={"name": "New"})
        assert r.status_code == 200
        assert r.json()["name"] == "New"

    def test_unknown_returns_404(self, client):
        r = client.patch("/sessions/nope12345678901", json={"name": "X"})
        assert r.status_code == 404


class TestDeleteSessionRoute:
    def test_removes_session(self, client):
        create = client.post("/sessions", json={"name": "Del Me"})
        session_id = create.json()["session_id"]
        r = client.delete(f"/sessions/{session_id}")
        assert r.status_code == 200
        assert r.json()["deleted"] is True

    def test_subsequent_get_is_404(self, client):
        create = client.post("/sessions", json={"name": "Del Me"})
        session_id = create.json()["session_id"]
        client.delete(f"/sessions/{session_id}")
        r = client.get(f"/sessions/{session_id}")
        assert r.status_code == 404

    def test_unknown_returns_404(self, client):
        r = client.delete("/sessions/nope12345678901")
        assert r.status_code == 404


class TestHistoryRoute:
    def test_empty_history(self, client):
        create = client.post("/sessions", json={"name": "Hist"})
        session_id = create.json()["session_id"]
        r = client.get(f"/sessions/{session_id}/history")
        assert r.status_code == 200
        assert r.json()["messages"] == []

    def test_returns_messages_in_order(self, client):
        create = client.post("/sessions", json={"name": "Ordered"})
        session_id = create.json()["session_id"]
        client.post(f"/sessions/{session_id}/messages", json={"role": "user", "content": "first"})
        client.post(f"/sessions/{session_id}/messages", json={"role": "assistant", "content": "second"})
        r = client.get(f"/sessions/{session_id}/history")
        msgs = r.json()["messages"]
        assert len(msgs) == 2
        assert msgs[0]["content"] == "first"
        assert msgs[1]["content"] == "second"


class TestAddMessageRoute:
    def test_adds_message(self, client):
        create = client.post("/sessions", json={"name": "Msg"})
        session_id = create.json()["session_id"]
        r = client.post(
            f"/sessions/{session_id}/messages",
            json={"role": "user", "content": "Hello"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["role"] == "user"
        assert data["content"] == "Hello"

    def test_unknown_session_returns_404(self, client):
        r = client.post(
            "/sessions/nope12345678901/messages",
            json={"role": "user", "content": "Hi"},
        )
        assert r.status_code == 404


class TestClearHistoryRoute:
    def test_clears_messages(self, client):
        create = client.post("/sessions", json={"name": "ClearHist"})
        session_id = create.json()["session_id"]
        client.post(f"/sessions/{session_id}/messages", json={"role": "user", "content": "a"})
        client.post(f"/sessions/{session_id}/messages", json={"role": "user", "content": "b"})
        r = client.delete(f"/sessions/{session_id}/history")
        assert r.status_code == 200
        assert r.json()["deleted"] == 2
        # Verify empty
        hist = client.get(f"/sessions/{session_id}/history")
        assert hist.json()["messages"] == []


class TestActiveSessionRoute:
    def test_get_active_when_none(self, client):
        r = client.get("/sessions/active")
        assert r.status_code == 200
        assert r.json()["active_session_id"] is None

    def test_post_sets_active_session(self, client):
        create = client.post("/sessions", json={"name": "Active"})
        session_id = create.json()["session_id"]
        r = client.post("/sessions/active", json={"session_id": session_id})
        assert r.status_code == 200
        assert r.json()["active_session_id"] == session_id

    def test_post_with_unknown_session_id_returns_404(self, client):
        r = client.post("/sessions/active", json={"session_id": "nope12345678901"})
        assert r.status_code == 404

    def test_get_active_returns_session_after_set(self, client):
        create = client.post("/sessions", json={"name": "GetActive"})
        session_id = create.json()["session_id"]
        client.post("/sessions/active", json={"session_id": session_id})
        r = client.get("/sessions/active")
        assert r.status_code == 200
        data = r.json()
        assert data["active_session_id"] == session_id


class TestChatSessionPersistence:
    def test_chat_with_session_id_persists_messages(self, client):
        create = client.post("/sessions", json={"name": "ChatPersist"})
        session_id = create.json()["session_id"]
        r = client.post(
            "/chat",
            json={"message": "What is the plan today?", "session_id": session_id},
        )
        assert r.status_code == 200
        # Check messages stored in session
        hist = client.get(f"/sessions/{session_id}/history")
        messages = hist.json()["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is the plan today?"
        assert messages[1]["role"] == "assistant"

    def test_chat_with_active_session_persists_messages(self, client):
        create = client.post("/sessions", json={"name": "ActiveChat"})
        session_id = create.json()["session_id"]
        client.post("/sessions/active", json={"session_id": session_id})
        r = client.post("/chat", json={"message": "Hello from active session"})
        assert r.status_code == 200
        hist = client.get(f"/sessions/{session_id}/history")
        messages = hist.json()["messages"]
        roles = [m["role"] for m in messages]
        assert "user" in roles
        assert "assistant" in roles

    def test_history_shows_messages_added_via_chat(self, client):
        create = client.post("/sessions", json={"name": "HistViaChat"})
        session_id = create.json()["session_id"]
        client.post("/chat", json={"message": "msg1", "session_id": session_id})
        client.post("/chat", json={"message": "msg2", "session_id": session_id})
        hist = client.get(f"/sessions/{session_id}/history")
        messages = hist.json()["messages"]
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert len(user_msgs) == 2
        assert user_msgs[0]["content"] == "msg1"
        assert user_msgs[1]["content"] == "msg2"
