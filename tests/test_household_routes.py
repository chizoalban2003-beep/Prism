"""Tests for household dashboard, analytics, and per-user identity routes."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prism_state import _set_state, _state

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_profile(user_id="alice", name="Alice", role="admin", last_active=None):
    profile = MagicMock()
    profile.user_id = user_id
    profile.name = name
    profile.role = role
    profile.last_active = last_active if last_active is not None else time.time()
    profile.created_at = time.time() - 3600
    profile.to_dict.return_value = {
        "user_id": user_id,
        "name": name,
        "role": role,
        "last_active": profile.last_active,
        "created_at": profile.created_at,
        "db_path": f"/tmp/{user_id}/memory_graph.db",
        "soul_path": f"/tmp/{user_id}/soul.db",
    }
    return profile


def _make_registry(profiles=None):
    reg = MagicMock()
    _profiles = profiles if profiles is not None else [_make_profile()]
    reg.list_users.return_value = _profiles
    reg.get.side_effect = lambda uid: next(
        (p for p in _profiles if p.user_id == uid), None
    )
    reg.register.side_effect = lambda user_id, name, role="member": _make_profile(
        user_id, name, role
    )
    reg.remove.return_value = True
    reg.get_soul.return_value = None
    return reg


def _make_bus(signals=None):
    bus = MagicMock()
    bus.signal_history.return_value = signals if signals is not None else []
    return bus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_state():
    _state.pop("user_registry", None)
    _state.pop("household_bus", None)
    _set_state(agent=None)
    yield
    _state.pop("user_registry", None)
    _state.pop("household_bus", None)
    _set_state(agent=None)


@pytest.fixture
def client():
    from prism_asgi import app

    return TestClient(app)


@pytest.fixture
def reg_client(client):
    """Client with a pre-wired UserRegistry."""
    reg = _make_registry()
    _state["user_registry"] = reg
    return client, reg


# ---------------------------------------------------------------------------
# GET /household/dashboard
# ---------------------------------------------------------------------------


class TestHouseholdDashboard:
    def test_returns_200(self, client):
        r = client.get("/household/dashboard")
        assert r.status_code == 200

    def test_content_type_html(self, client):
        r = client.get("/household/dashboard")
        assert "text/html" in r.headers["content-type"]

    def test_contains_prism_household(self, client):
        r = client.get("/household/dashboard")
        assert "PRISM Household" in r.text

    def test_contains_analytics_endpoint_ref(self, client):
        r = client.get("/household/dashboard")
        assert "household/analytics" in r.text

    def test_contains_section_titles(self, client):
        r = client.get("/household/dashboard")
        assert "Registered Users" in r.text
        assert "Recent Signals" in r.text


# ---------------------------------------------------------------------------
# GET /household/analytics
# ---------------------------------------------------------------------------


class TestHouseholdAnalytics:
    def test_no_registry_503(self, client):
        r = client.get("/household/analytics")
        assert r.status_code == 503

    def test_returns_200_with_registry(self, reg_client):
        client, _ = reg_client
        r = client.get("/household/analytics")
        assert r.status_code == 200

    def test_required_keys_present(self, reg_client):
        client, _ = reg_client
        d = client.get("/household/analytics").json()
        for key in (
            "total_users",
            "active_today",
            "active_this_week",
            "by_role",
            "recent_signals",
            "users",
        ):
            assert key in d, f"missing key: {key}"

    def test_total_users_count(self, client):
        profiles = [
            _make_profile("u1", "Alice", "admin"),
            _make_profile("u2", "Bob", "member"),
        ]
        _state["user_registry"] = _make_registry(profiles)
        d = client.get("/household/analytics").json()
        assert d["total_users"] == 2

    def test_active_today_counts_recent(self, client):
        now = time.time()
        profiles = [
            _make_profile("u1", "Alice", "admin", last_active=now - 100),  # active
            _make_profile("u2", "Bob", "member", last_active=now - 200_000),  # old
        ]
        _state["user_registry"] = _make_registry(profiles)
        d = client.get("/household/analytics").json()
        assert d["active_today"] == 1

    def test_by_role_breakdown(self, client):
        profiles = [
            _make_profile("u1", "Alice", "admin"),
            _make_profile("u2", "Bob", "member"),
            _make_profile("u3", "Carol", "guest"),
        ]
        _state["user_registry"] = _make_registry(profiles)
        d = client.get("/household/analytics").json()
        assert d["by_role"]["admin"] == 1
        assert d["by_role"]["member"] == 1
        assert d["by_role"]["guest"] == 1

    def test_users_list_has_expected_fields(self, reg_client):
        client, _ = reg_client
        d = client.get("/household/analytics").json()
        assert len(d["users"]) >= 1
        u = d["users"][0]
        for field in ("user_id", "name", "role", "last_active", "is_active_today"):
            assert field in u

    def test_recent_signals_from_bus(self, client):
        _state["user_registry"] = _make_registry()
        sig = {
            "signal_id": "s1",
            "source": "test",
            "signal_type": "ping",
            "ts": time.time(),
            "results": {},
        }
        _state["household_bus"] = _make_bus([sig])
        d = client.get("/household/analytics").json()
        assert len(d["recent_signals"]) == 1
        assert d["recent_signals"][0]["signal_type"] == "ping"

    def test_empty_signals_without_bus(self, reg_client):
        client, _ = reg_client
        d = client.get("/household/analytics").json()
        assert d["recent_signals"] == []


# ---------------------------------------------------------------------------
# GET /users
# ---------------------------------------------------------------------------


class TestListUsers:
    def test_no_registry_503(self, client):
        r = client.get("/users")
        assert r.status_code == 503

    def test_returns_user_list(self, reg_client):
        client, _ = reg_client
        d = client.get("/users").json()
        assert "users" in d
        assert "total" in d
        assert d["total"] == 1


# ---------------------------------------------------------------------------
# POST /users
# ---------------------------------------------------------------------------


class TestCreateUser:
    def test_creates_user(self, reg_client):
        client, reg = reg_client
        r = client.post("/users", json={"user_id": "bob", "name": "Bob", "role": "member"})
        assert r.status_code == 200
        assert r.json()["ok"] is True
        reg.register.assert_called_once()

    def test_missing_fields_400(self, reg_client):
        client, _ = reg_client
        r = client.post("/users", json={"user_id": "x"})
        assert r.status_code == 400

    def test_no_registry_503(self, client):
        r = client.post("/users", json={"user_id": "x", "name": "X"})
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# DELETE /users/{user_id}
# ---------------------------------------------------------------------------


class TestDeleteUser:
    def test_deletes_user(self, reg_client):
        client, reg = reg_client
        r = client.delete("/users/alice")
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_not_found_404(self, reg_client):
        client, reg = reg_client
        reg.remove.return_value = False
        r = client.delete("/users/nobody")
        assert r.status_code == 404

    def test_no_registry_503(self, client):
        r = client.delete("/users/alice")
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# GET /users/{user_id}/identity
# ---------------------------------------------------------------------------


class TestUserIdentity:
    def test_unknown_user_404(self, reg_client):
        client, reg = reg_client
        reg.get.side_effect = lambda uid: None
        r = client.get("/users/nobody/identity")
        assert r.status_code == 404

    def test_no_registry_503(self, client):
        r = client.get("/users/alice/identity")
        assert r.status_code == 503

    def test_known_user_200(self, reg_client):
        client, _ = reg_client
        r = client.get("/users/alice/identity")
        assert r.status_code == 200

    def test_response_has_required_keys(self, reg_client):
        client, _ = reg_client
        d = client.get("/users/alice/identity").json()
        for key in ("user_id", "name", "role", "soul_beliefs", "phase", "last_active"):
            assert key in d, f"missing key: {key}"

    def test_user_id_matches(self, reg_client):
        client, _ = reg_client
        d = client.get("/users/alice/identity").json()
        assert d["user_id"] == "alice"

    def test_soul_beliefs_populated_when_soul_available(self, client):
        belief = MagicMock()
        belief.text = "I value honesty"
        belief.belief_type = "value"
        belief.confidence = 0.9
        belief.source = "stated"

        soul_mock = MagicMock()
        soul_mock.list_beliefs.return_value = [belief]

        reg = _make_registry()
        reg.get_soul.return_value = soul_mock
        _state["user_registry"] = reg

        d = client.get("/users/alice/identity").json()
        assert len(d["soul_beliefs"]) == 1
        assert d["soul_beliefs"][0]["text"] == "I value honesty"

    def test_soul_beliefs_empty_when_no_soul(self, reg_client):
        client, _ = reg_client
        d = client.get("/users/alice/identity").json()
        assert d["soul_beliefs"] == []
