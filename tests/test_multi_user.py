"""
tests/test_multi_user.py
========================
Tests for prism_multi_user.py (UserRegistry, HouseholdBus) and
prism_routes_users.py (FastAPI endpoints).
"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from prism_multi_user import HouseholdBus, UserProfile, UserRegistry
from prism_organ_bus import OrganSignal
from prism_state import _set_state

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry(tmp_path):
    """Fresh UserRegistry backed by a tmp directory."""
    return UserRegistry(base_dir=str(tmp_path / "users"))


@pytest.fixture()
def client(tmp_path):
    """
    TestClient wired with a real UserRegistry and, optionally, HouseholdBus.
    Imports prism_asgi which includes all routers (including prism_routes_users).
    """
    reg = UserRegistry(base_dir=str(tmp_path / "users"))
    bus = HouseholdBus(registry=reg)
    _set_state(user_registry=reg, household_bus=bus)

    from prism_asgi import app  # noqa: PLC0415

    return TestClient(app)


# ---------------------------------------------------------------------------
# UserRegistry unit tests
# ---------------------------------------------------------------------------


class TestRegisterAndListUsers:
    def test_register_and_list_users(self, registry: UserRegistry):
        registry.register("alice", "Alice Doe")
        registry.register("bob", "Bob Smith", role="admin")
        users = registry.list_users()
        assert len(users) == 2
        ids = {u.user_id for u in users}
        assert ids == {"alice", "bob"}

    def test_registered_profile_fields(self, registry: UserRegistry):
        p = registry.register("charlie", "Charlie Brown", role="guest")
        assert isinstance(p, UserProfile)
        assert p.user_id == "charlie"
        assert p.name == "Charlie Brown"
        assert p.role == "guest"
        assert "charlie" in p.db_path
        assert "charlie" in p.soul_path

    def test_get_returns_none_for_unknown(self, registry: UserRegistry):
        assert registry.get("nobody") is None

    def test_get_returns_profile_after_register(self, registry: UserRegistry):
        registry.register("dave", "Dave Jones")
        p = registry.get("dave")
        assert p is not None
        assert p.name == "Dave Jones"

    def test_remove_returns_true_for_known(self, registry: UserRegistry):
        registry.register("eve", "Eve Adams")
        assert registry.remove("eve") is True
        assert registry.get("eve") is None

    def test_remove_returns_false_for_unknown(self, registry: UserRegistry):
        assert registry.remove("ghost") is False

    def test_list_empty_initially(self, registry: UserRegistry):
        assert registry.list_users() == []


class TestDuplicateUserRaises:
    def test_duplicate_user_raises(self, registry: UserRegistry):
        registry.register("frank", "Frank Castle")
        with pytest.raises(ValueError, match="already registered"):
            registry.register("frank", "Frank Castle 2")

    def test_invalid_role_raises(self, registry: UserRegistry):
        with pytest.raises(ValueError, match="role must be"):
            registry.register("grace", "Grace Hopper", role="superuser")


class TestGetUserMemoryIsolated:
    def test_get_user_memory_isolated(self, registry: UserRegistry):
        """Different users receive distinct PrismMemoryGraph instances backed
        by different DB paths."""
        registry.register("u1", "User One")
        registry.register("u2", "User Two")

        mem1 = registry.get_memory("u1")
        mem2 = registry.get_memory("u2")

        assert mem1 is not mem2

        # DB paths must differ
        assert str(mem1._cold._conn.execute("PRAGMA database_list").fetchone()) != str(
            mem2._cold._conn.execute("PRAGMA database_list").fetchone()
        )

    def test_same_user_returns_same_instance(self, registry: UserRegistry):
        """get_memory returns the same object on subsequent calls."""
        registry.register("u3", "User Three")
        mem_a = registry.get_memory("u3")
        mem_b = registry.get_memory("u3")
        assert mem_a is mem_b

    def test_unknown_user_raises_key_error(self, registry: UserRegistry):
        with pytest.raises(KeyError):
            registry.get_memory("nobody")

    def test_memory_paths_contain_user_id(self, registry: UserRegistry):
        registry.register("heidi", "Heidi K")
        p = registry.get("heidi")
        assert "heidi" in p.db_path


class TestHouseholdBusBroadcast:
    def test_broadcast_empty_registry(self, registry: UserRegistry):
        bus = HouseholdBus(registry=registry)
        signal = OrganSignal(source="test", signal_type="ping", payload={"x": 1})
        results = bus.broadcast(signal)
        assert results == {}

    def test_broadcast_reaches_all_users(self, registry: UserRegistry):
        registry.register("ivan", "Ivan D")
        registry.register("judy", "Judy G")
        bus = HouseholdBus(registry=registry)
        signal = OrganSignal(source="test", signal_type="ping", payload={"x": 1})
        results = bus.broadcast(signal)
        assert set(results.keys()) == {"ivan", "judy"}

    def test_route_to_single_user(self, registry: UserRegistry):
        registry.register("karl", "Karl M")
        bus = HouseholdBus(registry=registry)
        signal = OrganSignal(source="test", signal_type="alert", payload={"msg": "hi"})
        result = bus.route_to("karl", signal)
        assert result is not None

    def test_route_to_unknown_raises(self, registry: UserRegistry):
        bus = HouseholdBus(registry=registry)
        signal = OrganSignal(source="test", signal_type="alert", payload={})
        with pytest.raises(KeyError):
            bus.route_to("nobody", signal)

    def test_signal_history_recorded(self, registry: UserRegistry):
        registry.register("lena", "Lena S")
        bus = HouseholdBus(registry=registry)
        signal = OrganSignal(source="test", signal_type="event", payload={"k": "v"})
        bus.broadcast(signal)
        history = bus.signal_history(n=5)
        assert len(history) == 1
        assert history[0]["signal_type"] == "event"

    def test_active_users_within_24h(self, registry: UserRegistry, tmp_path):
        registry.register("mia", "Mia T")
        bus = HouseholdBus(registry=registry)
        bus.broadcast(
            OrganSignal(source="test", signal_type="ping", payload={})
        )
        active = bus.active_users()
        assert "mia" in active

    def test_inactive_user_not_in_active_list(self, registry: UserRegistry, tmp_path):
        registry.register("noah", "Noah A")
        # Manually set last_active to 25 hours ago
        p = registry.get("noah")
        p.last_active = time.time() - (25 * 3600)
        bus = HouseholdBus(registry=registry)
        active = bus.active_users()
        assert "noah" not in active


# ---------------------------------------------------------------------------
# FastAPI endpoint tests
# ---------------------------------------------------------------------------


class TestUsersEndpointEmpty:
    def test_users_endpoint_empty(self, client: TestClient, tmp_path):
        # client fixture resets state for each test via its own registry
        # We need a fresh client with empty registry
        from prism_asgi import app  # noqa: PLC0415

        reg = UserRegistry(base_dir=str(tmp_path / "fresh_users"))
        _set_state(user_registry=reg, household_bus=None)
        c = TestClient(app)
        resp = c.get("/users")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["users"] == []


class TestRegisterUserEndpoint:
    def test_register_user_endpoint(self, client: TestClient):
        resp = client.post(
            "/users",
            json={"user_id": "oscar", "name": "Oscar W", "role": "member"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["user"]["user_id"] == "oscar"
        assert data["user"]["name"] == "Oscar W"

    def test_register_missing_name_returns_400(self, client: TestClient):
        resp = client.post("/users", json={"user_id": "pam"})
        assert resp.status_code == 400

    def test_register_missing_user_id_returns_400(self, client: TestClient):
        resp = client.post("/users", json={"name": "Pam B"})
        assert resp.status_code == 400

    def test_register_duplicate_returns_409(self, client: TestClient):
        client.post(
            "/users",
            json={"user_id": "quinn", "name": "Quinn A", "role": "guest"},
        )
        resp = client.post(
            "/users",
            json={"user_id": "quinn", "name": "Quinn B"},
        )
        assert resp.status_code == 409

    def test_list_after_register(self, client: TestClient):
        client.post(
            "/users",
            json={"user_id": "ruth", "name": "Ruth G", "role": "admin"},
        )
        resp = client.get("/users")
        assert resp.status_code == 200
        ids = {u["user_id"] for u in resp.json()["users"]}
        assert "ruth" in ids

    def test_delete_user(self, client: TestClient):
        client.post("/users", json={"user_id": "sam", "name": "Sam P"})
        resp = client.delete("/users/sam")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_delete_unknown_user_returns_404(self, client: TestClient):
        resp = client.delete("/users/ghost")
        assert resp.status_code == 404


class TestSwitchUserEndpoint:
    def test_switch_user_endpoint(self, client: TestClient):
        client.post(
            "/users",
            json={"user_id": "tina", "name": "Tina F", "role": "member"},
        )
        resp = client.post("/users/tina/activate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["active_user_id"] == "tina"
        assert data["profile"]["user_id"] == "tina"

    def test_activate_unknown_user_returns_404(self, client: TestClient):
        resp = client.post("/users/nobody/activate")
        assert resp.status_code == 404

    def test_activate_updates_state(self, client: TestClient):
        client.post(
            "/users",
            json={"user_id": "uma", "name": "Uma T"},
        )
        client.post("/users/uma/activate")
        from prism_state import _state  # noqa: PLC0415

        assert _state.get("active_user_id") == "uma"


class TestHouseholdSignalsEndpoint:
    def test_signals_empty_initially(self, client: TestClient):
        resp = client.get("/household/signals")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["signals"] == []

    def test_broadcast_endpoint(self, client: TestClient):
        resp = client.post(
            "/household/broadcast",
            json={"signal_type": "test_event", "payload": {"key": "value"}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["signal_type"] == "test_event"
        assert "signal_id" in data

    def test_broadcast_missing_signal_type_returns_400(self, client: TestClient):
        resp = client.post(
            "/household/broadcast",
            json={"payload": {"key": "value"}},
        )
        assert resp.status_code == 400

    def test_signals_appear_after_broadcast(self, client: TestClient):
        client.post(
            "/household/broadcast",
            json={"signal_type": "health_check", "payload": {}},
        )
        resp = client.get("/household/signals")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        types = [s["signal_type"] for s in data["signals"]]
        assert "health_check" in types

    def test_no_registry_returns_503(self, tmp_path):
        from prism_asgi import app  # noqa: PLC0415

        _set_state(user_registry=None, household_bus=None)
        c = TestClient(app)
        resp = c.get("/household/signals")
        # signals endpoint falls back to 503 only when reg is None
        # (our impl returns empty list when bus=None but reg is present,
        #  and 503 when reg is None)
        assert resp.status_code == 503
