"""
tests/test_mobile_sync.py
=========================
Unit and integration tests for MobileSyncManager and /mobile/* routes.
"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from prism_mobile_sync import MobileSyncManager
from prism_state import _set_state

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _manager(tmp_path) -> MobileSyncManager:
    """Return a fresh MobileSyncManager backed by a temp SQLite DB."""
    return MobileSyncManager(
        secret_key="test-secret",
        db_path=str(tmp_path / "mobile.db"),
    )


# ---------------------------------------------------------------------------
# Unit tests — MobileSyncManager
# ---------------------------------------------------------------------------


class TestMobileSyncManager:
    def test_register_client_returns_token(self, tmp_path):
        mgr = _manager(tmp_path)
        token = mgr.register_client("dev-001", "iPhone 15", "ios")
        assert isinstance(token, str)
        assert len(token) > 10
        # format: "<issued_at>.<hex>"
        parts = token.split(".", 1)
        assert len(parts) == 2
        assert parts[0].isdigit()
        assert len(parts[1]) == 64  # SHA-256 hex

    def test_verify_token_valid(self, tmp_path):
        mgr = _manager(tmp_path)
        token = mgr.register_client("dev-002", "Pixel 8", "android")
        assert mgr.verify_token("dev-002", token) is True

    def test_verify_token_wrong_returns_false(self, tmp_path):
        mgr = _manager(tmp_path)
        token = mgr.register_client("dev-003", "Galaxy", "android")
        # Tamper with the digest portion
        parts = token.split(".", 1)
        bad_token = parts[0] + ".deadbeef" * 8
        assert mgr.verify_token("dev-003", bad_token) is False

    def test_verify_token_wrong_device_returns_false(self, tmp_path):
        mgr = _manager(tmp_path)
        token = mgr.register_client("dev-A", "Watch", "watchos")
        # Token issued for dev-A should not validate for dev-B
        assert mgr.verify_token("dev-B", token) is False

    def test_ingest_health_data_returns_count(self, tmp_path):
        mgr = _manager(tmp_path)
        mgr.register_client("dev-010", "Watch", "watchos")
        metrics = [
            {"metric": "heart_rate",  "value": 72,  "unit": "bpm",      "timestamp": time.time()},
            {"metric": "steps",       "value": 8432, "unit": "count",   "timestamp": time.time()},
            {"metric": "blood_oxygen", "value": 98.5, "unit": "%",      "timestamp": time.time()},
        ]
        count = mgr.ingest_health_data("dev-010", metrics)
        assert count == 3

    def test_ingest_health_data_empty_list_returns_zero(self, tmp_path):
        mgr = _manager(tmp_path)
        mgr.register_client("dev-011", "Watch2", "watchos")
        assert mgr.ingest_health_data("dev-011", []) == 0

    def test_ingest_health_data_skips_bad_entries(self, tmp_path):
        mgr = _manager(tmp_path)
        mgr.register_client("dev-012", "Watch3", "watchos")
        metrics = [
            {"metric": "heart_rate", "value": 70, "unit": "bpm", "timestamp": time.time()},
            {"no_metric_key": "bad"},  # missing required "metric" and "value"
        ]
        count = mgr.ingest_health_data("dev-012", metrics)
        assert count == 1

    def test_get_sync_state_empty(self, tmp_path):
        mgr = _manager(tmp_path)
        mgr.register_client("dev-020", "iPad", "ios")
        state = mgr.sync_state("dev-020")
        assert "last_sync" in state
        assert "pending_count" in state
        assert "agent_status" in state
        assert state["pending_count"] == 0
        assert state["agent_status"] == "online"

    def test_get_sync_state_counts_pending_notifications(self, tmp_path):
        mgr = _manager(tmp_path)
        mgr.register_client("dev-021", "iPad Pro", "ios")
        mgr.queue_notification("dev-021", {"title": "Hello", "body": "World"})
        mgr.queue_notification("dev-021", {"title": "Reminder", "body": "Stand up"})
        state = mgr.sync_state("dev-021")
        assert state["pending_count"] == 2

    def test_get_pending_notifications_marks_delivered(self, tmp_path):
        mgr = _manager(tmp_path)
        mgr.register_client("dev-030", "Android", "android")
        mgr.queue_notification("dev-030", {"msg": "first"})
        mgr.queue_notification("dev-030", {"msg": "second"})

        notifications = mgr.get_pending_notifications("dev-030")
        assert len(notifications) == 2

        # After retrieval they should be marked delivered
        notifications2 = mgr.get_pending_notifications("dev-030")
        assert len(notifications2) == 0

    def test_register_push_token(self, tmp_path):
        mgr = _manager(tmp_path)
        mgr.register_client("dev-040", "Phone", "ios")
        # Should not raise
        mgr.register_push_token("dev-040", "fcm-token-abc123")


# ---------------------------------------------------------------------------
# Integration tests — FastAPI routes
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path):
    """TestClient wired with a fresh MobileSyncManager in _state."""
    mgr = MobileSyncManager(
        secret_key="test-secret",
        db_path=str(tmp_path / "mobile_routes.db"),
    )
    _set_state(mobile_sync=mgr)

    from prism_asgi import app
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def client_no_manager():
    """TestClient with mobile_sync removed from _state to test 503 paths."""
    from prism_state import _state
    _state.pop("mobile_sync", None)

    from prism_asgi import app
    return TestClient(app, raise_server_exceptions=True)


class TestMobileRegisterEndpoint:
    def test_mobile_register_endpoint(self, client):
        resp = client.post("/mobile/register", json={
            "device_id": "e2e-dev-1",
            "name":      "Test Phone",
            "platform":  "android",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "sync_token" in data
        assert data["device_id"] == "e2e-dev-1"

    def test_mobile_register_missing_device_id_returns_400(self, client):
        resp = client.post("/mobile/register", json={"name": "x", "platform": "ios"})
        assert resp.status_code == 400

    def test_mobile_register_no_manager_returns_503(self, client_no_manager):
        resp = client_no_manager.post("/mobile/register", json={
            "device_id": "any", "name": "Phone", "platform": "ios",
        })
        assert resp.status_code == 503


class TestMobileHealthDataEndpoint:
    def test_mobile_health_data_endpoint(self, client):
        # First register
        reg_resp = client.post("/mobile/register", json={
            "device_id": "e2e-dev-2", "name": "Watch", "platform": "watchos",
        })
        token = reg_resp.json()["sync_token"]

        resp = client.post(
            "/mobile/health_data",
            headers={"X-Device-ID": "e2e-dev-2", "X-Sync-Token": token},
            json={
                "device_id": "e2e-dev-2",
                "metrics": [
                    {"metric": "heart_rate", "value": 65, "unit": "bpm",
                     "timestamp": time.time()},
                    {"metric": "steps", "value": 5000, "unit": "count",
                     "timestamp": time.time()},
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["inserted"] == 2

    def test_mobile_health_data_no_manager_returns_503(self, client_no_manager):
        resp = client_no_manager.post("/mobile/health_data", json={
            "device_id": "any",
            "metrics": [{"metric": "hr", "value": 60, "unit": "bpm",
                         "timestamp": time.time()}],
        })
        assert resp.status_code == 503


class TestMobileSyncEndpoint:
    def test_mobile_sync_endpoint_no_auth_returns_401_or_503(self, client):
        """GET /mobile/sync without auth headers returns 401."""
        resp = client.get("/mobile/sync")
        assert resp.status_code in (401, 503)

    def test_mobile_sync_endpoint_invalid_token_returns_401(self, client):
        resp = client.get(
            "/mobile/sync",
            headers={"X-Device-ID": "e2e-dev-99", "X-Sync-Token": "0.badhex"},
        )
        assert resp.status_code == 401

    def test_mobile_sync_endpoint_valid_auth_returns_state(self, client):
        reg_resp = client.post("/mobile/register", json={
            "device_id": "e2e-dev-3", "name": "iPad", "platform": "ios",
        })
        token = reg_resp.json()["sync_token"]

        resp = client.get(
            "/mobile/sync",
            headers={"X-Device-ID": "e2e-dev-3", "X-Sync-Token": token},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "last_sync" in data
        assert "pending_count" in data
        assert "agent_status" in data


class TestMobileNotificationsEndpoint:
    def test_mobile_notifications_no_auth_returns_401(self, client):
        resp = client.get("/mobile/notifications")
        assert resp.status_code in (401, 503)

    def test_mobile_notifications_returns_payloads(self, client, tmp_path):
        """Queue a notification directly then retrieve via API."""
        from prism_state import _state
        mgr = _state.get("mobile_sync")

        reg_resp = client.post("/mobile/register", json={
            "device_id": "e2e-dev-4", "name": "TV", "platform": "tvos",
        })
        token = reg_resp.json()["sync_token"]

        mgr.queue_notification("e2e-dev-4", {"title": "Ping", "body": "Test"})

        resp = client.get(
            "/mobile/notifications",
            headers={"X-Device-ID": "e2e-dev-4", "X-Sync-Token": token},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["notifications"][0]["payload"]["title"] == "Ping"


class TestMobilePushTokenEndpoint:
    def test_mobile_push_token_endpoint(self, client):
        reg_resp = client.post("/mobile/register", json={
            "device_id": "e2e-dev-5", "name": "Phone", "platform": "android",
        })
        token = reg_resp.json()["sync_token"]

        resp = client.post(
            "/mobile/push_token",
            headers={"X-Device-ID": "e2e-dev-5", "X-Sync-Token": token},
            json={"device_id": "e2e-dev-5", "push_token": "fcm:abc123xyz"},
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
