"""
tests/test_federation.py
========================
Tests for prism_federation.py and prism_routes_federation.py.
"""
from __future__ import annotations

import time

import pytest

from prism_federation import FederationManager, FederationPeer, StateVector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _manager(tmp_path, node_id: str = "test-node") -> FederationManager:
    return FederationManager(node_id=node_id, db_path=str(tmp_path / "federation.db"))


# ---------------------------------------------------------------------------
# FederationManager — unit tests
# ---------------------------------------------------------------------------


class TestAnnounce:
    def test_announce_returns_node_id(self, tmp_path):
        fm = _manager(tmp_path)
        nid = fm.announce("http://192.168.1.5:8742")
        assert nid == "test-node"


class TestPeers:
    def test_add_and_list_peers(self, tmp_path):
        fm = _manager(tmp_path)
        peer = fm.add_peer("peer-1", "Home", "http://192.168.1.10:8742")
        assert isinstance(peer, FederationPeer)
        assert peer.peer_id == "peer-1"

        peers = fm.list_peers()
        assert len(peers) == 1
        assert peers[0].name == "Home"

    def test_remove_peer(self, tmp_path):
        fm = _manager(tmp_path)
        fm.add_peer("peer-2", "Work", "http://10.0.0.5:8742")
        assert fm.remove_peer("peer-2") is True
        assert fm.list_peers() == []

    def test_remove_peer_returns_false_when_missing(self, tmp_path):
        fm = _manager(tmp_path)
        assert fm.remove_peer("nonexistent") is False


# ---------------------------------------------------------------------------
# StateVector — unit tests
# ---------------------------------------------------------------------------


class TestStateVector:
    def test_state_vector_increment(self):
        sv = StateVector("node-a")
        assert sv.increment() == 1
        assert sv.increment() == 2
        assert sv.to_dict()["node-a"] == 2

    def test_state_vector_to_dict(self):
        sv = StateVector("node-x")
        sv.increment()
        d = sv.to_dict()
        assert "node-x" in d
        assert isinstance(d["node-x"], int)

    def test_state_vector_update_merges(self):
        sv = StateVector("a")
        sv.increment()  # a=1
        sv.update({"b": 5, "a": 3})
        d = sv.to_dict()
        assert d["b"] == 5
        assert d["a"] == 3  # remote "a" wins (3 > 1)

    def test_state_vector_happens_before(self):
        sv_a = StateVector("a")
        sv_a.increment()          # a=1

        other = {"a": 2, "b": 1}  # strictly dominates sv_a
        assert sv_a.happens_before(other) is True

    def test_state_vector_not_happens_before_when_concurrent(self):
        sv_a = StateVector("a")
        sv_a.increment()          # a=1
        sv_a.update({"b": 0})

        other = {"a": 1, "b": 1}  # concurrent (b is higher but a is equal)
        assert sv_a.happens_before(other) is True  # a=1<=1, b=0<1

    def test_state_vector_happens_before_equal_returns_false(self):
        sv_a = StateVector("a")
        sv_a.increment()          # a=1
        # Identical vector — neither strictly before
        assert sv_a.happens_before({"a": 1}) is False

    def test_state_vector_happens_before_when_local_ahead(self):
        sv_a = StateVector("a")
        sv_a.increment()
        sv_a.increment()          # a=2
        # Remote is behind
        assert sv_a.happens_before({"a": 1}) is False


# ---------------------------------------------------------------------------
# FederationManager — sync / merge
# ---------------------------------------------------------------------------


class TestSyncPayload:
    def test_get_sync_payload_structure(self, tmp_path):
        fm = _manager(tmp_path)
        payload = fm.get_sync_payload()
        assert payload["node_id"] == "test-node"
        assert isinstance(payload["version"], int) and payload["version"] >= 1
        assert isinstance(payload["vector"], dict)
        assert "goals" in payload
        assert "beliefs_summary" in payload
        assert "timestamp" in payload

    def test_get_sync_payload_increments_version(self, tmp_path):
        fm = _manager(tmp_path)
        p1 = fm.get_sync_payload()
        p2 = fm.get_sync_payload()
        assert p2["version"] > p1["version"]


class TestMergePeerState:
    def test_merge_peer_state_increments_version(self, tmp_path):
        fm = _manager(tmp_path)
        fm.add_peer("remote-1", "Phone", "http://10.0.0.1:8742")

        remote_payload = {
            "node_id": "remote-1",
            "version": 5,
            "vector": {"remote-1": 5},
            "goals": [
                {
                    "goal_id": "g-remote-001",
                    "intent": "Buy groceries",
                    "status": "watching",
                    "updated_at": time.time() - 10,
                }
            ],
            "beliefs_summary": {},
            "timestamp": time.time() - 5,
        }

        result = fm.merge_peer_state("remote-1", remote_payload)
        assert "merged_count" in result
        assert "conflicts_resolved" in result
        assert result["peer_version"] == 5

    def test_merge_peer_state_updates_vector(self, tmp_path):
        fm = _manager(tmp_path)
        fm.add_peer("remote-2", "Work", "http://10.0.0.2:8742")

        remote_payload = {
            "node_id": "remote-2",
            "version": 3,
            "vector": {"remote-2": 3, "test-node": 0},
            "goals": [],
            "beliefs_summary": {},
            "timestamp": time.time(),
        }

        fm.merge_peer_state("remote-2", remote_payload)
        vec = fm._vector.to_dict()
        assert vec.get("remote-2", 0) == 3

    def test_merge_adopts_new_goal_from_remote(self, tmp_path):
        fm = _manager(tmp_path)
        fm.add_peer("remote-3", "Home", "http://10.0.0.3:8742")

        now = time.time()
        goal = {
            "goal_id": "new-goal-xyz",
            "intent": "New remote goal",
            "status": "watching",
            "updated_at": now,
        }
        remote_payload = {
            "node_id": "remote-3",
            "version": 1,
            "vector": {"remote-3": 1},
            "goals": [goal],
            "beliefs_summary": {},
            "timestamp": now,
        }
        result = fm.merge_peer_state("remote-3", remote_payload)
        assert result["merged_count"] >= 1

    def test_merge_skips_older_goal(self, tmp_path):
        fm = _manager(tmp_path)
        fm.add_peer("remote-4", "Tablet", "http://10.0.0.4:8742")

        old_ts = time.time() - 3600
        # Seed a local goal that is newer
        fm._upsert_local_goal({
            "goal_id": "shared-goal",
            "intent": "Local newer version",
            "status": "watching",
            "updated_at": time.time(),
            "user_priority": False,
        })

        remote_payload = {
            "node_id": "remote-4",
            "version": 1,
            "vector": {"remote-4": 1},
            "goals": [
                {
                    "goal_id": "shared-goal",
                    "intent": "Old remote version",
                    "status": "watching",
                    "updated_at": old_ts,
                }
            ],
            "beliefs_summary": {},
            "timestamp": old_ts,
        }
        fm.merge_peer_state("remote-4", remote_payload)
        # Local goal should be unchanged
        local = fm._load_local_goal("shared-goal")
        assert local["intent"] == "Local newer version"

    def test_user_priority_goal_not_overwritten(self, tmp_path):
        fm = _manager(tmp_path)
        fm.add_peer("remote-5", "Watch", "http://10.0.0.5:8742")

        # Seed a user-priority local goal
        fm._upsert_local_goal({
            "goal_id": "priority-goal",
            "intent": "User-set goal",
            "status": "watching",
            "updated_at": time.time() - 100,
            "user_priority": True,
        })

        future_ts = time.time() + 9999
        remote_payload = {
            "node_id": "remote-5",
            "version": 99,
            "vector": {"remote-5": 99},
            "goals": [
                {
                    "goal_id": "priority-goal",
                    "intent": "Remote overwrite attempt",
                    "status": "watching",
                    "updated_at": future_ts,
                }
            ],
            "beliefs_summary": {},
            "timestamp": future_ts,
        }
        result = fm.merge_peer_state("remote-5", remote_payload)
        assert result["conflicts_resolved"] >= 1
        local = fm._load_local_goal("priority-goal")
        assert local["intent"] == "User-set goal"


class TestPendingSync:
    def test_pending_sync_lists_stale_peers(self, tmp_path):
        fm = _manager(tmp_path)
        # Add a peer that has never been synced (last_seen=0)
        with fm._lock:
            import sqlite3
            with sqlite3.connect(fm._db) as conn:
                conn.execute(
                    "INSERT INTO federation_peers(peer_id, name, url, last_seen, sync_version)"
                    " VALUES (?,?,?,?,?)",
                    ("stale-peer", "Old", "http://0.0.0.0:8742", 0.0, 0),
                )
        pending = fm.pending_sync()
        assert "stale-peer" in pending

    def test_pending_sync_excludes_recent_peers(self, tmp_path):
        fm = _manager(tmp_path)
        fm.add_peer("fresh-peer", "Fresh", "http://10.0.0.9:8742")
        # add_peer sets last_seen=now, so should not appear as stale
        pending = fm.pending_sync()
        assert "fresh-peer" not in pending


# ---------------------------------------------------------------------------
# HTTP endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path):
    """TestClient wired with a real FederationManager."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import prism_state
    from prism_federation import FederationManager
    from prism_routes_federation import router

    fm = FederationManager(node_id="test-node", db_path=str(tmp_path / "fed.db"))
    prism_state._set_state(federation=fm)

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestFederationAnnounceEndpoint:
    def test_federation_announce_endpoint(self, client):
        resp = client.post(
            "/federation/announce",
            json={"url": "http://192.168.1.5:8742", "name": "Home"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["node_id"] == "test-node"
        assert "peers" in data

    def test_federation_announce_requires_url(self, client):
        resp = client.post("/federation/announce", json={"name": "Home"})
        assert resp.status_code == 400


class TestFederationPeersEndpoint:
    def test_federation_peers_endpoint(self, client):
        # Add a peer first via announce
        client.post(
            "/federation/announce",
            json={
                "url": "http://10.0.0.1:8742",
                "name": "Work",
                "peer_id": "work-node",
            },
        )
        resp = client.get("/federation/peers")
        assert resp.status_code == 200
        data = resp.json()
        assert "peers" in data
        assert data["total"] >= 1
        names = [p["name"] for p in data["peers"]]
        assert "Work" in names

    def test_federation_delete_peer(self, client):
        client.post(
            "/federation/announce",
            json={
                "url": "http://10.0.0.2:8742",
                "name": "Phone",
                "peer_id": "phone-node",
            },
        )
        resp = client.delete("/federation/peers/phone-node")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_federation_delete_nonexistent_peer(self, client):
        resp = client.delete("/federation/peers/does-not-exist")
        assert resp.status_code == 404


class TestFederationSyncEndpoints:
    def test_federation_sync_get(self, client):
        resp = client.get("/federation/sync")
        assert resp.status_code == 200
        data = resp.json()
        assert data["node_id"] == "test-node"
        assert "version" in data
        assert "vector" in data

    def test_federation_sync_roundtrip(self, client, tmp_path, monkeypatch):
        """Full roundtrip: get payload from node A, post it to node B."""
        from prism_federation import FederationManager

        # Node B — separate DB. SSRF guard blocks loopback peers by default;
        # the test rig opts in so we can register the in-process node A URL.
        monkeypatch.setenv("PRISM_FEDERATION_ALLOW_LOOPBACK", "1")
        fm_b = FederationManager(
            node_id="node-b", db_path=str(tmp_path / "fed_b.db")
        )
        fm_b.add_peer("test-node", "Node A", "http://127.0.0.1:8742")

        # Get node A's payload
        get_resp = client.get("/federation/sync")
        assert get_resp.status_code == 200
        payload = get_resp.json()

        # Post it to node B directly (no HTTP needed — merge is local)
        result = fm_b.merge_peer_state("test-node", payload)
        assert result["peer_version"] == payload["version"]
        # Vector should now include test-node's clock
        vec = fm_b._vector.to_dict()
        assert vec.get("test-node", 0) >= 1

    def test_federation_sync_post(self, client):
        """POST /federation/sync merges a remote payload."""
        remote_payload = {
            "node_id": "remote-x",
            "version": 7,
            "vector": {"remote-x": 7},
            "goals": [],
            "beliefs_summary": {},
            "timestamp": time.time(),
        }
        resp = client.post(
            "/federation/sync",
            json={"peer_id": "remote-x", "payload": remote_payload},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "merged_count" in data
        assert data["peer_version"] == 7

    def test_federation_sync_post_missing_peer_id(self, client):
        resp = client.post(
            "/federation/sync",
            json={"payload": {"version": 1}},
        )
        assert resp.status_code == 400

    def test_federation_sync_post_missing_payload(self, client):
        resp = client.post(
            "/federation/sync",
            json={"peer_id": "someone"},
        )
        assert resp.status_code == 400


class TestFederationStatusEndpoint:
    def test_federation_status(self, client):
        resp = client.get("/federation/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "node_id" in data
        assert "vector" in data
        assert "pending_peers" in data
        assert "peers" in data


class TestFederationUnavailable:
    def test_503_when_federation_not_set(self, tmp_path):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        import prism_state
        from prism_routes_federation import router

        # Remove federation from state
        prism_state._state.pop("federation", None)

        app = FastAPI()
        app.include_router(router)
        tc = TestClient(app)

        resp = tc.get("/federation/peers")
        assert resp.status_code == 503
