"""
Tests for prism_routes_chain — /chain/*, /organs/*, /organ_bus/* endpoints.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import prism_state
from prism_routes_chain import router


def _app_no_agent():
    app = FastAPI()
    app.include_router(router)
    prism_state._state.clear()
    return TestClient(app)


def _app_with_agent():
    app = FastAPI()
    app.include_router(router)

    chain = MagicMock()
    chain.recent_chains.return_value = [{"chain_id": "c1", "summary": "did stuff"}]
    chain.MAX_STEPS = 12
    chain._db = "/tmp/chains.db"

    organ_loader = MagicMock()
    organ_loader.known_intents.return_value = {"email_send": "Send an email"}
    organ_loader.list_organs.return_value = ["email_organ"]

    organ_bus = MagicMock()
    organ_bus.history.return_value = [{"signal": "test"}]
    sub = MagicMock()
    sub.organ_name = "test_organ"
    sub.signal_types = ["info"]
    sub.vocabulary = "test vocab"[:120]
    organ_bus._subscribers = [sub]
    organ_bus._lock = __import__("threading").Lock()

    agent = MagicMock()
    agent._chain = chain
    agent._organ_loader = organ_loader
    agent._organ_bus = organ_bus

    prism_state._state.clear()
    prism_state._state["agent"] = agent
    return TestClient(app), agent


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    prism_state._state.clear()


# ---------------------------------------------------------------------------
# /chain
# ---------------------------------------------------------------------------

class TestChainRecent:
    def test_chain_recent_no_agent(self):
        data = _app_no_agent().get("/chain/recent").json()
        assert data["chains"] == []

    def test_chain_recent_200(self):
        client, _ = _app_with_agent()
        r = client.get("/chain/recent")
        assert r.status_code == 200
        assert "chains" in r.json()

    def test_chain_recent_n_param(self):
        client, agent = _app_with_agent()
        client.get("/chain/recent?n=3")
        agent._chain.recent_chains.assert_called_with(n=3)


class TestChainStatus:
    def test_chain_status_no_agent(self):
        data = _app_no_agent().get("/chain/status").json()
        assert data == {"configured": False}

    def test_chain_status_200(self):
        client, _ = _app_with_agent()
        data = client.get("/chain/status").json()
        assert "max_steps" in data
        assert data["max_steps"] == 12

    def test_chain_status_has_db(self):
        client, _ = _app_with_agent()
        data = client.get("/chain/status").json()
        assert "db" in data


# /organs/intents lives in prism_routes_infra now (issue #28-45). Its
# tests moved to tests/test_organs_intents_route_shadow_issue_28.py.


# ---------------------------------------------------------------------------
# /organ_bus
# ---------------------------------------------------------------------------

class TestOrganBusHistory:
    def test_organ_bus_no_agent(self):
        data = _app_no_agent().get("/organ_bus/history").json()
        assert data["available"] is False

    def test_organ_bus_history_200(self):
        client, _ = _app_with_agent()
        data = client.get("/organ_bus/history").json()
        assert data["available"] is True
        assert isinstance(data["signals"], list)

    def test_organ_bus_n_param(self):
        client, agent = _app_with_agent()
        client.get("/organ_bus/history?n=5")
        agent._organ_bus.history.assert_called_with(n=5)


class TestOrganBusSubscribers:
    def test_subscribers_no_agent(self):
        data = _app_no_agent().get("/organ_bus/subscribers").json()
        assert data["available"] is False

    def test_subscribers_200(self):
        client, _ = _app_with_agent()
        data = client.get("/organ_bus/subscribers").json()
        assert data["available"] is True
        assert data["count"] == 1
        assert data["subscribers"][0]["organ"] == "test_organ"
