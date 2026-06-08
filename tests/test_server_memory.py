"""
test_server_memory.py
=====================
Tests for memory, proactive/pending, and smarthome/status routes.
Uses FastAPI TestClient.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prism_asgi import app
from prism_state import _set_state


def _make_agent() -> MagicMock:
    agent = MagicMock()
    agent.status.return_value = {"ok": True, "role": "test"}
    agent.morning_briefing.return_value = MagicMock()
    agent.ask.return_value = MagicMock(
        task="t", method="m", success=True, elapsed_ms=1.0, output={}
    )
    agent.reflect.return_value = {}
    agent.identity.return_value = {}
    agent.identity_domains.return_value = []
    agent.recent_artifacts.return_value = []
    agent._assistant = MagicMock()
    agent._assistant.history.return_value = []
    agent._hub = MagicMock()
    agent._hub.list_devices.return_value = []
    agent._profile = MagicMock()
    agent._profile.name = "TestUser"
    return agent


@pytest.fixture()
def client():
    agent = _make_agent()
    _set_state(agent=agent)
    return TestClient(app, raise_server_exceptions=False)


def test_memory_ingest_200(client):
    """POST /memory/ingest with a mock memory agent returns 200 with entry_id."""
    from prism_state import _state
    mock_mem = MagicMock()
    mock_mem.ingest.return_value = "abc123"
    _state["agent"]._memory = mock_mem

    r = client.post("/memory/ingest",
                    json={"content": "test content", "source": "note", "title": "Test"})
    assert r.status_code == 200
    assert r.json().get("entry_id") == "abc123"


def test_memory_search_200(client):
    """GET /memory/search?q=test returns 200 with a results list."""
    from prism_memory import MemoryEntry, MemoryResult
    from prism_state import _state

    mock_entry = MemoryEntry(
        entry_id="e1", content="test content", source="note", title="Test Note"
    )
    mock_result = MemoryResult(entry=mock_entry, score=0.9, excerpt="test content")
    mock_mem = MagicMock()
    mock_mem.search.return_value = [mock_result]
    _state["agent"]._memory = mock_mem

    r = client.get("/memory/search?q=test")
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert isinstance(data["results"], list)


def test_proactive_pending_200(client):
    """GET /proactive/pending returns 200 with an events list."""
    from prism_state import _state
    _state["agent"]._proactive_buffer = []

    r = client.get("/proactive/pending")
    assert r.status_code == 200
    data = r.json()
    assert "events" in data
    assert isinstance(data["events"], list)
