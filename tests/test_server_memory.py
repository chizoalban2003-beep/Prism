"""Tests for memory, proactive/pending, and smarthome/status routes — Gap Prompt 9b."""
from __future__ import annotations

import json
import time
import urllib.request
from unittest.mock import MagicMock

from kde_server import KDEServer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent() -> MagicMock:
    agent = MagicMock()
    agent.status.return_value = {"ok": True, "role": "test"}
    agent.morning_briefing.return_value = MagicMock(
        time="2026-01-01T09:00:00",
        plan=MagicMock(primary_focus="Test", activation=0.5, fulcrum=0.5,
                       tasks=[], warnings=[], rationale="ok"),
        wearable_summary="",
        device_status=[],
        priority_tasks=[],
        alerts=[],
        match_intelligence=None,
    )
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


def _start_server(port: int) -> KDEServer:
    server = KDEServer(agent=_make_agent(), port=port)
    server.start(blocking=False)
    time.sleep(0.15)
    return server


def _get(url: str) -> tuple[int, dict]:
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.status, json.loads(resp.read())


def _post(url: str, data: dict) -> tuple[int, dict]:
    body = json.dumps(data).encode()
    req  = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, json.loads(resp.read())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_memory_ingest_200():
    """POST /memory/ingest with a mock memory agent returns 200 with entry_id."""
    port   = 19800
    server = _start_server(port)
    try:
        # Attach a mock prism_agent with _memory
        mock_agent          = MagicMock()
        mock_agent._memory  = MagicMock()
        mock_agent._memory.ingest.return_value = "abc123"
        server._server.prism_agent = mock_agent

        status, data = _post(
            f"http://127.0.0.1:{port}/memory/ingest",
            {"content": "test content", "source": "note", "title": "Test"},
        )
        assert status == 200
        assert "entry_id" in data
        assert data["entry_id"] == "abc123"
    finally:
        server.stop()


def test_memory_search_200():
    """GET /memory/search?q=test returns 200 with a results list."""
    port   = 19801
    server = _start_server(port)
    try:
        from prism_memory import MemoryEntry, MemoryResult
        mock_entry  = MemoryEntry(
            entry_id="e1", content="test content", source="note",
            title="Test Note",
        )
        mock_result = MemoryResult(entry=mock_entry, score=0.9,
                                   excerpt="test content")

        mock_agent         = MagicMock()
        mock_agent._memory = MagicMock()
        mock_agent._memory.search.return_value = [mock_result]
        server._server.prism_agent = mock_agent

        status, data = _get(f"http://127.0.0.1:{port}/memory/search?q=test")
        assert status == 200
        assert "results" in data
        assert isinstance(data["results"], list)
    finally:
        server.stop()


def test_proactive_pending_200():
    """GET /proactive/pending returns 200 with an events list."""
    port   = 19802
    server = _start_server(port)
    try:
        mock_agent = MagicMock()
        mock_agent._proactive_buffer = []
        server._server.prism_agent = mock_agent

        status, data = _get(f"http://127.0.0.1:{port}/proactive/pending")
        assert status == 200
        assert "events" in data
        assert isinstance(data["events"], list)
    finally:
        server.stop()
