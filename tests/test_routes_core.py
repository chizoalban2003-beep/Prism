"""
test_routes_core.py
===================
Tests for the core agent routes.
Uses FastAPI TestClient — no real port binding, no sleep().
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prism_asgi import app
from prism_state import _set_state

# ---------------------------------------------------------------------------
# Shared mocks
# ---------------------------------------------------------------------------

def _make_agent() -> MagicMock:
    agent = MagicMock()
    agent.status.return_value = {
        "profile": "TestAthlete",
        "role":    "athlete",
        "sport":   "football",
        "ok":      True,
    }
    agent.morning_briefing.return_value = MagicMock(
        time="2026-05-28T09:00:00",
        plan=MagicMock(
            primary_focus="Speed Training", activation=0.75, fulcrum=0.52,
            tasks=[], warnings=[], rationale="Focus on speed today",
        ),
        wearable_summary="HRV: 65ms",
        device_status=[],
        priority_tasks=["Morning run", "Gym"],
        alerts=[],
        match_intelligence=None,
    )
    agent.ask.return_value = MagicMock(
        task="create_training_plan", method="keyword",
        success=True, elapsed_ms=42.0, output={"content": "Mock plan"},
    )
    agent.reflect.return_value = {"fixed_fulcrum": 0.5, "drift": 0.02}
    agent.identity.return_value = {
        "domains": [{"label": "sport", "value": 0.5, "crystallised": False}],
        "insight": "Identity still crystallising",
        "confidence": 0.4,
        "n_decisions": 2,
    }
    agent.identity_domains.return_value = [
        {"domain": "sport", "fixed_fulcrum": 0.5, "variance": 0.1,
         "n_observations": 2, "crystallised": False, "confidence": 0.3,
         "last_updated": 1.0}
    ]
    agent.observe_identity.return_value = {
        "domains": [{"label": "sport", "value": 0.55, "crystallised": False}],
        "insight": "Identity still crystallising",
        "confidence": 0.5, "n_decisions": 3,
    }
    agent.reset_identity_domain.return_value = {
        "domains": [{"label": "sport", "value": 0.5, "crystallised": False}],
        "insight": "Identity still crystallising",
        "confidence": 0.2, "n_decisions": 0,
    }
    agent.recent_artifacts.return_value = [
        {"artifact_id": "a1", "domain": "sport", "artifact_type": "plan",
         "title": "Mock artifact", "content": {"ok": True},
         "fulcrum_at_time": 0.5, "identity_version": 1,
         "created_at": 1.0, "rating": None, "user_name": "TestAthlete"}
    ]
    agent.rate_artifact.return_value = {"artifact_id": "a1", "rating": 0.9}
    agent._assistant = MagicMock()
    agent._assistant.history.return_value = []
    agent._hub = MagicMock()
    agent._hub.list_devices.return_value = []
    agent._profile = MagicMock()
    agent._profile.name = "TestAthlete"
    return agent




@pytest.fixture()
def client():
    agent = _make_agent()
    _set_state(agent=agent)
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_status_endpoint_returns_json(client):
    from unittest.mock import patch
    with patch("urllib.request.urlopen", side_effect=Exception("no ollama")):
        r = client.get("/status")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict)
    assert "profile" in data or "ok" in data


def test_server_localhost_only():
    """ASGI server must always bind to 127.0.0.1."""
    import pytest as _pt

    import prism_asgi
    with _pt.raises(AssertionError):
        prism_asgi.serve(host="0.0.0.0", port=19999)


def test_unknown_route_returns_404(client):
    r = client.get("/this_route_does_not_exist_xyz")
    assert r.status_code == 404


