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


def _make_platform() -> MagicMock:
    from prediction_engine import PredictionPlatform
    platform = MagicMock(spec=PredictionPlatform)
    mp = MagicMock()
    mp.predict.return_value = MagicMock(
        subject="Arsenal vs Chelsea", prediction="Arsenal win", confidence=0.62,
        p_home_win=0.55, p_draw=0.25, p_away_win=0.20, predicted_margin=1.2,
        distribution={"home_win": 0.55, "draw": 0.25, "away_win": 0.20},
        risk=0.3, risk_adj=0.28, fulcrum=0.5,
        key_factors=[("form", 0.7, "positive")], expected_value=1.0,
        home_team="Arsenal", away_team="Chelsea",
    )
    platform.match = mp
    ip = MagicMock()
    ip.predict.return_value = MagicMock(
        subject="TestAthlete", prediction="low risk", confidence=0.8,
        athlete_name="TestAthlete", risk_level="low", days_to_risk=30,
        recommendations=["Monitor load"],
        distribution={"low": 0.8, "moderate": 0.15, "high": 0.05},
        risk=0.1, risk_adj=0.1, fulcrum=0.5, key_factors=[], expected_value=0.1,
    )
    platform.injury = ip
    pp = MagicMock()
    pp.predict.return_value = MagicMock(
        subject="TestAthlete", prediction="good form", confidence=0.75,
        athlete_name="TestAthlete", period="next 30 days", expected_rating=7.5,
        form_trend="improving",
        distribution={"improving": 0.75, "stable": 0.2, "declining": 0.05},
        risk=0.2, risk_adj=0.18, fulcrum=0.5, key_factors=[], expected_value=7.5,
    )
    platform.performance = pp
    tp = MagicMock()
    tp.predict.return_value = MagicMock(
        subject="TestAthlete", prediction="mid-range value", confidence=0.65,
        athlete_name="TestAthlete", value_band="10M-20M",
        value_low_m=10.0, value_high_m=20.0,
        distribution={"10M-20M": 0.65, "5M-10M": 0.35},
        risk=0.3, risk_adj=0.28, fulcrum=0.5, key_factors=[], expected_value=15.0,
    )
    platform.transfer = tp
    return platform


@pytest.fixture()
def client():
    agent = _make_agent()
    platform = _make_platform()
    _set_state(agent=agent, platform=platform)
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


def test_predict_match_endpoint(client):
    r = client.get("/predict/match?home=Arsenal&away=Chelsea&sport=football")
    assert r.status_code == 200
    assert "p_home_win" in r.json()


def test_server_localhost_only():
    """ASGI server must always bind to 127.0.0.1."""
    import pytest as _pt

    import prism_asgi
    with _pt.raises(AssertionError):
        prism_asgi.serve(host="0.0.0.0", port=19999)


def test_unknown_route_returns_404(client):
    r = client.get("/this_route_does_not_exist_xyz")
    assert r.status_code == 404


def test_devices_endpoint(client):
    r = client.get("/devices")
    assert r.status_code == 200
    assert "devices" in r.json()


