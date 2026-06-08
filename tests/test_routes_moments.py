"""
test_routes_moments.py
======================
Tests for /moment/* and /duel/* endpoints — uses FastAPI TestClient.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from duel_analyzer import DuelAnalyzer
from moment_analyzer import MomentAnalyzer
from prism_asgi import app
from prism_state import _set_state


@pytest.fixture()
def client():
    from moment_pipeline import LiveMomentPipeline
    agent = MagicMock()
    agent.status.return_value = {"ok": True}
    agent._assistant = MagicMock()
    agent._assistant.history.return_value = []
    agent._hub = MagicMock()
    agent._hub.list_devices.return_value = []
    agent._profile = MagicMock()
    agent._profile.name = "Tester"
    ma = MomentAnalyzer()
    _set_state(
        agent=agent,
        platform=MagicMock(),
        moment_analyzer=ma,
        duel_analyzer=DuelAnalyzer(),
        live_pipeline=LiveMomentPipeline(ma),
    )
    return TestClient(app, raise_server_exceptions=False)


def test_get_moment_configs_200(client):
    """GET /moment/configs returns 200 and a list of sport/moment_type pairs."""
    r = client.get("/moment/configs")
    assert r.status_code == 200
    data = r.json()
    assert "configs" in data
    configs = data["configs"]
    assert isinstance(configs, list)
    assert len(configs) > 0
    assert all("sport" in c and "moment_type" in c for c in configs)
    sports = {c["sport"] for c in configs}
    assert "Football" in sports


def test_get_moment_analyze_football(client):
    """GET /moment/analyze with Football params returns recommended key."""
    r = client.get(
        "/moment/analyze"
        "?sport=Football&moment_type=1v1_keeper"
        "&player=Haaland&base=0.82&pitch_x=0.91&pitch_y=0.48"
        "&xg_raw=0.65&fatigue=0.30&confidence=0.88"
        "&gk_name=Alisson&gk_distance=6.0"
    )
    assert r.status_code == 200
    data = r.json()
    assert "recommended" in data
    assert "xg_contextual" in data
    assert "options" in data
    assert isinstance(data["options"], list)
    assert len(data["options"]) > 0


def test_get_moment_analyze_tennis(client):
    """GET /moment/analyze with Tennis serve_deuce works."""
    from moment_configs_ext import register_extended_configs
    register_extended_configs()

    r = client.get(
        "/moment/analyze"
        "?sport=Tennis&moment_type=serve_deuce"
        "&player=Djokovic&base=0.72&pitch_x=0.60&pitch_y=0.50"
        "&confidence=0.85&fatigue=0.10"
    )
    assert r.status_code == 200
    assert "recommended" in r.json()


def test_post_calibrate_success(client):
    """POST /moment/calibrate returns {'status': 'calibrated'}."""
    # First create a moment in history
    client.get(
        "/moment/analyze"
        "?sport=Football&moment_type=1v1_keeper"
        "&player=Kane&base=0.75&pitch_x=0.88&pitch_y=0.50"
    )
    r = client.post("/moment/calibrate", json={
        "moment_id":    "some-id",
        "action_taken": "Power shot",
        "success":      True,
        "xg_realized":  1.0,
        "notes":        "top corner",
    })
    assert r.status_code == 200
    assert r.json().get("status") == "calibrated"


def test_post_live_frame_no_moment(client):
    """A frame with ball-carrier deep in own half returns {'moment': null}."""
    frame = {
        "timestamp": 100.0,
        "sport":     "Football",
        "players": [
            {"name": "Keeper", "team": "Home", "x": 20.0, "y": 34.0,
             "speed": 2.0, "has_ball": True},
            {"name": "OppFwd", "team": "Away", "x": 80.0, "y": 34.0,
             "speed": 4.0, "has_ball": False},
        ],
    }
    r = client.post("/moment/live_frame", json=frame)
    assert r.status_code == 200
    assert r.json().get("moment") is None


def test_post_live_frame_detects(client):
    """A frame with ball-carrier in attacking zone near an opponent detects a moment."""
    from moment_pipeline import LiveMomentPipeline

    frame = {
        "timestamp": 200.0,
        "sport":     "Football",
        "players": [
            {"name": "Striker", "team": "Home", "x": 88.0, "y": 34.0,
             "speed": 5.0, "has_ball": True},
            {"name": "Defender", "team": "Away", "x": 90.0, "y": 35.0,
             "speed": 4.0, "has_ball": False},
        ],
    }
    result = None
    for _ in range(LiveMomentPipeline.MIN_FRAMES + 2):
        r = client.post("/moment/live_frame", json=frame)
        assert r.status_code == 200
        if r.json().get("moment") is not None:
            result = r.json()["moment"]
            break

    assert result is not None, "Expected a detected moment after MIN_FRAMES frames"
    assert "recommended" in result


def test_duel_network_endpoint(client):
    """GET /duel/network returns edges list (may be empty initially)."""
    r = client.get("/duel/network?match_id=match_001")
    assert r.status_code == 200
    data = r.json()
    assert "edges" in data
    assert isinstance(data["edges"], list)


def test_duel_network_after_add_match(client):
    """POST /duel/add_match then GET /duel/network returns non-empty edges."""
    events = [{
        "id": "evt-1",
        "type": {"name": "Duel"},
        "player": {"name": "Salah"},
        "team": {"name": "Liverpool"},
        "location": [95.0, 40.0],
        "timestamp": "00:01:30.000",
        "duel": {
            "type": {"name": "ground"},
            "outcome": {"name": "won"},
            "counterpart": {"name": "Reece James", "team": {"name": "Chelsea"}},
        },
    }]
    r = client.post("/duel/add_match",
                    json={"match_id": "match_001", "events": events})
    assert r.status_code == 200
    assert r.json().get("n_duels") == 1

    r2 = client.get("/duel/network?match_id=match_001")
    assert r2.status_code == 200
    assert len(r2.json()["edges"]) >= 1


def test_cors_headers_present(client):
    """FastAPI CORS middleware must include Access-Control-Allow-Origin."""
    r = client.get("/moment/configs", headers={"Origin": "http://localhost"})
    assert r.headers.get("access-control-allow-origin") is not None
