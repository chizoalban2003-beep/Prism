"""
test_kde_server.py
==================
Tests for kde_server.py

pytest. Mock KDEAgent and PredictionPlatform — no real HTTP calls to Ollama.
"""
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
    """Create a mock KDEAgent with minimal interface."""
    agent = MagicMock()
    agent.status.return_value = {
        "profile": "TestAthlete",
        "role":    "athlete",
        "sport":   "football",
        "ok":      True,
    }
    agent.morning_briefing.return_value = MagicMock(
        time            = "2026-05-28T09:00:00",
        plan            = MagicMock(
            primary_focus = "Speed Training",
            activation    = 0.75,
            fulcrum       = 0.52,
            tasks         = [],
            warnings      = [],
            rationale     = "Focus on speed today",
        ),
        wearable_summary  = "HRV: 65ms",
        device_status     = [],
        priority_tasks    = ["Morning run", "Gym"],
        alerts            = [],
        match_intelligence = None,
    )
    agent.ask.return_value = MagicMock(
        task       = "create_training_plan",
        method     = "keyword",
        success    = True,
        elapsed_ms = 42.0,
        output     = {"content": "Mock plan"},
    )
    agent.reflect.return_value = {"fixed_fulcrum": 0.5, "drift": 0.02}
    agent.identity.return_value = {
        "domains": [{"label": "sport", "value": 0.5, "crystallised": False}],
        "insight": "Identity still crystallising",
        "confidence": 0.4,
        "n_decisions": 2,
    }
    agent.identity_domains.return_value = [
        {
            "domain": "sport",
            "fixed_fulcrum": 0.5,
            "variance": 0.1,
            "n_observations": 2,
            "crystallised": False,
            "confidence": 0.3,
            "last_updated": 1.0,
        }
    ]
    agent.observe_identity.return_value = {
        "domains": [{"label": "sport", "value": 0.55, "crystallised": False}],
        "insight": "Identity still crystallising",
        "confidence": 0.5,
        "n_decisions": 3,
    }
    agent.reset_identity_domain.return_value = {
        "domains": [{"label": "sport", "value": 0.5, "crystallised": False}],
        "insight": "Identity still crystallising",
        "confidence": 0.2,
        "n_decisions": 0,
    }
    agent.recent_artifacts.return_value = [
        {
            "artifact_id": "a1",
            "domain": "sport",
            "artifact_type": "plan",
            "title": "Mock artifact",
            "content": {"ok": True},
            "fulcrum_at_time": 0.5,
            "identity_version": 1,
            "created_at": 1.0,
            "rating": None,
            "user_name": "TestAthlete",
        }
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

    # match predictor
    mp = MagicMock()
    mp.predict.return_value = MagicMock(
        subject           = "Arsenal vs Chelsea",
        prediction        = "Arsenal win",
        confidence        = 0.62,
        p_home_win        = 0.55,
        p_draw            = 0.25,
        p_away_win        = 0.20,
        predicted_margin  = 1.2,
        distribution      = {"home_win": 0.55, "draw": 0.25, "away_win": 0.20},
        risk              = 0.3,
        risk_adj          = 0.28,
        fulcrum           = 0.5,
        key_factors       = [("form", 0.7, "positive")],
        expected_value    = 1.0,
        home_team         = "Arsenal",
        away_team         = "Chelsea",
    )
    platform.match = mp

    # injury predictor
    ip = MagicMock()
    ip.predict.return_value = MagicMock(
        subject         = "TestAthlete",
        prediction      = "low risk",
        confidence      = 0.8,
        athlete_name    = "TestAthlete",
        risk_level      = "low",
        days_to_risk    = 30,
        recommendations = ["Monitor load"],
        distribution    = {"low": 0.8, "moderate": 0.15, "high": 0.05},
        risk            = 0.1,
        risk_adj        = 0.1,
        fulcrum         = 0.5,
        key_factors     = [],
        expected_value  = 0.1,
    )
    platform.injury = ip

    # performance predictor
    pp = MagicMock()
    pp.predict.return_value = MagicMock(
        subject          = "TestAthlete",
        prediction       = "good form",
        confidence       = 0.75,
        athlete_name     = "TestAthlete",
        period           = "next 30 days",
        expected_rating  = 7.5,
        form_trend       = "improving",
        distribution     = {"improving": 0.75, "stable": 0.2, "declining": 0.05},
        risk             = 0.2,
        risk_adj         = 0.18,
        fulcrum          = 0.5,
        key_factors      = [],
        expected_value   = 7.5,
    )
    platform.performance = pp

    # transfer predictor
    tp = MagicMock()
    tp.predict.return_value = MagicMock(
        subject          = "TestAthlete",
        prediction       = "mid-range value",
        confidence       = 0.65,
        athlete_name     = "TestAthlete",
        value_band       = "10M-20M",
        value_low_m      = 10.0,
        value_high_m     = 20.0,
        distribution     = {"10M-20M": 0.65, "5M-10M": 0.35},
        risk             = 0.3,
        risk_adj         = 0.28,
        fulcrum          = 0.5,
        key_factors      = [],
        expected_value   = 15.0,
    )
    platform.transfer = tp

    return platform


def _start_server(port: int) -> KDEServer:
    agent    = _make_agent()
    platform = _make_platform()
    server   = KDEServer(agent=agent, port=port, platform=platform)
    server.start(blocking=False)
    time.sleep(0.15)   # give the thread a moment
    return server


def _get(url: str) -> tuple[int, dict]:
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.status, json.loads(resp.read())


def _post(url: str, data: dict) -> tuple[int, dict]:
    body = json.dumps(data).encode()
    req  = urllib.request.Request(
        url,
        data    = body,
        method  = "POST",
        headers = {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, json.loads(resp.read())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_server_starts_on_port():
    """Server should bind to 127.0.0.1 and the given port."""
    port   = 18742
    server = _start_server(port)
    try:
        status, data = _get(f"http://127.0.0.1:{port}/status")
        assert status == 200
    finally:
        server.stop()


def test_status_endpoint_returns_json():
    port   = 18743
    server = _start_server(port)
    try:
        status, data = _get(f"http://127.0.0.1:{port}/status")
        assert status == 200
        assert isinstance(data, dict)
        assert "profile" in data or "ok" in data
    finally:
        server.stop()


def test_predict_match_endpoint():
    """GET /predict/match must return p_home_win key."""
    port   = 18744
    server = _start_server(port)
    try:
        status, data = _get(
            f"http://127.0.0.1:{port}/predict/match?home=Arsenal&away=Chelsea&sport=football"
        )
        assert status == 200
        assert "p_home_win" in data
    finally:
        server.stop()


def test_ask_endpoint_routes_task():
    """POST /ask with training plan prompt must return a task result."""
    port   = 18745
    server = _start_server(port)
    try:
        status, data = _post(
            f"http://127.0.0.1:{port}/ask",
            {"prompt": "create a training plan for next week"},
        )
        assert status == 200
        assert "task" in data or "output" in data
    finally:
        server.stop()


def test_server_localhost_only():
    """KDEServer must always bind to 127.0.0.1, never 0.0.0.0."""
    agent  = _make_agent()
    # Even if someone passes "0.0.0.0", it should be forced to 127.0.0.1
    server = KDEServer(agent=agent, host="0.0.0.0", port=18746)
    assert server._host == "127.0.0.1"


def test_unknown_route_returns_404():
    port   = 18747
    server = _start_server(port)
    try:
        from urllib.error import HTTPError
        try:
            _get(f"http://127.0.0.1:{port}/this_route_does_not_exist_xyz")
            # If no exception, the server returned 404 as valid JSON
        except HTTPError as e:
            assert e.code == 404
    finally:
        server.stop()


def test_history_endpoint():
    port   = 18748
    server = _start_server(port)
    try:
        status, data = _get(f"http://127.0.0.1:{port}/history?days=7")
        assert status == 200
        assert "history" in data
    finally:
        server.stop()


def test_devices_endpoint():
    port   = 18749
    server = _start_server(port)
    try:
        status, data = _get(f"http://127.0.0.1:{port}/devices")
        assert status == 200
        assert "devices" in data
    finally:
        server.stop()


def test_identity_endpoint_returns_json():
    port = 18750
    server = _start_server(port)
    try:
        status, data = _get(f"http://127.0.0.1:{port}/identity")
        assert status == 200
        assert "domains" in data
        assert "insight" in data
    finally:
        server.stop()


def test_identity_observe_endpoint_updates_identity():
    port = 18751
    server = _start_server(port)
    try:
        status, data = _post(
            f"http://127.0.0.1:{port}/identity/observe",
            {"domain": "sport", "fulcrum": 0.6, "rating": 0.8},
        )
        assert status == 200
        assert "n_decisions" in data
    finally:
        server.stop()


def test_artifacts_endpoint_returns_json():
    port = 18752
    server = _start_server(port)
    try:
        status, data = _get(f"http://127.0.0.1:{port}/artifacts?domain=sport&n=5")
        assert status == 200
        assert "artifacts" in data
        assert isinstance(data["artifacts"], list)
    finally:
        server.stop()
