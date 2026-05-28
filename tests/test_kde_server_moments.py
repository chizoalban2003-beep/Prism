"""
test_kde_server_moments.py
==========================
Tests for the new /moment/* and /duel/* endpoints added to kde_server.py.

Uses a real in-process HTTP server (no external calls) with mocked agents.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request
from unittest.mock import MagicMock

import pytest

from kde_server import KDEServer, DEFAULT_HOST
from moment_analyzer import MomentAnalyzer
from duel_analyzer import DuelAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent() -> MagicMock:
    agent = MagicMock()
    agent.status.return_value = {"ok": True}
    agent.morning_briefing.return_value = MagicMock(
        time="2026-01-01T09:00:00",
        plan=MagicMock(
            primary_focus="Speed", activation=0.5, fulcrum=0.5,
            tasks=[], warnings=[], rationale="ok",
        ),
        wearable_summary="HRV: 60ms",
        device_status=[],
        priority_tasks=[],
        alerts=[],
        match_intelligence=None,
    )
    agent.ask.return_value = MagicMock(
        task="test", method="kw", success=True, elapsed_ms=1.0,
        output={}
    )
    agent.reflect.return_value = {}
    agent._assistant = MagicMock()
    agent._assistant.history.return_value = []
    agent._hub = MagicMock()
    agent._hub.list_devices.return_value = []
    agent._profile = MagicMock()
    agent._profile.name = "Tester"
    return agent


def _make_platform() -> MagicMock:
    platform = MagicMock()
    mp = MagicMock()
    mp.predict.return_value = MagicMock(
        subject="A vs B", prediction="A win", confidence=0.6,
        p_home_win=0.55, p_draw=0.25, p_away_win=0.20,
        predicted_margin=1.0, distribution={}, risk=0.3, risk_adj=0.28,
        fulcrum=0.5, key_factors=[], expected_value=1.0,
        home_team="A", away_team="B",
    )
    platform.match = mp
    platform.injury = MagicMock()
    platform.performance = MagicMock()
    platform.transfer = MagicMock()
    return platform


def _start(port: int) -> KDEServer:
    agent    = _make_agent()
    platform = _make_platform()
    server   = KDEServer(
        agent           = agent,
        port            = port,
        platform        = platform,
        moment_analyzer = MomentAnalyzer(),
        duel_analyzer   = DuelAnalyzer(),
    )
    server.start(blocking=False)
    time.sleep(0.15)
    return server


def _get(url: str) -> tuple[int, dict]:
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, json.loads(r.read())


def _post(url: str, data: dict) -> tuple[int, dict]:
    body = json.dumps(data).encode()
    req  = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, json.loads(r.read())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_get_moment_configs_200():
    """GET /moment/configs returns 200 and a list of sport/moment_type pairs."""
    port   = 19200
    server = _start(port)
    try:
        status, data = _get(f"http://127.0.0.1:{port}/moment/configs")
        assert status == 200
        assert "configs" in data
        configs = data["configs"]
        assert isinstance(configs, list)
        assert len(configs) > 0
        assert all("sport" in c and "moment_type" in c for c in configs)
        # Should contain at least Football/1v1_keeper
        sports = {c["sport"] for c in configs}
        assert "Football" in sports
    finally:
        server.stop()


def test_get_moment_analyze_football():
    """GET /moment/analyze with Football params returns recommended key."""
    port   = 19201
    server = _start(port)
    try:
        status, data = _get(
            f"http://127.0.0.1:{port}/moment/analyze"
            f"?sport=Football&moment_type=1v1_keeper"
            f"&player=Haaland&base=0.82&pitch_x=0.91&pitch_y=0.48"
            f"&xg_raw=0.65&fatigue=0.30&confidence=0.88"
            f"&gk_name=Alisson&gk_distance=6.0"
        )
        assert status == 200
        assert "recommended" in data
        assert "xg_contextual" in data
        assert "options" in data
        assert isinstance(data["options"], list)
        assert len(data["options"]) > 0
    finally:
        server.stop()


def test_get_moment_analyze_tennis():
    """GET /moment/analyze with Tennis serve_deuce works."""
    port   = 19202
    server = _start(port)
    try:
        # register extended configs first via the endpoint call (they're lazily available)
        from moment_configs_ext import register_extended_configs
        register_extended_configs()

        status, data = _get(
            f"http://127.0.0.1:{port}/moment/analyze"
            f"?sport=Tennis&moment_type=serve_deuce"
            f"&player=Djokovic&base=0.72&pitch_x=0.60&pitch_y=0.50"
            f"&confidence=0.85&fatigue=0.10"
        )
        assert status == 200
        assert "recommended" in data
    finally:
        server.stop()


def test_post_calibrate_success():
    """POST /moment/calibrate returns {'status': 'calibrated'}."""
    port   = 19203
    server = _start(port)
    try:
        # First create a moment in history
        _get(
            f"http://127.0.0.1:{port}/moment/analyze"
            f"?sport=Football&moment_type=1v1_keeper"
            f"&player=Kane&base=0.75&pitch_x=0.88&pitch_y=0.50"
        )
        status, data = _post(
            f"http://127.0.0.1:{port}/moment/calibrate",
            {
                "moment_id":    "some-id",
                "action_taken": "Power shot",
                "success":      True,
                "xg_realized":  1.0,
                "notes":        "top corner",
            },
        )
        assert status == 200
        assert data.get("status") == "calibrated"
    finally:
        server.stop()


def test_post_live_frame_no_moment():
    """A frame with ball-carrier deep in own half returns {'moment': null}."""
    port   = 19204
    server = _start(port)
    try:
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
        status, data = _post(f"http://127.0.0.1:{port}/moment/live_frame", frame)
        assert status == 200
        assert data.get("moment") is None
    finally:
        server.stop()


def test_post_live_frame_detects():
    """A frame with ball-carrier in attacking zone near an opponent detects a moment."""
    port   = 19205
    server = _start(port)
    try:
        from moment_pipeline import LiveMomentPipeline

        # Feed MIN_FRAMES consecutive frames to trigger detection
        frame = {
            "timestamp": 200.0,
            "sport":     "Football",
            "players": [
                # Attacker past DETECTION_THRESHOLD_X (0.72 * 105 = ~75.6 m)
                {"name": "Striker", "team": "Home", "x": 88.0, "y": 34.0,
                 "speed": 5.0, "has_ball": True},
                # Opponent within DETECTION_DIST (5 m)
                {"name": "Defender", "team": "Away", "x": 90.0, "y": 35.0,
                 "speed": 4.0, "has_ball": False},
            ],
        }
        result = None
        for _ in range(LiveMomentPipeline.MIN_FRAMES + 2):
            status, data = _post(f"http://127.0.0.1:{port}/moment/live_frame", frame)
            assert status == 200
            if data.get("moment") is not None:
                result = data["moment"]
                break

        assert result is not None, "Expected a detected moment after MIN_FRAMES frames"
        assert "recommended" in result
    finally:
        server.stop()


def test_duel_network_endpoint():
    """GET /duel/network returns edges list (may be empty initially)."""
    port   = 19206
    server = _start(port)
    try:
        status, data = _get(f"http://127.0.0.1:{port}/duel/network?match_id=match_001")
        assert status == 200
        assert "edges" in data
        assert isinstance(data["edges"], list)
    finally:
        server.stop()


def test_duel_network_after_add_match():
    """POST /duel/add_match then GET /duel/network returns non-empty edges."""
    port   = 19207
    server = _start(port)
    try:
        events = [
            {
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
            }
        ]
        status, data = _post(
            f"http://127.0.0.1:{port}/duel/add_match",
            {"match_id": "match_001", "events": events},
        )
        assert status == 200
        assert data.get("n_duels") == 1

        status2, net = _get(f"http://127.0.0.1:{port}/duel/network?match_id=match_001")
        assert status2 == 200
        assert len(net["edges"]) >= 1
    finally:
        server.stop()


def test_server_localhost_only():
    """KDEServer must always use 127.0.0.1 regardless of constructor arg."""
    agent  = _make_agent()
    server = KDEServer(agent=agent, host="0.0.0.0", port=19208)
    assert server._host == DEFAULT_HOST == "127.0.0.1"


def test_cors_headers_present():
    """Every response must include Access-Control-Allow-Origin header."""
    port   = 19209
    server = _start(port)
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/moment/configs", timeout=5
        ) as resp:
            assert resp.headers.get("Access-Control-Allow-Origin") is not None
    finally:
        server.stop()
