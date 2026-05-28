from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from unittest.mock import MagicMock

from kde_server import KDEServer


def _make_agent() -> MagicMock:
    agent = MagicMock()
    agent.status.return_value = {"ok": True}
    agent.morning_briefing.return_value = MagicMock()
    agent.ask.return_value = MagicMock(task="test", method="kw", success=True, elapsed_ms=1.0, output={})
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
    platform.match = MagicMock()
    platform.injury = MagicMock()
    platform.performance = MagicMock()
    platform.transfer = MagicMock()
    return platform


def _start(port: int) -> KDEServer:
    server = KDEServer(
        agent=_make_agent(),
        port=port,
        platform=_make_platform(),
    )
    server.start(blocking=False)
    time.sleep(0.15)
    return server


def _get(url: str) -> tuple[int, dict]:
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status, json.loads(response.read())


def _post(url: str, data: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(data).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read())


def test_domain_list_200():
    server = _start(19300)
    try:
        status, data = _get("http://127.0.0.1:19300/domain/list")
        assert status == 200
        assert "domains" in data
        assert len(data["domains"]) == 6
    finally:
        server.stop()


def test_domain_evaluate_medical():
    server = _start(19301)
    try:
        profile = urllib.parse.quote("Elderly (65+)")
        status, data = _get(
            "http://127.0.0.1:19301/domain/evaluate"
            f"?domain=Medical&profile={profile}&severity=0.85&vital_signs=0.70&deteriorating=0.60"
        )
        assert status == 200
        assert "recommended" in data
        assert "options" in data
    finally:
        server.stop()


def test_domain_sensitivity_5_steps():
    server = _start(19302)
    try:
        profile = urllib.parse.quote("Middle-aged")
        status, data = _get(
            "http://127.0.0.1:19302/domain/sensitivity"
            f"?domain=Medical&profile={profile}&factor=severity&steps=5"
        )
        assert status == 200
        assert len(data["sweep"]) == 5
    finally:
        server.stop()


def test_post_validate_returns_accuracy():
    server = _start(19303)
    try:
        status, data = _post(
            "http://127.0.0.1:19303/domain/validate",
            {
                "domain": "Medical",
                "cases": [{
                    "case_id": "001",
                    "profile": "Elderly (65+)",
                    "factor_values": {"severity": 0.85},
                    "expert_choice": "Emergency A&E now",
                }],
            },
        )
        assert status == 200
        assert 0.0 <= data["accuracy"] <= 1.0
    finally:
        server.stop()


def test_unknown_domain_returns_404():
    server = _start(19304)
    try:
        try:
            _get("http://127.0.0.1:19304/domain/evaluate?domain=Unknown&profile=Any")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
        else:
            raise AssertionError("Expected HTTPError for unknown domain")
    finally:
        server.stop()
