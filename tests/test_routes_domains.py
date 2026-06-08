"""
test_routes_domains.py
======================
Tests for /domain/* and related endpoints — uses FastAPI TestClient.
"""
from __future__ import annotations

import urllib.parse
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prism_asgi import app
from prism_state import _set_state


def _make_agent() -> MagicMock:
    agent = MagicMock()
    agent.status.return_value = {"ok": True}
    agent.morning_briefing.return_value = MagicMock()
    agent.ask.return_value = MagicMock(
        task="test", method="kw", success=True, elapsed_ms=1.0, output={}
    )
    agent.reflect.return_value = {}
    agent._assistant = MagicMock()
    agent._assistant.history.return_value = []
    agent._hub = MagicMock()
    agent._hub.list_devices.return_value = []
    agent._profile = MagicMock()
    agent._profile.name = "Tester"
    return agent


@pytest.fixture()
def client():
    from domain_configs import ALL_DOMAINS, DomainDecisionModel
    domain_models = {name: DomainDecisionModel(cfg) for name, cfg in ALL_DOMAINS.items()}
    _set_state(agent=_make_agent(), platform=MagicMock(), domain_models=domain_models)
    return TestClient(app, raise_server_exceptions=False)


def test_domain_list_200(client):
    r = client.get("/domain/list")
    assert r.status_code == 200
    data = r.json()
    assert "domains" in data
    assert len(data["domains"]) == 6


def test_domain_evaluate_medical(client):
    profile = urllib.parse.quote("Elderly (65+)")
    r = client.get(
        f"/domain/evaluate?domain=Medical&profile={profile}"
        "&severity=0.85&vital_signs=0.70&deteriorating=0.60"
    )
    assert r.status_code == 200
    data = r.json()
    assert "recommended" in data
    assert "options" in data


def test_domain_sensitivity_5_steps(client):
    profile = urllib.parse.quote("Middle-aged")
    r = client.get(
        f"/domain/sensitivity?domain=Medical&profile={profile}&factor=severity&steps=5"
    )
    assert r.status_code == 200
    assert len(r.json()["sweep"]) == 5


def test_post_validate_returns_accuracy(client):
    r = client.post("/domain/validate", json={
        "domain": "Medical",
        "cases": [{
            "case_id": "001",
            "profile": "Elderly (65+)",
            "factor_values": {"severity": 0.85},
            "expert_choice": "Emergency A&E now",
        }],
    })
    assert r.status_code == 200
    assert 0.0 <= r.json()["accuracy"] <= 1.0


def test_unknown_domain_returns_404(client):
    r = client.get("/domain/evaluate?domain=Unknown&profile=Any")
    assert r.status_code == 404
