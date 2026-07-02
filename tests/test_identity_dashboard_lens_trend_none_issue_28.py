"""
tests/test_identity_dashboard_lens_trend_none_issue_28.py
=========================================================
Regression for issue #28: a soul lens with no observations has
trend == None (SoulLens.trend is a property, so the old
`hasattr(ln, "trend")` guard was always True) — round(None, 3) then
raised TypeError and replaced the ENTIRE soul section of
GET /identity/dashboard with {"error": ...}, hiding seed, beliefs
and tensions from the identity UI.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prism_state import _set_state


@pytest.fixture(autouse=True)
def clean_state():
    _set_state(agent=None, federation=None)
    yield
    _set_state(agent=None, federation=None)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("PRISM_FEDERATION_REQUIRE_AUTH", "0")
    from prism_asgi import app
    return TestClient(app)


def _agent_with_empty_lens():
    soul = MagicMock()
    seed = MagicMock()
    seed.stated_values = ["honesty"]
    seed.stated_goals = ["ship"]
    seed.stated_constraints = []
    soul.get_seed.return_value = seed
    soul.list_beliefs.return_value = []
    soul.delta_report.return_value = []

    lens = MagicMock()
    lens.name = "Focus"
    lens.description = "Track depth of focused work sessions"
    lens.trend = None  # lens exists but has no observations yet
    soul.list_lenses.return_value = [lens]

    agent = MagicMock()
    agent._soul = soul
    agent._persona = None
    agent._reflection = None
    agent._phase = None
    agent._router = None
    return agent


def test_dashboard_soul_survives_lens_with_no_observations(client):
    _set_state(agent=_agent_with_empty_lens())
    r = client.get("/identity/dashboard")
    assert r.status_code == 200
    soul = r.json()["soul"]
    assert "error" not in soul, f"soul section collapsed to error: {soul}"
    assert soul["has_seed"] is True
    assert soul["lenses"] == [
        {"name": "Focus",
         "description": "Track depth of focused work sessions",
         "trend": None},
    ]
