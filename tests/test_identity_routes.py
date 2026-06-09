"""Tests for prism_routes_identity.py and federation identity routes."""
from __future__ import annotations

import time
import unittest.mock as _mock
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prism_state import _set_state, _state

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_state():
    _set_state(agent=None, federation=None)
    _state.pop("_last_weekly_report", None)
    yield
    _set_state(agent=None, federation=None)
    _state.pop("_last_weekly_report", None)


@pytest.fixture(autouse=True)
def clean_onboarding_file():
    p = Path("~/.prism/onboarding_state.json").expanduser()
    if p.exists():
        p.unlink()
    yield
    if p.exists():
        p.unlink()


@pytest.fixture
def client():
    from prism_asgi import app
    return TestClient(app)


def _make_soul(has_seed: bool = True):
    soul = MagicMock()
    soul.has_seed.return_value = has_seed
    if has_seed:
        seed = MagicMock()
        seed.stated_values = ["honesty", "autonomy", "excellence"]
        seed.stated_goals = ["build great products", "grow daily"]
        seed.stated_constraints = ["no shortcuts"]
        soul.get_seed.return_value = seed
    else:
        soul.get_seed.return_value = None

    belief = MagicMock()
    belief.text = "I value deep work"
    belief.belief_type = "value"
    belief.source = "stated"
    belief.confidence = 0.85
    belief.observation_count = 12
    soul.list_beliefs.return_value = [belief]

    lens = MagicMock()
    lens.name = "focus"
    lens.description = "Focus level lens"
    lens.trend = 0.72
    soul.list_lenses.return_value = [lens]
    soul.delta_report.return_value = []

    return soul


def _make_persona():
    persona = MagicMock()
    trait = MagicMock()
    trait.name = "response_length"
    trait.value = "concise"
    trait.confidence = 0.85
    trait.source = "inferred"
    trait.observation_count = 20
    persona.list_traits.return_value = [trait]
    persona.growth_since.return_value = {
        "new_traits": 2, "new_patterns": 1, "confidence_avg": 0.75, "days": 7,
    }
    persona.peak_hours.return_value = [9, 10, 14]
    return persona


def _make_reflection():
    reflection = MagicMock()
    report = MagicMock()
    report.ran_at = time.time()
    report.summary = "Productive week overall."
    report.patterns = ["works best in morning", "prefers concise replies"]
    report.belief_proposals = [{"node_id": "b1", "text": None, "new_confidence": 0.9, "rationale": "observed"}]
    report.unresolved_goals = ["finish project X"]
    report.applied = True
    report.error = None
    reflection.run.return_value = report
    return reflection


def _make_agent(has_soul=True, has_persona=True, has_reflection=True, soul_seed=True):
    agent = MagicMock()
    agent._soul = _make_soul(soul_seed) if has_soul else None
    agent._persona = _make_persona() if has_persona else None
    agent._reflection = _make_reflection() if has_reflection else None
    agent._phase = None
    agent._router = None
    return agent


# ---------------------------------------------------------------------------
# GET /identity/dashboard
# ---------------------------------------------------------------------------


class TestIdentityDashboard:
    def test_200_no_agent(self, client):
        r = client.get("/identity/dashboard")
        assert r.status_code == 200
        d = r.json()
        assert "phase" in d
        assert "generated_at" in d
        assert d["soul"] is None
        assert d["persona"] is None

    def test_200_with_soul_and_persona(self, client):
        _set_state(agent=_make_agent())
        r = client.get("/identity/dashboard")
        assert r.status_code == 200
        d = r.json()
        assert d["soul"]["has_seed"] is True
        assert "honesty" in d["soul"]["stated_values"]
        assert len(d["persona"]["traits"]) == 1
        assert d["persona"]["traits"][0]["name"] == "response_length"

    def test_crystallisation_pct_computed(self, client):
        _set_state(agent=_make_agent())
        r = client.get("/identity/dashboard")
        d = r.json()
        assert d["crystallisation_pct"] == pytest.approx(85.0)

    def test_no_soul(self, client):
        _set_state(agent=_make_agent(has_soul=False))
        r = client.get("/identity/dashboard")
        assert r.status_code == 200
        assert r.json()["soul"] is None

    def test_no_persona(self, client):
        _set_state(agent=_make_agent(has_persona=False))
        r = client.get("/identity/dashboard")
        assert r.status_code == 200
        d = r.json()
        assert d["persona"] is None
        assert d["crystallisation_pct"] == 0.0

    def test_soul_without_seed(self, client):
        _set_state(agent=_make_agent(soul_seed=False))
        r = client.get("/identity/dashboard")
        assert r.status_code == 200
        d = r.json()
        assert d["soul"]["has_seed"] is False
        assert d["soul"]["stated_values"] == []

    def test_phase_unknown_without_engine(self, client):
        _set_state(agent=_make_agent())
        r = client.get("/identity/dashboard")
        d = r.json()
        assert d["phase"]["current"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# GET /identity  (HTML dashboard)
# ---------------------------------------------------------------------------


class TestIdentityHTML:
    def test_returns_html(self, client):
        r = client.get("/identity/ui")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_html_contains_key_content(self, client):
        r = client.get("/identity/ui")
        assert "Your Prism" in r.text
        assert "identity/dashboard" in r.text

    def test_moat_statement_present(self, client):
        r = client.get("/identity/ui")
        assert "stay on this device" in r.text


# ---------------------------------------------------------------------------
# GET /identity/onboard  (HTML ceremony page)
# ---------------------------------------------------------------------------


class TestIdentityOnboardHTML:
    def test_returns_html(self, client):
        r = client.get("/identity/onboard")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_contains_ceremony_content(self, client):
        r = client.get("/identity/onboard")
        assert "Identity Ceremony" in r.text
        assert "onboarding/start" in r.text


# ---------------------------------------------------------------------------
# GET /reports/weekly
# ---------------------------------------------------------------------------


class TestWeeklyReport:
    def test_no_reflection_503(self, client):
        _set_state(agent=_make_agent(has_reflection=False))
        r = client.get("/reports/weekly")
        assert r.status_code == 503

    def test_no_cached_report_returns_info(self, client):
        _set_state(agent=_make_agent())
        r = client.get("/reports/weekly")
        assert r.status_code == 200
        assert "info" in r.json()

    def test_returns_cached_after_generate(self, client):
        _set_state(agent=_make_agent())
        client.post("/reports/weekly/generate")
        r = client.get("/reports/weekly")
        assert r.status_code == 200
        assert r.json()["summary"] == "Productive week overall."

    def test_no_agent_503(self, client):
        r = client.get("/reports/weekly")
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# POST /reports/weekly/generate
# ---------------------------------------------------------------------------


class TestWeeklyReportGenerate:
    def test_generates_report(self, client):
        _set_state(agent=_make_agent())
        r = client.post("/reports/weekly/generate")
        assert r.status_code == 200
        d = r.json()
        assert d["summary"] == "Productive week overall."
        assert d["applied"] is True
        assert len(d["patterns"]) == 2

    def test_caches_result(self, client):
        _set_state(agent=_make_agent())
        client.post("/reports/weekly/generate")
        assert "_last_weekly_report" in _state

    def test_no_reflection_503(self, client):
        _set_state(agent=_make_agent(has_reflection=False))
        r = client.post("/reports/weekly/generate")
        assert r.status_code == 503

    def test_belief_proposals_serialized(self, client):
        _set_state(agent=_make_agent())
        r = client.post("/reports/weekly/generate")
        d = r.json()
        assert len(d["belief_proposals"]) == 1
        assert d["belief_proposals"][0]["node_id"] == "b1"

    def test_unresolved_goals_included(self, client):
        _set_state(agent=_make_agent())
        r = client.post("/reports/weekly/generate")
        d = r.json()
        assert "finish project X" in d["unresolved_goals"]


# ---------------------------------------------------------------------------
# GET /onboarding/status
# ---------------------------------------------------------------------------


class TestOnboardingStatus:
    def test_no_agent(self, client):
        r = client.get("/onboarding/status")
        assert r.status_code == 200
        d = r.json()
        assert d["ceremony_complete"] is False
        assert d["questions_total"] == 7

    def test_with_soul_seed(self, client):
        _set_state(agent=_make_agent())
        r = client.get("/onboarding/status")
        d = r.json()
        assert d["ceremony_complete"] is True

    def test_observations_counted(self, client):
        _set_state(agent=_make_agent())
        r = client.get("/onboarding/status")
        d = r.json()
        assert d["observations"] == 20

    def test_message_with_seed(self, client):
        _set_state(agent=_make_agent())
        r = client.get("/onboarding/status")
        d = r.json()
        assert "ceremony complete" in d["message"].lower() or "crystallised" in d["message"].lower()

    def test_message_without_seed(self, client):
        r = client.get("/onboarding/status")
        d = r.json()
        assert "ceremony" in d["message"].lower()


# ---------------------------------------------------------------------------
# POST /onboarding/start
# ---------------------------------------------------------------------------


class TestOnboardingStart:
    def test_returns_first_question(self, client):
        r = client.post("/onboarding/start")
        assert r.status_code == 200
        d = r.json()
        assert d["started"] is True
        assert d["question_index"] == 0
        assert d["total_questions"] == 7
        assert "question" in d
        assert d["progress"] == "1/7"

    def test_question_key_is_identity(self, client):
        r = client.post("/onboarding/start")
        d = r.json()
        assert d["question_key"] == "identity"

    def test_restart_resets_state(self, client):
        client.post("/onboarding/start")
        client.post("/onboarding/answer", json={"answer": "I am a developer"})
        client.post("/onboarding/start")
        r = client.post("/onboarding/answer", json={"answer": "restarted"})
        # After restart, first question is submitted so we're at index 1
        assert r.json()["question_index"] == 1


# ---------------------------------------------------------------------------
# POST /onboarding/answer
# ---------------------------------------------------------------------------


class TestOnboardingAnswer:
    def test_empty_answer_400(self, client):
        client.post("/onboarding/start")
        r = client.post("/onboarding/answer", json={"answer": ""})
        assert r.status_code == 400

    def test_advances_question_index(self, client):
        client.post("/onboarding/start")
        r = client.post("/onboarding/answer", json={"answer": "I am a developer"})
        assert r.status_code == 200
        d = r.json()
        assert d["complete"] is False
        assert d["question_index"] == 1

    def test_progress_string(self, client):
        client.post("/onboarding/start")
        r = client.post("/onboarding/answer", json={"answer": "I am a developer"})
        assert r.json()["progress"] == "2/7"

    def test_full_ceremony_no_soul(self, client):
        """Completing without soul wired returns complete=True and seed=None."""
        client.post("/onboarding/start")
        answers = [
            "Developer", "Technical decisions", "Excellence",
            "Procrastination", "More productive", "Detail-oriented", "Nothing",
        ]
        result = None
        for ans in answers:
            result = client.post("/onboarding/answer", json={"answer": ans})
        assert result.status_code == 200
        d = result.json()
        assert d["complete"] is True
        assert "ceremony complete" in d["message"].lower()

    def test_ceremony_complete_with_soul(self, client):
        agent = _make_agent()
        agent._soul.has_seed.return_value = False
        seed_mock = MagicMock()
        seed_mock.stated_values = ["honesty", "growth"]
        seed_mock.stated_goals = ["build great things"]
        ceremony_mock = MagicMock()
        ceremony_mock.run_from_answers.return_value = seed_mock
        with _mock.patch("prism_routes_identity.IdentityCeremony", return_value=ceremony_mock, create=True):
            _set_state(agent=agent)
            client.post("/onboarding/start")
            answers = [
                "Developer", "Technical", "Excellence",
                "Procrastination", "Productive", "Detail-oriented", "Nothing",
            ]
            result = None
            for ans in answers:
                result = client.post("/onboarding/answer", json={"answer": ans})
        assert result.status_code == 200
        assert result.json()["complete"] is True

    def test_already_complete_400(self, client):
        client.post("/onboarding/start")
        answers = ["A", "B", "C", "D", "E", "F", "G"]
        for ans in answers:
            client.post("/onboarding/answer", json={"answer": ans})
        r = client.post("/onboarding/answer", json={"answer": "extra"})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# GET /federation/identity
# ---------------------------------------------------------------------------


class TestFederationIdentityGet:
    def test_no_agent_returns_null_soul(self, client):
        r = client.get("/federation/identity")
        assert r.status_code == 200
        d = r.json()
        assert d["soul"] is None
        assert d["persona"] is None

    def test_with_soul_exports(self, client):
        agent = _make_agent()
        agent._soul.export_json.return_value = {"beliefs": [{"text": "honesty", "confidence": 0.9}]}
        _set_state(agent=agent)
        r = client.get("/federation/identity")
        assert r.status_code == 200
        d = r.json()
        assert d["soul"] is not None
        assert len(d["soul"]["beliefs"]) == 1

    def test_with_persona_exports_traits(self, client):
        _set_state(agent=_make_agent())
        r = client.get("/federation/identity")
        d = r.json()
        assert d["persona"] is not None
        assert len(d["persona"]["traits"]) == 1
        assert d["persona"]["traits"][0]["name"] == "response_length"

    def test_timestamp_present(self, client):
        r = client.get("/federation/identity")
        d = r.json()
        assert "timestamp" in d
        assert d["timestamp"] > 0


# ---------------------------------------------------------------------------
# POST /federation/identity/merge
# ---------------------------------------------------------------------------


class TestFederationIdentityMerge:
    def test_empty_body_400(self, client):
        r = client.post("/federation/identity/merge", json={})
        assert r.status_code == 400

    def test_merge_no_agent_returns_ok(self, client):
        payload = {
            "node_id": "peer-123",
            "timestamp": time.time(),
            "soul": {"beliefs": [{"text": "courage", "confidence": 0.8, "belief_type": "value"}]},
            "persona": {"traits": []},
        }
        r = client.post("/federation/identity/merge", json=payload)
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert d["merged_beliefs"] == 0

    def test_merge_new_belief(self, client):
        agent = _make_agent()
        agent._soul.list_beliefs.return_value = []
        _set_state(agent=agent)
        payload = {
            "node_id": "peer-abc",
            "timestamp": time.time(),
            "soul": {"beliefs": [{"text": "courage", "confidence": 0.8, "belief_type": "value"}]},
            "persona": {"traits": []},
        }
        r = client.post("/federation/identity/merge", json=payload)
        assert r.status_code == 200
        d = r.json()
        assert d["merged_beliefs"] == 1
        agent._soul.add_belief.assert_called_once()

    def test_merge_higher_confidence_updates(self, client):
        agent = _make_agent()
        existing = MagicMock()
        existing.text = "courage"
        existing.node_id = "b-1"
        existing.confidence = 0.5
        agent._soul.list_beliefs.return_value = [existing]
        _set_state(agent=agent)

        payload = {
            "node_id": "peer-abc",
            "timestamp": time.time(),
            "soul": {"beliefs": [{"text": "courage", "confidence": 0.9, "belief_type": "value"}]},
            "persona": {"traits": []},
        }
        r = client.post("/federation/identity/merge", json=payload)
        assert r.status_code == 200
        agent._soul.update_belief.assert_called_once_with("b-1", 0.9, notes="federated from peer")

    def test_merge_lower_confidence_skipped(self, client):
        agent = _make_agent()
        existing = MagicMock()
        existing.text = "courage"
        existing.node_id = "b-1"
        existing.confidence = 0.95
        agent._soul.list_beliefs.return_value = [existing]
        _set_state(agent=agent)

        payload = {
            "node_id": "peer-abc",
            "timestamp": time.time(),
            "soul": {"beliefs": [{"text": "courage", "confidence": 0.5}]},
            "persona": {"traits": []},
        }
        r = client.post("/federation/identity/merge", json=payload)
        assert r.status_code == 200
        agent._soul.update_belief.assert_not_called()

    def test_merge_traits(self, client):
        agent = _make_agent()
        agent._persona.get_trait.return_value = None
        _set_state(agent=agent)

        payload = {
            "node_id": "peer-abc",
            "timestamp": time.time(),
            "soul": None,
            "persona": {"traits": [{"name": "risk_tolerance", "value": "moderate", "confidence": 0.7}]},
        }
        r = client.post("/federation/identity/merge", json=payload)
        assert r.status_code == 200
        d = r.json()
        assert d["merged_traits"] == 1

    def test_peer_node_id_echoed(self, client):
        payload = {
            "node_id": "peer-xyz",
            "timestamp": time.time(),
            "soul": None,
            "persona": None,
        }
        r = client.post("/federation/identity/merge", json=payload)
        assert r.json()["peer_node_id"] == "peer-xyz"
