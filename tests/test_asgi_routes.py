"""
FastAPI route tests using TestClient — no real HTTP port binding.
Covers all 11 router groups. Mock agent supplied via prism_state._set_state().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prism_asgi import app
from prism_state import _set_state

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@dataclass
class _FakeCard:
    body: str = "answer"
    title: str = "Title"
    card_data: dict = field(default_factory=dict)

    class _CT:
        value = "text"
    card_type: object = field(default_factory=_CT)

    def to_json(self):
        return {"body": self.body, "title": self.title}


@dataclass
class _FakeTask:
    task_id: str = "t1"
    title: str = "Test task"
    status: str = "done"
    progress: float = 1.0
    current_step: str = ""
    steps_done: int = 1
    steps_total: int = 1
    result: str = ""
    error: str = ""
    started_at: str = ""
    completed_at: str = ""


@pytest.fixture()
def client():
    agent = MagicMock()
    agent.status.return_value = {"phase": "STABLE", "version": "0.1"}
    agent.reflect.return_value = {"patterns": []}
    agent.identity.return_value = {"domains": []}
    agent.identity_domains.return_value = []
    agent.recent_artifacts.return_value = []
    agent.chat.return_value = _FakeCard()
    agent.ask.return_value = MagicMock(task="t", method="m", success=True,
                                        elapsed_ms=10.0, output="out")

    # task queue
    tq = MagicMock()
    tq.list_recent.return_value = [_FakeTask()]
    tq.get.return_value = _FakeTask()
    agent._task_queue = tq

    # chain
    chain = MagicMock()
    chain.MAX_STEPS = 8
    chain._db = "/tmp/chains.db"
    chain.recent_chains.return_value = []
    agent._chain = chain

    # horizon
    horizon = MagicMock()
    horizon.list_goals.return_value = []
    horizon.status.return_value = {"total": 0}
    agent._horizon = horizon

    # organ loader
    ol = MagicMock()
    ol.known_intents.return_value = {}
    ol.list_organs.return_value = []
    agent._organ_loader = ol

    # organ bus
    ob = MagicMock()
    ob.history.return_value = []
    ob._subscribers = []
    ob._lock = MagicMock().__enter__.return_value
    agent._organ_bus = ob

    # memory
    mem = MagicMock()
    mem.search.return_value = []
    mem.ingest.return_value = "entry-1"
    agent._memory = mem

    # LLM router
    llm = MagicMock()
    llm.status_summary.return_value = {"best": "none", "available": [], "stdlib_only": True}
    agent._router = llm

    _set_state(agent=agent, platform=None, task_queue=tq, llm_router=llm)
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health(client):
    r = client.get("/_health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


# ---------------------------------------------------------------------------
# Predict routes (no platform → 503)
# ---------------------------------------------------------------------------

def test_predict_match_no_platform(client):
    r = client.get("/predict/match")
    assert r.status_code == 503

def test_predict_injury_no_platform(client):
    r = client.get("/predict/injury")
    assert r.status_code == 503

def test_predict_performance_no_platform(client):
    r = client.get("/predict/performance")
    assert r.status_code == 503

def test_predict_transfer_no_platform(client):
    r = client.get("/predict/transfer")
    assert r.status_code == 503

def test_predict_brief_no_platform(client):
    r = client.get("/predict/brief")
    assert r.status_code == 503


def test_predict_match_with_platform(client):
    platform = MagicMock()
    platform.match.predict.return_value = MagicMock(
        home="A", away="B", home_win=0.5, draw=0.3, away_win=0.2, confidence=0.8
    )
    _set_state(platform=platform)
    r = client.get("/predict/match?home=Arsenal&away=Chelsea&sport=football")
    assert r.status_code == 200
    _set_state(platform=None)


# ---------------------------------------------------------------------------
# Agent routes
# ---------------------------------------------------------------------------

def test_status(client):
    with patch("urllib.request.urlopen", side_effect=Exception("no ollama")):
        r = client.get("/status")
    assert r.status_code == 200
    data = r.json()
    assert "phase" in data or "version" in data or "ollama" in data

def test_context_no_context_manager(client):
    r = client.get("/context")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Infra routes
# ---------------------------------------------------------------------------

def test_llm_status(client):
    r = client.get("/llm/status")
    assert r.status_code == 200
    assert "best" in r.json() or "available" in r.json()

def test_tasks_list(client):
    r = client.get("/tasks")
    assert r.status_code == 200
    data = r.json()
    assert "tasks" in data
    assert data["count"] >= 0

def test_tasks_by_id(client):
    r = client.get("/tasks/t1")
    assert r.status_code == 200
    data = r.json()
    assert data["task_id"] == "t1"

def test_tasks_invalid_n(client):
    r = client.get("/tasks?n=abc")
    assert r.status_code in (400, 422)

def test_metrics(client):
    with patch("prism_metrics.metrics") as m:
        m.report.return_value = {"psi": 0.0}
        r = client.get("/metrics")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Chain routes
# ---------------------------------------------------------------------------

def test_chain_recent(client):
    r = client.get("/chain/recent")
    assert r.status_code == 200
    assert "chains" in r.json()

def test_chain_status(client):
    r = client.get("/chain/status")
    assert r.status_code == 200

def test_organs(client):
    r = client.get("/organs")
    assert r.status_code == 200
    assert "organs" in r.json()

def test_organ_bus_history(client):
    r = client.get("/organ_bus/history")
    assert r.status_code == 200
    assert "signals" in r.json()

def test_organ_bus_subscribers(client):
    r = client.get("/organ_bus/subscribers")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Horizon routes
# ---------------------------------------------------------------------------

def test_horizon_goals(client):
    r = client.get("/horizon/goals")
    assert r.status_code == 200
    data = r.json()
    assert "goals" in data
    assert data["total"] == 0

def test_horizon_status(client):
    r = client.get("/horizon/status")
    assert r.status_code == 200
    assert "available" in r.json()

def test_horizon_goal_post_missing_fields(client):
    r = client.post("/horizon/goal", json={"intent": "learn"})
    assert r.status_code in (400, 422, 503)

def test_horizon_goal_post_no_horizon(client):
    agent = _state_get_agent()
    orig = agent._horizon
    agent._horizon = None
    r = client.post("/horizon/goal",
                    json={"intent": "learn", "trigger_condition": "when tired"})
    assert r.status_code in (503, 400, 422)
    agent._horizon = orig

def _state_get_agent():
    from prism_state import _state
    return _state.get("agent")


# ---------------------------------------------------------------------------
# Sensors routes
# ---------------------------------------------------------------------------

def test_memory_search_no_query(client):
    r = client.get("/memory/search")
    assert r.status_code in (400, 422)

def test_memory_search_with_query(client):
    r = client.get("/memory/search?q=test")
    assert r.status_code == 200
    assert "results" in r.json()

def test_memory_ingest_missing_content(client):
    r = client.post("/memory/ingest", json={})
    assert r.status_code in (400, 422)

def test_memory_ingest_ok(client):
    r = client.post("/memory/ingest", json={"content": "test note"})
    assert r.status_code == 200
    assert r.json().get("ok") is True

def test_perception_status_not_configured(client):
    r = client.get("/perception/status")
    assert r.status_code == 200

def test_proactive(client):
    r = client.get("/proactive")
    assert r.status_code == 200

def test_smarthome_status_not_configured(client):
    r = client.get("/smarthome/status")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Core routes
# ---------------------------------------------------------------------------

def test_chat_post(client):
    r = client.post("/chat", json={"message": "hello"})
    assert r.status_code == 200

def test_search_empty_query(client):
    r = client.get("/search?q=")
    assert r.status_code == 200

def test_device_capabilities(client):
    with patch("prism_device_agent.DeviceCapabilityScanner") as mock_scanner:
        caps = MagicMock()
        caps.platform = "linux"
        caps.has_browser = False
        caps.cli_tools = {}
        caps.py_packages = []
        caps.summary.return_value = "ok"
        mock_scanner.return_value.scan.return_value = caps
        r = client.get("/device/capabilities")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Integrations routes (not configured → safe fallback)
# ---------------------------------------------------------------------------

def test_email_status_not_configured(client):
    r = client.get("/email/status")
    assert r.status_code == 200

def test_calendar_status_not_configured(client):
    r = client.get("/calendar/status")
    assert r.status_code == 200

def test_browser_status_not_configured(client):
    r = client.get("/browser/status")
    assert r.status_code == 200

def test_instructions_empty(client):
    r = client.get("/instructions")
    assert r.status_code == 200
    assert "instructions" in r.json()

def test_discovery_services_empty(client):
    r = client.get("/discovery/services")
    assert r.status_code == 200

def test_push_status_not_configured(client):
    r = client.get("/push/status")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# UI routes
# ---------------------------------------------------------------------------

def test_ui_root(client):
    with patch("prism_chat.get_chat_html", return_value="<html>ok</html>"):
        r = client.get("/")
    assert r.status_code == 200
    assert "html" in r.headers.get("content-type", "").lower()

def test_ui_mobile(client):
    with patch("prism_pwa.get_mobile_html", return_value="<html>mobile</html>"):
        r = client.get("/mobile")
    assert r.status_code == 200

def test_ui_manifest(client):
    with patch("prism_pwa.get_manifest", return_value='{"name":"Prism"}'):
        r = client.get("/manifest.json")
    assert r.status_code == 200

def test_ui_sw(client):
    with patch("prism_pwa.get_service_worker", return_value="self.addEventListener('install',()=>{})"):
        r = client.get("/sw.js")
    assert r.status_code == 200

def test_ui_icon(client):
    with patch("prism_pwa.get_icon_svg", return_value="<svg/>"):
        r = client.get("/icon.svg")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Unknown route → 404
# ---------------------------------------------------------------------------

def test_unknown_route(client):
    r = client.get("/completely_unknown_route_xyz")
    assert r.status_code == 404

def test_unknown_post_route(client):
    r = client.post("/completely_unknown_route_xyz", json={})
    assert r.status_code == 404
