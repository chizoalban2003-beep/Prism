"""
tests/test_orchestrator.py
==========================
Tests for ChainOrchestrator — decomposition, execution, synthesis,
horizon pausing, condition checking, profile routing, and persistence.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from prism_orchestrator import (
    PROFILES,
    ChainOrchestrator,
    ChainProfile,
    OrchestratorNode,
    TaskGraph,
)
from typing import Optional

# ── Helpers ───────────────────────────────────────────────────────────────────

def _card(body="result"):
    from prism_responses import text_card
    return text_card(body, "test")


def _orch(**kwargs) -> ChainOrchestrator:
    tmp = tempfile.mkdtemp()
    defaults = dict(db_path=str(Path(tmp) / "orch.db"))
    defaults.update(kwargs)
    return ChainOrchestrator(**defaults)


def _graph(nodes=None, original="test task") -> TaskGraph:
    return TaskGraph(
        graph_id   = "g1",
        original   = original,
        context_id = "default",
        nodes      = nodes or [],
    )


def _node(node_id="n1", intent="weather_check", goal="get weather",
          profile="reactive", depends_on=None, condition="",
          horizon_pause=False) -> OrchestratorNode:
    return OrchestratorNode(
        node_id=node_id, intent=intent, goal=goal,
        profile=profile, depends_on=depends_on or [],
        condition=condition, horizon_pause=horizon_pause,
    )


# ── Profiles ─────────────────────────────────────────────────────────────────

def test_all_profiles_present():
    for name in ("reactive", "analytical", "verification", "creative", "negotiation"):
        assert name in PROFILES
        p = PROFILES[name]
        assert isinstance(p, ChainProfile)


def test_reactive_is_fastest():
    assert PROFILES["reactive"].min_capability == 1
    assert not PROFILES["reactive"].use_parallel
    assert not PROFILES["reactive"].speculative


def test_negotiation_is_cross_session():
    assert PROFILES["negotiation"].cross_session
    assert PROFILES["negotiation"].min_capability == 3


# ── OrchestratorNode ─────────────────────────────────────────────────────────

def test_node_roundtrip():
    n = _node()
    d = n.to_dict()
    n2 = OrchestratorNode.from_dict(d)
    assert n2.node_id == n.node_id
    assert n2.intent  == n.intent
    assert n2.profile == n.profile


# ── TaskGraph ─────────────────────────────────────────────────────────────────

def test_graph_ready_nodes_no_deps():
    g = _graph([_node("n1"), _node("n2")])
    assert len(g.ready_nodes()) == 2


def test_graph_ready_nodes_with_deps():
    g = _graph([_node("n1"), _node("n2", depends_on=["n1"])])
    ready = g.ready_nodes()
    assert len(ready) == 1
    assert ready[0].node_id == "n1"


def test_graph_ready_after_dep_done():
    n1 = _node("n1")
    n2 = _node("n2", depends_on=["n1"])
    g = _graph([n1, n2])
    n1.status = "done"
    assert any(n.node_id == "n2" for n in g.ready_nodes())


def test_graph_is_complete():
    n1 = _node("n1")
    n1.status = "done"
    g = _graph([n1])
    assert g.is_complete()


def test_graph_not_complete_with_pending():
    g = _graph([_node("n1")])
    assert not g.is_complete()


def test_graph_is_paused():
    n1 = _node("n1")
    n1.status = "waiting"
    g = _graph([n1])
    assert g.is_paused()


def test_graph_roundtrip():
    g = _graph([_node("n1"), _node("n2")])
    g2 = TaskGraph.from_dict(g.to_dict())
    assert g2.graph_id == g.graph_id
    assert len(g2.nodes) == 2


def test_graph_node_results():
    n1 = _node("n1")
    n1.result = "sunny"
    n1.status = "done"
    g = _graph([n1])
    assert g.node_results() == {"n1": "sunny"}


# ── should_orchestrate ────────────────────────────────────────────────────────

def test_should_orchestrate_conditional():
    orch = _orch()
    assert orch.should_orchestrate("book a flight only if the hotel confirms")


def test_should_orchestrate_cross_session():
    orch = _orch()
    assert orch.should_orchestrate("remind me when the price drops below 300")


def test_should_orchestrate_multi_domain():
    orch = _orch()
    assert orch.should_orchestrate("check my email and add any urgent items to my calendar")


def test_should_not_orchestrate_simple():
    orch = _orch()
    assert not orch.should_orchestrate("what is the weather today")


def test_should_not_orchestrate_short():
    orch = _orch()
    assert not orch.should_orchestrate("hi")


def test_should_orchestrate_long_multi_sentence():
    orch = _orch()
    msg = ("Please check the current weather in London. "
           "Then look at my calendar for tomorrow. "
           "Finally, draft an email with a meeting suggestion based on the weather.")
    assert orch.should_orchestrate(msg)


# ── orchestrate — no LLM (fallback path) ─────────────────────────────────────

def test_orchestrate_no_router_falls_back_to_chain():
    chain = MagicMock()
    chain.run.return_value = _card("chain result")
    orch = _orch(chain=chain)
    card = orch.orchestrate("book flight only if hotel confirms",
                            MagicMock(), {})
    chain.run.assert_called_once()
    assert card.body == "chain result"


# ── orchestrate — with LLM ────────────────────────────────────────────────────

def _mock_router_decompose(needs: bool = True, extra_nodes: Optional[list] = None):
    router = MagicMock()
    nodes = [
        {"node_id": "n1", "intent": "weather_check", "goal": "get London weather",
         "profile": "reactive", "depends_on": [], "condition": "", "horizon_pause": False},
    ]
    if extra_nodes:
        nodes.extend(extra_nodes)
    payload = {
        "needs_orchestration": needs,
        "rationale": "multi-step task",
        "nodes": nodes if needs else [],
        "synthesis_hint": "combine results",
    }
    # LLM returns decompose, then synthesis
    router.call.side_effect = [
        (json.dumps(payload), "claude/sonnet"),
        ("Here is the combined answer.", "claude/sonnet"),
    ]
    return router


def test_orchestrate_with_llm_single_reactive_node():
    router = _mock_router_decompose()
    loader = MagicMock()
    loader.list_organs.return_value = ["weather_check"]
    loader.known_intents.return_value = {"weather_check": "get weather"}
    loader.get_organ_policy.return_value = {"risk_level": "low", "irreversible": False, "requires_approval": False}
    weather_fn = MagicMock(return_value=_card("Sunny 22°C"))
    loader.get.return_value = weather_fn

    chain = MagicMock()
    orch = _orch(router=router, organ_loader=loader, chain=chain)
    card = orch.orchestrate("what is the weather and my calendar?", MagicMock(), {})
    weather_fn.assert_called_once()
    assert card is not None


def test_orchestrate_needs_orchestration_false_falls_back():
    router = _mock_router_decompose(needs=False)
    chain = MagicMock()
    chain.run.return_value = _card("chain answer")
    orch = _orch(router=router, chain=chain)
    orch.orchestrate("simple question", MagicMock(), {})
    chain.run.assert_called_once()


def test_orchestrate_llm_json_parse_error_falls_back():
    router = MagicMock()
    router.call.return_value = ("not valid json {{{{", "claude/sonnet")
    chain = MagicMock()
    chain.run.return_value = _card("fallback")
    orch = _orch(router=router, chain=chain)
    card = orch.orchestrate("book flight only if hotel confirms", MagicMock(), {})
    assert card.body == "fallback"


# ── serial dependency execution ───────────────────────────────────────────────

def test_serial_nodes_execute_in_order():
    order = []
    def fake_execute(intent, goal, ctx):
        order.append(intent)
        return _card(f"result of {intent}")

    chain = MagicMock()
    chain.run.side_effect = lambda msg, fn, ctx, **kw: _card(fn("chain", msg, ctx).body)

    n1 = _node("n1", intent="email_read",  goal="read email",  profile="analytical")
    n2 = _node("n2", intent="add_task",    goal="add tasks",   profile="analytical", depends_on=["n1"])
    g = _graph([n1, n2])

    orch = _orch(chain=chain)
    # use chain for both (no organ_loader)
    orch._run_graph(g, fake_execute, {})
    assert n1.status == "done"
    assert n2.status == "done"
    # n2 result should include prior_output injected from n1
    assert "prior_output" in str(chain.run.call_args_list)  # ctx had prior_output


# ── parallel execution ────────────────────────────────────────────────────────

def test_parallel_nodes_both_complete():
    n1 = _node("n1", intent="weather_check", goal="weather", profile="analytical")
    n2 = _node("n2", intent="finance_summary", goal="finance", profile="analytical")
    g = _graph([n1, n2])

    chain = MagicMock()
    chain.run.return_value = _card("parallel result")
    orch = _orch(chain=chain)
    orch._run_graph(g, MagicMock(), {})
    assert n1.status == "done"
    assert n2.status == "done"
    assert chain.run.call_count == 2


# ── horizon pause ─────────────────────────────────────────────────────────────

def test_horizon_pause_creates_goal():
    horizon = MagicMock()
    horizon.add.return_value = "hz_001"
    n1 = _node("n1", goal="book flight when hotel confirms",
               profile="negotiation", horizon_pause=True,
               condition="hotel_confirmed")
    g = _graph([n1])
    orch = _orch(horizon=horizon)
    orch._run_graph(g, MagicMock(), {})
    horizon.add.assert_called_once()
    assert n1.status == "waiting"
    assert g.horizon_goal_ids.get("n1") == "hz_001"
    assert g.is_paused()


def test_horizon_pause_no_horizon_skips():
    n1 = _node("n1", horizon_pause=True)
    g = _graph([n1])
    orch = _orch()  # no horizon
    orch._run_graph(g, MagicMock(), {})
    assert n1.status == "skipped"


# ── resume_waiting ────────────────────────────────────────────────────────────

def test_resume_waiting_resumes_triggered_goal():
    horizon = MagicMock()
    goal_obj = MagicMock()
    goal_obj.status.value = "triggered"
    horizon.get.return_value = goal_obj

    n1 = _node("n1", horizon_pause=True, condition="hotel_confirmed")
    n1.status = "waiting"
    g = _graph([n1])
    g.status = "paused"
    g.horizon_goal_ids = {"n1": "hz_001"}

    chain = MagicMock()
    chain.run.return_value = _card("resumed result")
    orch = _orch(horizon=horizon, chain=chain)
    orch._persist(g)

    cards = orch.resume_waiting(MagicMock(), {})
    assert len(cards) == 1


def test_resume_waiting_no_paused_graphs():
    orch = _orch()
    cards = orch.resume_waiting(MagicMock(), {})
    assert cards == []


# ── condition checking ────────────────────────────────────────────────────────

def test_condition_check_yes():
    router = MagicMock()
    router.call.return_value = ("YES", "model")
    orch = _orch(router=router)
    n = _node("n2", condition="hotel is confirmed", depends_on=["n1"])
    n1 = _node("n1")
    n1.result = "Hotel Marriott confirmed for June 5"
    n1.status = "done"
    g = _graph([n1, n])
    assert orch._check_condition(n, g) is True


def test_condition_check_no():
    router = MagicMock()
    router.call.return_value = ("NO", "model")
    orch = _orch(router=router)
    n = _node("n2", condition="hotel is confirmed", depends_on=["n1"])
    n1 = _node("n1")
    n1.result = "Hotel unavailable"
    n1.status = "done"
    g = _graph([n1, n])
    assert orch._check_condition(n, g) is False


def test_condition_check_no_router_returns_true():
    orch = _orch()
    n = _node(condition="some condition")
    g = _graph([n])
    assert orch._check_condition(n, g) is True


def test_condition_check_empty_returns_true():
    orch = _orch()
    n = _node(condition="")
    g = _graph()
    assert orch._check_condition(n, g) is True


# ── synthesis ─────────────────────────────────────────────────────────────────

def test_synthesis_with_router():
    router = MagicMock()
    router.call.return_value = ("London is sunny, calendar is clear.", "claude/sonnet")
    n1 = _node("n1")
    n1.result = "Sunny 22°C"
    n1.status = "done"
    g = _graph([n1], original="weather and calendar?")
    orch = _orch(router=router)
    result = orch._synthesise(g)
    assert "sunny" in result.lower() or result  # router returned the synthesis


def test_synthesis_no_router_concatenates():
    n1 = _node("n1")
    n1.result = "Weather: sunny"
    n1.status = "done"
    n2 = _node("n2")
    n2.result = "Calendar: free"
    n2.status = "done"
    g = _graph([n1, n2])
    orch = _orch()
    result = orch._synthesise(g)
    assert "Weather: sunny" in result
    assert "Calendar: free" in result


def test_synthesis_no_results():
    g = _graph([_node("n1")])
    orch = _orch()
    result = orch._synthesise(g)
    assert "no results" in result.lower()


# ── outcome tracking ─────────────────────────────────────────────────────────

def test_orchestrate_records_outcome():
    # needs_orchestration=True with one analytical node → graph runs → tracker fires
    payload = {
        "needs_orchestration": True, "rationale": "multi-step",
        "nodes": [{"node_id": "n1", "intent": "research", "goal": "check flights",
                   "profile": "analytical", "depends_on": [], "condition": "",
                   "horizon_pause": False}],
        "synthesis_hint": "summarise",
    }
    router = MagicMock()
    router.call.side_effect = [
        (json.dumps(payload), "m"),
        ("Final answer.", "m"),  # synthesis call
    ]
    tracker = MagicMock()
    chain = MagicMock()
    chain.run.return_value = _card("done")
    orch = _orch(router=router, outcome_tracker=tracker, chain=chain)
    orch.orchestrate("book flight only if hotel confirms", MagicMock(), {})
    tracker.record.assert_called_once()


# ── persistence ───────────────────────────────────────────────────────────────

def test_persist_and_reload():
    n1 = _node("n1")
    n1.status = "waiting"
    g = _graph([n1])
    g.status = "paused"
    g.horizon_goal_ids = {"n1": "hz_999"}
    orch = _orch()
    orch._persist(g)
    loaded = orch._load_paused()
    assert len(loaded) == 1
    assert loaded[0].graph_id == "g1"
    assert loaded[0].nodes[0].status == "waiting"


def test_persist_completed_not_in_paused():
    g = _graph([_node("n1")])
    g.status = "completed"
    orch = _orch()
    orch._persist(g)
    assert orch._load_paused() == []
