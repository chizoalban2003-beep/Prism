"""
tests/test_causality.py
========================
Tests for CausalGraph, CausalReasoner, and /causality/* endpoints.
No LLM calls — llm_router is always None in these tests.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from prism_causality import CausalEdge, CausalGraph, CausalReasoner, CounterfactualResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def graph(tmp_path):
    return CausalGraph(db_path=str(tmp_path / "causality.db"))


@pytest.fixture()
def reasoner(graph):
    return CausalReasoner(graph=graph)


# ---------------------------------------------------------------------------
# CausalGraph — unit tests
# ---------------------------------------------------------------------------

def test_add_and_get_edges(graph):
    edge = graph.add_edge("sleep_debt", "performance_drop", strength=0.8, direction="positive")
    assert isinstance(edge, CausalEdge)
    assert edge.cause_id == "sleep_debt"
    assert edge.effect_id == "performance_drop"
    assert edge.strength == pytest.approx(0.8)
    assert edge.evidence_count == 1


def test_add_edge_increments_evidence_count(graph):
    graph.add_edge("A", "B", strength=0.5)
    graph.add_edge("A", "B", strength=0.7)  # same edge → upsert
    edges = graph.all_edges()
    assert len(edges) == 1
    assert edges[0].evidence_count == 2


def test_get_causes_and_effects(graph):
    graph.add_edge("X", "Y")
    graph.add_edge("X", "Z")
    graph.add_edge("W", "Y")

    causes_of_y = graph.get_causes("Y")
    assert {e.cause_id for e in causes_of_y} == {"X", "W"}

    effects_of_x = graph.get_effects("X")
    assert {e.effect_id for e in effects_of_x} == {"Y", "Z"}


def test_remove_edge_returns_true(graph):
    graph.add_edge("A", "B")
    removed = graph.remove_edge("A", "B")
    assert removed is True
    assert graph.all_edges() == []


def test_remove_nonexistent_edge_returns_false(graph):
    assert graph.remove_edge("no", "such") is False


def test_causal_chain_depth_limit(graph):
    # Chain: A → B → C → D
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    graph.add_edge("C", "D")

    chains = graph.causal_chain("A", depth=2)
    # With depth=2, max path length from A is 3 nodes (A→B→C)
    for path in chains:
        assert path[0] == "A"
        assert len(path) <= 3


def test_causal_chain_full_depth(graph):
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    graph.add_edge("C", "D")

    chains = graph.causal_chain("A", depth=5)
    # Should find path A→B→C→D
    all_paths = [p for chain in chains for p in [chain]]
    longest = max(all_paths, key=len)
    assert longest == ["A", "B", "C", "D"]


def test_detect_no_cycles_in_dag(graph):
    graph.add_edge("X", "Y")
    graph.add_edge("Y", "Z")
    assert graph.detect_cycles() == []


def test_cycle_detection_finds_cycle(graph):
    # Manually insert a cycle: A → B → C → A
    import time
    for cause, effect in [("A", "B"), ("B", "C"), ("C", "A")]:
        graph._conn.execute(
            "INSERT OR REPLACE INTO causal_edges "
            "(cause_id, effect_id, strength, direction, evidence_count, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (cause, effect, 0.5, "positive", 1, time.time()),
        )
    graph._conn.commit()
    cycles = graph.detect_cycles()
    assert len(cycles) >= 1
    # Each cycle path contains at least one node repeated
    cycle = cycles[0]
    assert len(cycle) >= 2


def test_strongest_causes(graph):
    graph.add_edge("A", "Z", strength=0.3)
    graph.add_edge("B", "Z", strength=0.9)
    graph.add_edge("C", "Z", strength=0.6)
    top = graph.strongest_causes("Z", top_n=2)
    assert len(top) == 2
    assert top[0].strength == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# CausalReasoner — unit tests
# ---------------------------------------------------------------------------

def test_explain_no_causes(reasoner):
    text = reasoner.explain("orphan_belief")
    assert "no registered causal" in text.lower() or "orphan_belief" in text


def test_explain_with_causes(graph, reasoner):
    graph.add_edge("stress", "insomnia", strength=0.75)
    text = reasoner.explain("insomnia")
    assert "insomnia" in text or "stress" in text
    assert "0.75" in text


def test_counterfactual_returns_result(graph, reasoner):
    graph.add_edge("sleep_debt", "fatigue")
    graph.add_edge("fatigue", "errors")
    result = reasoner.counterfactual(
        "What if sleep debt didn't exist?",
        remove_belief_id="sleep_debt",
    )
    assert isinstance(result, CounterfactualResult)
    assert result.query == "What if sleep debt didn't exist?"
    assert isinstance(result.changed_beliefs, list)
    assert 0.0 <= result.confidence <= 1.0


def test_counterfactual_no_downstream_chain(reasoner):
    result = reasoner.counterfactual("what if?", remove_belief_id="isolated_node")
    assert isinstance(result, CounterfactualResult)
    assert result.changed_beliefs == []


def test_infer_edges_from_soul_no_soul(reasoner):
    assert reasoner.infer_edges_from_soul(None) == 0


def test_infer_edges_from_soul_with_mock(graph, reasoner):
    from unittest.mock import MagicMock
    edge = MagicMock()
    edge.relation = "supports"
    edge.strength = 0.6
    edge.from_id  = "cause_belief"
    edge.to_id    = "effect_belief"

    soul = MagicMock()
    soul.list_edges.return_value = [edge]

    added = reasoner.infer_edges_from_soul(soul)
    assert added == 1
    effects = graph.get_effects("cause_belief")
    assert any(e.effect_id == "effect_belief" for e in effects)


def test_build_explanation_tree(graph, reasoner):
    graph.add_edge("root", "child_a")
    graph.add_edge("root", "child_b")
    graph.add_edge("parent", "root")

    tree = reasoner.build_explanation_tree("root")
    assert tree["belief_id"] == "root"
    assert isinstance(tree["causes"], list)
    assert isinstance(tree["effects"], list)
    effect_ids = {e["belief_id"] for e in tree["effects"]}
    assert {"child_a", "child_b"}.issubset(effect_ids)
    cause_ids = {e["belief_id"] for e in tree["causes"]}
    assert "parent" in cause_ids


# ---------------------------------------------------------------------------
# /causality/* endpoints
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path):
    from prism_asgi import app
    from prism_state import _set_state
    cg = CausalGraph(db_path=str(tmp_path / "test_causality.db"))
    cr = CausalReasoner(graph=cg)
    _set_state(causal_reasoner=cr)
    return TestClient(app, raise_server_exceptions=False)


def test_causality_graph_endpoint_empty(client):
    r = client.get("/causality/graph")
    assert r.status_code == 200
    assert r.json()["edges"] == []


def test_causality_add_edge_endpoint(client):
    r = client.post("/causality/edges", json={
        "cause_id": "stress", "effect_id": "insomnia", "strength": 0.8,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["cause_id"] == "stress"
    assert data["effect_id"] == "insomnia"


def test_causality_graph_after_add(client):
    client.post("/causality/edges", json={"cause_id": "A", "effect_id": "B"})
    r = client.get("/causality/graph")
    assert r.status_code == 200
    assert len(r.json()["edges"]) >= 1


def test_causality_explain_endpoint(client):
    client.post("/causality/edges", json={"cause_id": "X", "effect_id": "Y"})
    r = client.get("/causality/explain/Y")
    assert r.status_code == 200
    data = r.json()
    assert "explanation" in data


def test_causality_counterfactual_endpoint(client):
    client.post("/causality/edges", json={"cause_id": "P", "effect_id": "Q"})
    r = client.post("/causality/counterfactual", json={
        "query": "What if P didn't exist?",
        "remove_belief_id": "P",
    })
    assert r.status_code == 200
    data = r.json()
    assert "counterfactual_outcome" in data
    assert "confidence" in data


def test_causality_chain_endpoint(client):
    client.post("/causality/edges", json={"cause_id": "A", "effect_id": "B"})
    client.post("/causality/edges", json={"cause_id": "B", "effect_id": "C"})
    r = client.get("/causality/chain/A")
    assert r.status_code == 200
    assert "chains" in r.json()


def test_causality_no_state_503():
    from prism_asgi import app
    from prism_state import _set_state
    _set_state(causal_reasoner=None)
    c = TestClient(app, raise_server_exceptions=False)
    r = c.get("/causality/graph")
    assert r.status_code == 503


def test_causality_infer_endpoint(client):
    r = client.post("/causality/infer")
    assert r.status_code == 200
    assert "edges_added" in r.json()
