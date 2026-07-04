"""Tests for the conversation-memory -> WAL graph durability bridge."""
from __future__ import annotations

import types

from prism_agent import PrismAgent
from prism_memory import PrismMemory
from prism_memory_graph import PrismMemoryGraph


def _graph(tmp_path) -> PrismMemoryGraph:
    return PrismMemoryGraph(
        db_path=str(tmp_path / "graph.db"),
        wal_path=str(tmp_path / "wal.db"),
    )


# ── PrismMemory.ingest_conversation now returns the entry id ─────────────────────

def test_ingest_conversation_returns_entry_id(tmp_path):
    mem = PrismMemory(db_path=str(tmp_path / "mem.db"))
    eid = mem.ingest_conversation("user", "x" * 60)
    assert isinstance(eid, str) and eid


def test_ingest_conversation_short_returns_none(tmp_path):
    mem = PrismMemory(db_path=str(tmp_path / "mem.db"))
    assert mem.ingest_conversation("user", "hi") is None


# ── _mirror_turn_to_graph (tested without a full agent) ──────────────────────────

def test_mirror_writes_node(tmp_path):
    g = _graph(tmp_path)
    ns = types.SimpleNamespace(_memory_graph=g)
    PrismAgent._mirror_turn_to_graph(ns, "user", "hello there friend", "abc123")
    node = g.aggregator.get_node("conv_abc123")
    assert node is not None
    assert node.node_type == "observation"
    assert node.value["role"] == "user"
    assert node.value["source"] == "conversation"
    # The write went through the WAL → pending until the shadow pipeline commits.
    assert g.consistency_psi() > 0


def test_mirror_links_user_and_assistant(tmp_path):
    g = _graph(tmp_path)
    ns = types.SimpleNamespace(_memory_graph=g)
    PrismAgent._mirror_turn_to_graph(ns, "user", "a question", "u1")
    PrismAgent._mirror_turn_to_graph(ns, "assistant", "an answer", "a1")
    edges = g.aggregator.edges_for("conv_u1")
    assert any(e.dst == "conv_a1" and e.relation == "answered_by" for e in edges)


def test_graph_recall_merges_into_context(tmp_path):
    g = _graph(tmp_path)
    ns = types.SimpleNamespace(_memory_graph=g)
    # seed a past turn in the graph
    PrismAgent._mirror_turn_to_graph(ns, "user", "the migration must finish first", "p1")
    ctx: dict = {}
    PrismAgent._graph_recall(ns, "tell me about the migration", ctx)
    mc = ctx.get("memory_context", [])
    assert any("migration" in (e.get("excerpt", "")) for e in mc)


def test_graph_recall_noop_without_graph():
    ns = types.SimpleNamespace()
    ctx: dict = {}
    PrismAgent._graph_recall(ns, "anything", ctx)  # must not raise
    assert ctx == {}


def test_mirror_noop_without_graph():
    ns = types.SimpleNamespace()  # no _memory_graph
    # Must not raise.
    PrismAgent._mirror_turn_to_graph(ns, "user", "x" * 60, "id1")


def test_mirror_noop_without_entry_id(tmp_path):
    g = _graph(tmp_path)
    ns = types.SimpleNamespace(_memory_graph=g)
    PrismAgent._mirror_turn_to_graph(ns, "user", "content", None)
    assert g.consistency_psi() == 0


# ── End-to-end: chat() drives the bridge, pipeline commits, recovery works ───────

def test_chat_populates_graph_and_recovers(tmp_path, offline_llm):
    # offline_llm: the statement below routes to the chat LLM path; on a
    # dev machine with a slow local model that's a real multi-minute
    # generation (timed out locally while fail-fast passing in CI). The
    # graph mirroring under test happens regardless of what the LLM says.
    agent = PrismAgent()
    g = _graph(tmp_path)
    agent._memory_graph = g
    # A plain statement (not a "remember/always" standing instruction, which
    # short-circuits chat() before the memory step) that exceeds 50 chars.
    long_msg = "the quarterly numbers looked unusual across several regions this period"
    agent.chat(long_msg, {"source": "test"})

    # The user turn (and likely the assistant turn) are now in the graph WAL.
    assert g.consistency_psi() > 0

    # Commit drains hot -> cold; then a fresh graph on the same files recovers.
    committed = g.commit_pending()
    assert committed > 0
    assert g.consistency_psi() == 0

    g2 = PrismMemoryGraph(
        db_path=str(tmp_path / "graph.db"),
        wal_path=str(tmp_path / "wal.db"),
    )
    nodes = g2.aggregator.query_nodes(node_type="observation", limit=50)
    assert any(n.value.get("source") == "conversation" for n in nodes)
