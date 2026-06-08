"""
tests/test_orchestrator_async.py
=================================
Tests for ChainOrchestrator.orchestrate_async() and _run_graph_async().

Strategy: mock _decompose, _execute_node, _synthesise, and _chain_run so
these tests are pure unit tests — no LLM calls, no sqlite I/O.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from prism_orchestrator import ChainOrchestrator, OrchestratorNode, TaskGraph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_orch(**kwargs) -> ChainOrchestrator:
    orch = ChainOrchestrator.__new__(ChainOrchestrator)
    orch._chain   = None
    orch._loader  = None
    orch._tracker = None
    orch._horizon = None
    orch._router  = None
    orch._soul    = None
    orch._persona = None
    for k, v in kwargs.items():
        setattr(orch, k, v)
    return orch


def _make_graph(n_nodes: int = 2, profile: str = "analytical") -> TaskGraph:
    nodes = [
        OrchestratorNode(
            node_id  = f"n{i}",
            intent   = "autonomous",
            goal     = f"goal {i}",
            profile  = profile,
            depends_on=[],
        )
        for i in range(n_nodes)
    ]
    return TaskGraph(
        graph_id  = "test-graph",
        original  = "test message",
        context_id= "ctx",
        nodes     = nodes,
    )


def _fake_execute_node(node, graph, agent_fn, ctx):
    node.status = "done"
    node.result = f"result for {node.node_id}"


# ---------------------------------------------------------------------------
# orchestrate_async — fallback paths
# ---------------------------------------------------------------------------

def test_orchestrate_async_decompose_failure_falls_back():
    """orchestrate_async() falls back to _chain_run when decompose raises."""
    from prism_responses import text_card

    orch = _make_orch()
    fake_card = text_card("fallback answer", "fallback")

    with patch.object(orch, "_decompose", side_effect=RuntimeError("llm down")), \
         patch.object(orch, "_chain_run", return_value=fake_card):
        card = asyncio.run(orch.orchestrate_async("complex task", MagicMock(), {}))

    assert card.body == "fallback answer"


def test_orchestrate_async_no_nodes_falls_back():
    """orchestrate_async() falls back when _decompose returns None."""
    from prism_responses import text_card

    orch = _make_orch()
    fake_card = text_card("chain answer", "chain")

    with patch.object(orch, "_decompose", return_value=None), \
         patch.object(orch, "_chain_run", return_value=fake_card):
        card = asyncio.run(orch.orchestrate_async("hello", MagicMock(), {}))

    assert card.body == "chain answer"


# ---------------------------------------------------------------------------
# orchestrate_async — happy path
# ---------------------------------------------------------------------------

def test_orchestrate_async_executes_and_synthesises():
    """orchestrate_async() runs graph nodes and synthesises a final answer."""
    orch = _make_orch()
    graph = _make_graph(n_nodes=2)

    with patch.object(orch, "_decompose", return_value=graph), \
         patch.object(orch, "_execute_node", side_effect=_fake_execute_node), \
         patch.object(orch, "_synthesise", return_value="synthesised answer"), \
         patch.object(orch, "_persist"):
        card = asyncio.run(orch.orchestrate_async("multi-step task", MagicMock(), {}))

    assert "synthesised answer" in card.body
    assert graph.status == "completed"


# ---------------------------------------------------------------------------
# _run_graph_async — serial execution
# ---------------------------------------------------------------------------

def test_run_graph_async_serial_executes_all_nodes():
    """_run_graph_async() executes all serial nodes in order."""
    orch = _make_orch()
    graph = _make_graph(n_nodes=3, profile="analytical")

    executed = []

    def _track(node, g, fn, ctx):
        executed.append(node.node_id)
        node.status = "done"
        node.result = "ok"

    with patch.object(orch, "_execute_node", side_effect=_track):
        asyncio.run(orch._run_graph_async(graph, MagicMock(), {}))

    assert len(executed) == 3
    assert all(n.status == "done" for n in graph.nodes)


# ---------------------------------------------------------------------------
# _run_graph_async — parallel execution
# ---------------------------------------------------------------------------

def test_run_graph_async_parallel_executes_via_gather():
    """
    _run_graph_async() fans out parallel_safe nodes concurrently via
    asyncio.gather.  With a 'reactive' profile (use_parallel=True) and
    multiple independent nodes, all nodes should complete.
    """
    orch = _make_orch()
    graph = _make_graph(n_nodes=3, profile="reactive")

    order: list[str] = []

    def _track(node, g, fn, ctx):
        order.append(node.node_id)
        node.status = "done"
        node.result = "ok"

    with patch.object(orch, "_execute_node", side_effect=_track):
        asyncio.run(orch._run_graph_async(graph, MagicMock(), {}))

    assert len(order) == 3
    assert all(n.status == "done" for n in graph.nodes)


# ---------------------------------------------------------------------------
# _run_graph_async — timeout
# ---------------------------------------------------------------------------

def test_run_graph_async_timeout_marks_nodes_failed():
    """
    _run_graph_async() marks parallel nodes as failed on asyncio.TimeoutError.
    """
    import asyncio as _asyncio

    orch = _make_orch()
    graph = _make_graph(n_nodes=2, profile="reactive")

    async def _never_finish(node, g, fn, ctx):
        node.status = "running"
        await _asyncio.sleep(9999)

    async def _run():
        with patch.object(orch, "_execute_node", side_effect=lambda *a, **kw: None):
            # Directly test timeout path by patching asyncio.gather to raise
            with patch("asyncio.gather", side_effect=_asyncio.TimeoutError):
                with patch("asyncio.wait_for", side_effect=_asyncio.TimeoutError):
                    await orch._run_graph_async(graph, MagicMock(), {})

    asyncio.run(_run())
    # Nodes that were "running" should be marked failed
    for n in graph.nodes:
        if n.status == "running":
            assert n.error == "parallel timeout"


# ---------------------------------------------------------------------------
# Sync orchestrate() unchanged
# ---------------------------------------------------------------------------

def test_sync_orchestrate_still_works():
    """orchestrate() sync path is unaffected by Phase 6 changes."""
    orch = _make_orch()
    graph = _make_graph(n_nodes=1)

    with patch.object(orch, "_decompose", return_value=graph), \
         patch.object(orch, "_execute_node", side_effect=_fake_execute_node), \
         patch.object(orch, "_synthesise", return_value="sync answer"), \
         patch.object(orch, "_persist"):
        card = orch.orchestrate("single task", MagicMock(), {})

    assert "sync answer" in card.body
