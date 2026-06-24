"""Synthesis-budget fix for issue #28 bug 4 — 3000-char cap silently dropped tail nodes.

Live test: a 5-node chain whose late nodes produced rich output had
those nodes silently truncated from the synthesis prompt because the
old code did ``results_text[:3000]`` on the whole concatenation. The
user got a synthesis that only referenced the first two or three nodes.

The fix has two pieces, both pinned here:

  1. Per-node budget so a chatty node can't monopolise the prompt — each
     node gets ``max(800, TOTAL // n_done)`` chars, the rest is dropped
     with an explicit ``[truncated]`` marker.
  2. Total budget raised from 3000 to 8000 chars so a typical 3-5 node
     chain has room to land without truncation.

We also pin the no-router fallback path so it still returns the
concatenated results unchanged.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from prism_orchestrator import (
    _SYNTHESIS_TOTAL_BUDGET,
    ChainOrchestrator,
    OrchestratorNode,
    TaskGraph,
)


def _orch(router=None) -> ChainOrchestrator:
    tmp = tempfile.mkdtemp()
    return ChainOrchestrator(db_path=str(Path(tmp) / "orch.db"), router=router)


def _done_node(node_id: str, result: str, intent: str = "x") -> OrchestratorNode:
    n = OrchestratorNode(
        node_id=node_id, intent=intent, goal="g",
        profile="reactive", depends_on=[],
    )
    n.status = "done"
    n.result = result
    return n


def _graph(nodes) -> TaskGraph:
    return TaskGraph(
        graph_id="g1",
        original="o",
        context_id="default",
        nodes=nodes,
    )


class TestPerNodeBudget:
    def test_chatty_node_does_not_monopolise(self):
        """Five nodes, one of them very chatty. Synthesis prompt must
        include all five — the chatty node gets truncated, the quiet ones
        survive intact."""
        captured: dict = {}

        def _fake_call(prompt: str, **kwargs):
            captured["prompt"] = prompt
            return ("synth", {"provider": "mock"})

        router = MagicMock()
        router.call.side_effect = _fake_call
        orch = _orch(router=router)
        nodes = [
            _done_node("n1", "A" * 20000, intent="chatty"),   # would have swallowed 3000-char budget alone
            _done_node("n2", "BBB quiet result two"),
            _done_node("n3", "CCC quiet result three"),
            _done_node("n4", "DDD quiet result four"),
            _done_node("n5", "EEE quiet result five"),
        ]
        orch._synthesise(_graph(nodes))
        prompt = captured["prompt"]
        # All five node markers reach the prompt — that's the regression.
        for marker in ("[n1 — chatty]", "[n2 — x]", "[n3 — x]", "[n4 — x]", "[n5 — x]"):
            assert marker in prompt, f"missing {marker} in synthesis prompt"
        # Quiet nodes survive intact.
        assert "BBB quiet result two" in prompt
        assert "EEE quiet result five" in prompt
        # Chatty node gets the truncation marker.
        assert "[truncated]" in prompt

    def test_total_budget_constant_matches_docstring_claim(self):
        # Pin the constant so a casual edit to a smaller value triggers a
        # test failure rather than silently re-introducing the bug.
        assert _SYNTHESIS_TOTAL_BUDGET >= 8000


class TestNoRouterFallback:
    def test_concatenated_results_returned_without_router(self):
        orch = _orch(router=None)
        nodes = [
            _done_node("n1", "alpha"),
            _done_node("n2", "beta"),
        ]
        out = orch._synthesise(_graph(nodes))
        assert "alpha" in out and "beta" in out
        assert "[n1" in out and "[n2" in out

    def test_no_done_nodes_returns_explanatory_string(self):
        orch = _orch(router=None)
        n = OrchestratorNode(
            node_id="n1", intent="x", goal="g",
            profile="reactive", depends_on=[],
        )
        n.status = "failed"
        n.result = ""
        out = orch._synthesise(_graph([n]))
        assert "no results to synthesise" in out
