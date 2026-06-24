"""Failure-card fix for issue #28 bug 3 — orchestrator said "Task complete" on blocked synthesis.

Live test: a chain whose synthesis step errored returned the card
``"Task completed."`` because the previous code path was a flat
``text_card(graph.final_answer or "Task completed.", ...)`` at the
end of both :meth:`ChainOrchestrator.orchestrate` and
:meth:`ChainOrchestrator.orchestrate_async`. That ignored
``graph.status == "failed"`` entirely.

These tests pin the new :meth:`ChainOrchestrator._result_card` branch:
when status is "failed" the card title becomes ``[Failed] ...`` and the
body explains which sub-tasks failed (or fell back to partial-completion
copy) so the user is never told a partially-broken chain succeeded.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from prism_orchestrator import (
    ChainOrchestrator,
    OrchestratorNode,
    TaskGraph,
)


def _orch() -> ChainOrchestrator:
    tmp = tempfile.mkdtemp()
    return ChainOrchestrator(db_path=str(Path(tmp) / "orch.db"))


def _node(node_id="n1", status="done", error="", intent="weather_check") -> OrchestratorNode:
    n = OrchestratorNode(
        node_id=node_id, intent=intent, goal="x",
        profile="reactive", depends_on=[],
    )
    n.status = status
    n.error = error
    return n


def _graph(nodes, *, status="failed", final_answer="") -> TaskGraph:
    g = TaskGraph(
        graph_id="g1",
        original="run weather then summarise",
        context_id="default",
        nodes=nodes,
    )
    g.status = status
    g.final_answer = final_answer
    return g


class TestFailedGraphCard:
    def test_failed_status_never_says_task_completed(self):
        orch = _orch()
        g = _graph([_node("n1", status="failed", error="HTTP 500 from weather API")])
        card = orch._result_card(g, "what's the weather and summarise it")
        assert "Task completed." not in card.body
        assert "could not complete" in card.body.lower()

    def test_failed_card_title_prefixed_with_failed(self):
        orch = _orch()
        g = _graph([_node("n1", status="failed", error="boom")])
        card = orch._result_card(g, "do the thing")
        assert card.title.startswith("[Failed]")

    def test_failed_card_lists_failed_node_errors(self):
        orch = _orch()
        g = _graph([
            _node("n1", status="failed", error="HTTP 500 from weather API",
                  intent="weather_check"),
            _node("n2", status="failed", error="timeout after 30s",
                  intent="calendar_check"),
        ])
        card = orch._result_card(g, "test")
        assert "n1" in card.body
        assert "weather_check" in card.body
        assert "HTTP 500" in card.body
        assert "n2" in card.body
        assert "timeout after 30s" in card.body

    def test_failed_card_mentions_partial_completion(self):
        orch = _orch()
        g = _graph([
            _node("n1", status="done"),
            _node("n2", status="done"),
            _node("n3", status="failed", error="oops"),
        ])
        card = orch._result_card(g, "test")
        # Two nodes did complete — the card should say so.
        assert "2 step" in card.body

    def test_failed_card_with_no_error_info_still_renders(self):
        orch = _orch()
        g = _graph([_node("n1", status="failed", error="")])
        card = orch._result_card(g, "test")
        # Empty error string falls back to "no error info" rather than
        # producing a broken/empty bullet.
        assert "no error info" in card.body

    def test_failed_card_includes_partial_final_answer_when_present(self):
        orch = _orch()
        g = _graph(
            [_node("n1", status="failed", error="bad")],
            final_answer="Partial synthesis: weather data unavailable.",
        )
        card = orch._result_card(g, "test")
        assert "Partial synthesis" in card.body


class TestCompletedGraphCard:
    """Happy path must still work — completed graphs get the orchestrated card."""

    def test_completed_status_uses_orchestrated_title(self):
        orch = _orch()
        g = _graph([_node("n1", status="done")],
                   status="completed",
                   final_answer="It's sunny in London.")
        card = orch._result_card(g, "weather?")
        assert card.title.startswith("[Orchestrated]")
        assert "It's sunny in London." in card.body

    def test_completed_without_final_answer_falls_back(self):
        orch = _orch()
        g = _graph([_node("n1", status="done")],
                   status="completed",
                   final_answer="")
        card = orch._result_card(g, "test")
        # Original "Task completed." fallback survives for genuinely-completed
        # graphs that simply had no synthesiser output.
        assert "Task completed." in card.body
