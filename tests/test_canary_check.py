"""Tests for organs/canary_check.py — pipeline health probe."""
import time

import pytest

from prism_memory_graph import PrismMemoryGraph
from prism_metrics import PrismMetrics


@pytest.fixture()
def graph(tmp_path):
    g = PrismMemoryGraph(tmp_path / "g.db", tmp_path / "w.db")
    yield g
    g.close()


@pytest.fixture()
def metrics_inst(tmp_path, monkeypatch):
    inst = PrismMetrics(tmp_path / "metrics.db")
    monkeypatch.setattr("prism_metrics.metrics", inst)
    yield inst
    inst.close()


def _run_canary(graph, metrics_inst=None):
    from organs.canary_check import execute
    ctx = {"memory_graph": graph}
    return execute("canary_check", "check system health", ctx)


class TestCanaryExecute:
    def test_returns_card(self, graph, metrics_inst):
        card = _run_canary(graph, metrics_inst)
        assert card is not None

    def test_card_contains_ok(self, graph, metrics_inst):
        card = _run_canary(graph, metrics_inst)
        assert "OK" in str(card)

    def test_node_committed_to_cold_layer(self, graph, metrics_inst):
        _run_canary(graph, metrics_inst)
        # canary node should have been written and committed
        nodes = graph._cold.query_nodes(node_type="_canary")
        assert len(nodes) >= 1

    def test_canary_run_recorded_in_metrics(self, graph, metrics_inst):
        _run_canary(graph, metrics_inst)
        stats = metrics_inst.canary_stats()
        assert stats["n"] == 1
        assert stats["mean_ms"] is not None
        assert stats["mean_ms"] > 0

    def test_canary_counter_incremented(self, graph, metrics_inst):
        _run_canary(graph, metrics_inst)
        assert metrics_inst.get("canary_runs") == 1

    def test_multiple_runs_accumulate(self, graph, metrics_inst):
        for _ in range(3):
            _run_canary(graph, metrics_inst)
        assert metrics_inst.canary_stats()["n"] == 3
        assert metrics_inst.get("canary_runs") == 3

    def test_success_rate_one_on_clean_run(self, graph, metrics_inst):
        _run_canary(graph, metrics_inst)
        assert metrics_inst.canary_stats()["success_rate"] == pytest.approx(1.0)

    def test_duration_ms_is_positive(self, graph, metrics_inst):
        _run_canary(graph, metrics_inst)
        stats = metrics_inst.canary_stats()
        assert stats["mean_ms"] > 0

    def test_card_contains_duration(self, graph, metrics_inst):
        card = _run_canary(graph, metrics_inst)
        # card body should include ms timing
        assert "ms" in str(card)

    def test_rho_available_after_multiple_runs(self, graph, metrics_inst):
        for _ in range(3):
            _run_canary(graph, metrics_inst)
            time.sleep(0.001)
        rho = metrics_inst.performance_rho()
        assert rho is not None


class TestCanaryOrganMeta:
    def test_organ_meta_present(self):
        from organs.canary_check import ORGAN_META
        assert ORGAN_META["intent"] == "canary_check"
        assert ORGAN_META["capabilities"] == []

    def test_organ_policy_not_irreversible(self):
        from organs.canary_check import ORGAN_POLICY
        assert not ORGAN_POLICY["irreversible"]
        assert not ORGAN_POLICY["requires_approval"]
        assert ORGAN_POLICY["risk_level"] == "low"


class TestCanaryWithoutGraph:
    def test_creates_own_graph_when_ctx_empty(self, metrics_inst):
        from organs.canary_check import execute
        card = execute("canary_check", "health check", {})
        # Should succeed or fail gracefully — not raise
        assert card is not None
