"""
CI Performance Gate — TAD §8

Asserts that the full write→WAL→commit→read round-trip completes within SLO_MS.
Break-glass: create DEBT_WAIVER.json with {"skip_perf_gate": true, "reason": "..."}.

SLO is configurable via PRISM_PERF_SLO_MS env var (default 1000 ms).
Production target is 500 ms; CI environments may need the higher default
due to VM/coverage overhead and cold SQLite startup.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

# SLO: configurable via env so CI can tune without code changes.
# Default 1000 ms covers cold SQLite start + coverage instrumentation overhead.
SLO_MS = float(os.environ.get("PRISM_PERF_SLO_MS", "1000"))

_WAIVER_PATH = Path(__file__).parent.parent / "DEBT_WAIVER.json"


def _waiver_active() -> bool:
    if not _WAIVER_PATH.exists():
        return False
    try:
        data = json.loads(_WAIVER_PATH.read_text())
        return bool(data.get("skip_perf_gate"))
    except Exception:
        return False


@pytest.fixture()
def graph(tmp_path):
    from prism_memory_graph import PrismMemoryGraph
    g = PrismMemoryGraph(
        db_path=tmp_path / "cold.db",
        wal_path=tmp_path / "wal.db",
    )
    yield g
    g.close()


class TestPerformanceGate:
    def test_canary_round_trip_within_slo(self, graph):
        """Full write→WAL→commit→read round-trip must complete under SLO_MS."""
        if _waiver_active():
            pytest.skip("DEBT_WAIVER.json: skip_perf_gate is active")

        from prism_memory_graph import GraphNode

        node = GraphNode(
            node_id="perf_canary",
            node_type="_perf_gate",
            value={"probe": True},
            ts=time.time(),
        )

        t0 = time.monotonic()
        graph.write_node(node)
        graph.commit_pending()
        found = graph._cold.get_node("perf_canary")
        elapsed_ms = (time.monotonic() - t0) * 1000

        assert found is not None, "Canary node not found after commit"
        assert elapsed_ms < SLO_MS, (
            f"Round-trip took {elapsed_ms:.1f} ms — exceeds SLO of {SLO_MS} ms. "
            f"Create DEBT_WAIVER.json with skip_perf_gate=true to bypass."
        )

    def test_ten_sequential_writes_within_slo(self, graph):
        """10 sequential node writes + single commit must stay under 3× SLO."""
        if _waiver_active():
            pytest.skip("DEBT_WAIVER.json: skip_perf_gate is active")

        from prism_memory_graph import GraphNode

        t0 = time.monotonic()
        for i in range(10):
            graph.write_node(GraphNode(
                node_id=f"perf_{i}",
                node_type="_perf_gate",
                value={"i": i},
                ts=time.time(),
            ))
        graph.commit_pending()
        elapsed_ms = (time.monotonic() - t0) * 1000

        assert elapsed_ms < SLO_MS * 3, (
            f"10-write batch took {elapsed_ms:.1f} ms — exceeds 3×SLO ({SLO_MS * 3} ms)"
        )

    def test_consistency_psi_zero_after_commit(self, graph):
        """Ψ must reach 0 after commit — invariant for pipeline correctness."""
        from prism_memory_graph import GraphNode

        graph.write_node(GraphNode(
            node_id="psi_check",
            node_type="_perf_gate",
            value={},
            ts=time.time(),
        ))
        assert graph.consistency_psi() > 0
        graph.commit_pending()
        assert graph.consistency_psi() == 0
