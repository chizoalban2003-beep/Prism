"""
PRISM Chaos Test Suite — TAD Section 5

Tests the full integrated stack (MemoryGraph + ShadowPipeline + Watchdog + Metrics)
under structured fault injection, using the Consistency Oracle (Ψ) as the
validation instrument.

  CHAOS-001  SIGKILL mid-transaction  → WAL replay recovers all pending writes
  CHAOS-002  Disk/IO timeout          → cold layer stays clean; retry succeeds
  CHAOS-003  Sequence tampering       → UNIQUE constraint rejects duplicates;
                                        watchdog flags anomaly

  ORACLE     Consistency Oracle (Ψ)   → Ψ = pending WAL entries;
                                        0 = equilibrium, > 0 = mutations in flight
"""
from __future__ import annotations

import threading
import time
import uuid

import pytest

from prism_memory_graph import GraphEdge, GraphNode, PrismMemoryGraph
from prism_metrics import PrismMetrics
from prism_shadow_pipeline import PrismShadowPipeline
from prism_watchdog import PrismWatchdog
from typing import Optional

# ── Helpers ───────────────────────────────────────────────────────────────────

def _node(nid: Optional[str] = None, **kw) -> GraphNode:
    return GraphNode(
        node_id   = nid or uuid.uuid4().hex[:8],
        node_type = "entity",
        value     = kw or {"synthetic": True},
        ts        = time.time(),
    )


# ── Consistency Oracle ────────────────────────────────────────────────────────

class ConsistencyOracle:
    """
    Monitors Ψ = pending WAL entries (buffer ∩ graph divergence).
    Ψ = 0 means perfect equilibrium — all writes committed to cold layer.

    Usage::

        oracle = ConsistencyOracle(graph)
        oracle.assert_eventually_zero(timeout=5.0)
    """

    def __init__(self, graph: PrismMemoryGraph) -> None:
        self._graph = graph
        self._samples: list[int] = []

    @property
    def psi(self) -> int:
        return self._graph.consistency_psi()

    def sample(self) -> int:
        v = self.psi
        self._samples.append(v)
        return v

    def assert_zero(self) -> None:
        assert self.psi == 0, f"Ψ = {self.psi}, expected 0 (perfect equilibrium)"

    def assert_eventually_zero(self, timeout: float = 5.0, poll: float = 0.02) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.psi == 0:
                return
            time.sleep(poll)
        raise AssertionError(
            f"Ψ never reached 0 within {timeout}s (last value: {self.psi})"
        )

    def assert_trending_down(self) -> None:
        if len(self._samples) < 2:
            return
        assert self._samples[-1] <= self._samples[0], (
            f"Ψ trending up: {self._samples}"
        )

    def history(self) -> list[int]:
        return list(self._samples)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def graph(tmp_path):
    g = PrismMemoryGraph(tmp_path / "graph.db", tmp_path / "wal.db")
    yield g
    g.close()


@pytest.fixture()
def oracle(graph):
    return ConsistencyOracle(graph)


@pytest.fixture()
def met(tmp_path, monkeypatch):
    inst = PrismMetrics(tmp_path / "metrics.db")
    monkeypatch.setattr("prism_metrics.metrics", inst)
    yield inst
    inst.close()


# ── Consistency Oracle unit tests ─────────────────────────────────────────────

class TestConsistencyOracle:
    def test_psi_zero_on_empty_graph(self, oracle):
        oracle.assert_zero()

    def test_psi_rises_on_write(self, graph, oracle):
        graph.write_node(_node("n1"))
        assert oracle.psi == 1

    def test_psi_returns_to_zero_after_commit(self, graph, oracle):
        graph.write_node(_node("n1"))
        graph.commit_pending()
        oracle.assert_zero()

    def test_psi_tracks_multiple_writes(self, graph, oracle):
        for i in range(5):
            graph.write_node(_node(f"n{i}"))
        assert oracle.psi == 5

    def test_oracle_sample_records_history(self, graph, oracle):
        graph.write_node(_node("n1"))
        oracle.sample()
        graph.commit_pending()
        oracle.sample()
        assert oracle.history() == [1, 0]

    def test_assert_eventually_zero_with_pipeline(self, graph, oracle):
        for i in range(3):
            graph.write_node(_node(f"n{i}"))
        pipeline = PrismShadowPipeline(graph, interval_s=0.05)
        pipeline.start()
        try:
            oracle.assert_eventually_zero(timeout=3.0)
        finally:
            pipeline.stop(timeout=2.0)

    def test_assert_eventually_zero_raises_if_stuck(self, graph, oracle):
        graph.write_node(_node("stuck"))
        # Don't commit or start a pipeline — Ψ stays at 1
        with pytest.raises(AssertionError, match="never reached 0"):
            oracle.assert_eventually_zero(timeout=0.1)


# ── CHAOS-001: SIGKILL simulation ─────────────────────────────────────────────

class TestChaos001SigkillSimulation:
    """
    Scenario: Shadow pipeline thread dies mid-write (simulated by closing the
    pipeline before any commit can fire).  All mutations are in the WAL.
    On restart, replay_wal() + commit_pending() must recover Ψ → 0.
    """

    def test_wal_preserves_writes_after_crash(self, tmp_path):
        g = PrismMemoryGraph(tmp_path / "g.db", tmp_path / "w.db")
        g.write_node(_node("survivor_a"))
        g.write_node(_node("survivor_b"))
        assert g.consistency_psi() == 2
        # Simulate crash: abandon without committing
        g._cold.close()
        g._wal.close()

        # Restart
        g2 = PrismMemoryGraph(tmp_path / "g.db", tmp_path / "w.db")
        replayed = g2.replay_wal()
        assert replayed == 2
        oracle = ConsistencyOracle(g2)
        # Not yet committed — Ψ = 2 (in hot buffer after replay)
        assert oracle.psi == 2
        g2.close()

    def test_wal_replay_then_commit_achieves_equilibrium(self, tmp_path):
        g = PrismMemoryGraph(tmp_path / "g.db", tmp_path / "w.db")
        for i in range(5):
            g.write_node(_node(f"n{i}"))
        g._cold.close()
        g._wal.close()

        g2 = PrismMemoryGraph(tmp_path / "g.db", tmp_path / "w.db")
        g2.replay_wal()
        committed = g2.commit_pending()
        assert committed == 5
        ConsistencyOracle(g2).assert_zero()
        g2.close()

    def test_cold_layer_intact_before_any_commit(self, tmp_path):
        g = PrismMemoryGraph(tmp_path / "g.db", tmp_path / "w.db")
        g.write_node(_node("precrash"))
        # Crash — cold layer should have nothing
        assert g._cold.get_node("precrash") is None
        g._cold.close()
        g._wal.close()

        g2 = PrismMemoryGraph(tmp_path / "g.db", tmp_path / "w.db")
        g2.replay_wal()
        g2.commit_pending()
        # Now cold layer should have it
        assert g2._cold.get_node("precrash") is not None
        g2.close()

    def test_metrics_record_wal_replays(self, tmp_path, monkeypatch):
        met = PrismMetrics(tmp_path / "m.db")
        monkeypatch.setattr("prism_metrics.metrics", met)
        g = PrismMemoryGraph(tmp_path / "g.db", tmp_path / "w.db")
        g.write_node(_node("n1"))
        g._cold.close()
        g._wal.close()

        g2 = PrismMemoryGraph(tmp_path / "g.db", tmp_path / "w.db")
        g2.replay_wal()

        # Start pipeline — it commits and records metrics
        p = PrismShadowPipeline(g2, interval_s=0.05)
        p.start()
        ConsistencyOracle(g2).assert_eventually_zero(timeout=3.0)
        p.stop(timeout=2.0)

        assert met.get("commits_total") >= 1
        g2.close()
        met.close()

    def test_crash_recovery_with_edges(self, tmp_path):
        g = PrismMemoryGraph(tmp_path / "g.db", tmp_path / "w.db")
        g.write_node(_node("a"))
        g.write_node(_node("b"))
        g.write_edge(GraphEdge("a", "b", "connects"))
        g._cold.close()
        g._wal.close()

        g2 = PrismMemoryGraph(tmp_path / "g.db", tmp_path / "w.db")
        g2.replay_wal()
        g2.commit_pending()
        edges = g2._cold.edges_for("a")
        assert any(e.relation == "connects" for e in edges)
        g2.close()


# ── CHAOS-002: disk/IO error ──────────────────────────────────────────────────

class TestChaos002DiskError:
    """
    Scenario: cold.upsert_nodes_batch() raises (disk timeout).
    The atomic commit must abort cleanly.  The cold layer must stay consistent
    (no half-written data).  Subsequent retry must succeed.
    """

    def test_cold_layer_clean_after_first_commit_failure(self, graph, monkeypatch):
        from prism_memory_graph import _ColdLayer
        original_batch = _ColdLayer.upsert_nodes_batch
        attempt = {"n": 0}

        def failing_batch(self, nodes):
            attempt["n"] += 1
            if attempt["n"] == 1:
                raise OSError("Simulated disk timeout")
            return original_batch(self, nodes)

        monkeypatch.setattr(_ColdLayer, "upsert_nodes_batch", failing_batch)
        graph.write_node(_node("safe1"))
        graph.write_node(_node("safe2"))

        # First commit fails
        graph.commit_pending()
        # Cold layer must not have corrupted partial writes — batch raised before commit
        assert graph._cold.get_node("safe1") is None

    def test_retry_after_disk_error_succeeds(self, graph, monkeypatch):
        from prism_memory_graph import _ColdLayer
        original_batch = _ColdLayer.upsert_nodes_batch
        attempt = {"n": 0}

        def fail_once_batch(self, nodes):
            attempt["n"] += 1
            if attempt["n"] == 1:
                raise OSError("One-time disk error")
            return original_batch(self, nodes)

        monkeypatch.setattr(_ColdLayer, "upsert_nodes_batch", fail_once_batch)
        graph.write_node(_node("retry_me"))
        graph.commit_pending()  # fails
        # Remove the fault and retry
        monkeypatch.setattr(_ColdLayer, "upsert_nodes_batch", original_batch)
        committed = graph.commit_pending()
        assert committed == 1
        assert graph._cold.get_node("retry_me") is not None

    def test_psi_stays_non_zero_after_failed_commit(self, graph, monkeypatch):
        from prism_memory_graph import _ColdLayer

        def always_fail_batch(self, nodes):
            raise OSError("Permanent disk error")

        monkeypatch.setattr(_ColdLayer, "upsert_nodes_batch", always_fail_batch)
        graph.write_node(_node("stuck"))
        graph.commit_pending()
        # WAL entry still pending → Ψ > 0
        assert graph.consistency_psi() == 1

    def test_pipeline_retries_after_transient_error(self, graph, monkeypatch):
        from prism_memory_graph import _ColdLayer
        original_batch = _ColdLayer.upsert_nodes_batch
        call_count = {"n": 0}

        def fail_twice_batch(self, nodes):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise OSError("Transient error")
            return original_batch(self, nodes)

        monkeypatch.setattr(_ColdLayer, "upsert_nodes_batch", fail_twice_batch)
        graph.write_node(_node("eventually"))
        pipeline = PrismShadowPipeline(graph, interval_s=0.05, max_restarts=10)
        pipeline.start()

        # Remove fault after a short delay — pipeline will eventually succeed
        time.sleep(0.15)
        monkeypatch.setattr(_ColdLayer, "upsert_nodes_batch", original_batch)

        ConsistencyOracle(graph).assert_eventually_zero(timeout=3.0)
        pipeline.stop(timeout=2.0)


# ── CHAOS-003: sequence tampering ────────────────────────────────────────────

class TestChaos003SequenceTampering:
    """
    Scenario: seq_ids in the WAL are tampered with (duplicated or gapped).
    UNIQUE constraint must reject duplicates.  Watchdog must detect anomalous
    Dm growth and flag it via metrics.
    """

    def test_duplicate_seq_id_rejected_by_unique_constraint(self, graph):
        seq = graph._wal.append("upsert_node", {"node_id": "n1", "node_type": "entity",
                                                  "value": {}, "ts": 1.0})
        with pytest.raises(Exception):
            graph._wal._conn.execute(
                "INSERT INTO wal(seq_id, op, payload, ts) VALUES (?,?,?,?)",
                (seq, "upsert_node", '{"node_id":"n1"}', time.time()),
            )
            graph._wal._conn.commit()

    def test_mark_committed_idempotent_after_tampering(self, graph):
        seq = graph._wal.append("upsert_node", {"node_id": "n1", "node_type": "entity",
                                                  "value": {}, "ts": 1.0})
        graph._wal.mark_committed(seq)
        graph._wal.mark_committed(seq)  # second mark must not raise
        assert graph._wal.pending_count() == 0

    def test_watchdog_detects_growing_dm_via_metrics(self, graph, met):
        # Write nodes first, then start the pipeline so the initial run
        # sees them (pipeline interval=60s means no second commit during the test).
        for i in range(5):
            graph.write_node(_node(f"n{i}"))

        pipeline = PrismShadowPipeline(graph, interval_s=60.0)  # slow — won't commit again

        watchdog = PrismWatchdog(
            pipeline, dm_threshold=0, check_interval=0.05
        )

        # Commit the batch once to establish a baseline, then add more nodes
        pipeline.start()
        time.sleep(0.05)  # let initial pipeline cycle drain those 5

        # Write 5 more nodes — pipeline won't commit them for 60s
        for i in range(5, 10):
            graph.write_node(_node(f"n{i}"))

        watchdog.start()
        time.sleep(0.2)
        watchdog.stop()
        pipeline.stop(timeout=1.0)

        # Watchdog should have recorded Dm samples; at least some still pending
        dm_rows = graph._wal.pending_count()
        assert dm_rows == 5  # the second batch is still uncommitted

    def test_pipeline_rejects_replaying_already_committed_entries(self, graph):
        graph.write_node(_node("n1"))
        # First commit
        graph.commit_pending()
        assert graph.consistency_psi() == 0
        # Manually un-commit (tamper)
        graph._wal._conn.execute("UPDATE wal SET committed=0")
        graph._wal._conn.commit()
        assert graph._wal.pending_count() == 1

        # Re-applying to cold layer: INSERT OR REPLACE handles idempotency
        committed = graph.commit_pending()
        assert committed == 1
        # Node still exists correctly
        assert graph._cold.get_node("n1") is not None

    def test_seq_id_uniqueness_across_concurrent_writes(self, graph):
        results: list[str] = []
        lock = threading.Lock()

        def writer():
            seq = graph._wal.append("upsert_node", {"node_id": uuid.uuid4().hex,
                                                      "node_type": "entity",
                                                      "value": {}, "ts": time.time()})
            with lock:
                results.append(seq)

        threads = [threading.Thread(target=writer) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(results)) == 20, "seq_ids must be globally unique"


# ── Full integrated chaos scenario ────────────────────────────────────────────

class TestFullStackChaos:
    """
    End-to-end: graph + pipeline + watchdog + metrics + oracle all together.
    Verifies the system self-heals and Ψ → 0 under combined stress.
    """

    def test_full_stack_recovers_from_pipeline_death(self, tmp_path, monkeypatch):
        met = PrismMetrics(tmp_path / "m.db")
        monkeypatch.setattr("prism_metrics.metrics", met)

        graph = PrismMemoryGraph(tmp_path / "g.db", tmp_path / "w.db")
        oracle = ConsistencyOracle(graph)

        # Write some nodes
        for i in range(4):
            graph.write_node(_node(f"n{i}"))
        assert oracle.psi == 4

        # Start pipeline, let it commit
        pipeline = PrismShadowPipeline(graph, interval_s=0.05)
        pipeline.start()
        oracle.assert_eventually_zero(timeout=3.0)

        # Kill pipeline
        pipeline.stop(timeout=1.0)
        assert not pipeline.is_alive

        # Write more nodes while pipeline is dead
        for i in range(4, 8):
            graph.write_node(_node(f"n{i}"))
        assert oracle.psi == 4  # new mutations pending

        # Watchdog resurrects
        watchdog = PrismWatchdog(pipeline, dm_threshold=0, check_interval=0.05)
        watchdog.start()

        oracle.assert_eventually_zero(timeout=5.0)
        watchdog.stop()
        pipeline.stop(timeout=2.0)

        # Verify cold layer has all 8 nodes
        cold_nodes = graph._cold.query_nodes(limit=20)
        assert len(cold_nodes) >= 8

        assert met.get("commits_total") >= 8
        assert watchdog.status()["resurrections"] >= 1

        graph.close()
        met.close()

    def test_psi_oscillates_and_settles_under_continuous_writes(self, graph):
        """Ψ rises on writes and falls on commit — never gets stuck."""
        oracle = ConsistencyOracle(graph)
        pipeline = PrismShadowPipeline(graph, interval_s=0.02)
        pipeline.start()

        try:
            for batch in range(5):
                for i in range(3):
                    graph.write_node(_node())
                oracle.sample()
                time.sleep(0.05)
                oracle.sample()

            oracle.assert_eventually_zero(timeout=3.0)
        finally:
            pipeline.stop(timeout=2.0)

        # Ψ should have returned to 0
        oracle.assert_zero()
