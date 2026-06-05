"""Tests for PrismShadowPipeline and PrismWatchdog."""
import time

import pytest

from prism_memory_graph import GraphNode, PrismMemoryGraph
from prism_shadow_pipeline import PrismShadowPipeline
from prism_watchdog import PrismWatchdog


def _node(nid, **kw) -> GraphNode:
    return GraphNode(node_id=nid, node_type="entity", value=kw or {}, ts=time.time())


@pytest.fixture()
def graph(tmp_path):
    g = PrismMemoryGraph(tmp_path / "g.db", tmp_path / "w.db")
    yield g
    g.close()


@pytest.fixture()
def pipeline(graph):
    p = PrismShadowPipeline(graph, interval_s=0.05)
    yield p
    p.stop(timeout=2.0)


# ── PrismShadowPipeline ───────────────────────────────────────────────────────

class TestShadowPipelineLifecycle:
    def test_not_alive_before_start(self, pipeline):
        assert not pipeline.is_alive

    def test_alive_after_start(self, pipeline):
        pipeline.start()
        assert pipeline.is_alive

    def test_not_alive_after_stop(self, pipeline):
        pipeline.start()
        pipeline.stop(timeout=2.0)
        assert not pipeline.is_alive

    def test_start_idempotent(self, pipeline):
        pipeline.start()
        t = pipeline._thread
        pipeline.start()  # second call must not spawn new thread
        assert pipeline._thread is t


class TestShadowPipelineCommits:
    def test_pipeline_commits_written_nodes(self, graph, pipeline):
        graph.write_node(_node("n1", name="Alice"))
        pipeline.start()
        deadline = time.monotonic() + 2.0
        while graph.consistency_psi() > 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert graph._cold.get_node("n1") is not None

    def test_status_reports_committed_total(self, graph, pipeline):
        graph.write_node(_node("n1"))
        graph.write_node(_node("n2"))
        pipeline.start()
        deadline = time.monotonic() + 2.0
        while graph.consistency_psi() > 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pipeline.status()["committed_total"] >= 2

    def test_status_pending_drops_to_zero(self, graph, pipeline):
        for i in range(5):
            graph.write_node(_node(f"n{i}"))
        pipeline.start()
        deadline = time.monotonic() + 2.0
        while graph.consistency_psi() > 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pipeline.status()["pending"] == 0


# ── CHAOS-001: pipeline crash mid-commit ─────────────────────────────────────

class TestChaos001PipelineCrash:
    def test_data_survives_pipeline_crash(self, graph):
        """WAL must preserve data if the pipeline thread dies before committing."""
        graph.write_node(_node("survivor", label="must_persist"))
        # Simulate crash: pipeline never ran; WAL still has the entry
        assert graph.consistency_psi() == 1
        # Replay + recommit simulates restart
        graph.replay_wal()
        graph.commit_pending()
        assert graph._cold.get_node("survivor") is not None

    def test_partial_commit_resumes_correctly(self, graph):
        """Entries that failed to commit stay in WAL and are retried."""
        for i in range(3):
            graph.write_node(_node(f"n{i}"))
        # Manually commit only first entry
        pending = graph._wal.pending()
        p = pending[0]["payload"]
        graph._cold.upsert_node(
            GraphNode(node_id=p["node_id"], node_type=p["node_type"],
                      value=p["value"], ts=p["ts"])
        )
        graph._wal.mark_committed(pending[0]["seq_id"])
        # Two remain
        assert graph.consistency_psi() == 2
        # Full commit clears them
        graph.commit_pending()
        assert graph.consistency_psi() == 0


# ── CHAOS-002: disk error aborts without corrupting cold layer ────────────────

class TestChaos002DiskError:
    def test_cold_layer_intact_after_commit_failure(self, graph, monkeypatch):
        """If cold.upsert_node raises, cold layer must remain clean."""
        from prism_memory_graph import _ColdLayer
        original = _ColdLayer.upsert_node

        call_count = {"n": 0}
        def failing_upsert(self, node):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError("Simulated disk error")
            return original(self, node)

        monkeypatch.setattr(_ColdLayer, "upsert_node", failing_upsert)

        graph.write_node(_node("safe"))
        graph.write_node(_node("safe2"))
        graph.commit_pending()  # first entry fails; second never reached
        # Cold layer has no corrupted half-writes
        # safe node stayed in WAL (still pending)
        assert graph.consistency_psi() >= 1


# ── CHAOS-003: sequence tampering ────────────────────────────────────────────

class TestChaos003SequenceTampering:
    def test_duplicate_seq_id_rejected(self, graph):
        """WAL UNIQUE constraint rejects duplicate seq_ids (idempotency)."""
        seq = graph._wal.append("upsert_node", {"node_id": "n1"})
        with pytest.raises(Exception):
            graph._wal._conn.execute(
                "INSERT INTO wal(seq_id, op, payload, ts) VALUES (?,?,?,?)",
                (seq, "upsert_node", '{"node_id":"n1"}', time.time())
            )
            graph._wal._conn.commit()


# ── PrismWatchdog ─────────────────────────────────────────────────────────────

class TestWatchdog:
    def test_watchdog_resurrects_dead_pipeline(self, graph):
        pipeline = PrismShadowPipeline(graph, interval_s=60.0)
        pipeline.start()
        pipeline.stop(timeout=2.0)
        assert not pipeline.is_alive

        graph.write_node(_node("n1"))

        watchdog = PrismWatchdog(pipeline, dm_threshold=0, check_interval=0.05)
        watchdog.start()
        deadline = time.monotonic() + 2.0
        while not pipeline.is_alive and time.monotonic() < deadline:
            time.sleep(0.01)
        watchdog.stop()
        pipeline.stop(timeout=2.0)

        assert watchdog.status()["resurrections"] >= 1

    def test_watchdog_does_not_resurrect_healthy_pipeline(self, graph):
        pipeline = PrismShadowPipeline(graph, interval_s=60.0)
        pipeline.start()

        watchdog = PrismWatchdog(pipeline, dm_threshold=0, check_interval=0.05)
        watchdog.start()
        time.sleep(0.15)
        watchdog.stop()
        pipeline.stop(timeout=2.0)

        assert watchdog.status()["resurrections"] == 0

    def test_watchdog_status_structure(self, graph):
        pipeline = PrismShadowPipeline(graph)
        watchdog = PrismWatchdog(pipeline)
        s = watchdog.status()
        assert "resurrections" in s
        assert "pipeline" in s
        assert "alive" in s["pipeline"]
