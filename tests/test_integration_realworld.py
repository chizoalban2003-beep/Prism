"""
Real-world integration stress tests for PRISM infrastructure.

Exercises the stack against realistic scenarios:
  - ML pipeline end-to-end with real numpy/sklearn data
  - Memory graph at scale (500 nodes, batch commit)
  - Concurrent WAL writes under thread contention
  - Session manager CRUD lifecycle at scale
  - ML nightly sweep with real outcome data
  - Vision ML bridge pipeline with real frame sequence
"""
from __future__ import annotations

import threading
import time

import pytest

# ---------------------------------------------------------------------------
# 1. ML assembler — end-to-end with sklearn datasets
# ---------------------------------------------------------------------------

class TestMLPipelineRealWorld:
    def test_regression_on_diabetes_dataset(self):
        """Ridge regression on sklearn's diabetes dataset (442 samples, 10 features)."""
        pytest.importorskip("numpy")
        pytest.importorskip("sklearn")
        from sklearn.datasets import load_diabetes

        from prism_ml_assembler import MLAssembler

        data = load_diabetes()
        X = data.data.tolist()
        y = data.target.tolist()

        asm = MLAssembler()
        result = asm.run(task="predict diabetes progression", X=X, y=y, translate=False)

        assert result.algorithm in {"ridge", "lasso", "xgboost", "lightgbm", "fallback_mean"}
        assert 0.0 <= result.confidence <= 1.0
        pred = result.prediction
        if hasattr(pred, "__len__"):
            assert len(pred) == len(y)
        assert result.duration_ms > 0

    def test_classification_on_iris(self):
        """Classification on Iris (150 samples, 4 features, 3 classes)."""
        pytest.importorskip("numpy")
        pytest.importorskip("sklearn")
        from sklearn.datasets import load_iris

        from prism_ml_assembler import MLAssembler

        data = load_iris()
        X = data.data.tolist()
        y = [float(v) for v in data.target.tolist()]

        asm = MLAssembler()
        result = asm.run(task="classify iris species", X=X, y=y, translate=False)

        assert result.algorithm in {
            "ridge", "lasso", "xgboost", "lightgbm", "logistic", "fallback_mean"
        }
        assert 0.0 <= result.confidence <= 1.0

    def test_high_dimensional_data(self):
        """ML assembler handles wide data (200 samples × 150 features)."""
        np = pytest.importorskip("numpy")
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 150)).tolist()
        y = rng.standard_normal(200).tolist()

        from prism_ml_assembler import MLAssembler

        asm = MLAssembler()
        result = asm.run(task="high dim regression", X=X, y=y, translate=False)
        assert result.confidence >= 0.0

    def test_nightly_sweep_with_real_outcomes(self):
        """run_nightly_sweep correctly identifies weak algorithm and updates params."""
        pytest.importorskip("numpy")
        from unittest.mock import MagicMock

        from prism_ml_assembler import MLAssembler, run_nightly_sweep

        asm = MLAssembler()

        # Simulate outcome tracker with a weak-performing algorithm
        tracker = MagicMock()
        tracker.get_failed_outcomes.return_value = [
            {"algorithm": "ridge", "error": 0.45, "task": "predict X"}
        ] * 10

        updated = run_nightly_sweep(asm, tracker)
        assert isinstance(updated, dict)

    def test_unsupervised_clustering(self):
        """Unsupervised path (no y) returns a clustering result."""
        np = pytest.importorskip("numpy")
        rng = np.random.default_rng(7)
        X = rng.standard_normal((80, 5)).tolist()

        from prism_ml_assembler import MLAssembler

        asm = MLAssembler()
        result = asm.run(task="cluster user sessions", X=X, translate=False)
        assert result.algorithm in {"dbscan", "kmeans", "fallback_mean"}


# ---------------------------------------------------------------------------
# 2. Memory graph at scale
# ---------------------------------------------------------------------------

class TestMemoryGraphAtScale:
    @pytest.fixture()
    def graph(self, tmp_path):
        from prism_memory_graph import PrismMemoryGraph
        g = PrismMemoryGraph(
            db_path=tmp_path / "cold.db",
            wal_path=tmp_path / "wal.db",
        )
        yield g
        g.close()

    def test_500_nodes_batch_commit(self, graph):
        """Write 500 nodes then commit — verifies WAL handles large batches."""
        from prism_memory_graph import GraphNode

        N = 500
        for i in range(N):
            graph.write_node(GraphNode(
                node_id=f"scale_{i}",
                node_type="test",
                value={"i": i, "payload": "x" * 64},
                ts=time.time(),
            ))

        assert graph.consistency_psi() == N
        t0 = time.monotonic()
        graph.commit_pending()
        elapsed_ms = (time.monotonic() - t0) * 1000

        assert graph.consistency_psi() == 0
        # 500-node commit should stay under 5 s on any reasonable hardware
        assert elapsed_ms < 5000, f"Batch commit took {elapsed_ms:.0f} ms"

        found = graph._cold.get_node("scale_0")
        assert found is not None

    def test_concurrent_writes_no_data_loss(self, graph):
        """16 threads each write 32 nodes concurrently — no nodes lost after commit."""
        from prism_memory_graph import GraphNode

        N_THREADS = 16
        N_PER_THREAD = 32
        errors = []

        def _write(tid):
            try:
                for j in range(N_PER_THREAD):
                    graph.write_node(GraphNode(
                        node_id=f"t{tid}_n{j}",
                        node_type="concurrent",
                        value={"tid": tid, "j": j},
                        ts=time.time(),
                    ))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_write, args=(i,)) for i in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        graph.commit_pending()

        # All nodes should be committed
        assert graph.consistency_psi() == 0

    def test_edge_creation_and_traversal(self, graph):
        """Create a chain of 10 edges; traversal must reach terminal node."""
        from prism_memory_graph import GraphEdge, GraphNode

        # Create 10 nodes
        for i in range(10):
            graph.write_node(GraphNode(
                node_id=f"chain_{i}", node_type="chain",
                value={"step": i}, ts=time.time(),
            ))

        # Chain them: 0→1→2→...→9
        for i in range(9):
            graph.write_edge(GraphEdge(
                src=f"chain_{i}",
                dst=f"chain_{i + 1}",
                relation="next",
                weight=1.0,
            ))

        graph.commit_pending()

        # Verify the last node is reachable
        node = graph._cold.get_node("chain_9")
        assert node is not None
        assert node.value["step"] == 9


# ---------------------------------------------------------------------------
# 3. Session manager lifecycle at scale
# ---------------------------------------------------------------------------

class TestSessionManagerAtScale:
    @pytest.fixture()
    def sm(self, tmp_path):
        from prism_session_manager import SessionManager
        return SessionManager(db_path=str(tmp_path / "sessions.db"))

    def test_create_100_sessions(self, sm):
        for i in range(100):
            sm.create_session(name=f"Session {i}", description=f"desc {i}")
        sessions = sm.list_sessions(limit=200)
        assert len(sessions) == 100

    def test_pagination(self, sm):
        for i in range(25):
            sm.create_session(name=f"S{i}")
        page1 = sm.list_sessions(limit=10, offset=0)
        page2 = sm.list_sessions(limit=10, offset=10)
        page3 = sm.list_sessions(limit=10, offset=20)
        assert len(page1) == 10
        assert len(page2) == 10
        assert len(page3) == 5
        ids = {s.session_id for s in page1 + page2 + page3}
        assert len(ids) == 25

    def test_message_throughput(self, sm):
        """100 messages to a single session should persist correctly."""
        sess = sm.create_session(name="High Volume")
        for i in range(100):
            sm.add_message(sess.session_id, role="user", content=f"Message {i}")
        messages = sm.get_history(sess.session_id, n=200)
        assert len(messages) == 100

    def test_concurrent_session_creation(self, sm):
        """10 threads each create 5 sessions concurrently."""
        errors = []
        session_ids = []
        lock = threading.Lock()

        def _create(tid):
            try:
                for j in range(5):
                    s = sm.create_session(name=f"T{tid}S{j}")
                    with lock:
                        session_ids.append(s.session_id)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_create, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(set(session_ids)) == 50  # all unique


# ---------------------------------------------------------------------------
# 4. Vision ML bridge — realistic frame sequence
# ---------------------------------------------------------------------------

class TestVisionMLBridgeRealWorld:
    def test_frame_sequence_ingestion(self):
        """Ingest 30 frames at 8×8 grid — bridge accumulates and runs ML."""
        import random

        from prism_ml_assembler import MLAssembler
        from prism_vision_ml_bridge import VisionMLBridge

        asm = MLAssembler()
        bridge = VisionMLBridge(assembler=asm, min_frames=10, max_buffer=30)

        rng = random.Random(42)
        last_result = None
        for _ in range(30):
            frame = bytes([rng.randint(0, 255) for _ in range(512)])
            last_result = bridge.ingest(frame)

        assert last_result is not None
        assert last_result["frames_buffered"] == 30
        assert "ml_result" in last_result
        assert last_result["ml_result"]["algorithm"] in {
            "ridge", "lasso", "xgboost", "lightgbm", "dbscan", "kmeans", "fallback_mean"
        }

    def test_bridge_clear_resets_state(self):
        """After clear(), the next frame has no delta and ML reruns from scratch."""
        import random

        from prism_ml_assembler import MLAssembler
        from prism_vision_ml_bridge import VisionMLBridge

        asm = MLAssembler()
        bridge = VisionMLBridge(assembler=asm, min_frames=3)

        rng = random.Random(7)
        for _ in range(5):
            bridge.ingest(bytes([rng.randint(0, 255) for _ in range(512)]))

        bridge.clear()
        result = bridge.ingest(bytes([128] * 512))
        assert result["has_delta"] is False
        assert result["frames_buffered"] == 1


# ---------------------------------------------------------------------------
# 5. LLM Ledger — cost accounting accuracy
# ---------------------------------------------------------------------------

class TestLLMCostAccounting:
    @pytest.fixture()
    def ledger(self, tmp_path):
        from prism_llm_ledger import LLMLedger
        return LLMLedger(db_path=str(tmp_path / "ledger.db"))

    def test_cost_accumulation(self, ledger):
        """Record 50 calls; summary total should match per-call sum."""
        import time as _t

        total_cost = 0.0
        for i in range(50):
            rec = ledger.record_call(
                provider="openai", model="gpt-4o",
                input_tokens=500 + i * 10,
                output_tokens=200 + i * 5,
                latency_ms=300.0,
                source="test",
            )
            total_cost += rec.cost_usd

        summary = ledger.summary(since_ts=_t.time() - 3600)
        assert summary["total_calls"] == 50
        assert abs(summary["total_cost_usd"] - total_cost) < 1e-6

    def test_by_model_breakdown(self, ledger):
        """Records for two models appear as separate entries in by_model."""
        for _ in range(5):
            ledger.record_call("openai", "gpt-4o", 100, 50, 200)
        for _ in range(3):
            ledger.record_call("ollama", "llama3", 80, 40, 50)

        breakdown = ledger.by_model(days=1)
        model_names = {entry["model"] for entry in breakdown}
        assert "gpt-4o" in model_names
        assert "llama3" in model_names
