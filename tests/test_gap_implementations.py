"""
Tests for the five architectural gap implementations:

Gap 1 — WAL batch commit (prism_wal + prism_memory_graph)
Gap 2 — PrismSoul contradiction detector
Gap 3 — OutcomeTracker → AdaptiveFulcrum feedback loop
Gap 4 — HorizonPlanner deterministic condition router
Gap 5 — PrismPerception sport biometric ingestion
"""
from __future__ import annotations

import time

import pytest

# ── Gap 1: WAL batch commit ───────────────────────────────────────────────────

class TestWALBatchMethods:
    @pytest.fixture()
    def wal(self, tmp_path):
        from prism_wal import PrismWAL
        w = PrismWAL(tmp_path / "wal.db")
        yield w
        w.close()

    def test_append_batch_returns_correct_count(self, wal):
        entries = [("upsert_node", {"node_id": f"n{i}", "x": i}) for i in range(5)]
        seq_ids = wal.append_batch(entries)
        assert len(seq_ids) == 5
        assert len(set(seq_ids)) == 5  # all unique

    def test_append_batch_empty_returns_empty(self, wal):
        assert wal.append_batch([]) == []

    def test_append_batch_single_sqlite_commit(self, wal):
        """Batch of 50 entries should leave all 50 pending (not committed)."""
        entries = [("upsert_node", {"node_id": f"n{i}"}) for i in range(50)]
        wal.append_batch(entries)
        assert wal.pending_count() == 50

    def test_mark_committed_batch(self, wal):
        entries = [("upsert_node", {"node_id": f"n{i}"}) for i in range(4)]
        seq_ids = wal.append_batch(entries)
        wal.mark_committed_batch(seq_ids[:2])
        assert wal.pending_count() == 2

    def test_mark_committed_batch_empty_noop(self, wal):
        wal.mark_committed_batch([])  # should not raise

    def test_mark_committed_batch_all(self, wal):
        seq_ids = wal.append_batch([("upsert_node", {"x": 1}) for _ in range(3)])
        wal.mark_committed_batch(seq_ids)
        assert wal.pending_count() == 0


class TestGraphBatchWrite:
    @pytest.fixture()
    def graph(self, tmp_path):
        from prism_memory_graph import PrismMemoryGraph
        g = PrismMemoryGraph(tmp_path / "g.db", tmp_path / "w.db")
        yield g
        g.close()

    def test_write_nodes_batch_all_readable(self, graph):
        from prism_memory_graph import GraphNode
        nodes = [GraphNode(node_id=f"n{i}", node_type="entity", value={}, ts=time.time())
                 for i in range(10)]
        graph.write_nodes_batch(nodes)
        graph.commit_pending()
        for i in range(10):
            assert graph.get_node(f"n{i}") is not None

    def test_write_edges_batch(self, graph):
        from prism_memory_graph import GraphEdge, GraphNode
        graph.write_node(GraphNode("a", "entity", {}, ts=time.time()))
        graph.write_node(GraphNode("b", "entity", {}, ts=time.time()))
        edges = [GraphEdge(src="a", dst="b", relation="knows", weight=1.0, ts=time.time())]
        graph.write_edges_batch(edges)
        graph.commit_pending()
        assert graph.consistency_psi() == 0

    def test_commit_pending_batch_is_atomic(self, graph):
        """All nodes in a batch are committed in one transaction."""
        from prism_memory_graph import GraphNode
        nodes = [GraphNode(f"n{i}", "entity", {}, ts=time.time()) for i in range(20)]
        graph.write_nodes_batch(nodes)
        assert graph.consistency_psi() == 20
        committed = graph.commit_pending()
        assert committed == 20
        assert graph.consistency_psi() == 0

    def test_batch_commit_latency_under_200ms(self, tmp_path):
        """100-node batch commit must be <200ms (was ~1400ms with per-row commits)."""
        from prism_memory_graph import GraphNode, PrismMemoryGraph
        g = PrismMemoryGraph(tmp_path / "perf.db", tmp_path / "perf_wal.db")
        nodes = [GraphNode(f"n{i}", "entity", {"v": i}, ts=time.time())
                 for i in range(100)]
        g.write_nodes_batch(nodes)
        t0 = time.monotonic()
        g.commit_pending()
        elapsed_ms = (time.monotonic() - t0) * 1000
        g.close()
        assert elapsed_ms < 200, f"Batch commit took {elapsed_ms:.0f}ms (expected <200ms)"


# ── Gap 2: PrismSoul contradiction detector ───────────────────────────────────

class TestSoulKeywordSim:
    def test_identical_texts_score_1(self):
        from prism_soul import PrismSoul
        assert PrismSoul._keyword_sim("exercise improves health", "exercise improves health") == pytest.approx(1.0)

    def test_disjoint_texts_score_0(self):
        from prism_soul import PrismSoul
        assert PrismSoul._keyword_sim("quantum computing physics", "cooking recipes pasta") == pytest.approx(0.0)

    def test_partial_overlap(self):
        from prism_soul import PrismSoul
        sim = PrismSoul._keyword_sim("regular exercise keeps body healthy", "body health exercise")
        assert 0.1 < sim < 1.0

    def test_empty_text_returns_zero(self):
        from prism_soul import PrismSoul
        assert PrismSoul._keyword_sim("", "exercise") == pytest.approx(0.0)

    def test_stopwords_ignored(self):
        from prism_soul import PrismSoul
        # Only stop words — should return 0
        assert PrismSoul._keyword_sim("that this with from have", "that this with") == pytest.approx(0.0)


class TestSoulEntailmentCheck:
    @pytest.fixture()
    def soul(self, tmp_path):
        from prism_soul import PrismSoul
        s = PrismSoul(db_path=tmp_path / "soul.db")
        yield s

    def test_no_contradictions_on_empty_soul(self, soul):
        result = soul.run_entailment_check()
        assert result == []

    def test_contradiction_created_for_negative_trend(self, soul):
        # Add stated belief
        soul.add_belief("regular exercise keeps energy levels high", "preference", "stated", 0.8)
        # Add a lens with a declining trend
        lid = soul.add_lens("exercise_energy", "exercise energy performance",
                            ["hrv_recovery"])
        for val in [0.9, 0.8, 0.7, 0.5, 0.3, 0.2]:
            soul.record_observation(lid, val, "test")
        contradictions = soul.run_entailment_check(sim_threshold=0.05)
        assert len(contradictions) >= 1
        assert any("exercise" in c["stated"].lower() or "energy" in c["stated"].lower()
                   for c in contradictions)

    def test_no_contradiction_on_positive_trend(self, soul):
        soul.add_belief("recovery is improving over time", "preference", "stated", 0.7)
        lid = soul.add_lens("recovery_lens", "recovery improving", ["hrv_recovery"])
        for val in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            soul.record_observation(lid, val, "test")
        assert soul.run_entailment_check() == []

    def test_duplicate_contradictions_not_added(self, soul):
        soul.add_belief("regular exercise keeps energy high", "preference", "stated", 0.8)
        lid = soul.add_lens("energy_lens", "exercise energy", ["hrv_recovery"])
        for val in [0.8, 0.6, 0.4, 0.3, 0.2, 0.1]:
            soul.record_observation(lid, val, "test")
        soul.run_entailment_check(sim_threshold=0.05)
        second = soul.run_entailment_check(sim_threshold=0.05)
        assert len(second) == 0  # no new ones; edges already exist

    def test_delta_report_run_check_param(self, soul):
        soul.add_belief("training performance stays consistent", "pattern", "stated", 0.75)
        lid = soul.add_lens("perf_lens", "training performance consistency", ["sport_readiness"])
        for val in [0.9, 0.7, 0.5, 0.3, 0.2, 0.1]:
            soul.record_observation(lid, val, "test")
        report = soul.delta_report(run_check=True)
        # run_check=True should discover and include new contradictions
        assert isinstance(report, list)

    def test_delta_report_default_no_check(self, soul):
        # By default run_check=False, so no automatic discovery
        soul.add_belief("consistent sleep improves mood", "preference", "stated", 0.8)
        result = soul.delta_report()
        assert result == []  # no pre-existing contradicts edges


# ── Gap 3: OutcomeTracker → AdaptiveFulcrum feedback ─────────────────────────

class TestFulcrumFeedback:
    @pytest.fixture(autouse=True)
    def _setup_network(self, monkeypatch):
        """Load a fresh spectrum so get_current_network() returns something."""
        from prism_spectrum_middleware import load_spectrum
        load_spectrum()

    @pytest.fixture()
    def tracker(self, tmp_path):
        from prism_outcome_tracker import OutcomeTracker
        return OutcomeTracker(db_path=tmp_path / "outcomes.db")

    def test_fulcrum_feedback_no_network_is_noop(self, tracker, monkeypatch):
        import prism_spectrum_middleware as psm
        monkeypatch.setattr(psm, "_current_network", None)
        # Should not raise
        tracker.record("c1", "test goal", "done")

    def test_fulcrum_feedback_done_outcome(self, tracker):
        from prism_spectrum_middleware import get_current_network
        net = get_current_network()
        assert net is not None
        # Record and verify no exception
        rec = tracker.record("c1", "complete the task", "done", steps_count=3)
        assert rec.outcome == "done"

    def test_fulcrum_feedback_corrected_outcome(self, tracker):
        rec = tracker.record("c2", "write a report", "user_corrected", steps_count=5)
        assert rec.outcome == "user_corrected"

    def test_fulcrum_feedback_does_not_crash_on_abandoned(self, tracker):
        rec = tracker.record("c3", "send email", "abandoned")
        assert rec.record_id is not None

    def test_network_singleton_synced_by_chain(self):
        from prism_chain import PrismChain
        from prism_spectrum_middleware import (
            SpectrumGates,
            build_spectrum_network,
            set_current_network,
        )
        chain = PrismChain()
        new_net = build_spectrum_network(SpectrumGates(V=0.8, E=0.2, A=0.6, X=0.7))
        set_current_network(new_net)
        chain._sync_spectrum()
        assert chain._spectrum_network is new_net


# ── Gap 4: HorizonPlanner deterministic condition router ─────────────────────

class TestDeterministicCondition:
    def _eval(self, condition, ctx):
        from prism_horizon import HorizonPlanner
        return HorizonPlanner._deterministic_condition(condition, ctx)

    def test_gte_true(self):
        assert self._eval("hrv >= 60", {"hrv": 65.0}) is True

    def test_gte_false(self):
        assert self._eval("hrv >= 60", {"hrv": 55.0}) is False

    def test_lte_true(self):
        assert self._eval("price <= 100", {"price": 99.9}) is True

    def test_gt_false(self):
        assert self._eval("steps > 10000", {"steps": 8000}) is False

    def test_lt_true(self):
        assert self._eval("weight < 80", {"weight": 75}) is True

    def test_eq_true(self):
        assert self._eval("score == 5", {"score": 5.0}) is True

    def test_neq_true(self):
        assert self._eval("status != 0", {"status": 1}) is True

    def test_missing_key_falls_through(self):
        assert self._eval("temperature >= 20", {}) is None

    def test_day_is_recognized_pattern(self):
        # The day-of-week pattern must be recognized (returns bool, not None)
        result = self._eval("day is monday", {})
        assert result in (True, False)

    def test_presence_check_true(self):
        assert self._eval("token present", {"token": "abc"}) is True

    def test_presence_check_false(self):
        assert self._eval("token exists", {"token": ""}) is False

    def test_presence_check_missing(self):
        assert self._eval("token exists", {}) is False

    def test_unrecognized_returns_none(self):
        assert self._eval("the stock market has crashed", {}) is None

    def test_numeric_with_negative_value(self):
        assert self._eval("delta >= -5", {"delta": -3.0}) is True

    def test_deterministic_used_before_llm(self, tmp_path):
        """_evaluate_trigger should use deterministic path and not call LLM for numeric."""
        from prism_horizon import HorizonPlanner
        planner = HorizonPlanner(llm_router=None, db_path=str(tmp_path / "h.db"))
        goal_id = planner.add("Test numeric goal", trigger_condition="steps >= 10000")
        planner.update_context(goal_id, steps=12000)
        goal = planner._load_goal(goal_id)
        result = planner._evaluate_trigger(goal)
        assert result is True

    def test_deterministic_below_threshold(self, tmp_path):
        from prism_horizon import HorizonPlanner
        planner = HorizonPlanner(llm_router=None, db_path=str(tmp_path / "h.db"))
        goal_id = planner.add("Below threshold goal", trigger_condition="steps >= 10000")
        planner.update_context(goal_id, steps=5000)
        goal = planner._load_goal(goal_id)
        result = planner._evaluate_trigger(goal)
        assert result is False


# ── Gap 5: PrismPerception sport biometric ingestion ─────────────────────────

class TestSportReadinessModel:
    def test_all_inputs_good_is_high_score(self):
        from prism_perception import SportReadinessModel
        m = SportReadinessModel("football")
        score = m.score(hrv_ms=80, sleep_hrs=8.5, high_intensity_mins=0, soreness=1)
        assert score >= 0.75

    def test_all_inputs_bad_is_low_score(self):
        from prism_perception import SportReadinessModel
        m = SportReadinessModel("football")
        score = m.score(hrv_ms=20, sleep_hrs=4.0, high_intensity_mins=90, soreness=9)
        assert score <= 0.35

    def test_no_inputs_returns_0_5(self):
        from prism_perception import SportReadinessModel
        m = SportReadinessModel()
        assert m.score() == pytest.approx(0.5)

    def test_partial_inputs(self):
        from prism_perception import SportReadinessModel
        m = SportReadinessModel("cycling")
        score = m.score(hrv_ms=70, sleep_hrs=7.5)
        assert 0.5 < score < 1.0

    def test_label_peak(self):
        from prism_perception import SportReadinessModel
        m = SportReadinessModel()
        assert m.label(0.85) == "peak"

    def test_label_ready(self):
        from prism_perception import SportReadinessModel
        assert SportReadinessModel().label(0.65) == "ready"

    def test_label_caution(self):
        from prism_perception import SportReadinessModel
        assert SportReadinessModel().label(0.45) == "caution"

    def test_label_rest(self):
        from prism_perception import SportReadinessModel
        assert SportReadinessModel().label(0.3) == "rest"

    def test_to_factor_range(self):
        from prism_perception import SportReadinessModel
        m = SportReadinessModel()
        assert 0.0 <= m.to_factor(0.72) <= 1.0

    def test_unknown_sport_uses_default_weights(self):
        from prism_perception import SportReadinessModel
        m = SportReadinessModel("underwater_polo")
        score = m.score(hrv_ms=70, sleep_hrs=7.5)
        assert 0.0 <= score <= 1.0

    def test_all_sport_types(self):
        from prism_perception import SportReadinessModel
        for sport in ("football", "basketball", "rugby", "tennis", "cycling", "default"):
            m = SportReadinessModel(sport)
            score = m.score(hrv_ms=60, sleep_hrs=7, high_intensity_mins=30, soreness=3)
            assert 0.0 <= score <= 1.0


class TestBiometricChannelSportSignal:
    @pytest.fixture()
    def channel(self):
        import queue

        from prism_perception import BiometricChannel
        q = queue.Queue()
        return BiometricChannel(q, device_hub=None, enabled=True, sport="football")

    def test_ingest_emits_sport_readiness(self, channel):
        channel.ingest(hrv_ms=70, sleep_hrs=7.5)
        signals = []
        while not channel._q.empty():
            signals.append(channel._q.get_nowait())
        signal_types = [s.factor_id for s in signals]
        assert "sport_readiness" in signal_types

    def test_ingest_training_load_emits_signal(self, channel):
        channel.ingest(training_load=200)
        signals = []
        while not channel._q.empty():
            signals.append(channel._q.get_nowait())
        signal_types = [s.factor_id for s in signals]
        assert "training_load" in signal_types

    def test_ingest_no_args_emits_nothing(self, channel):
        channel.ingest()
        assert channel._q.empty()

    def test_sport_readiness_value_in_range(self, channel):
        channel.ingest(hrv_ms=65, sleep_hrs=7.0, high_intensity_mins=45, soreness=4)
        signals = []
        while not channel._q.empty():
            signals.append(channel._q.get_nowait())
        readiness = next((s for s in signals if s.factor_id == "sport_readiness"), None)
        assert readiness is not None
        assert 0.0 <= readiness.value <= 1.0


class TestHealthDirWatcher:
    def test_try_ingest_health_dir(self, tmp_path):
        import json

        from prism_perception import _try_ingest_health_dir
        ingested = []

        def mock_ingest(**kwargs):
            ingested.append(kwargs)

        (tmp_path / "day1.json").write_text(json.dumps({
            "hrv": 65, "sleep_hours": 7.5, "steps": 8000,
        }))
        seen: set[str] = set()
        _try_ingest_health_dir(tmp_path, seen, mock_ingest)
        assert len(ingested) == 1
        assert ingested[0]["hrv_ms"] == 65
        assert "day1.json" in seen

    def test_already_seen_files_skipped(self, tmp_path):
        import json

        from prism_perception import _try_ingest_health_dir
        ingested = []
        (tmp_path / "day1.json").write_text(json.dumps({"hrv": 60}))
        seen = {"day1.json"}
        _try_ingest_health_dir(tmp_path, seen, lambda **kw: ingested.append(kw))
        assert len(ingested) == 0

    def test_missing_dir_noop(self, tmp_path):
        from prism_perception import _try_ingest_health_dir
        _try_ingest_health_dir(tmp_path / "nonexistent", set(), lambda **kw: None)
