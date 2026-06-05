"""Tests for PrismMetrics — three-layered observability funnel."""
import time

import pytest

from prism_metrics import DM_GROWTH_THRESHOLD, LR_WARN_THRESHOLD_S, PrismMetrics


@pytest.fixture()
def m(tmp_path):
    inst = PrismMetrics(tmp_path / "metrics.db")
    yield inst
    inst.close()


# ── Layer 1: counters ─────────────────────────────────────────────────────────

class TestCounters:
    def test_get_missing_returns_zero(self, m):
        assert m.get("nonexistent") == 0

    def test_inc_creates_counter(self, m):
        m.inc("wal_replays")
        assert m.get("wal_replays") == 1

    def test_inc_amount(self, m):
        m.inc("commits_total", 5)
        assert m.get("commits_total") == 5

    def test_inc_accumulates(self, m):
        m.inc("wal_replays")
        m.inc("wal_replays")
        m.inc("wal_replays", 3)
        assert m.get("wal_replays") == 5

    def test_all_counters_returns_dict(self, m):
        m.inc("a")
        m.inc("b", 2)
        c = m.all_counters()
        assert c["a"] == 1
        assert c["b"] == 2

    def test_multiple_counters_independent(self, m):
        m.inc("x")
        m.inc("y", 10)
        assert m.get("x") == 1
        assert m.get("y") == 10


# ── Layer 2: reconciliation latency ──────────────────────────────────────────

class TestLatency:
    def test_mean_latency_none_when_empty(self, m):
        assert m.mean_latency() is None

    def test_mean_latency_single_value(self, m):
        m.record_latency(10.0)
        assert m.mean_latency() == pytest.approx(10.0)

    def test_mean_latency_multiple_values(self, m):
        m.record_latency(4.0)
        m.record_latency(6.0)
        assert m.mean_latency() == pytest.approx(5.0)

    def test_lr_alert_false_below_threshold(self, m):
        m.record_latency(LR_WARN_THRESHOLD_S - 1)
        assert not m.lr_alert()

    def test_lr_alert_true_above_threshold(self, m):
        m.record_latency(LR_WARN_THRESHOLD_S + 1)
        assert m.lr_alert()

    def test_old_latency_excluded_from_window(self, m):
        # Manually insert old record
        m._conn.execute(
            "INSERT INTO latency_log(ts, value) VALUES(?,?)",
            (time.time() - 400, LR_WARN_THRESHOLD_S + 10),
        )
        m._conn.commit()
        # Window=300s should exclude it
        assert m.mean_latency(window_s=300) is None


# ── Layer 3: drift magnitude ──────────────────────────────────────────────────

class TestDriftMagnitude:
    def test_dm_trend_zero_with_no_data(self, m):
        assert m.dm_trend() == pytest.approx(0.0)

    def test_dm_trend_zero_with_single_sample(self, m):
        m.record_dm(5)
        assert m.dm_trend() == pytest.approx(0.0)

    def test_dm_trend_positive_when_growing(self, m):
        for v in [1, 3, 7, 12, 20]:
            m.record_dm(v)
        assert m.dm_trend() > 0

    def test_dm_trend_negative_when_shrinking(self, m):
        for v in [20, 12, 7, 3, 1]:
            m.record_dm(v)
        assert m.dm_trend() < 0

    def test_critical_alert_false_when_dm_stable(self, m):
        for _ in range(5):
            m.record_dm(2)
        m.record_latency(1.0)
        assert not m.critical_alert()

    def test_critical_alert_true_when_dm_grows_and_lr_high(self, m):
        for v in range(0, DM_GROWTH_THRESHOLD * 6, DM_GROWTH_THRESHOLD):
            m.record_dm(v)
        m.record_latency(LR_WARN_THRESHOLD_S + 1)
        assert m.critical_alert()

    def test_critical_alert_false_when_only_lr_high(self, m):
        # Dm stable, only Lr is high
        for _ in range(5):
            m.record_dm(2)
        m.record_latency(LR_WARN_THRESHOLD_S + 1)
        assert not m.critical_alert()


# ── Canary tracking ───────────────────────────────────────────────────────────

class TestCanaryTracking:
    def test_canary_stats_empty(self, m):
        s = m.canary_stats()
        assert s["n"] == 0
        assert s["mean_ms"] is None

    def test_canary_stats_mean(self, m):
        m.record_canary(100.0)
        m.record_canary(200.0)
        s = m.canary_stats()
        assert s["mean_ms"] == pytest.approx(150.0)
        assert s["n"] == 2

    def test_canary_stats_max(self, m):
        m.record_canary(50.0)
        m.record_canary(300.0)
        assert m.canary_stats()["max_ms"] == pytest.approx(300.0)

    def test_canary_success_rate(self, m):
        m.record_canary(50.0, success=True)
        m.record_canary(50.0, success=False)
        m.record_canary(50.0, success=True)
        assert m.canary_stats()["success_rate"] == pytest.approx(2 / 3, abs=0.01)

    def test_performance_rho_none_with_single_run(self, m):
        m.record_canary(100.0)
        assert m.performance_rho() is None

    def test_performance_rho_positive_when_degrading(self, m):
        for ms in [10, 20, 30, 40, 50]:
            m.record_canary(float(ms))
            time.sleep(0.001)  # ensure distinct timestamps
        rho = m.performance_rho()
        assert rho is not None and rho > 0

    def test_performance_rho_negative_when_improving(self, m):
        for ms in [50, 40, 30, 20, 10]:
            m.record_canary(float(ms))
            time.sleep(0.001)
        rho = m.performance_rho()
        assert rho is not None and rho < 0


# ── Full report ───────────────────────────────────────────────────────────────

class TestReport:
    def test_report_structure(self, m):
        r = m.report()
        assert "layer1_counters" in r
        assert "layer2_lr_mean_s" in r
        assert "layer2_lr_alert" in r
        assert "layer3_dm_trend" in r
        assert "layer3_critical" in r
        assert "canary" in r
        assert "performance_rho" in r

    def test_report_reflects_recorded_data(self, m):
        m.inc("wal_replays", 3)
        m.record_latency(5.0)
        r = m.report()
        assert r["layer1_counters"].get("wal_replays") == 3
        assert r["layer2_lr_mean_s"] == pytest.approx(5.0)


# ── Prune ─────────────────────────────────────────────────────────────────────

class TestPrune:
    def test_prune_removes_old_records(self, m):
        old_ts = time.time() - 40 * 86400
        m._conn.execute("INSERT INTO latency_log(ts, value) VALUES(?,?)", (old_ts, 1.0))
        m._conn.execute("INSERT INTO dm_log(ts, value) VALUES(?,?)", (old_ts, 5))
        m._conn.execute("INSERT INTO canary_log(ts, duration_ms, success) VALUES(?,?,1)", (old_ts, 50.0))
        m._conn.commit()
        result = m.prune(older_than_days=30)
        assert result["latency_rows"] == 1
        assert result["dm_rows"] == 1
        assert result["canary_rows"] == 1

    def test_prune_preserves_recent_records(self, m):
        m.record_latency(1.0)
        m.record_dm(2)
        m.record_canary(50.0)
        result = m.prune(older_than_days=30)
        assert result["latency_rows"] == 0
        assert result["dm_rows"] == 0
        assert result["canary_rows"] == 0
