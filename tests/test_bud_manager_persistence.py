"""
Tests for prism_bud_manager — persistent execution log.
"""
from __future__ import annotations

import time

from prism_bud_manager import BudManager, BudStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mgr(tmp_path) -> BudManager:
    """BudManager backed by a temp SQLite DB."""
    return BudManager(db_path=tmp_path / "buds.db")


def _spawn_and_run(mgr: BudManager, intent: str = "test_intent", fail: bool = False):
    handle = mgr.spawn(intent, "msg", {}, [])
    if fail:
        try:
            mgr.execute(handle, lambda i, m, c: (_ for _ in ()).throw(RuntimeError("oops")))
        except RuntimeError:
            pass
    else:
        mgr.execute(handle, lambda i, m, c: "ok")
    return handle


# ---------------------------------------------------------------------------
# DB initialisation
# ---------------------------------------------------------------------------

class TestBudManagerDBInit:
    def test_db_file_created(self, tmp_path):
        _mgr(tmp_path)
        assert (tmp_path / "buds.db").exists()

    def test_conn_is_open(self, tmp_path):
        mgr = _mgr(tmp_path)
        assert mgr._conn is not None

    def test_schema_version_set(self, tmp_path):
        mgr = _mgr(tmp_path)
        ver = mgr._conn.execute("PRAGMA user_version").fetchone()[0]
        assert ver == 1

    def test_table_exists(self, tmp_path):
        mgr = _mgr(tmp_path)
        tables = {r[0] for r in mgr._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "bud_executions" in tables

    def test_indexes_exist(self, tmp_path):
        mgr = _mgr(tmp_path)
        indexes = {r[0] for r in mgr._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()}
        assert "idx_be_intent" in indexes
        assert "idx_be_ts" in indexes

    def test_no_db_path_uses_in_memory_fallback(self):
        """Passing a read-only path should not raise — falls back gracefully."""
        mgr = BudManager(db_path="/this/path/does/not/exist/bud.db")
        # Either conn is None (fallback) or an exception was suppressed
        # Either way, spawn/execute/decommission must not raise
        handle = mgr.spawn("x", "m", {}, [])
        mgr.execute(handle, lambda i, m, c: None)


# ---------------------------------------------------------------------------
# Persistence — write path
# ---------------------------------------------------------------------------

class TestBudExecutionPersistence:
    def test_completed_bud_persisted(self, tmp_path):
        mgr = _mgr(tmp_path)
        _spawn_and_run(mgr, "web_search")
        rows = mgr._conn.execute("SELECT * FROM bud_executions").fetchall()
        assert len(rows) == 1

    def test_failed_bud_persisted(self, tmp_path):
        mgr = _mgr(tmp_path)
        _spawn_and_run(mgr, "web_search", fail=True)
        rows = mgr._conn.execute("SELECT * FROM bud_executions").fetchall()
        assert len(rows) == 1

    def test_status_written_correctly(self, tmp_path):
        mgr = _mgr(tmp_path)
        _spawn_and_run(mgr, "x")
        row = mgr._conn.execute(
            "SELECT status FROM bud_executions"
        ).fetchone()
        assert row[0] == BudStatus.COMPLETED.value

    def test_failed_status_written(self, tmp_path):
        mgr = _mgr(tmp_path)
        _spawn_and_run(mgr, "x", fail=True)
        row = mgr._conn.execute(
            "SELECT status FROM bud_executions"
        ).fetchone()
        assert row[0] == BudStatus.FAILED.value

    def test_error_written_for_failures(self, tmp_path):
        mgr = _mgr(tmp_path)
        _spawn_and_run(mgr, "x", fail=True)
        row = mgr._conn.execute(
            "SELECT error FROM bud_executions"
        ).fetchone()
        assert row[0] is not None and "oops" in row[0]

    def test_error_null_for_successes(self, tmp_path):
        mgr = _mgr(tmp_path)
        _spawn_and_run(mgr, "x")
        row = mgr._conn.execute(
            "SELECT error FROM bud_executions"
        ).fetchone()
        assert row[0] is None

    def test_duration_ms_positive(self, tmp_path):
        mgr = _mgr(tmp_path)
        _spawn_and_run(mgr, "x")
        row = mgr._conn.execute(
            "SELECT duration_ms FROM bud_executions"
        ).fetchone()
        assert row[0] >= 0.0

    def test_capabilities_persisted_as_json(self, tmp_path):
        mgr = _mgr(tmp_path)
        handle = mgr.spawn("x", "m", {}, ["internet_read", "filesystem_read"])
        mgr.execute(handle, lambda i, m, c: None)
        row = mgr._conn.execute(
            "SELECT capabilities FROM bud_executions"
        ).fetchone()
        import json
        caps = json.loads(row[0])
        assert "internet_read" in caps
        assert "filesystem_read" in caps

    def test_multiple_buds_persisted(self, tmp_path):
        mgr = _mgr(tmp_path)
        for _ in range(5):
            _spawn_and_run(mgr)
        count = mgr._conn.execute(
            "SELECT COUNT(*) FROM bud_executions"
        ).fetchone()[0]
        assert count == 5

    def test_survives_restart(self, tmp_path):
        mgr1 = _mgr(tmp_path)
        for _ in range(3):
            _spawn_and_run(mgr1)
        mgr1._conn.close()

        mgr2 = BudManager(db_path=tmp_path / "buds.db")
        count = mgr2._conn.execute(
            "SELECT COUNT(*) FROM bud_executions"
        ).fetchone()[0]
        assert count == 3


# ---------------------------------------------------------------------------
# execution_history
# ---------------------------------------------------------------------------

class TestExecutionHistory:
    def test_returns_list(self, tmp_path):
        mgr = _mgr(tmp_path)
        hist = mgr.execution_history()
        assert isinstance(hist, list)

    def test_empty_on_fresh_db(self, tmp_path):
        mgr = _mgr(tmp_path)
        assert mgr.execution_history() == []

    def test_returns_records_after_runs(self, tmp_path):
        mgr = _mgr(tmp_path)
        _spawn_and_run(mgr, "search")
        hist = mgr.execution_history()
        assert len(hist) == 1

    def test_record_has_required_keys(self, tmp_path):
        mgr = _mgr(tmp_path)
        _spawn_and_run(mgr, "search")
        rec = mgr.execution_history()[0]
        for key in ("bud_id", "intent", "status", "duration_ms", "error",
                    "capabilities", "timestamp"):
            assert key in rec

    def test_filter_by_intent(self, tmp_path):
        mgr = _mgr(tmp_path)
        _spawn_and_run(mgr, "search")
        _spawn_and_run(mgr, "calendar")
        _spawn_and_run(mgr, "search")
        results = mgr.execution_history(intent="search")
        assert len(results) == 2
        assert all(r["intent"] == "search" for r in results)

    def test_limit_respected(self, tmp_path):
        mgr = _mgr(tmp_path)
        for _ in range(10):
            _spawn_and_run(mgr)
        results = mgr.execution_history(limit=3)
        assert len(results) == 3

    def test_ordered_newest_first(self, tmp_path):
        mgr = _mgr(tmp_path)
        for _ in range(3):
            _spawn_and_run(mgr)
        results = mgr.execution_history()
        timestamps = [r["timestamp"] for r in results]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_days_filter_excludes_old(self, tmp_path):
        mgr = _mgr(tmp_path)
        _spawn_and_run(mgr)
        # Manually backdate the row to 10 days ago
        mgr._conn.execute(
            "UPDATE bud_executions SET timestamp = ?",
            (time.time() - 10 * 86400,),
        )
        mgr._conn.commit()
        results = mgr.execution_history(days=7)
        assert len(results) == 0

    def test_capabilities_deserialized_to_list(self, tmp_path):
        mgr = _mgr(tmp_path)
        h = mgr.spawn("x", "m", {}, ["internet_read"])
        mgr.execute(h, lambda i, m, c: None)
        rec = mgr.execution_history()[0]
        assert isinstance(rec["capabilities"], list)


# ---------------------------------------------------------------------------
# intent_stats
# ---------------------------------------------------------------------------

class TestIntentStats:
    def test_returns_dict(self, tmp_path):
        mgr = _mgr(tmp_path)
        stats = mgr.intent_stats("search")
        assert isinstance(stats, dict)

    def test_zero_stats_on_no_data(self, tmp_path):
        mgr = _mgr(tmp_path)
        stats = mgr.intent_stats("nonexistent")
        assert stats["total"] == 0
        assert stats["success_rate"] == 0.0

    def test_total_count(self, tmp_path):
        mgr = _mgr(tmp_path)
        for _ in range(4):
            _spawn_and_run(mgr, "search")
        stats = mgr.intent_stats("search")
        assert stats["total"] == 4

    def test_success_rate_all_pass(self, tmp_path):
        mgr = _mgr(tmp_path)
        for _ in range(5):
            _spawn_and_run(mgr, "search")
        stats = mgr.intent_stats("search")
        assert stats["success_rate"] == 1.0

    def test_success_rate_mixed(self, tmp_path):
        mgr = _mgr(tmp_path)
        for _ in range(3):
            _spawn_and_run(mgr, "search")
        for _ in range(1):
            _spawn_and_run(mgr, "search", fail=True)
        stats = mgr.intent_stats("search")
        assert abs(stats["success_rate"] - 0.75) < 1e-6

    def test_avg_duration_positive(self, tmp_path):
        mgr = _mgr(tmp_path)
        _spawn_and_run(mgr, "search")
        stats = mgr.intent_stats("search")
        assert stats["avg_duration_ms"] >= 0.0

    def test_last_seen_set(self, tmp_path):
        mgr = _mgr(tmp_path)
        _spawn_and_run(mgr, "search")
        stats = mgr.intent_stats("search")
        assert stats["last_seen"] is not None
        assert stats["last_seen"] <= time.time() + 1

    def test_intent_field_matches(self, tmp_path):
        mgr = _mgr(tmp_path)
        stats = mgr.intent_stats("my_intent")
        assert stats["intent"] == "my_intent"


# ---------------------------------------------------------------------------
# session_stats — enhanced with total_all_time
# ---------------------------------------------------------------------------

class TestSessionStats:
    def test_has_total_all_time(self, tmp_path):
        mgr = _mgr(tmp_path)
        stats = mgr.session_stats()
        assert "total_all_time" in stats

    def test_total_all_time_matches_persistent_count(self, tmp_path):
        mgr = _mgr(tmp_path)
        for _ in range(3):
            _spawn_and_run(mgr)
        stats = mgr.session_stats()
        assert stats["total_all_time"] == 3

    def test_session_count_resets_between_instances(self, tmp_path):
        mgr1 = _mgr(tmp_path)
        for _ in range(4):
            _spawn_and_run(mgr1)
        mgr1._conn.close()

        mgr2 = BudManager(db_path=tmp_path / "buds.db")
        stats = mgr2.session_stats()
        # New session: session count=0, but all_time=4
        assert stats["total_spawned"] == 0
        assert stats["total_all_time"] == 4
