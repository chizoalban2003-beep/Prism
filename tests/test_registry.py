"""
tests/test_registry.py
======================
Unit tests for ksa_registry.py — SnapshotRegistry, PerformanceMetrics.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ksa_lever import ThreeBarSystem
from ksa_registry import PerformanceMetrics, SnapshotRecord, SnapshotRegistry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry(tmp_path):
    """A fresh SnapshotRegistry backed by a temp SQLite file."""
    return SnapshotRegistry(str(tmp_path / "test.db"))


@pytest.fixture
def default_system():
    return ThreeBarSystem.from_defaults()


@pytest.fixture
def good_metrics():
    return PerformanceMetrics(
        execution_time_ms=100.0,
        cpu_peak_pct=5.0,
        ram_peak_mb=30.0,
        success=True,
        override_fired=False,
        notes="unit test",
    )


# ---------------------------------------------------------------------------
# PerformanceMetrics tests
# ---------------------------------------------------------------------------

class TestPerformanceMetrics:
    def test_score_is_positive_on_success(self, good_metrics):
        assert good_metrics.score() > 0

    def test_score_zero_on_failure(self):
        m = PerformanceMetrics(
            execution_time_ms=50.0, success=False
        )
        assert m.score() == pytest.approx(0.0)

    def test_override_fired_reduces_score(self):
        fast = PerformanceMetrics(execution_time_ms=50.0, success=True, override_fired=False)
        slow = PerformanceMetrics(execution_time_ms=50.0, success=True, override_fired=True)
        assert fast.score() > slow.score()

    def test_serialisation_roundtrip(self, good_metrics):
        d  = good_metrics.to_dict()
        m2 = PerformanceMetrics.from_dict(d)
        assert m2.execution_time_ms == pytest.approx(good_metrics.execution_time_ms)
        assert m2.success           == good_metrics.success
        assert m2.override_fired    == good_metrics.override_fired


# ---------------------------------------------------------------------------
# SnapshotRegistry — save / load
# ---------------------------------------------------------------------------

class TestRegistrySaveLoad:
    def test_save_returns_version_1(self, registry, default_system):
        v = registry.save("task_a", default_system)
        assert v == 1

    def test_save_increments_version(self, registry, default_system):
        v1 = registry.save("task_a", default_system)
        v2 = registry.save("task_a", default_system)
        assert v2 == v1 + 1

    def test_load_returns_system(self, registry, default_system):
        registry.save("task_a", default_system)
        loaded = registry.load("task_a")
        assert isinstance(loaded, ThreeBarSystem)

    def test_load_specific_version(self, registry, default_system):
        v1 = registry.save("task_a", default_system)

        # Mutate and save second version
        default_system.levers[0].set_weights(left=9.0, right=1.0)
        registry.save("task_a", default_system)

        loaded_v1 = registry.load("task_a", version=v1)
        # v1 had default weights (0, 0)
        assert loaded_v1.levers[0].left_weight == pytest.approx(0.0)

    def test_load_unknown_task_raises(self, registry):
        with pytest.raises(KeyError):
            registry.load("nonexistent_task")

    def test_load_unknown_version_raises(self, registry, default_system):
        registry.save("task_a", default_system)
        with pytest.raises(KeyError):
            registry.load("task_a", version=99)


# ---------------------------------------------------------------------------
# SnapshotRegistry — record_outcome / best_version / auto_promote_best
# ---------------------------------------------------------------------------

class TestRegistryMetrics:
    def test_record_outcome_succeeds(self, registry, default_system, good_metrics):
        v = registry.save("task_a", default_system)
        registry.record_outcome("task_a", v, good_metrics)  # should not raise

    def test_record_outcome_unknown_version_raises(self, registry, good_metrics):
        with pytest.raises(KeyError):
            registry.record_outcome("task_a", 99, good_metrics)

    def test_best_version_returns_highest_score(self, registry, default_system):
        v1 = registry.save("task_a", default_system)
        registry.record_outcome(
            "task_a", v1,
            PerformanceMetrics(execution_time_ms=500.0, success=True),
        )

        v2 = registry.save("task_a", default_system)
        registry.record_outcome(
            "task_a", v2,
            PerformanceMetrics(execution_time_ms=50.0, success=True),
        )

        best = registry.best_version("task_a")
        # v2 is faster, so its score should be higher
        assert best == v2

    def test_auto_promote_best_changes_current(self, registry, default_system):
        v1 = registry.save("task_a", default_system)
        registry.record_outcome(
            "task_a", v1,
            PerformanceMetrics(execution_time_ms=500.0, success=True),
        )

        v2 = registry.save("task_a", default_system)
        registry.record_outcome(
            "task_a", v2,
            PerformanceMetrics(execution_time_ms=50.0, success=True),
        )

        promoted = registry.auto_promote_best("task_a")
        assert promoted == v2

        # Loading without version should now return v2
        system = registry.load("task_a")
        # Both snapshots are identical in this test, but confirm no error
        assert system is not None


# ---------------------------------------------------------------------------
# SnapshotRegistry — promote / rollback
# ---------------------------------------------------------------------------

class TestRegistryVersionControl:
    def test_promote(self, registry, default_system):
        v1 = registry.save("task_a", default_system)
        _v2 = registry.save("task_a", default_system)  # now current
        registry.promote("task_a", v1)
        tasks = {t["task_name"]: t for t in registry.list_tasks()}
        assert tasks["task_a"]["current_version"] == v1

    def test_promote_unknown_raises(self, registry, default_system):
        registry.save("task_a", default_system)
        with pytest.raises(KeyError):
            registry.promote("task_a", 99)

    def test_rollback(self, registry, default_system):
        registry.save("task_a", default_system)
        v2 = registry.save("task_a", default_system)
        prev = registry.rollback("task_a")
        assert prev == v2 - 1

    def test_rollback_at_v1_raises(self, registry, default_system):
        registry.save("task_a", default_system)
        with pytest.raises(ValueError):
            registry.rollback("task_a")


# ---------------------------------------------------------------------------
# SnapshotRegistry — list_tasks / history / prune / delete_task
# ---------------------------------------------------------------------------

class TestRegistryManagement:
    def test_list_tasks_empty(self, registry):
        assert registry.list_tasks() == []

    def test_list_tasks_after_save(self, registry, default_system):
        registry.save("task_a", default_system)
        registry.save("task_b", default_system)
        names = [t["task_name"] for t in registry.list_tasks()]
        assert "task_a" in names
        assert "task_b" in names

    def test_history_returns_records(self, registry, default_system):
        registry.save("task_a", default_system)
        registry.save("task_a", default_system)
        records = registry.history("task_a")
        assert len(records) == 2
        assert all(isinstance(r, SnapshotRecord) for r in records)

    def test_prune_removes_old_versions(self, registry, default_system):
        for _ in range(7):
            registry.save("task_a", default_system)
        removed = registry.prune("task_a", keep=3)
        assert removed == 4
        assert len(registry.history("task_a")) == 3

    def test_delete_task(self, registry, default_system):
        registry.save("task_a", default_system)
        registry.delete_task("task_a")
        with pytest.raises(KeyError):
            registry.load("task_a")
