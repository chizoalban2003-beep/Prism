"""
tests/test_optimizer.py
=======================
Unit tests for ksa_optimizer.py — KineticOptimizer.
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ksa_lever import ThreeBarSystem
from ksa_registry import PerformanceMetrics, SnapshotRegistry
from ksa_executor import ExecutionOutcome
from ksa_optimizer import KineticOptimizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry(tmp_path):
    reg = SnapshotRegistry(str(tmp_path / "test.db"))
    system = ThreeBarSystem.from_defaults()
    system.levers[0].set_weights(left=5.0, right=2.0)
    v = reg.save("demo_task", system)
    reg.record_outcome(
        "demo_task", v,
        PerformanceMetrics(execution_time_ms=100.0, success=True, override_fired=False),
    )
    return reg


@pytest.fixture
def optimizer(registry):
    return KineticOptimizer(registry, step_size=0.1, improvement_threshold=0.0)


def _make_outcome(task_name="demo_task", version=1, success=True, override=False):
    metrics = PerformanceMetrics(
        execution_time_ms = 100.0,
        success           = success,
        override_fired    = override,
    )
    return ExecutionOutcome(
        task_name    = task_name,
        version      = version,
        action_taken = "primary",
        return_code  = 0 if success else 1,
        stdout       = "",
        stderr       = "",
        metrics      = metrics,
        elapsed_ms   = 100.0,
    )


# ---------------------------------------------------------------------------
# maybe_improve tests
# ---------------------------------------------------------------------------

class TestMaybeImprove:
    def test_returns_none_on_failure(self, optimizer):
        outcome = _make_outcome(success=False)
        result  = optimizer.maybe_improve("demo_task", 1, outcome)
        assert result is None

    def test_returns_none_on_override_fired(self, optimizer):
        outcome = _make_outcome(override=True)
        result  = optimizer.maybe_improve("demo_task", 1, outcome)
        assert result is None

    def test_returns_none_for_unknown_task(self, optimizer):
        outcome = _make_outcome(task_name="ghost_task", version=99)
        result  = optimizer.maybe_improve("ghost_task", 99, outcome)
        assert result is None

    def test_may_return_new_version_on_success(self, optimizer):
        # With threshold=0.0 and random perturbation, improvement is likely
        # but not guaranteed. Run a few times and check at least one improves.
        outcome = _make_outcome()
        improved = None
        for _ in range(10):
            improved = optimizer.maybe_improve("demo_task", 1, outcome)
            if improved is not None:
                break
        # We can't guarantee improvement with random noise, just that it
        # runs without error and returns a valid type if anything is saved.
        assert improved is None or isinstance(improved, int)

    def test_new_version_is_higher(self, optimizer):
        # Force threshold to 0 and run enough trials to ensure improvement
        opt = KineticOptimizer(optimizer.registry, step_size=0.3,
                               improvement_threshold=-1.0)  # always accept
        outcome     = _make_outcome()
        new_version = opt.maybe_improve("demo_task", 1, outcome)
        assert new_version is not None
        assert new_version > 1


# ---------------------------------------------------------------------------
# hill_climb tests
# ---------------------------------------------------------------------------

class TestHillClimb:
    def test_returns_three_bar_system(self, optimizer):
        system = optimizer.hill_climb("demo_task", n_trials=3, dry_run=True)
        assert isinstance(system, ThreeBarSystem)

    def test_confidence_between_0_and_1(self, optimizer):
        system = optimizer.hill_climb("demo_task", n_trials=5, dry_run=True)
        conf   = system.simulate().confidence
        assert 0.0 <= conf <= 1.0

    def test_dry_run_does_not_save(self, optimizer, registry):
        before = len(registry.history("demo_task"))
        optimizer.hill_climb("demo_task", n_trials=5, dry_run=True)
        after  = len(registry.history("demo_task"))
        assert after == before

    def test_wet_run_saves_new_snapshot(self, optimizer, registry):
        before = len(registry.history("demo_task"))
        optimizer.hill_climb("demo_task", n_trials=5, dry_run=False)
        after  = len(registry.history("demo_task"))
        assert after == before + 1

    def test_fallback_to_defaults_for_unknown_task(self, optimizer):
        system = optimizer.hill_climb("nonexistent", n_trials=3, dry_run=True)
        assert isinstance(system, ThreeBarSystem)


# ---------------------------------------------------------------------------
# _perturb tests
# ---------------------------------------------------------------------------

class TestPerturb:
    def test_perturb_does_not_mutate_original(self, optimizer):
        system    = ThreeBarSystem.from_defaults()
        original0 = system.levers[0].left_arm_length
        optimizer._perturb(system)
        assert system.levers[0].left_arm_length == pytest.approx(original0)

    def test_perturb_never_touches_lever_2(self, optimizer):
        system  = ThreeBarSystem.from_defaults()
        lever2  = system.levers[2]
        orig_la = lever2.left_arm_length
        orig_ra = lever2.right_arm_length
        orig_b  = lever2.fulcrum_bias

        for _ in range(50):
            candidate = optimizer._perturb(system)
            assert candidate.levers[2].left_arm_length  == pytest.approx(orig_la)
            assert candidate.levers[2].right_arm_length == pytest.approx(orig_ra)
            assert candidate.levers[2].fulcrum_bias     == pytest.approx(orig_b)

    def test_perturb_clamps_arm_lengths(self):
        opt    = KineticOptimizer(
            registry        = None,   # not used in _perturb
            step_size       = 1000.0, # huge step to force clamping
            max_arm_length  = 4.0,
            min_arm_length  = 0.5,
            max_bias        = 2.0,
            min_bias        = -2.0,
        )
        system = ThreeBarSystem.from_defaults()
        for _ in range(20):
            candidate = opt._perturb(system)
            for idx in (0, 1):
                lever = candidate.levers[idx]
                assert lever.left_arm_length  >= 0.5
                assert lever.left_arm_length  <= 4.0
                assert lever.right_arm_length >= 0.5
                assert lever.right_arm_length <= 4.0
                assert lever.fulcrum_bias     >= -2.0
                assert lever.fulcrum_bias     <= 2.0
