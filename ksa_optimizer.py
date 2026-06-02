"""
ksa_optimizer.py
================
Kinetic State Agent — Kinetic Optimizer

After a task run, perturbs lever geometry with Gaussian noise if the new
configuration's simulated confidence is meaningfully better than the current
best. Gradient-free hill-climbing over arm lengths and fulcrum biases.

Hard rules:
    - Never mutate Lever 2 (Balancer). Only levers[0] and levers[1].
    - Arm lengths clamped to [min_arm_length, max_arm_length] = [0.5, 4.0].
    - Biases clamped to [min_bias, max_bias] = [-2.0, 2.0].
    - Skip if outcome.metrics.success is False OR override_active was True.
    - After saving a new snapshot, call registry.auto_promote_best().

Usage:
    opt = KineticOptimizer(registry)
    new_version = opt.maybe_improve("file_index_stealth", version=1, outcome=outcome)
    if new_version:
        print(f"Improved to v{new_version}")

    best_system = opt.hill_climb("file_index_stealth", n_trials=10)
"""

from __future__ import annotations

import copy
import logging
import random
from typing import Optional

from ksa_executor import ExecutionOutcome
from ksa_lever import ThreeBarSystem
from ksa_registry import SnapshotRegistry

logger = logging.getLogger(__name__)


class KineticOptimizer:
    """
    Gradient-free optimiser that perturbs Lever 0 and Lever 1 geometry
    using Gaussian noise, retaining improvements above a score threshold.

    Parameters
    ----------
    registry:
        The shared SnapshotRegistry.
    step_size:
        Standard deviation for the Gaussian noise applied to arm lengths
        and fulcrum biases during each perturbation.
    max_arm_length / min_arm_length:
        Clamp range for every arm length after perturbation.
    max_bias / min_bias:
        Clamp range for every fulcrum bias after perturbation.
    improvement_threshold:
        Minimum fractional improvement in score required to accept and save
        the perturbed configuration (default 0.02 = 2%).
    """

    def __init__(
        self,
        registry:              SnapshotRegistry,
        step_size:             float = 0.05,
        max_arm_length:        float = 4.0,
        min_arm_length:        float = 0.5,
        max_bias:              float = 2.0,
        min_bias:              float = -2.0,
        improvement_threshold: float = 0.02,
    ) -> None:
        self.registry              = registry
        self.step_size             = step_size
        self.max_arm_length        = max_arm_length
        self.min_arm_length        = min_arm_length
        self.max_bias              = max_bias
        self.min_bias              = min_bias
        self.improvement_threshold = improvement_threshold

    # ── Public API ────────────────────────────────────────────────────────────

    def maybe_improve(
        self,
        task_name: str,
        version:   int,
        outcome:   ExecutionOutcome,
    ) -> Optional[int]:
        """
        Attempt one perturbation step after a completed task run.

        Returns the new version number if an improved snapshot was saved,
        or None if the guards prevented optimisation or no improvement was found.

        Guard conditions that cause an immediate None return:
            - outcome.metrics.success is False
            - outcome.metrics.override_fired is True  (system was unstable)
        """
        # ── Guards ────────────────────────────────────────────────────────────
        if not outcome.metrics.success:
            logger.debug(
                "Skipping optimisation for '%s': task did not succeed.", task_name
            )
            return None

        if outcome.metrics.override_fired:
            logger.debug(
                "Skipping optimisation for '%s': Balancer override fired (unstable).",
                task_name,
            )
            return None

        # ── Load current best configuration ──────────────────────────────────
        try:
            current_system = self.registry.load(task_name, version)
        except KeyError:
            logger.warning(
                "Cannot optimise '%s' v%d: snapshot not found in registry.",
                task_name,
                version,
            )
            return None

        current_conf  = current_system.simulate().confidence

        # ── Perturb and evaluate ──────────────────────────────────────────────
        candidate = self._perturb(current_system)
        cand_conf = candidate.simulate().confidence

        # We use confidence as the proxy for "goodness" of lever geometry,
        # and require it to improve by at least improvement_threshold.
        required = current_conf * (1.0 + self.improvement_threshold)
        if cand_conf < required:
            logger.debug(
                "No improvement for '%s': %.4f → %.4f (required ≥ %.4f)",
                task_name,
                current_conf,
                cand_conf,
                required,
            )
            return None

        # ── Save and promote ──────────────────────────────────────────────────
        new_version = self.registry.save(task_name, candidate)
        logger.info(
            "Optimised '%s': conf %.4f → %.4f saved as v%d",
            task_name,
            current_conf,
            cand_conf,
            new_version,
        )

        promoted = self.registry.auto_promote_best(task_name)
        if promoted is not None:
            logger.debug("Auto-promoted '%s' to v%d", task_name, promoted)

        return new_version

    def hill_climb(
        self,
        task_name: str,
        n_trials:  int  = 5,
        dry_run:   bool = False,
    ) -> ThreeBarSystem:
        """
        Generate ``n_trials`` independent perturbations of the current best
        snapshot, simulate each, and return the one with the highest confidence.

        If ``dry_run`` is False (default) the winning system is saved to the
        registry and auto_promote_best is called.

        Always returns a ThreeBarSystem (falls back to from_defaults if the
        task has no snapshot yet).
        """
        try:
            base = self.registry.load(task_name)
        except KeyError:
            logger.info(
                "hill_climb: no snapshot for '%s', starting from defaults.", task_name
            )
            base = ThreeBarSystem.from_defaults()

        best_system = copy.deepcopy(base)
        best_conf   = best_system.simulate().confidence

        for i in range(n_trials):
            candidate = self._perturb(base)
            conf      = candidate.simulate().confidence
            logger.debug(
                "hill_climb trial %d/%d for '%s': conf=%.4f", i + 1, n_trials, task_name, conf
            )
            if conf > best_conf:
                best_conf   = conf
                best_system = candidate

        logger.info(
            "hill_climb result for '%s': best conf=%.4f after %d trials",
            task_name,
            best_conf,
            n_trials,
        )

        if not dry_run:
            new_ver = self.registry.save(task_name, best_system)
            self.registry.auto_promote_best(task_name)
            logger.info("hill_climb: saved '%s' as v%d", task_name, new_ver)

        return best_system

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _perturb(self, system: ThreeBarSystem) -> ThreeBarSystem:
        """
        Return a deep copy of ``system`` with Gaussian noise applied to
        levers[0] and levers[1]. Lever 2 (Balancer) is never touched.
        """
        candidate = copy.deepcopy(system)

        for idx in (0, 1):
            lever = candidate.levers[idx]

            lever.left_arm_length = self._clamp(
                lever.left_arm_length + random.gauss(0, self.step_size),
                self.min_arm_length,
                self.max_arm_length,
            )
            lever.right_arm_length = self._clamp(
                lever.right_arm_length + random.gauss(0, self.step_size),
                self.min_arm_length,
                self.max_arm_length,
            )
            lever.fulcrum_bias = self._clamp(
                lever.fulcrum_bias + random.gauss(0, self.step_size),
                self.min_bias,
                self.max_bias,
            )

        return candidate

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def __repr__(self) -> str:
        return (
            f"KineticOptimizer("
            f"step={self.step_size}, "
            f"arm=[{self.min_arm_length}, {self.max_arm_length}], "
            f"bias=[{self.min_bias}, {self.max_bias}], "
            f"threshold={self.improvement_threshold:.0%})"
        )


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import tempfile

    print("=== KSA Kinetic Optimizer Demo ===\n")

    from ksa_executor import ExecutionOutcome
    from ksa_registry import PerformanceMetrics

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        reg = SnapshotRegistry(db_path)

        # Seed a starting snapshot
        system  = ThreeBarSystem.from_defaults()
        system.levers[0].set_weights(left=5.0, right=3.0)
        version = reg.save("demo_task", system)

        metrics = PerformanceMetrics(
            execution_time_ms=120.0,
            cpu_peak_pct=10.0,
            ram_peak_mb=50.0,
            success=True,
            override_fired=False,
        )
        reg.record_outcome("demo_task", version, metrics)

        # Fake outcome wrapping the metrics
        outcome = ExecutionOutcome(
            task_name    = "demo_task",
            version      = version,
            action_taken = "primary",
            return_code  = 0,
            stdout       = "",
            stderr       = "",
            metrics      = metrics,
            elapsed_ms   = 120.0,
        )

        opt = KineticOptimizer(reg, step_size=0.1, improvement_threshold=0.0)
        print(f"Optimizer: {opt}\n")

        new_ver = opt.maybe_improve("demo_task", version, outcome)
        print(f"maybe_improve result: {'improved to v' + str(new_ver) if new_ver else 'no improvement'}")

        best = opt.hill_climb("demo_task", n_trials=10, dry_run=True)
        print(f"hill_climb confidence: {best.simulate().confidence:.4f}")

    finally:
        os.unlink(db_path)
        print("\nTemp DB cleaned up. ✓")
