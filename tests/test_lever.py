"""
tests/test_lever.py
===================
Unit tests for ksa_lever.py — Lever, ThreeBarSystem, EquilibriumResult.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ksa_lever import (
    Lever,
    ThreeBarSystem,
    TiltDirection,
)

# ---------------------------------------------------------------------------
# Lever tests
# ---------------------------------------------------------------------------

class TestLever:
    def test_tilt_left_when_left_heavy(self):
        lever = Lever(lever_id=0, left_arm_length=1.0, right_arm_length=1.0)
        lever.set_weights(left=8.0, right=2.0)
        state = lever.evaluate()
        assert state.tilt == TiltDirection.LEFT
        assert state.net_torque > 0

    def test_tilt_right_when_right_heavy(self):
        lever = Lever(lever_id=0, left_arm_length=1.0, right_arm_length=1.0)
        lever.set_weights(left=2.0, right=8.0)
        state = lever.evaluate()
        assert state.tilt == TiltDirection.RIGHT
        assert state.net_torque < 0

    def test_balanced_equal_weights(self):
        lever = Lever(lever_id=0)
        lever.set_weights(left=5.0, right=5.0)
        state = lever.evaluate()
        assert state.tilt == TiltDirection.BALANCED

    def test_fulcrum_bias_shifts_tilt(self):
        lever = Lever(lever_id=0, fulcrum_bias=10.0)
        lever.set_weights(left=0.0, right=0.0)
        state = lever.evaluate()
        assert state.tilt == TiltDirection.LEFT

    def test_arm_length_affects_torque(self):
        lever = Lever(lever_id=0, left_arm_length=2.0, right_arm_length=1.0)
        lever.set_weights(left=5.0, right=5.0)
        # left torque = 10, right torque = 5 → tilts left
        state = lever.evaluate()
        assert state.tilt == TiltDirection.LEFT

    def test_add_weight_left(self):
        lever = Lever(lever_id=0)
        lever.add_weight("left", 3.0)
        assert lever.left_weight == pytest.approx(3.0)

    def test_add_weight_right(self):
        lever = Lever(lever_id=0)
        lever.add_weight("right", 4.5)
        assert lever.right_weight == pytest.approx(4.5)

    def test_add_weight_invalid_side(self):
        lever = Lever(lever_id=0)
        with pytest.raises(ValueError):
            lever.add_weight("center", 1.0)

    def test_add_weight_floor_zero(self):
        lever = Lever(lever_id=0, left_weight=1.0)
        lever.add_weight("left", -100.0)
        assert lever.left_weight == pytest.approx(0.0)

    def test_tilt_magnitude_is_absolute(self):
        lever = Lever(lever_id=0)
        lever.set_weights(left=0.0, right=5.0)
        state = lever.evaluate()
        assert state.tilt_magnitude == pytest.approx(abs(state.net_torque))

    def test_serialisation_roundtrip(self):
        lever = Lever(lever_id=1, left_arm_length=1.5, right_arm_length=2.0,
                      fulcrum_bias=0.3, left_weight=4.0, right_weight=2.0)
        d      = lever.to_dict()
        lever2 = Lever.from_dict(d)
        assert lever2.left_arm_length  == pytest.approx(lever.left_arm_length)
        assert lever2.right_arm_length == pytest.approx(lever.right_arm_length)
        assert lever2.fulcrum_bias     == pytest.approx(lever.fulcrum_bias)
        assert lever2.left_weight      == pytest.approx(lever.left_weight)
        assert lever2.right_weight     == pytest.approx(lever.right_weight)


# ---------------------------------------------------------------------------
# ThreeBarSystem tests
# ---------------------------------------------------------------------------

class TestThreeBarSystem:
    def test_requires_exactly_3_levers(self):
        with pytest.raises(ValueError):
            ThreeBarSystem(levers=[Lever(0), Lever(1)])

    def test_from_defaults_returns_valid_system(self):
        sys = ThreeBarSystem.from_defaults()
        assert len(sys.levers) == 3

    def test_simulate_non_destructive(self):
        sys = ThreeBarSystem.from_defaults()
        sys.levers[0].set_weights(left=5.0, right=2.0)
        original_left = sys.levers[0].left_weight
        sys.simulate()
        assert sys.levers[0].left_weight == pytest.approx(original_left)

    def test_heavy_left_produces_left_tilt(self):
        sys = ThreeBarSystem.from_defaults()
        sys.levers[0].set_weights(left=10.0, right=0.0)
        result = sys.simulate()
        assert result.final_tilt == TiltDirection.LEFT

    def test_heavy_right_produces_right_tilt(self):
        sys = ThreeBarSystem.from_defaults()
        sys.levers[0].set_weights(left=0.0, right=10.0)
        result = sys.simulate()
        assert result.final_tilt == TiltDirection.RIGHT

    def test_balancer_override_on_extreme_load(self):
        sys = ThreeBarSystem.from_defaults()
        sys.levers[0].set_weights(left=50.0, right=0.0)
        sys.levers[1].set_weights(left=20.0, right=0.0)
        result = sys.simulate()
        assert result.override_active is True
        assert result.final_tilt == TiltDirection.BALANCED

    def test_confidence_between_0_and_1(self):
        sys = ThreeBarSystem.from_defaults()
        sys.levers[0].set_weights(left=5.0, right=2.0)
        result = sys.simulate()
        assert 0.0 <= result.confidence <= 1.0

    def test_snapshot_hydrate_roundtrip(self):
        sys = ThreeBarSystem.from_defaults()
        sys.levers[0].set_weights(left=3.0, right=7.0)
        snap  = sys.snapshot()
        sys2  = ThreeBarSystem()
        sys2.hydrate(snap)
        r1 = sys.simulate()
        r2 = sys2.simulate()
        assert r1.final_tilt == r2.final_tilt
        assert r1.confidence == pytest.approx(r2.confidence)

    def test_snapshot_json_roundtrip(self, tmp_path):
        sys  = ThreeBarSystem.from_defaults()
        sys.levers[0].set_weights(left=4.0, right=1.0)
        path = str(tmp_path / "snap.json")
        sys.save_snapshot(path)
        sys2 = ThreeBarSystem.load_snapshot(path)
        assert sys.simulate().final_tilt == sys2.simulate().final_tilt

    def test_linkage_matrix_propagates_weight(self):
        # Only Lever 0 has non-zero weight; coupling should affect Lever 1
        sys = ThreeBarSystem.from_defaults()
        sys.levers[0].set_weights(left=5.0, right=0.0)
        sys.levers[1].set_weights(left=0.0, right=0.0)
        result = sys.simulate()
        # Lever 1 should receive some weight from Lever 0 via coupling [0][1]=0.5
        assert result.states[1].tilt_magnitude > 0


# ---------------------------------------------------------------------------
# EquilibriumResult tests
# ---------------------------------------------------------------------------

class TestEquilibriumResult:
    def test_str_representation_contains_tilt(self):
        sys    = ThreeBarSystem.from_defaults()
        result = sys.simulate()
        s      = str(result)
        assert result.final_tilt.value.upper() in s

    def test_three_lever_states_returned(self):
        sys    = ThreeBarSystem.from_defaults()
        result = sys.simulate()
        assert len(result.states) == 3
        assert result.states[0].lever_id == 0
        assert result.states[1].lever_id == 1
        assert result.states[2].lever_id == 2
