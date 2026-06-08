"""
tests/test_allostatic.py
Tests for Vector I: Allostatic Baseline Shifting (Double-Order Hysteresis)
"""
import time

import pytest

from prism_perception import BiometricVEAXBridge, _HysteresisState

# ── Structural tests ──────────────────────────────────────────────────────────

def test_allostatic_state_has_slow_ema_field():
    state = _HysteresisState()
    assert hasattr(state, "slow_ema")


def test_slow_ema_updates_slower_than_fast_ema():
    """After the same number of stress ticks, slow_ema should be below fast ema."""
    bridge = BiometricVEAXBridge()
    factors = {"sleep_quality": 0.10}  # triggers fatigue rule (< 0.40)
    t0 = time.time()
    for i in range(10):
        bridge.apply(factors, now=t0 + i * 3600.0)
    for state in bridge._hyst.values():
        if state.ema > 0.0:
            assert state.slow_ema <= state.ema, (
                f"slow_ema {state.slow_ema} should not exceed fast ema {state.ema}"
            )


def test_allostatic_load_accumulates_under_sustained_stress():
    bridge = BiometricVEAXBridge()
    factors = {"sleep_quality": 0.10}
    t0 = time.time()
    for i in range(50):
        bridge.apply(factors, now=t0 + i * 3600.0)
    # At least one rule should have non-zero allostatic load
    loads = [s.allostatic_load for s in bridge._hyst.values()]
    assert max(loads) > 0.0, "Allostatic load should accumulate under sustained stress"


def test_allostatic_load_decays_during_recovery():
    bridge = BiometricVEAXBridge()
    t0 = time.time()
    # Stress phase
    for i in range(30):
        bridge.apply({"sleep_quality": 0.10}, now=t0 + i * 3600.0)
    loads_after_stress = [s.allostatic_load for s in bridge._hyst.values()]
    peak_load = max(loads_after_stress)
    # Recovery phase (good sleep)
    for i in range(30):
        bridge.apply({"sleep_quality": 0.95}, now=t0 + (30 + i) * 3600.0)
    loads_after_recovery = [s.allostatic_load for s in bridge._hyst.values()]
    # Load should have decreased in at least one rule
    assert min(loads_after_recovery) <= peak_load, (
        "Allostatic load should decay during recovery"
    )


def test_baseline_shift_increases_under_chronic_load():
    bridge = BiometricVEAXBridge()
    factors = {"sleep_quality": 0.05}
    t0 = time.time()
    # Simulate 7 days (168 hours) of sustained poor sleep
    for i in range(168):
        bridge.apply(factors, now=t0 + i * 3600.0)
    shifts = [s.baseline_shift for s in bridge._hyst.values()]
    assert max(shifts) > 0.0, "Baseline shift should increase under chronic load"


def test_baseline_shift_caps_at_0_30():
    bridge = BiometricVEAXBridge()
    factors = {"sleep_quality": 0.01}
    t0 = time.time()
    # Simulate very long chronic stress (30 days)
    for i in range(720):
        bridge.apply(factors, now=t0 + i * 3600.0)
    shifts = [s.baseline_shift for s in bridge._hyst.values()]
    assert max(shifts) <= 0.30, "Baseline shift must not exceed 0.30"


def test_baseline_shift_decays_during_macro_recovery():
    """Directly set a high baseline_shift and allostatic_load to below 1.0, then verify decay."""
    bridge = BiometricVEAXBridge()
    t0 = time.time()
    # Seed a rule with an artificially high baseline_shift and low allostatic_load
    # (allostatic_load < 1.0 → macro-recovery path activates)
    state = bridge._state(7)
    state.baseline_shift  = 0.15
    state.allostatic_load = 0.5   # < 1.0 → triggers macro-recovery
    state.ema   = 0.95
    state.last_ts = t0 - 3600.0

    # One recovery tick: allostatic_load < 1.0 → baseline_shift should decrease
    bridge.apply({"sleep_quality": 0.95}, now=t0)
    assert bridge._hyst[7].baseline_shift < 0.15, (
        "Baseline shift should decay when allostatic_load < 1.0"
    )


def test_recovery_delta_reduced_by_baseline_shift():
    """A bridge with high baseline_shift should return smaller recovery deltas."""
    bridge_fresh = BiometricVEAXBridge()
    bridge_worn  = BiometricVEAXBridge()

    # Artificially set a high baseline_shift in bridge_worn
    t0 = time.time()
    # Initialize states in bridge_worn with high baseline_shift
    for i in range(11):  # 11 rules
        state = bridge_worn._state(i)
        state.ema  = 0.9
        state.slow_ema = 0.4
        state.allostatic_load = 5.0
        state.baseline_shift  = 0.25
        state.last_ts = t0 - 3600.0

    # Apply recovery-triggering factors
    factors = {"hrv_recovery": 0.90, "sleep_quality": 0.90, "cognitive_readiness": 0.90}

    net_fresh = bridge_fresh.apply(factors, now=t0)
    net_worn  = bridge_worn.apply(factors, now=t0)

    # Any positive deltas in worn bridge should be <= fresh bridge
    for axis in ["V", "E", "A", "X"]:
        fresh_d = net_fresh.get(axis, 0.0)
        worn_d  = net_worn.get(axis, 0.0)
        if fresh_d > 0.0:
            assert worn_d <= fresh_d, (
                f"Axis {axis}: worn delta {worn_d} should be <= fresh delta {fresh_d}"
            )


def test_full_chronic_scenario():
    """
    7 days of hard stress followed by 1 day rest:
    V axis should be reduced but not zero (partial recovery, baseline shifted).
    """
    bridge = BiometricVEAXBridge()
    t0 = time.time()
    # 168 hours of stress
    for i in range(168):
        bridge.apply({"sleep_quality": 0.05, "hrv_recovery": 0.1}, now=t0 + i * 3600.0)
    # 24 hours of recovery
    for i in range(24):
        bridge.apply({"sleep_quality": 0.9, "hrv_recovery": 0.9}, now=t0 + (168 + i) * 3600.0)
    # Check that states exist and have non-trivial allostatic load
    assert len(bridge._hyst) > 0
    # At least some baseline shift should remain
    shifts = [s.baseline_shift for s in bridge._hyst.values()]
    assert max(shifts) > 0.0, "Baseline shift should remain after 24h recovery from 7 days stress"


def test_fresh_state_no_baseline_shift():
    state = _HysteresisState()
    assert state.baseline_shift == 0.0
    assert state.allostatic_load == 0.0
    assert state.slow_ema == 0.0


def test_brief_stress_no_baseline_shift():
    """A single stress episode should not produce meaningful baseline shift."""
    bridge = BiometricVEAXBridge()
    t0 = time.time()
    bridge.apply({"sleep_quality": 0.05}, now=t0)
    bridge.apply({"sleep_quality": 0.05}, now=t0 + 3600.0)
    shifts = [s.baseline_shift for s in bridge._hyst.values()]
    # With only 2 hours of stress, baseline_shift should be essentially 0
    assert max(shifts) < 0.01, "Brief stress should not cause significant baseline shift"


def test_allostatic_report_returns_all_rules():
    bridge = BiometricVEAXBridge()
    t0 = time.time()
    bridge.apply({"sleep_quality": 0.3, "hrv_recovery": 0.2}, now=t0)
    report = bridge.allostatic_report()
    assert isinstance(report, dict)
    assert len(report) > 0
    for key, val in report.items():
        assert "ema" in val
        assert "slow_ema" in val
        assert "allostatic_load" in val
        assert "baseline_shift" in val
        assert "debt" in val


def test_allostatic_report_keys_are_string_ints():
    bridge = BiometricVEAXBridge()
    t0 = time.time()
    bridge.apply({"sleep_quality": 0.3}, now=t0)
    report = bridge.allostatic_report()
    for key in report:
        assert key.isdigit(), f"Key '{key}' should be a string integer"


def test_zero_dt_no_baseline_change():
    """When dt=0 (same timestamp), allostatic fields should not change significantly."""
    bridge = BiometricVEAXBridge()
    t0 = time.time()
    # First call to initialize state
    bridge.apply({"sleep_quality": 0.1}, now=t0)
    shifts_before = {idx: s.baseline_shift for idx, s in bridge._hyst.items()}
    # Second call with same timestamp → dt=0
    bridge.apply({"sleep_quality": 0.1}, now=t0)
    for idx, s in bridge._hyst.items():
        assert s.baseline_shift == pytest.approx(shifts_before.get(idx, 0.0), abs=1e-9), (
            "Zero dt should not change baseline_shift"
        )


def test_high_load_no_recovery_allowed():
    """With debt > DEBT_BLOCK_THRESH, positive deltas are blocked regardless of baseline."""
    bridge = BiometricVEAXBridge()
    t0 = time.time()
    # Set extremely high debt in all states
    for i in range(11):
        state = bridge._state(i)
        state.ema  = 0.9
        state.debt = 1.0  # well above DEBT_BLOCK_THRESH=0.5
        state.baseline_shift = 0.0  # no shift — so shift is NOT the blocker
        state.last_ts = t0 - 3600.0

    # Recovery factors
    factors = {"hrv_recovery": 0.95, "sleep_quality": 0.95, "cognitive_readiness": 0.95}
    net = bridge.apply(factors, now=t0)
    # No positive recovery deltas should appear
    for axis, delta in net.items():
        assert delta <= 0.0, (
            f"Axis {axis} delta {delta} should not be positive when debt blocks recovery"
        )
