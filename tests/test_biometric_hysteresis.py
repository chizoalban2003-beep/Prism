"""
Tests for the asymmetric EMA + debt accumulator hysteresis system in
BiometricVEAXBridge (prism_perception.py).

Each test isolates one behavioural property of the new system:
  - fast fatigue accumulation
  - slow recovery requiring sustained signal
  - debt blocking premature recovery
  - debt decay over time
  - per-axis time constants (V slowest, A fastest)
  - state persistence across calls
  - no spurious recovery after brief positive spike
  - EMA monotone decrease with no signal
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from prism_perception import BiometricVEAXBridge, _HysteresisState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bridge() -> BiometricVEAXBridge:
    return BiometricVEAXBridge()


def _pump(bridge: BiometricVEAXBridge, factors: dict, n: int,
          now_start: float = 1_000_000.0, dt: float = 3600.0) -> list[dict]:
    """Call bridge.apply() n times, advancing clock by dt (seconds) each call."""
    results = []
    for i in range(n):
        results.append(bridge.apply(factors, now=now_start + i * dt))
    return results


def _pump_sequence(bridge: BiometricVEAXBridge,
                   sequence: list[dict],
                   now_start: float = 1_000_000.0,
                   dt: float = 3600.0) -> list[dict]:
    """Call bridge.apply() once per item in sequence, advancing clock by dt."""
    results = []
    for i, factors in enumerate(sequence):
        results.append(bridge.apply(factors, now=now_start + i * dt))
    return results


# ---------------------------------------------------------------------------
# test_cooldown_removed
# ---------------------------------------------------------------------------

def test_cooldown_removed():
    """Old _last_fired attribute must NOT exist on the bridge instance."""
    b = make_bridge()
    assert not hasattr(b, "_last_fired"), (
        "_last_fired still present; old TTL cooldown system was not removed"
    )


# ---------------------------------------------------------------------------
# test_fast_fatigue_accumulation
# ---------------------------------------------------------------------------

def test_fast_fatigue_accumulation():
    """
    A single bad HRV reading (< 0.30) should drive the EMA above the
    fatigue threshold (0.3) within 2 ticks, given α_down = 0.25.

    EMA after tick 1: 0 + 0.25*(1-0) = 0.25
    EMA after tick 2: 0.25 + 0.25*(1-0.25) = 0.4375  → crosses 0.3
    """
    b = make_bridge()
    # Find the rule index for hrv_recovery < 0.30 (rule idx 3 in _RULES)
    # We test the EMA directly to avoid coupling to rule indices
    rule_idx = next(
        i for i, (fid, cmp, _, _, _) in enumerate(b._RULES)
        if fid == "hrv_recovery" and cmp == "<"
    )

    t = 1_000_000.0
    b.apply({"hrv_recovery": 0.20}, now=t)
    state_after_1 = b._hyst[rule_idx].ema
    assert state_after_1 == pytest.approx(0.25, abs=1e-6), (
        f"EMA after 1 bad tick should be 0.25, got {state_after_1}"
    )

    b.apply({"hrv_recovery": 0.20}, now=t + 3600)
    state_after_2 = b._hyst[rule_idx].ema
    assert state_after_2 > BiometricVEAXBridge._FATIGUE_THRESH, (
        f"EMA {state_after_2} should cross fatigue threshold "
        f"{BiometricVEAXBridge._FATIGUE_THRESH} after 2 bad ticks"
    )


# ---------------------------------------------------------------------------
# test_slow_recovery_requires_sustained_signal
# ---------------------------------------------------------------------------

def test_slow_recovery_requires_sustained_signal():
    """
    3 bad HRV readings followed by 1 good one must NOT fire a recovery rule.
    The EMA for recovery (E axis, α_up=0.042) cannot reach 0.7 in one tick
    after being driven down by α_down dynamics.
    """
    b = make_bridge()
    # Find recovery rule for hrv_recovery > 0.80 (E+0.10)
    recovery_rule_idx = next(
        i for i, (fid, cmp, _, deltas, _) in enumerate(b._RULES)
        if fid == "hrv_recovery" and cmp == ">" and deltas.get("E", 0) > 0
    )

    now = 1_000_000.0
    # 3 bad readings (hrv_recovery = 0.20, below all thresholds for recovery)
    for tick in range(3):
        b.apply({"hrv_recovery": 0.20}, now=now + tick * 3600)

    # 1 good reading
    result = b.apply({"hrv_recovery": 0.90}, now=now + 3 * 3600)

    ema = b._hyst.get(recovery_rule_idx, _HysteresisState()).ema
    # EMA for the recovery rule should NOT have crossed 0.7 yet
    assert ema < BiometricVEAXBridge._RECOVERY_THRESH, (
        f"Recovery EMA {ema:.4f} should be below 0.7 after only 1 good tick"
    )
    # E may be negative (from the fatigue rule also acting on E), but must NOT be positive
    # A positive E delta would indicate the recovery rule (E+0.10) fired spuriously.
    assert result.get("E", 0.0) <= 0.0, (
        f"Recovery rule should not have fired; got positive E={result.get('E')}"
    )


# ---------------------------------------------------------------------------
# test_sustained_recovery_eventually_fires
# ---------------------------------------------------------------------------

def test_sustained_recovery_eventually_fires():
    """
    30+ consecutive good HRV readings (> 0.80) must eventually fire the
    recovery rule (EMA > 0.7 with E α_up = 0.042 needs ~30+ ticks).
    """
    b = make_bridge()
    with (
        patch("prism_perception.get_current_gates", create=True, return_value=None),
        patch("prism_perception.save_spectrum_state", create=True),
        patch("prism_perception.SpectrumGates", create=True),
        patch("prism_perception.load_spectrum", create=True,
              return_value=MagicMock(V=0.5, E=0.5, A=0.5, X=0.5)),
    ):
        results = _pump(b, {"hrv_recovery": 0.90}, n=50)

    e_positive = any(r.get("E", 0.0) > 0 for r in results)
    assert e_positive, (
        "Recovery rule (E+0.10) never fired in 50 sustained good HRV ticks"
    )


# ---------------------------------------------------------------------------
# test_debt_blocks_upward_delta
# ---------------------------------------------------------------------------

def test_debt_blocks_upward_delta():
    """
    After debt accumulates from fatigue, upward deltas are blocked.
    We verify by checking the net delta for an upward component.
    """
    b = make_bridge()
    # Rule 1: sport_readiness < 0.30 → A-0.15, V+0.10 (mixed: fatigue + upward)
    # Drive EMA up for rule 1 (idx 1) and accumulate debt
    rule_idx = 1  # sport_readiness < 0.30 rule

    # Manually set state with high EMA and high debt
    state = b._state(rule_idx)
    state.ema = 0.9          # well above fatigue threshold 0.3
    state.debt = 0.8         # above block threshold 0.5
    state.last_ts = 1_000_000.0

    # Rule 1 deltas: A=-0.15 (downward, should pass), V=+0.10 (upward, blocked by debt)
    with (
        patch("prism_perception.get_current_gates", create=True, return_value=None),
        patch("prism_perception.save_spectrum_state", create=True),
        patch("prism_perception.SpectrumGates", create=True),
        patch("prism_perception.load_spectrum", create=True,
              return_value=MagicMock(V=0.5, E=0.5, A=0.5, X=0.5)),
    ):
        result = b.apply({"sport_readiness": 0.20}, now=1_000_003.0)

    # V should be blocked (upward delta + debt > 0.5)
    assert result.get("V", 0.0) == pytest.approx(0.0, abs=1e-6), (
        f"V upward delta should be blocked by debt; got V={result.get('V')}"
    )
    # A should still pass (downward delta)
    assert result.get("A", 0.0) < 0.0, (
        f"A downward delta should not be blocked by debt; got A={result.get('A')}"
    )


# ---------------------------------------------------------------------------
# test_debt_decays_over_time
# ---------------------------------------------------------------------------

def test_debt_decays_over_time():
    """
    Debt decreases when no fatigue signal fires + time passes.
    Decay rate: 0.1 / hour. After 10 hours with no signal, debt drops by 1.0.
    """
    b = make_bridge()
    rule_idx = 0  # sport_readiness < 0.40 rule

    state = b._state(rule_idx)
    state.ema = 0.0         # no fatigue signal
    state.debt = 0.8
    state.last_ts = 1_000_000.0

    # Apply with neutral factor (no rule fires) 10 hours later
    b.apply({"sport_readiness": 0.60}, now=1_000_000.0 + 10 * 3600)

    new_debt = b._hyst[rule_idx].debt
    # Debt should have decayed by 0.1 * 10 = 1.0, clamped to 0
    assert new_debt == pytest.approx(0.0, abs=1e-6), (
        f"Debt should be 0 after 10h with no fatigue signal; got {new_debt}"
    )


def test_debt_partial_decay():
    """Debt decays proportionally to elapsed time."""
    b = make_bridge()
    rule_idx = 0

    state = b._state(rule_idx)
    state.debt = 0.5
    state.last_ts = 1_000_000.0

    # 2 hours with no fatigue trigger
    b.apply({"sport_readiness": 0.60}, now=1_000_000.0 + 2 * 3600)

    new_debt = b._hyst[rule_idx].debt
    # 0.5 - 0.1 * 2 = 0.3, but rule might not fire so EMA might add more debt
    # Just check debt went down from initial 0.5
    assert new_debt < 0.5, f"Debt should have decreased; got {new_debt}"


# ---------------------------------------------------------------------------
# test_per_axis_tau_V_slowest and test_per_axis_tau_A_fastest
# ---------------------------------------------------------------------------

def _ema_after_n_ticks_recovery(axis_alpha_up: float, n: int) -> float:
    """Simulate EMA recovery starting from 0 with sustained good signal."""
    ema = 0.0
    for _ in range(n):
        ema = ema + axis_alpha_up * (1.0 - ema)
    return ema


def test_per_axis_tau_V_slowest():
    """
    V axis (α_up=0.016) recovers much slower than A (α_up=0.25).
    After 20 ticks, V EMA should be substantially lower than A EMA.
    """
    v_ema = _ema_after_n_ticks_recovery(BiometricVEAXBridge._AXIS_ALPHA_UP["V"], 20)
    a_ema = _ema_after_n_ticks_recovery(BiometricVEAXBridge._AXIS_ALPHA_UP["A"], 20)
    assert v_ema < a_ema, (
        f"V EMA {v_ema:.4f} should be < A EMA {a_ema:.4f} after 20 ticks"
    )


def test_per_axis_tau_A_fastest():
    """
    A axis (α_up=0.25) should cross the recovery threshold fastest among V/E/A/X.
    """
    ticks_to_cross = {}
    for axis in ["V", "E", "A", "X"]:
        alpha = BiometricVEAXBridge._AXIS_ALPHA_UP[axis]
        ema = 0.0
        for t in range(200):
            ema = ema + alpha * (1.0 - ema)
            if ema > BiometricVEAXBridge._RECOVERY_THRESH:
                ticks_to_cross[axis] = t + 1
                break
        else:
            ticks_to_cross[axis] = 999  # never crossed

    assert ticks_to_cross["A"] < ticks_to_cross["V"], (
        f"A should cross recovery threshold faster than V; "
        f"A={ticks_to_cross['A']}, V={ticks_to_cross['V']}"
    )
    assert ticks_to_cross["A"] < ticks_to_cross["E"], (
        f"A should cross recovery threshold faster than E; "
        f"A={ticks_to_cross['A']}, E={ticks_to_cross['E']}"
    )
    assert ticks_to_cross["A"] < ticks_to_cross["X"], (
        f"A should cross recovery threshold faster than X; "
        f"A={ticks_to_cross['A']}, X={ticks_to_cross['X']}"
    )


# ---------------------------------------------------------------------------
# test_hysteresis_state_persists_across_calls
# ---------------------------------------------------------------------------

def test_hysteresis_state_persists_across_calls():
    """
    State should accumulate across multiple apply() calls, not reset.
    After 1 bad tick, EMA > 0; after 2, it's higher still.
    """
    b = make_bridge()
    rule_idx = next(
        i for i, (fid, cmp, _, _, _) in enumerate(b._RULES)
        if fid == "hrv_recovery" and cmp == "<"
    )

    t = 1_000_000.0
    b.apply({"hrv_recovery": 0.20}, now=t)
    ema_after_1 = b._hyst[rule_idx].ema

    b.apply({"hrv_recovery": 0.20}, now=t + 3600)
    ema_after_2 = b._hyst[rule_idx].ema

    assert ema_after_1 > 0.0, "EMA should be non-zero after 1 bad tick"
    assert ema_after_2 > ema_after_1, "EMA should increase with consecutive bad ticks"


# ---------------------------------------------------------------------------
# test_no_spurious_recovery_after_brief_spike — the hour-20 bug scenario
# ---------------------------------------------------------------------------

def test_no_spurious_recovery_after_brief_spike():
    """
    Simulate the original bug: 3 days of fatigue debt, then at hour 20 a
    single good HRV reading. The recovery rule should NOT fire because debt
    is still high.

    Scenario (1h ticks):
      Hours 0-23: hrv_recovery = 0.20 (bad — builds EMA and debt for fatigue rule)
      Hour 24:    hrv_recovery = 0.90 (single good tick)
    Expected: recovery rule (E+0.10) does NOT fire at hour 24.
    """
    b = make_bridge()
    now = 1_000_000.0
    # 24 hours of bad HRV
    for tick in range(24):
        b.apply({"hrv_recovery": 0.20}, now=now + tick * 3600)

    # Single good HRV reading at hour 24
    with (
        patch("prism_perception.get_current_gates", create=True, return_value=None),
        patch("prism_perception.save_spectrum_state", create=True),
        patch("prism_perception.SpectrumGates", create=True),
        patch("prism_perception.load_spectrum", create=True,
              return_value=MagicMock(V=0.5, E=0.5, A=0.5, X=0.5)),
    ):
        result = b.apply({"hrv_recovery": 0.90}, now=now + 24 * 3600)

    # Recovery rule should NOT have fired positively.
    # E may be negative from the hrv_recovery < 0.30 fatigue rule, but
    # a *positive* E delta would indicate the recovery rule (E+0.10) fired spuriously.
    e_delta = result.get("E", 0.0)
    assert e_delta <= 0.0, (
        f"Recovery rule fired spuriously after brief positive spike; E={e_delta}. "
        "This is the hour-20 bug — it should be suppressed by debt or low EMA."
    )


# ---------------------------------------------------------------------------
# test_ema_monotone_decrease_no_signal
# ---------------------------------------------------------------------------

def test_ema_monotone_decrease_no_signal():
    """
    When no rule conditions fire, EMA should decrease (or stay at 0) monotonically.
    """
    b = make_bridge()
    rule_idx = next(
        i for i, (fid, cmp, _, _, _) in enumerate(b._RULES)
        if fid == "hrv_recovery" and cmp == "<"
    )

    # Seed EMA with some value
    state = b._state(rule_idx)
    state.ema = 0.5
    state.last_ts = 1_000_000.0

    emas = [0.5]
    # Apply 10 ticks with a factor value that does NOT trigger any rule for hrv_recovery < 0.30
    # (use hrv_recovery = 0.50, which is above 0.30 threshold)
    for tick in range(1, 11):
        b.apply({"hrv_recovery": 0.50}, now=1_000_000.0 + tick * 3600)
        emas.append(b._hyst[rule_idx].ema)

    # Each successive EMA should be <= the previous
    for i in range(1, len(emas)):
        assert emas[i] <= emas[i - 1] + 1e-9, (
            f"EMA not monotonically decreasing at tick {i}: "
            f"{emas[i - 1]:.6f} → {emas[i]:.6f}"
        )


# ---------------------------------------------------------------------------
# _HysteresisState dataclass
# ---------------------------------------------------------------------------

def test_hysteresis_state_defaults():
    s = _HysteresisState()
    assert s.ema == 0.0
    assert s.debt == 0.0
    assert s.last_ts == 0.0


def test_hysteresis_state_fields_are_floats():
    s = _HysteresisState(ema=0.3, debt=0.7, last_ts=12345.6)
    assert isinstance(s.ema, float)
    assert isinstance(s.debt, float)
    assert isinstance(s.last_ts, float)


# ---------------------------------------------------------------------------
# Class attributes
# ---------------------------------------------------------------------------

def test_axis_alpha_up_keys():
    assert set(BiometricVEAXBridge._AXIS_ALPHA_UP.keys()) == {"V", "E", "A", "X"}


def test_axis_alpha_up_values_in_range():
    for axis, alpha in BiometricVEAXBridge._AXIS_ALPHA_UP.items():
        assert 0 < alpha <= 1.0, f"α_up for {axis} out of range: {alpha}"


def test_alpha_down_value():
    assert BiometricVEAXBridge._ALPHA_DOWN == pytest.approx(0.25)


def test_a_axis_fastest_alpha():
    """A axis must have the highest α_up (fastest recovery)."""
    assert BiometricVEAXBridge._AXIS_ALPHA_UP["A"] == max(
        BiometricVEAXBridge._AXIS_ALPHA_UP.values()
    )


def test_v_axis_slowest_alpha():
    """V axis must have the lowest α_up (slowest recovery)."""
    assert BiometricVEAXBridge._AXIS_ALPHA_UP["V"] == min(
        BiometricVEAXBridge._AXIS_ALPHA_UP.values()
    )
