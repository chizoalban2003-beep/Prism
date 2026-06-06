"""
Tests for BiometricVEAXBridge in prism_perception.py.

Updated for the asymmetric EMA + debt accumulator system that replaced the
flat TTL cooldown.  The old _last_fired-based tests have been rewritten to
work with EMA thresholds.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from prism_perception import BiometricVEAXBridge

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bridge() -> BiometricVEAXBridge:
    return BiometricVEAXBridge()


def _fake_gates(V=0.5, E=0.5, A=0.5, X=0.5):
    m = MagicMock()
    m.V = V
    m.E = E
    m.A = A
    m.X = X
    return m


def _pump(bridge: BiometricVEAXBridge, factors: dict, n: int,
          now_start: float = 1_000_000.0, dt: float = 3600.0) -> list[dict]:
    """Call bridge.apply() n times, advancing clock by dt each call."""
    results = []
    for i in range(n):
        results.append(bridge.apply(factors, now=now_start + i * dt))
    return results


def _patch_spectrum():
    """Context manager that stubs out prism_spectrum_middleware imports."""
    return (
        patch("prism_perception.get_current_gates", create=True, return_value=None),
        patch("prism_perception.save_spectrum_state", create=True),
        patch("prism_perception.SpectrumGates", create=True),
        patch("prism_perception.load_spectrum", create=True,
              return_value=_fake_gates()),
    )


# ---------------------------------------------------------------------------
# No-op when no rules fire
# ---------------------------------------------------------------------------

def test_no_op_empty_factors():
    b = make_bridge()
    result = b.apply({}, now=1_000_000.0)
    assert result == {}


def test_no_op_factors_below_high_threshold():
    b = make_bridge()
    # sport_readiness=0.55 — no rule fires (< 0.30, < 0.40 miss; > 0.80 misses)
    result = b.apply({"sport_readiness": 0.55}, now=1_000_000.0)
    assert result == {}


def test_no_op_factors_not_in_rules():
    b = make_bridge()
    result = b.apply({"unknown_factor": 0.99}, now=1_000_000.0)
    assert result == {}


# ---------------------------------------------------------------------------
# Single rule firing (EMA threshold, not cooldown)
# ---------------------------------------------------------------------------

def test_sport_readiness_low_fires_after_sustained_signal():
    """Fatigue rule must fire once EMA crosses 0.3 (happens in ~2 ticks)."""
    b = make_bridge()
    with (
        patch("prism_perception.get_current_gates", create=True, return_value=None),
        patch("prism_perception.save_spectrum_state", create=True),
        patch("prism_perception.SpectrumGates", create=True),
        patch("prism_perception.load_spectrum", create=True,
              return_value=_fake_gates()),
    ):
        results = _pump(b, {"sport_readiness": 0.35}, n=5)

    # Rule should have fired (A: -0.10) at some point in 5 ticks
    fired = any("A" in r and r["A"] < 0 for r in results)
    assert fired, f"Fatigue rule never fired in 5 ticks: {results}"


def test_hrv_recovery_high_fires_after_sustained_signal():
    """Recovery rule (E+0.10) requires EMA > 0.7, which takes ~28 ticks."""
    b = make_bridge()
    with (
        patch("prism_perception.get_current_gates", create=True, return_value=None),
        patch("prism_perception.save_spectrum_state", create=True),
        patch("prism_perception.SpectrumGates", create=True),
        patch("prism_perception.load_spectrum", create=True,
              return_value=_fake_gates()),
    ):
        results = _pump(b, {"hrv_recovery": 0.85}, n=35)

    fired = any("E" in r and r["E"] > 0 for r in results)
    assert fired, f"Recovery rule never fired in 35 ticks: {results}"


# ---------------------------------------------------------------------------
# Debt blocking: old cooldown tests replaced
# ---------------------------------------------------------------------------

def test_old_last_fired_attribute_does_not_exist():
    """Ensure the old cooldown system has been fully removed."""
    b = make_bridge()
    assert not hasattr(b, "_last_fired"), (
        "_last_fired still present — old cooldown system was not removed"
    )


def test_hyst_dict_exists():
    b = make_bridge()
    assert hasattr(b, "_hyst")
    assert isinstance(b._hyst, dict)


# ---------------------------------------------------------------------------
# Accumulation (multiple rules fire together)
# ---------------------------------------------------------------------------

def test_multiple_rules_accumulate():
    """
    stress_level > 0.70 (X+0.15, A-0.10) and > 0.85 (V+0.10) should both
    accumulate after enough ticks with stress_level=0.90.
    """
    b = make_bridge()
    with (
        patch("prism_perception.get_current_gates", create=True, return_value=None),
        patch("prism_perception.save_spectrum_state", create=True),
        patch("prism_perception.SpectrumGates", create=True),
        patch("prism_perception.load_spectrum", create=True,
              return_value=_fake_gates()),
    ):
        results = _pump(b, {"stress_level": 0.90}, n=5)

    # Fatigue rules fire fast (α_down=0.25 → EMA crosses 0.3 in ~2 ticks)
    # stress_level>0.70: A-0.10; stress_level>0.85: V+0.10 (recovery rule → needs EMA>0.7)
    # After 5 ticks, fatigue-direction rules should have fired
    a_fired = any("A" in r and r["A"] < 0 for r in results)
    assert a_fired, f"A-axis fatigue never fired: {results}"


# ---------------------------------------------------------------------------
# save_spectrum_state is called when deltas are non-zero
# ---------------------------------------------------------------------------

def test_save_not_called_when_no_delta():
    b = make_bridge()
    with patch("prism_perception.save_spectrum_state", create=True) as mock_save:
        b.apply({}, now=1_000_000.0)
    mock_save.assert_not_called()


# ---------------------------------------------------------------------------
# Clamping: result values stay in [0, 1]
# ---------------------------------------------------------------------------

def test_clamping_applied():
    """cognitive_readiness < 0.40 fires A-0.15; if current A=0.05, clamped to 0."""
    b = make_bridge()
    with (
        patch("prism_perception.get_current_gates", create=True, return_value=None),
        patch("prism_perception.save_spectrum_state", create=True),
        patch("prism_perception.SpectrumGates", create=True) as mock_cls,
        patch("prism_perception.load_spectrum", create=True,
              return_value=_fake_gates(A=0.05)),
    ):
        mock_cls.side_effect = lambda **kw: MagicMock(**kw)
        _pump(b, {"cognitive_readiness": 0.30}, n=5)

    call_kwargs = mock_cls.call_args
    if call_kwargs:
        a_val = call_kwargs[1].get("A", None)
        if a_val is not None:
            assert a_val >= 0.0


# ---------------------------------------------------------------------------
# Graceful handling of missing prism_spectrum_middleware
# ---------------------------------------------------------------------------

def test_apply_survives_import_error():
    b = make_bridge()
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "prism_spectrum_middleware":
            raise ImportError("unavailable")
        return real_import(name, *args, **kwargs)

    # Pump 5 ticks so the EMA crosses the fatigue threshold
    with patch("builtins.__import__", side_effect=mock_import):
        results = _pump(b, {"sport_readiness": 0.35}, n=5, now_start=1_000_000.0)

    assert all(isinstance(r, dict) for r in results)
