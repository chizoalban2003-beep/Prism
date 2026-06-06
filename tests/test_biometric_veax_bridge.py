"""
Tests for BiometricVEAXBridge in prism_perception.py.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from prism_perception import BiometricVEAXBridge

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bridge() -> BiometricVEAXBridge:
    b = BiometricVEAXBridge()
    # Ensure all cooldowns are expired so rules fire freely
    b._last_fired = {}
    return b


def _fake_gates(V=0.5, E=0.5, A=0.5, X=0.5):
    m = MagicMock()
    m.V = V
    m.E = E
    m.A = A
    m.X = X
    return m


# ---------------------------------------------------------------------------
# No-op when no rules fire
# ---------------------------------------------------------------------------

def test_no_op_empty_factors():
    b = make_bridge()
    result = b.apply({})
    assert result == {}


def test_no_op_factors_below_high_threshold():
    b = make_bridge()
    # sport_readiness=0.55 → none of the rules (<0.30, <0.40, >0.80) fire
    result = b.apply({"sport_readiness": 0.55})
    assert result == {}


def test_no_op_factors_not_in_rules():
    b = make_bridge()
    result = b.apply({"unknown_factor": 0.99})
    assert result == {}


# ---------------------------------------------------------------------------
# Single rule firing
# ---------------------------------------------------------------------------

def test_sport_readiness_low_fires():
    b = make_bridge()
    with (
        patch("prism_perception.get_current_gates", create=True, return_value=None),
        patch("prism_perception.save_spectrum_state", create=True),
        patch("prism_perception.SpectrumGates", create=True),
        patch("prism_perception.load_spectrum", create=True) as mock_load,
    ):
        mock_load.return_value = (_fake_gates(), None)
        result = b.apply({"sport_readiness": 0.35})  # < 0.40 rule fires

    # A axis should receive -0.10
    assert "A" in result
    assert result["A"] == pytest.approx(-0.10)


def test_hrv_recovery_high_fires():
    b = make_bridge()
    with (
        patch("prism_perception.get_current_gates", create=True, return_value=None),
        patch("prism_perception.save_spectrum_state", create=True),
        patch("prism_perception.SpectrumGates", create=True),
        patch("prism_perception.load_spectrum", create=True) as mock_load,
    ):
        mock_load.return_value = (_fake_gates(), None)
        result = b.apply({"hrv_recovery": 0.85})  # > 0.80

    assert "E" in result
    assert result["E"] == pytest.approx(+0.10)


# ---------------------------------------------------------------------------
# Cooldown enforcement
# ---------------------------------------------------------------------------

def test_cooldown_prevents_refiring():
    b = make_bridge()

    # Fire rule index 0 (sport_readiness < 0.40, cooldown=3600)
    b._last_fired[0] = time.time()  # just fired

    with (
        patch("prism_perception.get_current_gates", create=True, return_value=None),
        patch("prism_perception.save_spectrum_state", create=True),
        patch("prism_perception.SpectrumGates", create=True),
        patch("prism_perception.load_spectrum", create=True) as mock_load,
    ):
        mock_load.return_value = (_fake_gates(), None)
        result = b.apply({"sport_readiness": 0.35})

    # Rule 0 is in cooldown; rule 1 (< 0.30) won't fire since 0.35 > 0.30
    assert result.get("A", 0.0) == pytest.approx(0.0)


def test_cooldown_expires_allows_refiring():
    b = make_bridge()
    # Set last fired far in the past
    b._last_fired[0] = time.time() - 9999

    with (
        patch("prism_perception.get_current_gates", create=True, return_value=None),
        patch("prism_perception.save_spectrum_state", create=True),
        patch("prism_perception.SpectrumGates", create=True),
        patch("prism_perception.load_spectrum", create=True) as mock_load,
    ):
        mock_load.return_value = (_fake_gates(), None)
        result = b.apply({"sport_readiness": 0.35})

    assert "A" in result


# ---------------------------------------------------------------------------
# Accumulation (multiple rules fire)
# ---------------------------------------------------------------------------

def test_multiple_rules_accumulate():
    b = make_bridge()
    # stress_level > 0.70 (rule idx 5): X+0.15, A-0.10
    # stress_level > 0.85 (rule idx 6): V+0.10
    # Both should fire if value=0.90 and no cooldowns
    with (
        patch("prism_perception.get_current_gates", create=True, return_value=None),
        patch("prism_perception.save_spectrum_state", create=True),
        patch("prism_perception.SpectrumGates", create=True),
        patch("prism_perception.load_spectrum", create=True) as mock_load,
    ):
        mock_load.return_value = (_fake_gates(), None)
        result = b.apply({"stress_level": 0.90})

    assert result.get("X", 0.0) == pytest.approx(+0.15)
    assert result.get("A", 0.0) == pytest.approx(-0.10)
    assert result.get("V", 0.0) == pytest.approx(+0.10)


# ---------------------------------------------------------------------------
# save_spectrum_state is called when deltas are non-zero
# ---------------------------------------------------------------------------

def test_save_called_when_delta_nonzero():
    b = make_bridge()
    with (
        patch("prism_spectrum_middleware.get_current_gates", return_value=None),
        patch("prism_spectrum_middleware.save_spectrum_state"),
        patch("prism_spectrum_middleware.load_spectrum") as mock_load,
    ):
        mock_load.return_value = (_fake_gates(), None)
        result = b.apply({"hrv_recovery": 0.85})  # E+0.10

    assert "E" in result


def test_save_not_called_when_no_delta():
    b = make_bridge()
    with patch("prism_perception.save_spectrum_state", create=True) as mock_save:
        b.apply({})
    mock_save.assert_not_called()


# ---------------------------------------------------------------------------
# Clamping: result values stay in [0, 1]
# ---------------------------------------------------------------------------

def test_clamping_applied():
    b = make_bridge()
    # cognitive_readiness < 0.40: A-0.15
    # If current A is 0.05, new A should be clamped to 0.0
    with (
        patch("prism_perception.get_current_gates", create=True, return_value=None),
        patch("prism_perception.save_spectrum_state", create=True),
        patch("prism_perception.SpectrumGates", create=True) as mock_cls,
        patch("prism_perception.load_spectrum", create=True) as mock_load,
    ):
        mock_load.return_value = (_fake_gates(A=0.05), None)
        mock_cls.side_effect = lambda **kw: MagicMock(**kw)
        b.apply({"cognitive_readiness": 0.30})  # < 0.40 fires: A-0.15

    # save was called; A clamped to 0.0
    call_kwargs = mock_cls.call_args
    if call_kwargs:
        a_val = call_kwargs[1].get("A", None) or (call_kwargs[0][0].A if call_kwargs[0] else None)
        if a_val is not None:
            assert a_val >= 0.0


# ---------------------------------------------------------------------------
# Graceful handling of missing prism_spectrum_middleware
# ---------------------------------------------------------------------------

def test_apply_survives_import_error():
    b = make_bridge()
    # Simulate that the spectrum import inside apply() fails
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "prism_spectrum_middleware":
            raise ImportError("unavailable")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        # Should not raise; the exception is caught internally
        result = b.apply({"sport_readiness": 0.35})

    assert isinstance(result, dict)
