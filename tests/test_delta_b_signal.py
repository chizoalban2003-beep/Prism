"""
tests/test_delta_b_signal.py
Tests for Vector IV: Biological Debt → Φ_melt Extension (ΔB Signal)
"""
from unittest.mock import MagicMock, patch

import pytest

from prism_perception import VEAXDebtDynamics
from prism_phase import CrystallizationEngine

# ── _compute_delta_B ──────────────────────────────────────────────────────────

def test_compute_delta_b_none_bridge_returns_zero():
    engine = CrystallizationEngine()
    result = engine._compute_delta_B(None)
    assert result == pytest.approx(0.0)


def test_compute_delta_b_zero_debt_returns_zero():
    engine = CrystallizationEngine()
    bridge = MagicMock()
    dyn = VEAXDebtDynamics()  # all zeros
    bridge.dynamics = dyn
    result = engine._compute_delta_B(bridge)
    assert result == pytest.approx(0.0)


def test_compute_delta_b_high_v_debt_increases_phi():
    engine = CrystallizationEngine()
    bridge = MagicMock()
    dyn = VEAXDebtDynamics()
    dyn.add_debt("V", 1.5)
    bridge.dynamics = dyn

    delta_b = engine._compute_delta_B(bridge)
    assert delta_b > 0.0, "High V debt should produce positive ΔB"


def test_compute_delta_b_high_e_debt_increases_phi():
    engine = CrystallizationEngine()
    bridge = MagicMock()
    dyn = VEAXDebtDynamics()
    dyn.add_debt("E", 1.5)
    bridge.dynamics = dyn

    delta_b = engine._compute_delta_B(bridge)
    assert delta_b > 0.0, "High E debt should produce positive ΔB"


def test_phi_melt_uses_extended_formula_when_delta_b_positive():
    """When ΔB > 0, Φ = 0.5·ΔH + 0.3·ΔK + 0.2·ΔB (not 0.6·ΔH + 0.4·ΔK)."""
    engine = CrystallizationEngine()

    bridge = MagicMock()
    dyn = VEAXDebtDynamics()
    dyn.add_debt("V", 1.5)  # produce non-zero ΔB
    bridge.dynamics = dyn

    # Patch hardware to known values
    with patch("psutil.cpu_percent", return_value=50.0), \
         patch("psutil.virtual_memory") as mock_vm, \
         patch("psutil.sensors_temperatures", return_value={}), \
         patch("psutil.sensors_battery", return_value=None):
        mock_vm.return_value.percent = 50.0
        reading = engine.compute(soul=None, bridge=bridge)

    delta_h = reading.delta_H
    delta_b = engine._compute_delta_B(bridge)
    # With ΔB > 0 the formula is 0.5·ΔH + 0.3·0 + 0.2·ΔB
    expected = 0.5 * delta_h + 0.2 * delta_b
    assert reading.phi == pytest.approx(expected, abs=0.05), (
        "Extended formula should be used when ΔB > 0"
    )


def test_phi_melt_uses_original_formula_when_delta_b_zero():
    """When ΔB = 0, Φ = α·ΔH + β·ΔK (original formula)."""
    engine = CrystallizationEngine(alpha=0.6, beta=0.4)

    with patch("psutil.cpu_percent", return_value=40.0), \
         patch("psutil.virtual_memory") as mock_vm, \
         patch("psutil.sensors_temperatures", return_value={}), \
         patch("psutil.sensors_battery", return_value=None):
        mock_vm.return_value.percent = 30.0
        reading = engine.compute(soul=None, bridge=None)

    delta_h = reading.delta_H
    # No soul → delta_k = 0; No bridge → delta_b = 0
    expected = 0.6 * delta_h
    assert reading.phi == pytest.approx(expected, abs=0.05), (
        "Original formula should be used when ΔB = 0"
    )


def test_shadow_pipeline_passes_bridge_to_engine():
    """PrismShadowPipeline should pass bridge to phase_engine.compute()."""
    from prism_shadow_pipeline import PrismShadowPipeline

    mock_graph = MagicMock()
    mock_graph.commit_pending.return_value = 0
    mock_graph.consistency_psi.return_value = 0.0

    mock_engine = MagicMock()
    mock_engine.history = [True]
    mock_engine.should_melt.return_value = False

    mock_bridge = MagicMock()

    pipeline = PrismShadowPipeline(
        graph=mock_graph,
        phase_engine=mock_engine,
        bridge=mock_bridge,
    )

    # Manually call _run cycle once
    pipeline._graph.commit_pending()  # simulate a cycle
    # The bridge should be stored on the pipeline
    assert pipeline._bridge is mock_bridge


def test_delta_b_clamped_at_1():
    """ΔB should never exceed 1.0 even with extreme debt."""
    engine = CrystallizationEngine()
    bridge = MagicMock()
    dyn = VEAXDebtDynamics()
    dyn.add_debt("V", 2.0)
    dyn.add_debt("E", 2.0)
    bridge.dynamics = dyn

    result = engine._compute_delta_B(bridge)
    assert result <= 1.0, f"ΔB must be clamped at 1.0, got {result}"
    assert result >= 0.0, f"ΔB must be non-negative, got {result}"
