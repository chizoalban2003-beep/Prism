"""
Tests for prism_phase.py — CrystallizationEngine and helpers.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from prism_phase import (
    CrystallizationEngine,
    PhaseReading,
    PhaseState,
    get_engine,
    set_engine,
)

# ---------------------------------------------------------------------------
# PhaseState enum
# ---------------------------------------------------------------------------

def test_phase_state_members():
    assert set(p.value for p in PhaseState) == {"CRYSTAL", "STABLE", "VISCOUS", "LIQUID"}


def test_phase_state_identity():
    assert PhaseState.CRYSTAL is PhaseState.CRYSTAL
    assert PhaseState.LIQUID is not PhaseState.STABLE


# ---------------------------------------------------------------------------
# PhaseReading dataclass
# ---------------------------------------------------------------------------

def test_phase_reading_fields():
    r = PhaseReading(phi=0.5, delta_H=0.4, delta_K=0.3, phase=PhaseState.STABLE)
    assert r.phi == pytest.approx(0.5)
    assert r.delta_H == pytest.approx(0.4)
    assert r.delta_K == pytest.approx(0.3)
    assert r.phase is PhaseState.STABLE
    assert isinstance(r.ts, float)


def test_phase_reading_ts_auto():
    t0 = time.time()
    r = PhaseReading(phi=0.1, delta_H=0.1, delta_K=0.0, phase=PhaseState.CRYSTAL)
    assert r.ts >= t0


# ---------------------------------------------------------------------------
# phase_from_phi
# ---------------------------------------------------------------------------

def test_phase_from_phi_crystal():
    e = CrystallizationEngine()
    assert e.phase_from_phi(0.0) is PhaseState.CRYSTAL
    assert e.phase_from_phi(0.39) is PhaseState.CRYSTAL


def test_phase_from_phi_stable():
    e = CrystallizationEngine()
    assert e.phase_from_phi(0.40) is PhaseState.STABLE
    assert e.phase_from_phi(0.59) is PhaseState.STABLE


def test_phase_from_phi_viscous():
    e = CrystallizationEngine()
    assert e.phase_from_phi(0.60) is PhaseState.VISCOUS
    assert e.phase_from_phi(0.69) is PhaseState.VISCOUS


def test_phase_from_phi_liquid():
    e = CrystallizationEngine()
    assert e.phase_from_phi(0.70) is PhaseState.LIQUID
    assert e.phase_from_phi(1.00) is PhaseState.LIQUID


# ---------------------------------------------------------------------------
# veax_delta
# ---------------------------------------------------------------------------

def test_veax_delta_crystal():
    e = CrystallizationEngine()
    d = e.veax_delta(PhaseState.CRYSTAL)
    assert d["A"] == pytest.approx(+0.05)
    assert d["X"] == pytest.approx(-0.05)


def test_veax_delta_stable():
    e = CrystallizationEngine()
    assert e.veax_delta(PhaseState.STABLE) == {}


def test_veax_delta_viscous():
    e = CrystallizationEngine()
    d = e.veax_delta(PhaseState.VISCOUS)
    assert d["V"] == pytest.approx(-0.10)
    assert d["A"] == pytest.approx(+0.10)


def test_veax_delta_liquid():
    e = CrystallizationEngine()
    d = e.veax_delta(PhaseState.LIQUID)
    assert d["V"] == pytest.approx(0.0)
    assert d["A"] == pytest.approx(1.0)
    assert d["X"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# model_hint
# ---------------------------------------------------------------------------

def test_model_hint_all_phases():
    e = CrystallizationEngine()
    assert e.model_hint(PhaseState.CRYSTAL) == "fast"
    assert e.model_hint(PhaseState.STABLE) == "standard"
    assert e.model_hint(PhaseState.VISCOUS) == "capable"
    assert e.model_hint(PhaseState.LIQUID) == "emergency"


# ---------------------------------------------------------------------------
# _compute_delta_H
# ---------------------------------------------------------------------------

def test_compute_delta_h_returns_float():
    e = CrystallizationEngine()
    delta = e._compute_delta_H()
    assert isinstance(delta, float)
    assert 0.0 <= delta <= 1.0


def test_compute_delta_h_psutil_mocked(monkeypatch):
    import psutil

    monkeypatch.setattr(psutil, "cpu_percent", lambda interval=None: 50.0)
    vm = MagicMock()
    vm.percent = 80.0
    monkeypatch.setattr(psutil, "virtual_memory", lambda: vm)
    monkeypatch.setattr(psutil, "sensors_temperatures", lambda: {})
    bat = MagicMock()
    bat.power_plugged = True
    monkeypatch.setattr(psutil, "sensors_battery", lambda: bat)

    e = CrystallizationEngine()
    delta = e._compute_delta_H()
    # cpu_norm=0.5, ram=0.8, thermal=0, battery_drain=0
    # ΔH = 0.5*0.3 + 0.8*0.4 + 0*0.2 + 0*0.1 = 0.15 + 0.32 = 0.47
    assert delta == pytest.approx(0.47, abs=0.01)


# ---------------------------------------------------------------------------
# _compute_delta_K
# ---------------------------------------------------------------------------

def test_compute_delta_k_no_soul():
    e = CrystallizationEngine()
    assert e._compute_delta_K(None) == pytest.approx(0.0)


def test_compute_delta_k_with_soul():
    soul = MagicMock()
    soul.run_entailment_check.return_value = ["c1", "c2", "c3"]

    e = CrystallizationEngine()
    # Force cache miss
    e._last_entailment_ts = 0.0
    delta = e._compute_delta_K(soul)
    assert delta == pytest.approx(3 / 5)


def test_compute_delta_k_cached():
    soul = MagicMock()
    soul.run_entailment_check.return_value = ["c1"]

    e = CrystallizationEngine()
    e._last_entailment_ts = time.time() + 9999  # far future
    e._last_entailment_result = 0.42
    delta = e._compute_delta_K(soul)
    assert delta == pytest.approx(0.42)
    soul.run_entailment_check.assert_not_called()


def test_compute_delta_k_soul_error():
    soul = MagicMock()
    soul.run_entailment_check.side_effect = RuntimeError("boom")

    e = CrystallizationEngine()
    e._last_entailment_ts = 0.0
    # Should return 0.0 gracefully on error
    delta = e._compute_delta_K(soul)
    assert delta == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------

def test_compute_no_soul(monkeypatch):
    import psutil

    monkeypatch.setattr(psutil, "cpu_percent", lambda interval=None: 0.0)
    vm = MagicMock()
    vm.percent = 0.0
    monkeypatch.setattr(psutil, "virtual_memory", lambda: vm)
    monkeypatch.setattr(psutil, "sensors_temperatures", lambda: {})
    monkeypatch.setattr(psutil, "sensors_battery", lambda: None)
    # Prevent pytest/python process from being detected as a heavy process
    monkeypatch.setattr(psutil, "process_iter", lambda attrs=None: iter([]))

    e = CrystallizationEngine()
    r = e.compute(soul=None)

    assert isinstance(r, PhaseReading)
    assert r.phi == pytest.approx(0.0)
    assert r.phase is PhaseState.CRYSTAL
    assert len(e.history) == 1


def test_compute_appends_history():
    e = CrystallizationEngine()
    with patch.object(e, "_compute_delta_H", return_value=0.5), patch.object(
        e, "_compute_delta_K", return_value=0.5
    ):
        e.compute()
        e.compute()
    assert len(e.history) == 2


def test_compute_history_rolling_window():
    e = CrystallizationEngine(history_size=3)
    with patch.object(e, "_compute_delta_H", return_value=0.1), patch.object(
        e, "_compute_delta_K", return_value=0.1
    ):
        for _ in range(5):
            e.compute()
    assert len(e.history) == 3


# ---------------------------------------------------------------------------
# current_phase property
# ---------------------------------------------------------------------------

def test_current_phase_no_history():
    e = CrystallizationEngine()
    assert e.current_phase is PhaseState.STABLE


def test_current_phase_after_compute():
    e = CrystallizationEngine()
    with patch.object(e, "_compute_delta_H", return_value=0.9), patch.object(
        e, "_compute_delta_K", return_value=0.9
    ):
        e.compute()
    assert e.current_phase is PhaseState.LIQUID


# ---------------------------------------------------------------------------
# should_melt()
# ---------------------------------------------------------------------------

def test_should_melt_false_no_history():
    e = CrystallizationEngine()
    assert e.should_melt() is False


def test_should_melt_false_stable():
    e = CrystallizationEngine()
    with patch.object(e, "_compute_delta_H", return_value=0.3), patch.object(
        e, "_compute_delta_K", return_value=0.3
    ):
        e.compute()
    assert e.should_melt() is False


def test_should_melt_true_liquid():
    e = CrystallizationEngine()
    with patch.object(e, "_compute_delta_H", return_value=0.9), patch.object(
        e, "_compute_delta_K", return_value=0.9
    ):
        e.compute()
    assert e.should_melt() is True


def test_should_melt_exactly_at_threshold():
    e = CrystallizationEngine(melt_threshold=0.70)
    with patch.object(e, "_compute_delta_H", return_value=0.70), patch.object(
        e, "_compute_delta_K", return_value=0.70
    ):
        # phi = 0.6*0.70 + 0.4*0.70 = 0.70
        e.compute()
    assert e.should_melt() is True


# ---------------------------------------------------------------------------
# Singleton get_engine / set_engine
# ---------------------------------------------------------------------------

def test_get_engine_creates_singleton():
    import prism_phase

    prism_phase._engine = None  # reset
    e1 = get_engine()
    e2 = get_engine()
    assert e1 is e2
    assert isinstance(e1, CrystallizationEngine)


def test_set_engine_replaces_singleton():
    import prism_phase

    custom = CrystallizationEngine(alpha=0.9)
    set_engine(custom)
    assert get_engine() is custom
    # Reset to avoid leaking into other tests
    prism_phase._engine = None


def test_set_engine_none():
    import prism_phase

    set_engine(None)  # type: ignore[arg-type]
    assert prism_phase._engine is None
    # get_engine now recreates a fresh one
    e = get_engine()
    assert isinstance(e, CrystallizationEngine)
    prism_phase._engine = None
