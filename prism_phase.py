"""
prism_phase.py
==============
PRISM Φ_melt Crystallization Engine

Hardware-aware phase-change engine that maps telemetry + soul-contradiction
pressure to a single scalar Φ_melt, then selects the appropriate phase.

Formula:
    Φ_melt = α·ΔH_telemetry + β·ΔK_context   (α=0.6, β=0.4)
    ΔH = cpu_norm×0.3 + ram_pressure×0.4 + thermal×0.2 + battery_drain×0.1
    ΔK = contradiction_rate (normalized, from soul entailment)

Phases:
    CRYSTAL  (Φ < 0.40): smallest model, Direct T3, A+0.05 X-0.05
    STABLE   (0.40–0.60): standard chain, unchanged VEAX
    VISCOUS  (0.60–0.70): skip evaluator, V-0.10 A+0.10
    LIQUID   (Φ ≥ 0.70): cloud/3B emergency, abort long chains, V=0 A=1 X=0
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PhaseState enum
# ---------------------------------------------------------------------------

class PhaseState(Enum):
    CRYSTAL = "CRYSTAL"
    STABLE  = "STABLE"
    VISCOUS = "VISCOUS"
    LIQUID  = "LIQUID"


# ---------------------------------------------------------------------------
# PhaseReading dataclass
# ---------------------------------------------------------------------------

@dataclass
class PhaseReading:
    phi:     float
    delta_H: float
    delta_K: float
    phase:   PhaseState
    delta_B: float = 0.0
    ts:      float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Phase ordering helper (used by PhasePredictor and CrystallizationEngine)
# ---------------------------------------------------------------------------

_PHASE_ORDER: dict[PhaseState, int] = {
    PhaseState.CRYSTAL: 0,
    PhaseState.STABLE:  1,
    PhaseState.VISCOUS: 2,
    PhaseState.LIQUID:  3,
}


def _phase_order(phase: PhaseState) -> int:
    return _PHASE_ORDER.get(phase, 1)


# ---------------------------------------------------------------------------
# PhasePredictor — anticipatory phase shifting (Vector III)
# ---------------------------------------------------------------------------

class PhasePredictor:
    """
    Predicts future phase state by analyzing ΔH slope and heavy process spawns.

    Two signals:
    1. ΔH slope: linear regression over rolling window → extrapolate to threshold
    2. Heavy process detection: known compilation/test patterns → immediate prediction
    """

    _HEAVY_PROCS: frozenset[str] = frozenset({
        "pytest", "cargo", "gcc", "g++", "cc", "make", "cmake",
        "npm", "node", "webpack", "tsc",
        "mvn", "gradle", "java",
        "rustc", "go", "ld",
        # ML/training processes
        "torchrun", "accelerate", "deepspeed", "unsloth",
    })
    _WINDOW_SIZE = 6    # samples
    _LOOKAHEAD_S = 30.0  # predict this far ahead

    def __init__(self, melt_threshold: float = 0.70, viscous_threshold: float = 0.60) -> None:
        self._melt_t    = melt_threshold
        self._viscous_t = viscous_threshold
        self._history: deque[tuple[float, float]] = deque(maxlen=self._WINDOW_SIZE)

    def observe(self, dh: float, ts: float | None = None) -> None:
        self._history.append((ts if ts is not None else time.monotonic(), dh))

    def predict(self, current_dh: float) -> PhaseState | None:
        """Return predicted phase if crossing is imminent, else None."""
        # Heavy process spawn takes priority
        if self._heavy_proc_running():
            return PhaseState.LIQUID

        if len(self._history) < 3:
            return None

        slope = self._slope()
        if slope <= 0:
            return None  # load declining

        # Time to threshold crossing: (threshold - current) / slope (in seconds)
        time_to_melt    = (self._melt_t    - current_dh) / slope
        time_to_viscous = (self._viscous_t - current_dh) / slope

        if 0 < time_to_melt <= self._LOOKAHEAD_S:
            return PhaseState.LIQUID
        if 0 < time_to_viscous <= self._LOOKAHEAD_S:
            return PhaseState.VISCOUS
        return None

    def _slope(self) -> float:
        """Slope of ΔH over time window (units per second)."""
        pts = list(self._history)
        n = len(pts)
        t0 = pts[0][0]
        ts_norm = [p[0] - t0 for p in pts]
        vals    = [p[1] for p in pts]
        mean_t  = sum(ts_norm) / n
        mean_v  = sum(vals) / n
        num = sum((t - mean_t) * (v - mean_v) for t, v in zip(ts_norm, vals))
        den = sum((t - mean_t) ** 2 for t in ts_norm)
        return num / den if den > 0 else 0.0

    def _heavy_proc_running(self) -> bool:
        try:
            import psutil
            for p in psutil.process_iter(["name"]):
                try:
                    # Use exact name match (strip extension on Windows) to avoid
                    # false positives like "ld-linux-x86-64.so.2" matching "ld"
                    raw = p.info.get("name") or ""
                    name = raw.lower().split(".")[0]  # strip .exe / .so suffix
                    if name in self._HEAVY_PROCS:
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception:
            pass
        return False


# ---------------------------------------------------------------------------
# CrystallizationEngine
# ---------------------------------------------------------------------------

class CrystallizationEngine:
    """
    Computes Φ_melt from psutil telemetry and soul contradiction rate,
    classifies the result into a PhaseState, and exposes VEAX deltas +
    model hints for the rest of PRISM to act on.
    """

    def __init__(
        self,
        alpha:              float = 0.6,
        beta:               float = 0.4,
        melt_threshold:     float = 0.70,
        viscous_threshold:  float = 0.60,
        crystal_ceiling:    float = 0.40,
        history_size:       int   = 20,
    ) -> None:
        self.alpha             = alpha
        self.beta              = beta
        self.melt_threshold    = melt_threshold
        self.viscous_threshold = viscous_threshold
        self.crystal_ceiling   = crystal_ceiling
        self.history:          deque[PhaseReading] = deque(maxlen=history_size)

        # Cache the last entailment check result to avoid hammering soul
        self._last_entailment_ts:     float         = 0.0
        self._last_entailment_result: float         = 0.0
        self._entailment_ttl:         float         = 60.0   # seconds
        self._lock = threading.Lock()

        # Anticipatory phase predictor (Vector III)
        self._predictor = PhasePredictor(melt_threshold, viscous_threshold)

    # ── Public API ────────────────────────────────────────────────────────

    def compute(
        self,
        soul: Optional[Any] = None,
        bridge: Optional[Any] = None,
        kinetic: Optional[Any] = None,
    ) -> PhaseReading:
        """
        Compute a fresh PhaseReading.
        soul.run_entailment_check() is used for ΔK; gracefully handles None.
        The entailment call is cached for _entailment_ttl seconds to avoid spam.
        bridge:  optional BiometricVEAXBridge for biological pressure (ΔB, Vector IV).
        kinetic: optional KineticEngine — compound personal signal pressure (ΔC, Vector V).
        """
        with self._lock:
            return self._compute_locked(soul, bridge, kinetic)

    def _compute_locked(
        self,
        soul: Optional[Any] = None,
        bridge: Optional[Any] = None,
        kinetic: Optional[Any] = None,
    ) -> PhaseReading:
        delta_h = self._compute_delta_H()
        delta_k = self._compute_delta_K(soul)
        delta_b = self._compute_delta_B(bridge)
        delta_c = self._compute_delta_C(kinetic)

        # Φ_melt formula — weights rebalanced when extra vectors are active.
        # ΔC weight is kept modest (0.10) so single-domain spikes don't dominate;
        # the compound engine already requires multi-signal convergence.
        if delta_b > 0.0 and delta_c > 0.0:
            phi = 0.40 * delta_h + 0.25 * delta_k + 0.15 * delta_b + 0.10 * delta_c
        elif delta_b > 0.0:
            phi = 0.5 * delta_h + 0.3 * delta_k + 0.2 * delta_b
        elif delta_c > 0.0:
            phi = 0.50 * delta_h + 0.35 * delta_k + 0.15 * delta_c
        else:
            phi = self.alpha * delta_h + self.beta * delta_k

        phi   = max(0.0, min(1.0, phi))
        phase = self.phase_from_phi(phi)

        # Anticipatory phase shifting (Vector III)
        self._predictor.observe(delta_h)
        predicted = self._predictor.predict(delta_h)
        if predicted is not None and _phase_order(predicted) > _phase_order(phase):
            phase = predicted

        reading = PhaseReading(phi=phi, delta_H=delta_h, delta_K=delta_k, phase=phase, delta_B=delta_b)
        self.history.append(reading)
        logger.debug(
            "[phase] ΔH=%.3f ΔK=%.3f ΔB=%.3f ΔC=%.3f Φ=%.3f phase=%s",
            delta_h, delta_k, delta_b, delta_c, phi, phase.value,
        )
        return reading

    def _compute_delta_H(self) -> float:
        """
        Hardware telemetry component:
            ΔH = cpu_norm×0.3 + ram_pressure×0.4 + thermal×0.2 + battery_drain×0.1
        """
        try:
            import psutil
        except ImportError:
            return 0.0

        # CPU
        try:
            cpu_norm = psutil.cpu_percent(interval=None) / 100.0
        except Exception:
            cpu_norm = 0.0

        # RAM
        try:
            ram_pressure = psutil.virtual_memory().percent / 100.0
        except Exception:
            ram_pressure = 0.0

        # Thermal — average of all core temperatures; 0 if unavailable
        thermal = 0.0
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                all_temps: list[float] = []
                for entries in temps.values():
                    for e in entries:
                        if e.current is not None:
                            all_temps.append(e.current)
                if all_temps:
                    avg_c = sum(all_temps) / len(all_temps)
                    # Normalise: 30°C=0, 90°C=1
                    thermal = max(0.0, min(1.0, (avg_c - 30.0) / 60.0))
        except (AttributeError, Exception):
            thermal = 0.0

        # Battery drain — 0 if plugged in or unavailable; high when draining fast
        battery_drain = 0.0
        try:
            bat = psutil.sensors_battery()
            if bat and not bat.power_plugged:
                # secsleft: remaining seconds; <1800 = draining fast
                if bat.secsleft is not None and bat.secsleft > 0:
                    battery_drain = max(0.0, min(1.0, 1.0 - bat.secsleft / 14400.0))
                else:
                    # battery present but no time estimate — assume moderate drain
                    battery_drain = 0.3
        except Exception:
            battery_drain = 0.0

        delta_h = (
            cpu_norm      * 0.3
            + ram_pressure  * 0.4
            + thermal       * 0.2
            + battery_drain * 0.1
        )
        return max(0.0, min(1.0, delta_h))

    def _compute_delta_K(self, soul: Any) -> float:
        """
        Soul contradiction component.
        delta_K = contradiction_count / max_expected (default 5), clamped [0, 1].
        Uses a TTL cache to avoid calling soul.run_entailment_check() too often.
        """
        if soul is None:
            return 0.0

        now = time.time()
        if now - self._last_entailment_ts < self._entailment_ttl:
            return self._last_entailment_result

        try:
            result = soul.run_entailment_check()
            count  = len(result) if result else 0
        except Exception as exc:
            logger.debug("[phase] entailment check error: %s", exc)
            count = 0

        max_expected = 5
        delta_k = max(0.0, min(1.0, count / max_expected))
        self._last_entailment_ts     = now
        self._last_entailment_result = delta_k
        return delta_k

    def _compute_delta_B(self, bridge: Any) -> float:
        """
        Biological pressure from BiometricVEAXBridge (Vector IV).
        ΔB = weighted combination of V and E debt (slow-τ axes only) to avoid
        transient A spikes dominating.
        Returns 0.0 if bridge is None or unavailable.
        """
        if bridge is None:
            return 0.0
        try:
            dyn = bridge.dynamics  # VEAXDebtDynamics
            # Weight V and E debt more (deep fatigue, not transient stress)
            v_debt = dyn.get_axis_debt("V")
            e_debt = dyn.get_axis_debt("E")
            return min(1.0, (v_debt * 0.6 + e_debt * 0.4) / 1.5)
        except Exception:
            return 0.0

    def _compute_delta_C(self, kinetic: Any) -> float:
        """
        Compound personal-signal pressure from KineticEngine (Vector V).
        ΔC = compound_phi_delta() / 5.0 (normalised to [0, 1] using action_threshold).
        Returns 0.0 if kinetic is None or unavailable.
        """
        if kinetic is None:
            return 0.0
        try:
            raw = kinetic.compound_phi_delta()
            return max(0.0, min(1.0, raw / 5.0))
        except Exception:
            return 0.0

    def phase_from_phi(self, phi: float) -> PhaseState:
        """Classify a Φ value into a PhaseState."""
        if phi >= self.melt_threshold:
            return PhaseState.LIQUID
        if phi >= self.viscous_threshold:
            return PhaseState.VISCOUS
        if phi < self.crystal_ceiling:
            return PhaseState.CRYSTAL
        return PhaseState.STABLE

    def veax_delta(self, phase: PhaseState) -> dict[str, float]:
        """
        Return the VEAX deltas to apply for a given phase.
        Empty dict means no change (STABLE).
        """
        if phase is PhaseState.CRYSTAL:
            return {"A": +0.05, "X": -0.05}
        if phase is PhaseState.STABLE:
            return {}
        if phase is PhaseState.VISCOUS:
            return {"V": -0.10, "A": +0.10}
        # LIQUID
        return {"V": 0.0, "A": 1.0, "X": 0.0}  # absolute values for LIQUID

    def model_hint(self, phase: PhaseState) -> str:
        """Return a model-size hint string for the LLM router."""
        return {
            PhaseState.CRYSTAL:  "fast",
            PhaseState.STABLE:   "standard",
            PhaseState.VISCOUS:  "capable",
            PhaseState.LIQUID:   "emergency",
        }[phase]

    @property
    def current_phase(self) -> PhaseState:
        """The phase from the most recent reading, or STABLE if no readings yet."""
        if self.history:
            return self.history[-1].phase
        return PhaseState.STABLE

    def should_melt(self) -> bool:
        """True if the latest Φ meets or exceeds the melt threshold."""
        if self.history:
            return self.history[-1].phi >= self.melt_threshold
        return False


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_engine: Optional[CrystallizationEngine] = None


def get_engine() -> CrystallizationEngine:
    """Return (and lazily create) the module-level CrystallizationEngine singleton."""
    global _engine
    if _engine is None:
        _engine = CrystallizationEngine()
    return _engine


def set_engine(e: CrystallizationEngine) -> None:
    """Replace the module-level singleton (used in tests and custom setups)."""
    global _engine
    _engine = e
