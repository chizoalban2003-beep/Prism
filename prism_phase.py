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
    ts:      float = field(default_factory=time.time)


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

    # ── Public API ────────────────────────────────────────────────────────

    def compute(self, soul: Any = None) -> PhaseReading:
        """
        Compute a fresh PhaseReading.
        soul.run_entailment_check() is used for ΔK; gracefully handles None.
        The entailment call is cached for _entailment_ttl seconds to avoid spam.
        """
        delta_h = self._compute_delta_H()
        delta_k = self._compute_delta_K(soul)
        phi     = self.alpha * delta_h + self.beta * delta_k
        phi     = max(0.0, min(1.0, phi))
        phase   = self.phase_from_phi(phi)
        reading = PhaseReading(phi=phi, delta_H=delta_h, delta_K=delta_k, phase=phase)
        self.history.append(reading)
        logger.debug(
            "[phase] ΔH=%.3f ΔK=%.3f Φ=%.3f phase=%s",
            delta_h, delta_k, phi, phase.value,
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
