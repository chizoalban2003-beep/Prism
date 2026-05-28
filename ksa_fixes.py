"""
ksa_fixes.py
============
Kinetic State Agent — Live Weight Injection & Ground-Truth Optimization

Two utilities for real-time contextual adjustment of ThreeBarSystem levers:

    LiveWeightInjector
        Reads hardware state (CPU, RAM, battery, time-of-day) or external
        context dicts and injects them as movable-fulcrum weights into a
        ThreeBarSystem before simulation.

    GroundTruthOptimizer
        Applies observed outcome scores to adjust lever fulcrum positions
        via gradient nudges, improving future decisions through direct
        feedback (complementary to the registry-based hill-climber).
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from typing import Any

import psutil

from ksa_lever import ThreeBarSystem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LiveWeightInjector
# ---------------------------------------------------------------------------

class LiveWeightInjector:
    """
    Injects real-time context signals as left/right lever weights.

    Signals (all optional):
      - hardware: CPU %, RAM usage, battery level
      - time_of_day: morning/afternoon/evening shift
      - external: arbitrary dict of {name: float} overrides

    Usage:
        injector = LiveWeightInjector()
        injector.inject(system)          # hardware + time-of-day
        injector.inject(system, ctx={"hrv": 0.8, "load": 0.3})
    """

    # Thresholds for hardware pressure signals
    _CPU_HIGH    = 70.0   # % — high CPU → reduce left weight (less aggressive)
    _RAM_HIGH    = 80.0   # % — high RAM → reduce left weight
    _BATT_LOW    = 20.0   # % — low battery → tilt toward conservative (right)

    def __init__(
        self,
        hw_weight:      float = 0.5,   # sensitivity of hardware signals
        time_weight:    float = 0.3,   # sensitivity of time-of-day signal
        context_weight: float = 1.0,   # sensitivity of external context
    ) -> None:
        self._hw_w   = hw_weight
        self._time_w = time_weight
        self._ctx_w  = context_weight

    # ── public API ────────────────────────────────────────────────────────

    def inject(
        self,
        system:    ThreeBarSystem,
        ctx:       dict[str, float] | None = None,
        lever_idx: int                     = 0,
    ) -> None:
        """
        Compute net left/right delta weights from all active signals and
        apply them to *system.levers[lever_idx]*.

        Parameters
        ----------
        system    : ThreeBarSystem to modify in-place.
        ctx       : Optional external context {signal_name: 0.0–1.0}.
                    Positive values increase left weight; negative increase right.
        lever_idx : Which lever to inject into (default 0).
        """
        lever = system.levers[lever_idx]

        left_delta  = 0.0
        right_delta = 0.0

        # ── hardware pressure ────────────────────────────────────────────
        hw = self._hardware_pressure()
        if hw["cpu_pct"] > self._CPU_HIGH:
            right_delta += self._hw_w * (hw["cpu_pct"] - self._CPU_HIGH) / 30.0
        if hw["ram_pct"] > self._RAM_HIGH:
            right_delta += self._hw_w * (hw["ram_pct"] - self._RAM_HIGH) / 20.0
        if hw["battery_pct"] is not None and hw["battery_pct"] < self._BATT_LOW:
            right_delta += self._hw_w * 0.5

        # ── time-of-day shift ────────────────────────────────────────────
        tod_left = self._time_of_day_weight()
        left_delta  += self._time_w * tod_left
        right_delta += self._time_w * (1.0 - tod_left)

        # ── external context ─────────────────────────────────────────────
        if ctx:
            for value in ctx.values():
                clamped = max(-1.0, min(1.0, float(value)))
                if clamped >= 0:
                    left_delta  += self._ctx_w * clamped
                else:
                    right_delta += self._ctx_w * abs(clamped)

        # Apply deltas
        lever.add_weight("left",  max(0.0, left_delta))
        lever.add_weight("right", max(0.0, right_delta))

        logger.debug(
            "LiveWeightInjector: injected Δleft=%.3f Δright=%.3f into lever %d",
            left_delta, right_delta, lever_idx,
        )

    def hardware_snapshot(self) -> dict:
        """Return current hardware state as a plain dict."""
        return self._hardware_pressure()

    # ── private ───────────────────────────────────────────────────────────

    @staticmethod
    def _hardware_pressure() -> dict:
        cpu_pct = psutil.cpu_percent(interval=None)
        ram_pct = psutil.virtual_memory().percent
        try:
            batt    = psutil.sensors_battery()
            bat_pct = batt.percent if batt else None
        except Exception:
            bat_pct = None
        return {"cpu_pct": cpu_pct, "ram_pct": ram_pct, "battery_pct": bat_pct}

    @staticmethod
    def _time_of_day_weight() -> float:
        """Return 0–1 weight: higher in morning (peak focus), lower late evening."""
        hour = datetime.now(tz=timezone.utc).hour
        # Morning 6–10 → 0.8, Afternoon 11–17 → 0.6, Evening/Night → 0.4
        if 6 <= hour < 11:
            return 0.8
        if 11 <= hour < 18:
            return 0.6
        return 0.4


# ---------------------------------------------------------------------------
# GroundTruthOptimizer
# ---------------------------------------------------------------------------

class GroundTruthOptimizer:
    """
    Applies observed outcome scores to nudge lever fulcrum positions.

    Unlike the registry hill-climber (which searches snapshot space), this
    optimizer performs single-step gradient nudges on a live ThreeBarSystem
    based on direct outcome feedback from the practitioner or executor.

    Each call to ``update`` adjusts the fulcrum of the specified lever by a
    small step in the direction that would have produced a better result:
        score > target → good → strengthen current direction (tighten fulcrum)
        score < target → bad  → nudge fulcrum toward the opposite extreme
    """

    def __init__(
        self,
        step_size:   float = 0.02,   # maximum fulcrum adjustment per call
        target_score: float = 0.75,  # what a "good" outcome looks like
        lever_idx:   int   = 1,      # which lever's fulcrum to adjust
    ) -> None:
        self._step      = step_size
        self._target    = target_score
        self._lever_idx = lever_idx

    # ── public API ────────────────────────────────────────────────────────

    def update(
        self,
        system: ThreeBarSystem,
        score:  float,
        clamp:  tuple[float, float] = (0.1, 0.9),
    ) -> float:
        """
        Nudge *system.levers[lever_idx].fulcrum* based on *score*.

        Parameters
        ----------
        system : The ThreeBarSystem to update in-place.
        score  : Observed outcome quality (0.0–1.0).
        clamp  : (min, max) bounds for the fulcrum after adjustment.

        Returns the updated fulcrum value.
        """
        lever   = system.levers[self._lever_idx]
        current = lever.fulcrum_bias
        error   = score - self._target

        # Positive error → score is above target → slight reinforcement
        # Negative error → score below target → push fulcrum_bias toward zero
        # Gradient via tanh; multiplier 3.0 controls sensitivity — higher values
        # make the update more aggressive for mid-range errors, lower values smoother.
        _TANH_SCALE = 3.0
        delta   = self._step * math.tanh(error * _TANH_SCALE)
        new_val = current + delta
        new_val = max(clamp[0], min(clamp[1], new_val))
        lever.fulcrum_bias = new_val

        logger.debug(
            "GroundTruthOptimizer: fulcrum_bias %d %.3f → %.3f (score=%.3f, err=%.3f)",
            self._lever_idx, current, new_val, score, error,
        )
        return new_val

    def batch_update(
        self,
        system: ThreeBarSystem,
        scores: list[float],
        clamp:  tuple[float, float] = (0.1, 0.9),
    ) -> float:
        """Apply multiple score updates sequentially and return final fulcrum."""
        for s in scores:
            self.update(system, s, clamp)
        return system.levers[self._lever_idx].fulcrum_bias

    def calibrate(
        self,
        system:  ThreeBarSystem,
        history: list[dict],
        score_key: str = "score",
        clamp:   tuple[float, float] = (0.1, 0.9),
    ) -> float:
        """
        Replay a list of historical outcome dicts containing *score_key*.
        Useful for bootstrapping a new profile from past data.
        """
        scores = [
            float(h[score_key])
            for h in history
            if score_key in h and h[score_key] is not None
        ]
        return self.batch_update(system, scores, clamp)
