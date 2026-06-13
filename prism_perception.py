"""
prism_perception.py
===================
PRISM Perceptual Context Engine

Continuous background sensing that converts raw input from all available
channels into normalised factor values (0-1) for the decision engine.

The core loop:
  Every channel runs in its own daemon thread.
  Each produces a stream of ContextSignal objects.
  The ContextFuser aggregates signals into a ContextState.
  The ContextState is a dict of factor_id → float used by PrismAgent
  to enrich every decision with real-time perceptual context.

Privacy principles (enforced in code):
  All processing is local — nothing leaves the device.
  Raw data (audio frames, camera frames) is never stored.
  Only derived factor values are stored.
  Every channel is opt-in and can be paused at any time.
  A visible indicator shows which channels are active.
"""

from __future__ import annotations

import json as _json_mod
import logging
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context signal — the unit of output from every channel
# ---------------------------------------------------------------------------

@dataclass
class ContextSignal:
    channel:    str          # "voice"|"biometric"|"screen"|"system"|"typing"
    factor_id:  str          # maps to a Factor name in the decision engine
    value:      float        # 0.0 to 1.0 normalised
    confidence: float        # how reliable is this reading
    timestamp:  float = field(default_factory=time.time)
    raw_label:  str   = ""   # human-readable description of the raw reading


@dataclass
class ContextState:
    """
    The fused, current-moment context built from all active channels.
    This is what the decision engine sees — not raw sensor data,
    but a clean map of normalised factor values.
    """
    factors:      dict[str, float]    # factor_id → value
    confidence:   dict[str, float]    # factor_id → confidence
    active_channels: list[str]
    last_updated: float = field(default_factory=time.time)
    summary:      str   = ""          # plain English state summary

    def to_factor_updates(self) -> dict[str, float]:
        """Return only factors with sufficient confidence."""
        return {k: v for k, v in self.factors.items()
                if self.confidence.get(k, 0) >= 0.4}


# ---------------------------------------------------------------------------
# Base channel
# ---------------------------------------------------------------------------

class PerceptionChannel:
    """
    Abstract base. Each channel runs in its own daemon thread,
    pushing ContextSignal objects to a shared queue.
    """
    NAME = "base"

    def __init__(self, signal_queue: queue.Queue, enabled: bool = True):
        self._q       = signal_queue
        self._enabled = enabled
        self._stop    = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self._enabled:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"prism-{self.NAME}")
        self._thread.start()
        logger.info("Perception channel started: %s", self.NAME)

    def stop(self) -> None:
        self._stop.set()

    def pause(self) -> None:
        self._enabled = False

    def resume(self) -> None:
        self._enabled = True

    def _emit(self, factor_id: str, value: float,
               confidence: float, raw_label: str = "") -> None:
        self._q.put(ContextSignal(
            channel    = self.NAME,
            factor_id  = factor_id,
            value      = max(0.0, min(1.0, value)),
            confidence = confidence,
            raw_label  = raw_label,
        ))

    def _run(self) -> None:
        """Override in subclasses. Must check self._stop periodically."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# System context channel — always available, no permissions needed
# ---------------------------------------------------------------------------

class SystemContextChannel(PerceptionChannel):
    """
    Reads system state: time of day, battery, CPU load, active apps,
    network location. No microphone or camera required.
    Polls every 30 seconds.
    """
    NAME = "system"

    def _run(self) -> None:
        while not self._stop.wait(30.0):
            if not self._enabled:
                continue
            try:
                self._emit_time_context()
                self._emit_system_load()
                self._emit_battery()
            except Exception as e:
                logger.debug("System channel error: %s", e)

    def _emit_time_context(self) -> None:
        """Circadian context — energy level follows a human daily rhythm."""
        from datetime import datetime
        h = datetime.now().hour
        # Rough human energy curve: peaks at 10am and 3pm, low at 2pm and night
        curve = {
            0:0.15, 1:0.10, 2:0.08, 3:0.08, 4:0.10, 5:0.20,
            6:0.40, 7:0.60, 8:0.75, 9:0.85, 10:0.90, 11:0.88,
            12:0.80, 13:0.70, 14:0.65, 15:0.82, 16:0.80, 17:0.72,
            18:0.65, 19:0.58, 20:0.50, 21:0.42, 22:0.32, 23:0.22,
        }
        self._emit("circadian_energy", curve.get(h, 0.5), 0.7,
                   f"{h:02d}:00 circadian phase")

        # Work hours context
        is_work_hours = 8 <= h <= 18
        self._emit("work_context", 0.8 if is_work_hours else 0.2,
                   0.9, "work hours" if is_work_hours else "off hours")

    def _emit_system_load(self) -> None:
        try:
            import psutil
            cpu   = psutil.cpu_percent(interval=1) / 100.0
            mem   = psutil.virtual_memory().percent / 100.0
            # High CPU = system is busy = user is likely actively working
            self._emit("system_busy", cpu, 0.8, f"CPU {cpu:.0%}")
            self._emit("memory_pressure", mem, 0.8, f"RAM {mem:.0%}")
        except ImportError:
            pass

    def _emit_battery(self) -> None:
        try:
            import psutil
            bat = psutil.sensors_battery()
            if bat:
                level   = bat.percent / 100.0
                on_power= 1.0 if bat.power_plugged else 0.0
                self._emit("battery_level",   level,    0.95, f"Battery {bat.percent:.0f}%")
                self._emit("on_power",         on_power, 0.95,
                           "plugged in" if bat.power_plugged else "on battery")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Typing pattern channel — infers focus and stress from keyboard behaviour
# ---------------------------------------------------------------------------

class TypingPatternChannel(PerceptionChannel):
    """
    Monitors typing cadence via the system keyboard buffer.
    Does NOT capture what is typed — only timing patterns.

    Inferred signals:
      typing_speed    → high = focused and energetic
      typing_regularity → irregular = stressed or distracted
      active_typing   → currently in keyboard-intensive work
    """
    NAME = "typing"

    def __init__(self, signal_queue: queue.Queue, enabled: bool = True):
        super().__init__(signal_queue, enabled)
        self._key_times: list[float] = []
        self._max_samples = 30

    def record_keypress(self) -> None:
        """Call this from a keyboard hook to record timing."""
        now = time.time()
        self._key_times.append(now)
        if len(self._key_times) > self._max_samples:
            self._key_times.pop(0)

    def _run(self) -> None:
        """Emit typing-derived signals every 10 seconds."""
        while not self._stop.wait(10.0):
            if not self._enabled or len(self._key_times) < 5:
                continue
            self._analyse()
            # Prune old timestamps
            cutoff = time.time() - 60.0
            self._key_times = [t for t in self._key_times if t > cutoff]

    def _analyse(self) -> None:
        if len(self._key_times) < 3:
            return
        times  = sorted(self._key_times[-20:])
        gaps   = [times[i+1]-times[i] for i in range(len(times)-1)]
        if not gaps:
            return
        avg_gap = sum(gaps) / len(gaps)
        std_gap = math.sqrt(sum((g-avg_gap)**2 for g in gaps) / len(gaps))

        # Speed: inverse of average gap, normalised to 0-1
        # 0.1s gap (10 kps) = high speed, >2s gap = slow
        speed = max(0.0, min(1.0, 1.0 - (avg_gap - 0.08) / 1.5))

        # Regularity: lower std relative to mean = more regular = more focused
        cv = std_gap / max(avg_gap, 0.001)   # coefficient of variation
        regularity = max(0.0, min(1.0, 1.0 - cv * 0.5))

        self._emit("typing_speed",       speed,       0.7, f"avg gap {avg_gap:.2f}s")
        self._emit("typing_regularity",  regularity,  0.6, f"CV {cv:.2f}")
        self._emit("keyboard_active",    1.0,         0.9, "actively typing")


# ---------------------------------------------------------------------------
# Sport readiness model
# ---------------------------------------------------------------------------

class SportReadinessModel:
    """
    Composite sport readiness score from wearable biometrics.
    Weights are calibrated per sport based on the primary physiological demands.
    Score in [0, 1]; labels: 'peak' ≥0.80, 'ready' ≥0.60, 'caution' ≥0.40, 'rest' <0.40.
    """

    # (hrv_weight, sleep_weight, intensity_weight, soreness_weight)
    _SPORT_WEIGHTS: dict[str, tuple[float, float, float, float]] = {
        "football":    (0.30, 0.25, 0.25, 0.20),
        "basketball":  (0.25, 0.25, 0.30, 0.20),
        "rugby":       (0.20, 0.25, 0.30, 0.25),
        "tennis":      (0.35, 0.30, 0.20, 0.15),
        "cycling":     (0.35, 0.25, 0.25, 0.15),
        "default":     (0.30, 0.30, 0.20, 0.20),
    }

    def __init__(self, sport: str = "default") -> None:
        self.sport = sport.lower()

    def _weights(self) -> tuple[float, float, float, float]:
        return self._SPORT_WEIGHTS.get(self.sport, self._SPORT_WEIGHTS["default"])

    def score(
        self,
        hrv_ms: float | None = None,
        sleep_hrs: float | None = None,
        high_intensity_mins: float | None = None,
        soreness: float | None = None,
    ) -> float:
        """Weighted composite readiness score in [0, 1]."""
        wh, ws, wi, wsor = self._weights()
        components: list[tuple[float, float]] = []

        if hrv_ms is not None:
            hrv_norm = max(0.0, min(1.0, (hrv_ms - 20.0) / 80.0))
            components.append((hrv_norm, wh))

        if sleep_hrs is not None:
            sleep_norm = max(0.0, min(1.0, (sleep_hrs - 4.0) / 5.0))
            components.append((sleep_norm, ws))

        if high_intensity_mins is not None:
            # 0 mins = fully rested; >90 mins in last 24h = fatigued
            intensity_fatigue = max(0.0, min(1.0, high_intensity_mins / 90.0))
            components.append((1.0 - intensity_fatigue, wi))

        if soreness is not None:
            soreness_norm = max(0.0, min(1.0, soreness / 10.0))
            components.append((1.0 - soreness_norm, wsor))

        if not components:
            return 0.5
        total_w = sum(w for _, w in components)
        return sum(v * w for v, w in components) / total_w

    def label(self, score: float) -> str:
        if score >= 0.80:
            return "peak"
        if score >= 0.60:
            return "ready"
        if score >= 0.40:
            return "caution"
        return "rest"

    def to_factor(self, score: float) -> float:
        """Convert readiness score to a 0–1 decision factor."""
        return round(score, 4)


# ---------------------------------------------------------------------------
# VEAXDebtDynamics — coupled ODE system for VEAX debt decay (Vector II)
# ---------------------------------------------------------------------------

class VEAXDebtDynamics:
    """
    Coupled ODE system for VEAX debt decay.
    dS/dt = M·S + U  where S = [V_debt, E_debt, A_debt, X_debt]^T

    Biological coupling rationale:
    - High A_debt (amygdala/stress) suppresses V and E recovery (PFC hijack)
    - High V_debt (executive fatigue) slightly disinhibits A (loss of PFC control)
    - High A_debt suppresses X recovery (stress kills articulation)
    - High E_debt suppresses X (can't explain unconsolidated learning)

    Stability: solution is clamped to [0, 2] after each Euler step, guaranteeing
    boundedness regardless of off-diagonal coupling magnitudes.
    """

    # Axis index mapping
    _IDX: dict[str, int] = {"V": 0, "E": 1, "A": 2, "X": 3}

    # Diagonal: natural decay rates (1/τ per hour)
    # Off-diagonal: coupling terms (how debt[col] affects recovery of debt[row])
    # Negative off-diagonal = suppression of recovery
    # Positive off-diagonal = facilitation (rare)
    _M: list[list[float]] = [
        [-1 / 72,   0.008,  -0.150,   0.000],   # V row: suppressed by A
        [ 0.008,  -1 / 24,  -0.100,  -0.040],   # E row: suppressed by A and X
        [ 0.040,   0.015,  -1 / 4,    0.000],   # A row: disinhibited by V/E fatigue
        [ 0.000,  -0.060,  -0.100,  -1 / 20],   # X row: suppressed by E and A
    ]

    def __init__(self) -> None:
        self._debt: list[float] = [0.0, 0.0, 0.0, 0.0]  # [V, E, A, X]
        self._last_ts: float = 0.0

    def add_debt(self, axis: str, amount: float) -> None:
        """Called when a fatigue rule fires for a given axis."""
        idx = self._IDX.get(axis)
        if idx is not None:
            self._debt[idx] = min(2.0, self._debt[idx] + abs(amount))

    def step(self, dt_hours: float, U: list[float] | None = None) -> list[float]:
        """
        Euler integration step. U = external biological input [V, E, A, X].
        Returns current debt vector [V, E, A, X].
        """
        if dt_hours <= 0:
            return list(self._debt)
        if U is None:
            U = [0.0, 0.0, 0.0, 0.0]

        # dS/dt = M·S + U
        S = self._debt
        dS = [0.0, 0.0, 0.0, 0.0]
        for i in range(4):
            coupled = sum(self._M[i][j] * S[j] for j in range(4))
            dS[i] = coupled + U[i]

        # Euler step, clamp to [0, 2]
        for i in range(4):
            self._debt[i] = max(0.0, min(2.0, S[i] + dS[i] * dt_hours))

        return list(self._debt)

    def get_axis_debt(self, axis: str) -> float:
        idx = self._IDX.get(axis, -1)
        return self._debt[idx] if idx >= 0 else 0.0

    def global_debt(self) -> float:
        """Normalized sum: 0=no debt, 1=fully loaded."""
        return min(1.0, sum(self._debt) / 4.0)


# ---------------------------------------------------------------------------
# BiometricVEAXBridge — asymmetric EMA + debt accumulator VEAX auto-update
# ---------------------------------------------------------------------------

@dataclass
class _HysteresisState:
    """Per-rule EMA + debt accumulator state (with allostatic fields)."""
    ema:             float = 0.0   # fast EMA (current signal strength in [0, 1])
    slow_ema:        float = 0.0   # long-term slow EMA (adapted baseline)
    debt:            float = 0.0   # fatigue debt accumulated from downward deltas
    allostatic_load: float = 0.0   # accumulated chronic stress area (fast > slow)
    baseline_shift:  float = 0.0   # how much "100%" has permanently degraded [0, 0.3]
    last_ts:         float = 0.0   # unix timestamp of last update (0 = never)


class BiometricVEAXBridge:
    """
    Applies VEAX deltas based on biometric signal values using an asymmetric
    EMA + debt accumulator system that prevents spurious recovery after brief
    positive spikes during sustained fatigue.

    Design:
      - α_down = 0.25 for all axes (fast fatigue accumulation, ~2 ticks to cross 0.3)
      - α_up varies per axis (slow recovery):
          V: 0.016 (τ ≈ 72h at 1h ticks)
          E: 0.042 (τ ≈ 24h)
          A: 0.25  (τ ≈ 4h, fastest)
          X: 0.05  (τ ≈ 20h)
      - Debt accumulates on downward VEAX deltas; decays at 0.1/hour
      - Upward VEAX deltas are blocked when debt > 0.5
      - Fatigue rules fire when EMA > 0.3; recovery rules require EMA > 0.7

    Call apply(factors, now=None) after each perception batch.
    """

    # (factor_id, comparison, threshold, veax_deltas, primary_axis)
    # primary_axis: the VEAX axis whose α_up governs this rule's recovery speed.
    # "fatigue" rules have negative deltas; "recovery" rules have positive deltas.
    _RULES: list[tuple[str, str, float, dict[str, float], str]] = [
        ("sport_readiness",    "<",  0.40, {"A": -0.10},              "A"),
        ("sport_readiness",    "<",  0.30, {"A": -0.15, "V": +0.10},  "A"),
        ("sport_readiness",    ">",  0.80, {"A": +0.05},              "A"),
        ("hrv_recovery",       "<",  0.30, {"V": -0.10, "E": -0.10},  "V"),
        ("hrv_recovery",       ">",  0.80, {"E": +0.10},              "E"),
        ("stress_level",       ">",  0.70, {"X": +0.15, "A": -0.10}, "X"),
        ("stress_level",       ">",  0.85, {"V": +0.10},              "V"),
        ("sleep_quality",      "<",  0.40, {"V": -0.10, "E": -0.10}, "V"),
        ("sleep_quality",      ">",  0.80, {"E": +0.05, "X": -0.05}, "E"),
        ("cognitive_readiness",">",  0.80, {"E": +0.10, "V": +0.05}, "E"),
        ("cognitive_readiness","<",  0.40, {"A": -0.15, "X": +0.10}, "A"),
    ]

    # Per-axis slow-recovery α values (τ in hours at 1h ticks = -1/ln(1-α_up))
    _AXIS_ALPHA_UP: dict[str, float] = {
        "V": 0.016,   # τ ≈ 72h — verification trust, very slow to rebuild
        "E": 0.042,   # τ ≈ 24h — evolution latitude
        "A": 0.25,    # τ ≈ 4h  — autonomy, fastest recovery
        "X": 0.05,    # τ ≈ 20h — explanation verbosity
    }
    _ALPHA_DOWN: float = 0.25  # fast fatigue accumulation for all axes

    # EMA thresholds: fatigue rules fire above FATIGUE_THRESH, recovery above RECOVERY_THRESH
    _FATIGUE_THRESH:  float = 0.3
    _RECOVERY_THRESH: float = 0.7

    # Debt threshold above which upward deltas are blocked
    _DEBT_BLOCK_THRESH: float = 0.5

    # Debt decay rate: units per hour
    _DEBT_DECAY_RATE: float = 0.1

    def __init__(self) -> None:
        # Maps rule index → per-rule hysteresis state
        self._hyst: dict[int, _HysteresisState] = {}
        # Coupled ODE dynamics for VEAX debt (Vector II)
        self._dynamics: VEAXDebtDynamics = VEAXDebtDynamics()

    @property
    def dynamics(self) -> VEAXDebtDynamics:
        """Expose VEAXDebtDynamics for external access (e.g. CrystallizationEngine)."""
        return self._dynamics

    def _state(self, idx: int) -> _HysteresisState:
        if idx not in self._hyst:
            self._hyst[idx] = _HysteresisState()
        return self._hyst[idx]

    def _is_recovery_rule(self, deltas: dict[str, float]) -> bool:
        """A rule is a recovery rule if any of its deltas is strictly positive."""
        return any(v > 0.0 for v in deltas.values())

    def apply(self, factors: dict[str, float], now: float | None = None) -> dict[str, float]:
        """
        Evaluate all rules against current factor values using asymmetric EMA.
        Returns the net delta dict (empty if nothing fired).
        """
        if now is None:
            now = time.time()

        net: dict[str, float] = {}

        for idx, (fid, cmp, thresh, deltas, primary_axis) in enumerate(self._RULES):
            value = factors.get(fid)
            if value is None:
                continue

            state = self._state(idx)

            # Default dt to 3600s (1 hour) on first call
            dt = (now - state.last_ts) if state.last_ts > 0.0 else 3600.0

            # Evaluate whether the rule condition is met
            condition_met = (cmp == "<" and value < thresh) or (cmp == ">" and value > thresh)

            # Asymmetric EMA update
            alpha_up = self._AXIS_ALPHA_UP.get(primary_axis, 0.05)
            if condition_met:
                state.ema = state.ema + self._ALPHA_DOWN * (1.0 - state.ema)
            else:
                state.ema = state.ema - alpha_up * state.ema

            # ── Allostatic tracking (Vector I) ────────────────────────────
            # α_slow is 10x slower than α_up: very long-term adaptation memory
            alpha_slow = alpha_up * 0.1
            dt_hours = dt / 3600.0

            # Update slow EMA (long-term adaptation baseline)
            state.slow_ema = state.slow_ema + alpha_slow * (state.ema - state.slow_ema)

            # Allostatic load: area where fast EMA chronically exceeds slow EMA
            if state.ema > state.slow_ema:
                # Stress incursion above adapted baseline
                state.allostatic_load += (state.ema - state.slow_ema) * dt_hours * 0.01
            else:
                # Recovery: load decreases at half the accumulation rate
                state.allostatic_load = max(
                    0.0,
                    state.allostatic_load - (state.slow_ema - state.ema) * dt_hours * 0.005,
                )

            # Baseline shift: very slow drift upward under sustained load, very slow recovery
            _BETA = 0.002  # τ ≈ 2 weeks
            if state.allostatic_load > 1.0:
                # Only shift when meaningfully loaded
                state.baseline_shift = min(
                    0.30,
                    state.baseline_shift + _BETA * (state.allostatic_load - 1.0) * dt_hours,
                )
            else:
                # Macro-recovery: baseline shifts back at 5x accumulation rate when load is low
                state.baseline_shift = max(
                    0.0,
                    state.baseline_shift - _BETA * 5.0 * dt_hours,
                )
            # ─────────────────────────────────────────────────────────────

            # Debt update
            # Decay debt regardless of whether condition fired
            state.debt = max(0.0, state.debt - self._DEBT_DECAY_RATE * (dt / 3600.0))
            # Accumulate debt when condition fires and rule has downward (fatigue) deltas
            if condition_met and any(v < 0.0 for v in deltas.values()):
                state.debt += sum(abs(v) for v in deltas.values()) * 0.5

            state.last_ts = now

            # Determine EMA threshold based on rule type
            is_recovery = self._is_recovery_rule(deltas)
            threshold = self._RECOVERY_THRESH if is_recovery else self._FATIGUE_THRESH

            # Check if EMA crosses the firing threshold
            if state.ema <= threshold:
                continue

            # Apply per-axis debt blocking for upward deltas
            # Allostatic capacity reduction: recovery is capped at (1 - baseline_shift)
            for axis, delta in deltas.items():
                if delta > 0.0:
                    if state.debt > self._DEBT_BLOCK_THRESH:
                        continue  # debt blocking
                    effective_delta = delta * (1.0 - state.baseline_shift)
                    net[axis] = net.get(axis, 0.0) + effective_delta
                else:
                    net[axis] = net.get(axis, 0.0) + delta

            # Wire fatigue events into VEAXDebtDynamics (Vector II)
            if condition_met and any(v < 0.0 for v in deltas.values()):
                for axis, delta in deltas.items():
                    if delta < 0.0:
                        self._dynamics.add_debt(axis, abs(delta))

        # Step VEAXDebtDynamics ODE once per apply() call (Vector II)
        # Use median dt across initialized rule states (fall back to 1h)
        all_dts = [
            (now - s.last_ts) / 3600.0
            for s in self._hyst.values()
            if s.last_ts > 0.0
        ]
        median_dt_hours = sorted(all_dts)[len(all_dts) // 2] if all_dts else 1.0
        self._dynamics.step(median_dt_hours)

        if not net:
            return {}

        # Apply deltas to current VEAX gates
        try:
            from prism_spectrum_middleware import (
                SpectrumGates,
                get_current_gates,
                save_spectrum_state,
            )
            current = get_current_gates()
            if current is None:
                from prism_spectrum_middleware import load_spectrum
                loaded = load_spectrum()
                # load_spectrum may return (gates, network) tuple or just gates
                current = loaded[0] if isinstance(loaded, tuple) else loaded
            new_vals: dict[str, float] = {
                "V": current.V,
                "E": current.E,
                "A": current.A,
                "X": current.X,
            }
            for axis, delta in net.items():
                if axis in new_vals:
                    new_vals[axis] = max(0.0, min(1.0, new_vals[axis] + delta))
            new_gates = SpectrumGates(**new_vals)
            save_spectrum_state(new_gates)
            logger.debug("[BiometricVEAXBridge] applied deltas %s → new VEAX %s", net, new_vals)
        except Exception as exc:
            logger.debug("[BiometricVEAXBridge] VEAX update failed: %s", exc)

        return net

    def biological_pressure(self) -> float:
        """
        ΔB signal: normalized biological pressure from slow-τ VEAX axes.
        V×0.6 + E×0.4, normalized to [0, 1]. Used by SiliconResponsePolicy.
        """
        v = self._dynamics.get_axis_debt("V")
        e = self._dynamics.get_axis_debt("E")
        return min(1.0, (v * 0.6 + e * 0.4) / 1.5)

    def allostatic_report(self) -> dict[str, dict[str, float]]:
        """Return current allostatic state per rule index. For diagnostics."""
        return {
            str(idx): {
                "ema": s.ema,
                "slow_ema": s.slow_ema,
                "allostatic_load": s.allostatic_load,
                "baseline_shift": s.baseline_shift,
                "debt": s.debt,
            }
            for idx, s in self._hyst.items()
        }


# ---------------------------------------------------------------------------
# Biometric channel — reads from device_hub wearable data
# ---------------------------------------------------------------------------

class BiometricChannel(PerceptionChannel):
    """
    Reads wearable data from device_hub.py (Apple Health, Garmin, etc.)
    and converts to decision-relevant factor values.
    Polls every 5 minutes (wearable data does not change faster).
    """
    NAME = "biometric"

    def __init__(self, signal_queue: queue.Queue,
                 device_hub=None, enabled: bool = True,
                 sport: str = "default"):
        super().__init__(signal_queue, enabled)
        self._hub = device_hub
        self._sport_model = SportReadinessModel(sport)

    def _run(self) -> None:
        while not self._stop.wait(300.0):   # every 5 minutes
            if not self._enabled or self._hub is None:
                continue
            try:
                self._read_wearables()
            except Exception as e:
                logger.debug("Biometric channel error: %s", e)

    def ingest(self, hrv_ms: Optional[float] = None, heart_rate: Optional[int] = None,
                sleep_hrs: Optional[float] = None, steps: Optional[int] = None,
                soreness: Optional[int] = None,
                high_intensity_mins: Optional[float] = None,
                training_load: Optional[float] = None) -> None:
        """
        Direct ingestion for manual or wearable-pushed data.
        Call this when a wearable sync occurs.
        high_intensity_mins: minutes of high-intensity exercise in the last 24h
        training_load: subjective session RPE * duration (arbitrary units)
        """
        if hrv_ms is not None:
            # HRV: <30ms = very stressed, >80ms = well recovered
            hrv_norm = max(0.0, min(1.0, (hrv_ms - 20) / 80.0))
            self._emit("hrv_recovery", hrv_norm, 0.9,
                       f"HRV {hrv_ms:.0f}ms")
            self._emit("stress_level", 1.0 - hrv_norm, 0.85,
                       "stress from HRV")

        if heart_rate is not None:
            # Resting HR: <55 = athletic, >90 = stressed/unfit
            hr_norm = max(0.0, min(1.0, 1.0 - (heart_rate - 45) / 60.0))
            self._emit("cardio_state", hr_norm, 0.8, f"HR {heart_rate}bpm")

        if sleep_hrs is not None:
            # Sleep: <5hrs = very poor, >8hrs = excellent
            sleep_norm = max(0.0, min(1.0, (sleep_hrs - 4.0) / 5.0))
            self._emit("sleep_quality", sleep_norm, 0.95,
                       f"sleep {sleep_hrs:.1f}hrs")
            self._emit("cognitive_readiness", sleep_norm * 0.7 + 0.15,
                       0.85, "readiness from sleep")

        if steps is not None:
            # Steps: 0 = sedentary, 10000+ = active
            steps_norm = min(1.0, steps / 10000.0)
            self._emit("activity_today", steps_norm, 0.8, f"{steps} steps")

        if soreness is not None:
            # Soreness: 1-10 scale
            soreness_norm = max(0.0, min(1.0, soreness / 10.0))
            self._emit("physical_soreness", soreness_norm, 0.85,
                       f"soreness {soreness}/10")

        if training_load is not None:
            # Normalise against a ~400 AU daily ceiling (e.g. RPE 8 × 50min)
            load_norm = max(0.0, min(1.0, training_load / 400.0))
            self._emit("training_load", load_norm, 0.80, f"load {training_load:.0f}au")

        # Sport readiness composite signal (only when at least one biometric is present)
        sport_inputs = [hrv_ms, sleep_hrs, high_intensity_mins, soreness]
        if any(v is not None for v in sport_inputs):
            readiness_score = self._sport_model.score(
                hrv_ms=hrv_ms,
                sleep_hrs=sleep_hrs,
                high_intensity_mins=high_intensity_mins,
                soreness=float(soreness) if soreness is not None else None,
            )
            label = self._sport_model.label(readiness_score)
            self._emit("sport_readiness", self._sport_model.to_factor(readiness_score),
                       0.88, f"{self._sport_model.sport} readiness: {label} ({readiness_score:.2f})")

    def _read_wearables(self) -> None:
        if not self._hub:
            return
        try:
            data = self._hub.latest_health_snapshot()
            if data:
                self.ingest(
                    hrv_ms              = data.get("hrv"),
                    heart_rate          = data.get("heart_rate"),
                    sleep_hrs           = data.get("sleep_hours"),
                    steps               = data.get("steps"),
                    soreness            = data.get("soreness"),
                    high_intensity_mins = data.get("high_intensity_mins"),
                    training_load       = data.get("training_load"),
                )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Voice channel — speech-to-text + tone analysis
# ---------------------------------------------------------------------------

class VoiceChannel(PerceptionChannel):
    """
    Microphone → speech-to-text via Whisper (local, via Ollama or openai-whisper).
    Also analyses speech rate and energy as stress/focus proxies.

    Privacy: raw audio frames are NEVER stored or transmitted.
    Only the transcribed text and derived factor values are used.

    Requires: either 'openai-whisper' pip package or Ollama with a speech model.
    Falls back gracefully if neither is available.
    """
    NAME = "voice"
    SAMPLE_RATE   = 16000
    CHUNK_SECONDS = 30    # process in 30-second chunks

    def __init__(self, signal_queue: queue.Queue,
                 whisper_model: str = "base",
                 wake_word:     str = "hey prism",
                 enabled:       bool = True,
                 on_transcript: Optional[Callable] = None):
        super().__init__(signal_queue, enabled)
        self._whisper_model  = whisper_model
        self._wake_word      = wake_word.lower()
        self._on_transcript  = on_transcript   # callback when speech detected
        self._whisper        = None
        self._microphone_ok  = False

    def _run(self) -> None:
        if not self._try_init():
            logger.info("Voice channel: no audio input available — skipping")
            return

        logger.info("Voice channel active (wake word: '%s')", self._wake_word)
        while not self._stop.wait(0.1):
            if not self._enabled:
                time.sleep(1.0)
                continue
            try:
                self._listen_chunk()
            except Exception as e:
                logger.debug("Voice channel error: %s", e)
                time.sleep(5.0)

    def _try_init(self) -> bool:
        """Try to initialise audio input and Whisper. Return True if ready."""
        try:
            import pyaudio  # noqa
            self._microphone_ok = True
        except ImportError:
            return False

        try:
            import whisper
            self._whisper = whisper.load_model(self._whisper_model)
            return True
        except ImportError:
            pass

        return False

    def _listen_chunk(self) -> None:
        """
        Record CHUNK_SECONDS of audio, transcribe, analyse.
        If wake word detected: trigger on_transcript callback.
        """
        try:
            import numpy as np
            import pyaudio
        except ImportError:
            time.sleep(60.0)
            return

        pa     = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1,
                         rate=self.SAMPLE_RATE, input=True,
                         frames_per_buffer=1024)
        frames = []
        for _ in range(0, int(self.SAMPLE_RATE / 1024 * self.CHUNK_SECONDS)):
            if self._stop.is_set():
                break
            frames.append(stream.read(1024, exception_on_overflow=False))
        stream.stop_stream()
        stream.close()
        pa.terminate()

        if not frames:
            return

        # Convert to numpy for Whisper — raw audio never written to disk
        audio = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32)
        audio = audio / 32768.0   # normalise to [-1, 1]

        # Voice activity — are they speaking at all?
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.005:    # silence threshold
            self._emit("voice_active", 0.0, 0.9, "silence detected")
            return

        self._emit("voice_active", 1.0, 0.9, "speech detected")

        # Speech rate proxy — rough estimate from zero-crossing rate
        zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2)
        speech_rate_norm = min(1.0, zcr * 20.0)
        self._emit("speech_rate", speech_rate_norm, 0.6,
                   f"ZCR proxy {zcr:.3f}")

        # Transcribe
        if self._whisper is None:
            return
        try:
            result = self._whisper.transcribe(audio, fp16=False, language="en")
            text   = result.get("text","").strip().lower()
            if not text:
                return

            # Stress signals from language
            stress_words = ["urgent","immediately","asap","deadline","help",
                            "worried","stressed","overwhelmed","behind"]
            calm_words   = ["fine","good","great","relaxed","done","finished"]
            stress_count = sum(1 for w in stress_words if w in text)
            calm_count   = sum(1 for w in calm_words   if w in text)
            if stress_count or calm_count:
                total = stress_count + calm_count
                stress_signal = stress_count / total
                self._emit("voice_stress", stress_signal, 0.5,
                           f"stress words: {stress_count}")

            # Wake word detection
            if self._wake_word in text:
                command = text.split(self._wake_word, 1)[-1].strip()
                if command and self._on_transcript:
                    self._on_transcript(command)

        except Exception as e:
            logger.debug("Whisper transcription error: %s", e)


# ---------------------------------------------------------------------------
# Screen context channel
# ---------------------------------------------------------------------------

class ScreenContextChannel(PerceptionChannel):
    """
    Periodic screenshot → Ollama LLaVA analysis.
    Infers what kind of work is happening and current focus level.
    Privacy: screenshots are never stored — only the analysis text.
    Polls every 2 minutes.
    """
    NAME = "screen"

    def __init__(self, signal_queue: queue.Queue,
                 ollama_host: str = "http://localhost:11434",
                 enabled: bool = False):   # opt-in only
        super().__init__(signal_queue, enabled)
        self._ollama = ollama_host

    def _run(self) -> None:
        while not self._stop.wait(120.0):   # every 2 minutes
            if not self._enabled:
                continue
            try:
                self._analyse_screen()
            except Exception as e:
                logger.debug("Screen channel error: %s", e)

    def _analyse_screen(self) -> None:
        # Capture screenshot
        try:
            from PIL import ImageGrab
            shot = ImageGrab.grab()
        except ImportError:
            try:
                import mss
                import PIL.Image
                with mss.mss() as sct:
                    raw  = sct.grab(sct.monitors[1])
                    shot = PIL.Image.frombytes("RGB", raw.size, raw.bgra, "raw","BGRX")
            except ImportError:
                return

        # Resize to reduce LLaVA processing time
        shot = shot.resize((640, 360))

        # Convert to base64 for LLaVA
        import base64
        import io
        buf = io.BytesIO()
        shot.save(buf, format="JPEG", quality=60)
        b64 = base64.b64encode(buf.getvalue()).decode()

        prompt = (
            "Analyse this screenshot in 2 sentences. State: "
            "1) what kind of work is shown (coding/writing/email/browsing/idle/meeting), "
            "2) estimated focus level (high/medium/low). "
            "Reply with JSON: {\"work_type\":\"...\",\"focus\":\"high|medium|low\"}"
        )

        try:
            import json
            import urllib.request
            payload = json.dumps({
                "model":  "llava",
                "prompt": prompt,
                "images": [b64],
                "stream": False,
            }).encode()
            req  = urllib.request.Request(
                f"{self._ollama}/api/generate",
                data=payload,
                headers={"Content-Type":"application/json"})
            resp = urllib.request.urlopen(req, timeout=15)
            outer = json.loads(resp.read())
            data = json.loads(outer.get("response","{}").strip())

            focus_map  = {"high": 0.9, "medium": 0.5, "low": 0.2}
            focus_val  = focus_map.get(data.get("focus","medium"), 0.5)
            work_type  = data.get("work_type","unknown")

            self._emit("screen_focus",     focus_val, 0.65, work_type)
            self._emit("screen_work_type",
                       0.9 if work_type != "idle" else 0.1, 0.7, work_type)

            # Work type → domain context
            type_signal = {
                "coding":    ("developer_context", 0.9),
                "email":     ("communication_context", 0.8),
                "writing":   ("creative_context", 0.8),
                "meeting":   ("meeting_active", 0.9),
                "browsing":  ("research_context", 0.6),
                "idle":      ("idle_context", 0.9),
            }.get(work_type)
            if type_signal:
                self._emit(type_signal[0], type_signal[1], 0.65, work_type)

        except Exception as e:
            logger.debug("LLaVA screen analysis failed: %s", e)


# ---------------------------------------------------------------------------
# Context fuser — aggregates all channel signals into one ContextState
# ---------------------------------------------------------------------------

class ContextFuser:
    """
    Subscribes to the shared signal queue.
    Maintains a rolling window of signals (last 10 minutes).
    Produces a ContextState by taking confidence-weighted averages.
    """
    WINDOW_SECONDS = 600     # 10-minute rolling window
    DECAY_HALF_LIFE = 120.0  # older signals count less (2-min half-life)

    def __init__(self, signal_queue: queue.Queue):
        self._q           = signal_queue
        self._signals: dict[str, list[ContextSignal]] = {}
        self._lock        = threading.Lock()
        self._stop        = threading.Event()
        self._thread      = threading.Thread(
            target=self._fuse_loop, daemon=True, name="prism-fuser")
        self._veax_bridge = BiometricVEAXBridge()
        # KineticEngine bridge — wired by prism_agent after init
        self._kinetic: Optional[Any] = None
        self._kinetic_baselines: dict[str, tuple[float, float]] = {}  # (mu, sigma)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def current_state(self) -> ContextState:
        """Return the current fused context state."""
        now  = time.time()
        cutoff = now - self.WINDOW_SECONDS
        factors: dict[str, float]    = {}
        confidence: dict[str, float] = {}

        with self._lock:
            for factor_id, sigs in self._signals.items():
                recent = [s for s in sigs if s.timestamp > cutoff]
                if not recent:
                    continue
                # Exponential time-weighted average
                weights = [
                    s.confidence * (0.5 ** ((now - s.timestamp) / self.DECAY_HALF_LIFE))
                    for s in recent
                ]
                total_w = sum(weights)
                if total_w < 1e-9:
                    continue
                fused_value = sum(s.value * w for s, w in zip(recent, weights)) / total_w
                avg_conf    = sum(s.confidence for s in recent) / len(recent)
                factors[factor_id]    = round(fused_value, 3)
                confidence[factor_id] = round(avg_conf,    3)

        active = list({s.channel for sigs in self._signals.values()
                       for s in sigs if s.timestamp > now - 60})

        state = ContextState(
            factors          = factors,
            confidence       = confidence,
            active_channels  = active,
            last_updated     = now,
            summary          = self._summarise(factors),
        )

        # Zero-latency VEAX auto-update from biometric factors
        try:
            self._veax_bridge.apply(factors)
        except Exception as _bve:
            logger.debug("[ContextFuser] veax_bridge error: %s", _bve)

        return state

    def _fuse_loop(self) -> None:
        while not self._stop.wait(0.1):
            try:
                sig = self._q.get(timeout=0.5)
                with self._lock:
                    if sig.factor_id not in self._signals:
                        self._signals[sig.factor_id] = []
                    self._signals[sig.factor_id].append(sig)
                    # Prune old
                    cutoff = time.time() - self.WINDOW_SECONDS
                    self._signals[sig.factor_id] = [
                        s for s in self._signals[sig.factor_id]
                        if s.timestamp > cutoff
                    ]
                if self._kinetic is not None:
                    self._feed_kinetic(sig)
            except queue.Empty:
                pass

    def _feed_kinetic(self, sig: Any) -> None:
        """Bridge: convert a ContextSignal to a PersonalSignal and ingest into KineticEngine."""
        try:
            from prism_kinetic_engine import FACTOR_DOMAIN_MAP, PersonalSignal
            mu, sigma = self._kinetic_baselines.get(sig.factor_id, (0.5, 0.15))
            # Online EMA baseline update (slow-moving personal reference)
            alpha = 0.05
            new_mu = alpha * sig.value + (1.0 - alpha) * mu
            new_sigma = max(0.05, alpha * abs(sig.value - mu) + (1.0 - alpha) * sigma)
            self._kinetic_baselines[sig.factor_id] = (new_mu, new_sigma)
            domain = FACTOR_DOMAIN_MAP.get(sig.factor_id, "cognitive")
            kinetic = self._kinetic
            if kinetic is None:
                return
            kinetic.ingest(PersonalSignal(
                domain=domain,
                signal_type=sig.factor_id,
                raw_value=sig.value,
                mu=new_mu,
                sigma=new_sigma,
                impact=sig.confidence,
                confidence=sig.confidence,
            ))
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _summarise(factors: dict[str, float]) -> str:
        parts = []
        if factors.get("stress_level", 0) > 0.7:
            parts.append("high stress")
        elif factors.get("hrv_recovery", 1) > 0.7:
            parts.append("well recovered")
        if factors.get("sleep_quality", 0.5) < 0.4:
            parts.append("sleep-deprived")
        if factors.get("screen_focus", 0.5) > 0.75:
            parts.append("focused")
        if factors.get("voice_active", 0) > 0.5:
            parts.append("actively speaking")
        return ", ".join(parts) if parts else "normal context"


# ---------------------------------------------------------------------------
# Main perception engine
# ---------------------------------------------------------------------------

def _try_ingest_health_dir(
    health_path: Path,
    seen: set[str],
    ingest_fn: Callable,
    sport: str = "default",
) -> None:
    """Read new JSON health dump files from a directory and ingest them."""
    if not health_path.is_dir():
        return
    for fp in health_path.glob("*.json"):
        key = fp.name
        if key in seen:
            continue
        try:
            data = _json_mod.loads(fp.read_text())
            ingest_fn(
                hrv_ms              = data.get("hrv"),
                heart_rate          = data.get("heart_rate"),
                sleep_hrs           = data.get("sleep_hours"),
                steps               = data.get("steps"),
                soreness            = data.get("soreness"),
                high_intensity_mins = data.get("high_intensity_mins"),
                training_load       = data.get("training_load"),
            )
            seen.add(key)
        except Exception as exc:
            logging.getLogger(__name__).debug("health dir ingest error %s: %s", fp.name, exc)


class PrismPerception:
    """
    Orchestrates all perception channels.
    Provides a single interface for the rest of PRISM to get context.

    Usage:
        perception = PrismPerception.setup(
            enable_voice  = True,
            enable_screen = False,  # off by default — explicit opt-in
            device_hub    = hub,
            on_voice_command = agent.chat,
        )
        perception.start()

        # Anywhere in the codebase:
        context = perception.current_context()
        # context.factors = {"stress_level":0.65,"sleep_quality":0.45,...}
        # Pass these to the decision engine as factor updates.
    """

    def __init__(
        self,
        enable_voice:    bool = False,
        enable_screen:   bool = False,
        enable_biometric:bool = True,
        enable_system:   bool = True,
        enable_typing:   bool = True,
        device_hub             = None,
        ollama_host:     str  = "http://localhost:11434",
        whisper_model:   str  = "base",
        wake_word:       str  = "hey prism",
        on_voice_command: Optional[Callable] = None,
    ):
        self._q: queue.Queue = queue.Queue()
        self._fuser   = ContextFuser(self._q)
        self._channels: list[PerceptionChannel] = []
        self._typing: Optional[TypingPatternChannel] = None

        if enable_system:
            self._channels.append(SystemContextChannel(self._q))

        if enable_typing:
            self._typing = TypingPatternChannel(self._q)
            self._channels.append(self._typing)

        if enable_biometric:
            self._channels.append(BiometricChannel(self._q, device_hub))

        if enable_voice:
            self._channels.append(VoiceChannel(
                self._q,
                whisper_model    = whisper_model,
                wake_word        = wake_word,
                on_transcript    = on_voice_command,
            ))

        if enable_screen:
            self._channels.append(ScreenContextChannel(
                self._q, ollama_host=ollama_host))

    @classmethod
    def setup(cls, **kwargs) -> PrismPerception:
        return cls(**kwargs)

    def start(self) -> None:
        self._fuser.start()
        for ch in self._channels:
            ch.start()
        logger.info("PRISM perception started. Active channels: %s",
                    [c.NAME for c in self._channels if c._enabled])

    def stop(self) -> None:
        self._fuser.stop()
        for ch in self._channels:
            ch.stop()

    def current_context(self) -> ContextState:
        return self._fuser.current_state()

    def ingest_biometrics(self, **kwargs) -> None:
        """Push wearable data directly: hrv_ms, sleep_hrs, steps, etc."""
        for ch in self._channels:
            if isinstance(ch, BiometricChannel):
                ch.ingest(**kwargs)
                break

    def watch_health_dir(self, path: str, sport: str = "default") -> None:
        """
        Poll a directory for JSON health dumps (e.g. Apple Health exports).
        Ingests any new file dropped into `path` since the last poll.
        Runs in a background thread; safe to call once after start().
        """
        import threading as _threading
        health_path = Path(path)
        seen: set[str] = set()

        def _poll() -> None:
            while True:
                _try_ingest_health_dir(health_path, seen, self.ingest_biometrics, sport)
                import time as _t
                _t.sleep(60.0)

        t = _threading.Thread(target=_poll, daemon=True, name="prism-health-watch")
        t.start()
        logger.info("PrismPerception: watching health dir %s (sport=%s)", path, sport)

    def record_keypress(self) -> None:
        """Call from keyboard hook to feed typing pattern channel."""
        if self._typing:
            self._typing.record_keypress()

    def status(self) -> dict:
        state = self.current_context()
        return {
            "active_channels": state.active_channels,
            "factor_count":    len(state.factors),
            "summary":         state.summary,
            "factors":         state.factors,
        }
