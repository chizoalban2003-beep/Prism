"""
prism_kinetic_engine.py
=======================
Cross-domain compound signal engine — the mathematical core of PRISM's
proactive intelligence.

Why this exists in PRISM
------------------------
PRISM's proactive triggers are single-signal: HRV drop → recovery alert.
Deadline → calendar warning. Each fires in isolation. A crystallised personal
AI should do better: recognise when *multiple personal signals converge* and
act on the compound pattern before any single threshold trips.

This engine is the mechanism for that. It normalises signals from PRISM's
perception channels into a universal Z-score currency, accumulates weighted
torque on decision "levers", and fires an ActionWindow when compound pressure
exceeds a threshold — with damping to prevent noise-driven whipsaw.

Signal domains (PRISM-native)
-----------------------------
  "health"    — biometrics: HRV, sleep quality, soreness, readiness score
  "cognitive" — mental load: task density, decision count, chain depth
  "temporal"  — time pressure: deadline proximity, meeting density, buffer
  "energy"    — physical state: activity level, fatigue index, recovery debt
  "social"    — relational load: pending messages, unanswered threads

Lever examples
--------------
  "intervene_now"   — fires when health + temporal + cognitive all spike
  "defer_decision"  — fires when energy is low + cognitive load is high
  "proactive_assist"— fires when temporal gap + good health + pending tasks align

Integration points
------------------
  1. prism_perception.py  → engine.ingest(personal_signal())  on each factor update
  2. engine.on_action()   → ProactiveEvent queued into PrismProactive
  3. ActionWindow.delta_a → contributed to prism_phase.py Φ_melt (compound stress)

Architecture note
-----------------
The math is identical to the Kinetic arbitrage spec:
  τᵢ = Σ(Wᵢⱼ × Lᵢⱼ)        ← torque from cross-domain signals
  Z = (x − μ) / σ            ← universal anomaly currency
  ΔA = V_potential − V_current − C_friction   ← net action value
  Dampers: EMA + hysteresis + dashpot + Bayesian clutch
  Crisis bypass: Z ≥ 8 or dτ/dt > V_critical
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Maps perception factor_ids → Kinetic signal domains
# Used by the ContextFuser bridge to classify signals without extra config.
FACTOR_DOMAIN_MAP: dict[str, str] = {
    # Health / biometrics
    "hrv_recovery":      "health",
    "sleep_quality":     "health",
    "stress_level":      "health",
    "cardio_state":      "health",
    "sport_readiness":   "health",
    "soreness":          "health",
    # Energy / physical state
    "circadian_energy":  "energy",
    "battery_level":     "energy",
    "on_power":          "energy",
    # Cognitive / mental load
    "system_busy":       "cognitive",
    "memory_pressure":   "cognitive",
    "typing_speed":      "cognitive",
    "typing_regularity": "cognitive",
    "keyboard_active":   "cognitive",
    # Temporal / time pressure
    "work_context":      "temporal",
    # Social
    "screen_activity":   "social",
}


@dataclass
class PersonalSignal:
    """
    A Z-scored signal from one of PRISM's personal perception domains.

    ``raw_value`` is a dimensioned reading (e.g. HRV in ms, deadline minutes).
    ``mu`` and ``sigma`` are the user's *personal* rolling baseline — not a
    population average. After months of crystallisation these become unique to
    this specific user.
    """

    domain: str          # "health" | "cognitive" | "temporal" | "energy" | "social"
    signal_type: str     # e.g. "hrv_drop", "deadline_proximity", "task_density"
    raw_value: float
    mu: float            # user's personal rolling mean
    sigma: float         # user's personal rolling std dev (> 0)
    impact: float = 1.0  # how much this matters to the user right now (0–1)
    confidence: float = 1.0  # signal confidence: 1.0 = measured, 0.5 = inferred
    timestamp: float = field(default_factory=time.time)

    @property
    def z_score(self) -> float:
        """Anomaly intensity relative to this user's personal baseline."""
        if self.sigma == 0:
            return 0.0
        return (self.raw_value - self.mu) / self.sigma

    @property
    def expected_value(self) -> float:
        """impact × confidence — how much torque this signal should contribute."""
        return self.impact * self.confidence


@dataclass
class DecisionLever:
    """
    One proactive decision beam. Accumulates torque from cross-domain signals.
    Only activates when sustained compound pressure exceeds thresholds —
    preventing noise-driven false fires.
    """

    lever_id: str
    name: str
    description: str
    # Accumulated torque state
    net_torque: float = 0.0
    torque_integral: float = 0.0    # dashpot ∫τ dt — requires sustained pressure
    ema_torque: float = 0.0         # EMA low-pass — filters high-frequency noise
    last_updated: float = field(default_factory=time.time)
    activated: bool = False
    activation_time: Optional[float] = None
    # Hysteresis: activate threshold > deactivate threshold → prevents toggling
    activate_threshold: float = 3.0
    deactivate_threshold: float = 1.5
    dashpot_threshold: float = 10.0   # ∫τ dt required before slow signals fire
    # Compound-signal gate: lever only accepts torque from these domains,
    # AND can only fire when every gate domain has contributed recently.
    # Empty tuple = ungated (legacy behaviour, accepts every signal).
    gate_domains: tuple[str, ...] = ()
    gate_window_sec: float = 300.0
    # Last (timestamp, torque) per gated domain — populated in ingest().
    domain_torque_history: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass
class CrossDomainLink:
    """
    λ = ∂target/∂source — how much a signal in one personal domain
    amplifies pressure on another. Updated from outcome history.

    Examples:
      health → cognitive : low HRV raises cognitive load sensitivity (λ ≈ 0.4)
      temporal → energy  : tight deadlines drain perceived energy  (λ ≈ 0.3)
    """

    source_domain: str
    target_domain: str
    lambda_base: float
    confidence: float = 1.0   # P(H|E) — Bayesian clutch, decays if link unvalidated
    last_updated: float = field(default_factory=time.time)

    @property
    def lambda_effective(self) -> float:
        """λ_eff = λ_base × P(H|E) — Bayesian slipping clutch."""
        return self.lambda_base * self.confidence


@dataclass
class ActionWindow:
    """
    A detected moment when PRISM should proactively act.
    Fires when compound personal-signal torque exceeds threshold.

    ΔA = V_potential − V_current − C_friction
    If ΔA > engine.action_threshold → PRISM should act.
    """

    window_id: str
    lever_id: str
    source_signal: PersonalSignal
    v_potential: float    # value of acting now
    v_current: float      # value of the current state (usually 0)
    c_friction: float     # cost of interrupting / acting
    delta_a: float        # net action value: ΔA = V_potential − V_current − C_friction
    is_crisis: bool = False  # True when Z ≥ 8 or velocity crisis bypassed dampers
    triggered_at: float = field(default_factory=time.time)

    def to_proactive_message(self) -> str:
        direction = "urgent — crisis bypass active" if self.is_crisis else "suggested"
        return (
            f"[Kinetic/{self.lever_id}] Compound signal threshold crossed "
            f"({self.source_signal.domain}/{self.source_signal.signal_type} "
            f"Z={self.source_signal.z_score:.1f}, ΔA={self.delta_a:.2f}) — {direction}"
        )


class KineticEngine:
    """
    PRISM's compound proactive signal engine.

    Thread-safe. Designed to run continuously in the background alongside
    PrismPerception and PrismProactive.

    Quick start::

        engine = KineticEngine.for_prism()

        # Wire perception updates in:
        engine.ingest(PersonalSignal("health", "hrv_drop",
                                     raw_value=42, mu=65, sigma=12,
                                     impact=0.8, confidence=0.9))

        # Register proactive callback:
        engine.on_action(lambda w: proactive.schedule(w.to_proactive_message(), ...))

        # Inspect state:
        windows = engine.active_windows()
    """

    EMA_ALPHA: float = 0.3          # low-pass smoothing coefficient
    DASHPOT_DT: float = 1.0         # time step for ∫τ dt accumulation
    BLACK_SWAN_Z: float = 8.0       # Z ≥ 8 → bypass all dampers
    VELOCITY_CRITICAL: float = 2.0  # dτ/dt > V_critical → velocity crisis bypass
    _HISTORY_CAP: int = 1000        # max signals stored per domain

    def __init__(self, action_threshold: float = 5.0) -> None:
        self._levers: dict[str, DecisionLever] = {}
        self._links: list[CrossDomainLink] = []
        self._windows: list[ActionWindow] = []
        self._signal_history: dict[str, list[PersonalSignal]] = {}
        self._lock = threading.Lock()
        self._action_threshold = action_threshold
        self._callbacks: list[Callable[[ActionWindow], None]] = []

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def for_prism(cls) -> KineticEngine:
        """
        Create a KineticEngine pre-configured with PRISM's personal levers
        and evidence-based cross-domain links.

        Levers are tuned for personal-AI use, not trading. The math is the
        same; only the signal domains differ.
        """
        engine = cls(action_threshold=4.0)

        # Levers
        engine.add_lever(DecisionLever(
            "intervene_now",
            "Intervene Now",
            "Fires when health + temporal + cognitive all spike simultaneously. "
            "PRISM should act proactively: draft, reschedule, or alert.",
            activate_threshold=3.5, deactivate_threshold=1.5, dashpot_threshold=8.0,
            gate_domains=("health", "temporal", "cognitive"),
        ))
        engine.add_lever(DecisionLever(
            "defer_decision",
            "Defer Decision",
            "Fires when energy is low AND cognitive load is high. "
            "PRISM should flag that now is a bad time for irreversible choices.",
            activate_threshold=2.5, deactivate_threshold=1.0, dashpot_threshold=6.0,
            gate_domains=("energy", "cognitive"),
        ))
        engine.add_lever(DecisionLever(
            "proactive_assist",
            "Proactive Assist",
            "Fires when health is good AND temporal gap exists AND tasks are pending. "
            "PRISM should surface deferred work or suggestions.",
            activate_threshold=2.0, deactivate_threshold=0.8, dashpot_threshold=5.0,
            gate_domains=("health", "temporal"),
        ))

        # Cross-domain links (evidence-based defaults; updated by outcome tracker)
        engine.add_link(CrossDomainLink("health",    "cognitive", lambda_base=0.40))
        engine.add_link(CrossDomainLink("health",    "energy",    lambda_base=0.55))
        engine.add_link(CrossDomainLink("temporal",  "cognitive", lambda_base=0.35))
        engine.add_link(CrossDomainLink("temporal",  "health",    lambda_base=0.20))
        engine.add_link(CrossDomainLink("energy",    "cognitive", lambda_base=0.30))
        engine.add_link(CrossDomainLink("social",    "cognitive", lambda_base=0.25))

        return engine

    # ── Public API ────────────────────────────────────────────────────────────

    def add_lever(self, lever: DecisionLever) -> None:
        with self._lock:
            self._levers[lever.lever_id] = lever

    def add_link(self, link: CrossDomainLink) -> None:
        with self._lock:
            self._links.append(link)

    def on_action(self, callback: Callable[[ActionWindow], None]) -> None:
        """Register callback(window) invoked whenever an ActionWindow fires."""
        self._callbacks.append(callback)

    def ingest(self, signal: PersonalSignal) -> list[ActionWindow]:
        """
        Process a personal perception signal.
        Returns any newly triggered ActionWindows.
        """
        with self._lock:
            history = self._signal_history.setdefault(signal.domain, [])
            history.append(signal)
            if len(history) > self._HISTORY_CAP:
                self._signal_history[signal.domain] = history[-self._HISTORY_CAP:]

            is_crisis = abs(signal.z_score) >= self.BLACK_SWAN_Z
            new_windows: list[ActionWindow] = []

            for lever in self._levers.values():
                # Compound-signal gate — skip levers whose gate set excludes
                # this signal's domain. Prevents one strong signal from firing
                # every lever at once (issue #27 bug 5).
                if lever.gate_domains and signal.domain not in lever.gate_domains:
                    continue

                lambda_eff = self._get_lambda(signal.domain)
                torque_raw = signal.z_score * lambda_eff * signal.expected_value

                # Record per-domain torque so the gate can require every
                # gate domain to have contributed within `gate_window_sec`.
                if lever.gate_domains:
                    lever.domain_torque_history[signal.domain] = (
                        time.time(), torque_raw,
                    )

                # ── Peacetime damping ────────────────────────────────────────
                crisis = is_crisis
                if not crisis:
                    lever.ema_torque = (
                        self.EMA_ALPHA * torque_raw
                        + (1 - self.EMA_ALPHA) * lever.ema_torque
                    )
                    effective = lever.ema_torque

                    # Velocity bypass
                    dt = time.time() - lever.last_updated
                    if dt > 0 and abs(effective - lever.net_torque) / dt > self.VELOCITY_CRITICAL:
                        crisis = True
                else:
                    effective = torque_raw

                # Crisis override — raw signal, no filtering
                if crisis:
                    effective = torque_raw

                lever.net_torque = effective
                lever.last_updated = time.time()
                lever.torque_integral += abs(effective) * self.DASHPOT_DT

                # ── Compound-domain readiness ───────────────────────────────
                # A gated lever can only fire when every gate domain has
                # contributed torque within `gate_window_sec`. This is the
                # "compound" in compound-signal — one strong signal isn't
                # enough; the multi-domain pattern must be present.
                # Black-swan (Z≥8) bypasses this — a true crisis still
                # alerts — but the velocity-bypass `crisis` does not, since
                # rapid-fire from a single domain is exactly the false-fire
                # we're trying to prevent.
                now_ts = time.time()
                compound_ready = True
                if lever.gate_domains and not is_crisis:
                    cutoff = now_ts - lever.gate_window_sec
                    compound_ready = all(
                        lever.domain_torque_history.get(d, (0.0, 0.0))[0] >= cutoff
                        for d in lever.gate_domains
                    )

                # ── Hysteresis gate ──────────────────────────────────────────
                fire = False
                if is_crisis:
                    # True black-swan (Z≥8): always alerts.
                    fire = True
                elif crisis and compound_ready:
                    # Velocity-promoted crisis still respects the gate so a
                    # single-domain rapid-fire can't masquerade as compound.
                    fire = True
                elif (not lever.activated
                      and compound_ready
                      and abs(lever.net_torque) >= lever.activate_threshold):
                    fire = lever.torque_integral >= lever.dashpot_threshold
                elif lever.activated and abs(lever.net_torque) < lever.deactivate_threshold:
                    lever.activated = False

                if fire and not lever.activated:
                    lever.activated = True
                    lever.activation_time = time.time()

                    v_potential = signal.expected_value * abs(lambda_eff)
                    c_friction = abs(lever.net_torque) * 0.1
                    delta_a = v_potential - c_friction

                    if delta_a > self._action_threshold or crisis:
                        window = ActionWindow(
                            window_id=str(uuid.uuid4())[:8],
                            lever_id=lever.lever_id,
                            source_signal=signal,
                            v_potential=v_potential,
                            v_current=0.0,
                            c_friction=c_friction,
                            delta_a=delta_a,
                            is_crisis=is_crisis,
                        )
                        self._windows.append(window)
                        new_windows.append(window)
                        for cb in self._callbacks:
                            try:
                                cb(window)
                            except Exception:  # noqa: BLE001
                                pass

            return new_windows

    def active_windows(self, max_age_seconds: float = 3600.0) -> list[ActionWindow]:
        cutoff = time.time() - max_age_seconds
        with self._lock:
            return [w for w in self._windows if w.triggered_at >= cutoff]

    def lever_status(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "lever_id":        lv.lever_id,
                    "name":            lv.name,
                    "net_torque":      round(lv.net_torque, 4),
                    "ema_torque":      round(lv.ema_torque, 4),
                    "torque_integral": round(lv.torque_integral, 4),
                    "activated":       lv.activated,
                }
                for lv in self._levers.values()
            ]

    def link_status(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "source_domain":    lk.source_domain,
                    "target_domain":    lk.target_domain,
                    "lambda_base":      lk.lambda_base,
                    "confidence":       lk.confidence,
                    "lambda_effective": round(lk.lambda_effective, 6),
                }
                for lk in self._links
            ]

    def update_link_confidence(
        self, source: str, target: str, new_confidence: float
    ) -> None:
        """Bayesian clutch update — called by outcome tracker when link accuracy is measured."""
        with self._lock:
            for link in self._links:
                if link.source_domain == source and link.target_domain == target:
                    link.confidence = max(0.0, min(1.0, new_confidence))
                    link.last_updated = time.time()

    def compound_phi_delta(self) -> float:
        """
        Contribution to Φ_melt from active compound signals.
        Crisis windows contribute more heavily than standard activations.
        Used by prism_phase.py to factor personal signal compound pressure
        into the crystallisation engine.
        """
        recent = self.active_windows(max_age_seconds=300.0)
        if not recent:
            return 0.0
        return sum(
            (w.delta_a * 2.0 if w.is_crisis else w.delta_a)
            for w in recent
        ) / max(len(recent), 1)

    def _get_lambda(self, domain: str) -> float:
        """Max effective cross-elasticity for signals originating from domain."""
        best = 0.0
        for link in self._links:
            if link.source_domain == domain:
                best = max(best, abs(link.lambda_effective))
        return best if best > 0 else 0.1
