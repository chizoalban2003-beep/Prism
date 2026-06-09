"""
prism_kinetic_engine.py
=======================
Project Kinetic: cross-domain stochastic decision engine.

Models decisions as mechanical levers accumulating torque from cross-domain
signals. Detects arbitrage windows before they reach mainstream markets by
monitoring the latency between physical-world events and financial reactions.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class DomainSignal:
    """A normalised signal from any domain (logistics, commodities, financial)."""

    domain: str          # "maritime", "commodities", "financial", "geopolitical"
    signal_type: str     # e.g. "port_delay", "copper_spot", "vol_spike"
    raw_value: float
    mu: float            # rolling mean
    sigma: float         # rolling std dev (>0)
    value_at_risk: float = 0.0    # USD value exposed
    probability: float = 1.0      # confidence 0-1
    timestamp: float = field(default_factory=time.time)

    @property
    def z_score(self) -> float:
        """Anomaly intensity — universal cross-domain currency."""
        if self.sigma == 0:
            return 0.0
        return (self.raw_value - self.mu) / self.sigma

    @property
    def esu(self) -> float:
        """Expected Systemic Utility = VaR × P."""
        return self.value_at_risk * self.probability


@dataclass
class DecisionLever:
    """One decision beam. Accumulates torque from incoming signals."""

    lever_id: str
    name: str
    description: str
    # Accumulated state
    net_torque: float = 0.0
    torque_integral: float = 0.0    # dashpot: ∫τ dt
    ema_torque: float = 0.0         # low-pass filtered torque
    last_updated: float = field(default_factory=time.time)
    activated: bool = False
    activation_time: Optional[float] = None
    # Hysteresis thresholds
    activate_threshold: float = 3.0      # higher threshold to activate
    deactivate_threshold: float = 1.5    # lower threshold to deactivate
    dashpot_threshold: float = 10.0      # ∫τ dt needed for slow signals


@dataclass
class CrossElasticityLink:
    """λ = ∂target/∂source — how much target domain moves per unit source change."""

    source_domain: str
    target_domain: str
    lambda_base: float        # base cross-elasticity
    confidence: float = 1.0   # P(H|E) — Bayesian confidence
    last_updated: float = field(default_factory=time.time)

    @property
    def lambda_effective(self) -> float:
        """Bayesian slipping clutch: λ_eff = λ_base × P(H|E)."""
        return self.lambda_base * self.confidence


@dataclass
class ArbitrageWindow:
    """A detected cross-domain arbitrage opportunity."""

    window_id: str
    lever_id: str
    source_signal: DomainSignal
    u_potential: float      # utility if acted on
    u_current: float        # utility of current state
    c_friction: float       # cost/friction of acting
    delta_a: float          # ΔA = U_potential - U_current - C_friction
    is_black_swan: bool = False
    triggered_at: float = field(default_factory=time.time)

    @property
    def net_opportunity(self) -> float:
        return self.delta_a


class KineticEngine:
    """
    Cross-domain stochastic decision engine.

    Usage::

        engine = KineticEngine()

        # Register decision levers
        engine.add_lever(DecisionLever("hedge_copper", "Hedge Copper Exposure", "..."))

        # Register cross-elasticity links
        engine.add_link(CrossElasticityLink("maritime", "commodities", lambda_base=0.34))

        # Ingest domain signals
        engine.ingest(DomainSignal("maritime", "port_delay", raw_value=3.2, mu=1.1, sigma=0.8))

        # Get active arbitrage windows
        windows = engine.active_windows()
    """

    EMA_ALPHA: float = 0.3          # low-pass filter coefficient
    DASHPOT_DT: float = 1.0         # time step for integral accumulation
    BLACK_SWAN_Z: float = 8.0       # Z > 8 bypasses all dampers
    VELOCITY_CRITICAL: float = 2.0  # dτ/dt threshold for crisis bypass

    def __init__(self, arbitrage_threshold: float = 5.0) -> None:
        self._levers: dict[str, DecisionLever] = {}
        self._links: list[CrossElasticityLink] = []
        self._windows: list[ArbitrageWindow] = []
        self._signal_history: dict[str, list[DomainSignal]] = {}
        self._lock = threading.Lock()
        self._arbitrage_threshold = arbitrage_threshold
        self._callbacks: list[Callable[[ArbitrageWindow], None]] = []

    def add_lever(self, lever: DecisionLever) -> None:
        with self._lock:
            self._levers[lever.lever_id] = lever

    def add_link(self, link: CrossElasticityLink) -> None:
        with self._lock:
            self._links.append(link)

    def on_arbitrage(self, callback: Callable[[ArbitrageWindow], None]) -> None:
        """Register callback(window: ArbitrageWindow) for new windows."""
        self._callbacks.append(callback)

    def ingest(self, signal: DomainSignal) -> list[ArbitrageWindow]:
        """
        Process a new domain signal. Returns any newly triggered ArbitrageWindows.
        """
        with self._lock:
            # Store history — trim to 1000 per domain
            history = self._signal_history.setdefault(signal.domain, [])
            history.append(signal)
            if len(history) > 1000:
                self._signal_history[signal.domain] = history[-1000:]

            is_black_swan = abs(signal.z_score) >= self.BLACK_SWAN_Z
            new_windows: list[ArbitrageWindow] = []

            for lever in self._levers.values():
                # Compute torque contribution from this signal via cross-elasticity
                lambda_eff = self._get_lambda(signal.domain, lever)
                torque_contribution = signal.z_score * lambda_eff * signal.esu

                # ── Damping (peacetime) ──────────────────────────────────────
                crisis = is_black_swan
                if not crisis:
                    # EMA low-pass filter
                    lever.ema_torque = (
                        self.EMA_ALPHA * torque_contribution
                        + (1 - self.EMA_ALPHA) * lever.ema_torque
                    )
                    effective_torque = lever.ema_torque

                    # Velocity trigger check
                    dt = time.time() - lever.last_updated
                    if dt > 0:
                        torque_velocity = abs(effective_torque - lever.net_torque) / dt
                        if torque_velocity > self.VELOCITY_CRITICAL:
                            crisis = True  # velocity crisis bypass
                else:
                    effective_torque = torque_contribution

                # ── Crisis bypass ────────────────────────────────────────────
                if crisis:
                    effective_torque = torque_contribution  # raw, no filtering

                # Update lever state
                lever.net_torque = effective_torque
                lever.last_updated = time.time()

                # Dashpot: accumulate integral
                lever.torque_integral += abs(effective_torque) * self.DASHPOT_DT

                # ── Hysteresis activation gate ───────────────────────────────
                should_fire = False
                if crisis:
                    should_fire = True
                elif not lever.activated and abs(lever.net_torque) >= lever.activate_threshold:
                    should_fire = lever.torque_integral >= lever.dashpot_threshold
                elif lever.activated and abs(lever.net_torque) < lever.deactivate_threshold:
                    lever.activated = False

                if should_fire and not lever.activated:
                    lever.activated = True
                    lever.activation_time = time.time()

                    # Compute arbitrage window
                    u_potential = signal.esu * abs(lambda_eff)
                    u_current = 0.0
                    c_friction = abs(lever.net_torque) * 0.1  # 10% friction cost
                    delta_a = u_potential - u_current - c_friction

                    if delta_a > self._arbitrage_threshold or crisis:
                        window = ArbitrageWindow(
                            window_id=str(uuid.uuid4())[:8],
                            lever_id=lever.lever_id,
                            source_signal=signal,
                            u_potential=u_potential,
                            u_current=u_current,
                            c_friction=c_friction,
                            delta_a=delta_a,
                            is_black_swan=crisis,
                        )
                        self._windows.append(window)
                        new_windows.append(window)
                        for cb in self._callbacks:
                            try:
                                cb(window)
                            except Exception:  # noqa: BLE001
                                pass

            return new_windows

    def active_windows(self, max_age_seconds: float = 3600.0) -> list[ArbitrageWindow]:
        """Return arbitrage windows created within max_age_seconds."""
        cutoff = time.time() - max_age_seconds
        with self._lock:
            return [w for w in self._windows if w.triggered_at >= cutoff]

    def lever_status(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "lever_id": lv.lever_id,
                    "name": lv.name,
                    "net_torque": round(lv.net_torque, 4),
                    "ema_torque": round(lv.ema_torque, 4),
                    "torque_integral": round(lv.torque_integral, 4),
                    "activated": lv.activated,
                }
                for lv in self._levers.values()
            ]

    def link_status(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "source_domain": lk.source_domain,
                    "target_domain": lk.target_domain,
                    "lambda_base": lk.lambda_base,
                    "confidence": lk.confidence,
                    "lambda_effective": round(lk.lambda_effective, 6),
                }
                for lk in self._links
            ]

    def update_link_confidence(
        self, source: str, target: str, new_confidence: float
    ) -> None:
        """Update Bayesian confidence for a cross-elasticity link (daily update)."""
        with self._lock:
            for link in self._links:
                if link.source_domain == source and link.target_domain == target:
                    link.confidence = max(0.0, min(1.0, new_confidence))
                    link.last_updated = time.time()

    def _get_lambda(self, domain: str, lever: DecisionLever) -> float:
        """Get effective cross-elasticity for a domain→lever pair."""
        best = 0.0
        for link in self._links:
            if link.source_domain == domain:
                best = max(best, abs(link.lambda_effective))
        return best if best > 0 else 0.1  # default minimal coupling
