"""
ksa_lever.py
============
Kinetic State Agent — Core 3-Bar Lever System

Models a cascade of three linked levers that simulate torque-based
decision-making. Each lever's tilt can propagate weight into the next
lever in the chain, producing a final equilibrium state that maps to
a concrete action or routing decision.

Architecture:
    Lever 0 (Input Lever)    — receives raw weighted inputs
    Lever 1 (Logic Lever)    — applies constraint bias via fulcrum offset
    Lever 2 (Balancer Bar)   — safety/health monitor; can override cascade

Usage:
    system = ThreeBarSystem.from_defaults()
    system.levers[0].set_weights(left=8.0, right=3.0)
    result = system.simulate()
    print(result)

    snapshot = system.snapshot()        # save state
    system2 = ThreeBarSystem()
    system2.hydrate(snapshot)           # restore state
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

# ---------------------------------------------------------------------------
# Enums & Result Types
# ---------------------------------------------------------------------------

class TiltDirection(str, Enum):
    LEFT      = "left"
    RIGHT     = "right"
    BALANCED  = "balanced"


@dataclass
class LeverState:
    """The computed outcome of a single lever after torque simulation."""
    lever_id:        int
    net_torque:      float          # positive = left-heavy, negative = right-heavy
    tilt:            TiltDirection
    tilt_magnitude:  float          # absolute torque, useful for propagation scaling
    is_locked:       bool = False   # True if the Linkage Matrix suppressed this lever


@dataclass
class EquilibriumResult:
    """Final output after the full 3-bar cascade has been simulated."""
    states:          list[LeverState]
    final_tilt:      TiltDirection   # Lever 2 (Balancer) outcome
    override_active: bool            # True if Balancer damped an upstream instability
    confidence:      float           # 0.0–1.0, derived from tilt_magnitude ratio

    def __str__(self) -> str:
        lines = ["── Equilibrium Result ──────────────────"]
        for s in self.states:
            lock = " [LOCKED]" if s.is_locked else ""
            lines.append(
                f"  Lever {s.lever_id}: {s.tilt.value:8s} | "
                f"torque={s.net_torque:+.3f}{lock}"
            )
        lines.append(f"  Final Decision : {self.final_tilt.value.upper()}")
        lines.append(f"  Override Active: {self.override_active}")
        lines.append(f"  Confidence     : {self.confidence:.2%}")
        lines.append("────────────────────────────────────────")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Lever
# ---------------------------------------------------------------------------

class Lever:
    """
    A single mechanical lever with a fulcrum, two arm lengths,
    and independent left/right weights.

    Torque model:
        torque_left  = left_weight  * left_arm_length
        torque_right = right_weight * right_arm_length
        net_torque   = torque_left - torque_right

        net_torque > +threshold  → tilts LEFT
        net_torque < -threshold  → tilts RIGHT
        otherwise                → BALANCED
    """

    # Below this absolute net torque the lever is considered balanced
    BALANCE_THRESHOLD: float = 0.01

    def __init__(
        self,
        lever_id:         int,
        left_arm_length:  float = 1.0,
        right_arm_length: float = 1.0,
        fulcrum_bias:     float = 0.0,
        left_weight:      float = 0.0,
        right_weight:     float = 0.0,
    ):
        """
        Args:
            lever_id:         Identifier (0, 1, or 2 in a 3-bar system).
            left_arm_length:  Distance from fulcrum to left end (increases sensitivity).
            right_arm_length: Distance from fulcrum to right end.
            fulcrum_bias:     A constant offset added to net_torque, representing
                              the lever's built-in operational predisposition
                              (positive = biased left, negative = biased right).
            left_weight:      Current load on the left arm.
            right_weight:     Current load on the right arm.
        """
        self.lever_id         = lever_id
        self.left_arm_length  = left_arm_length
        self.right_arm_length = right_arm_length
        self.fulcrum_bias     = fulcrum_bias
        self.left_weight      = left_weight
        self.right_weight     = right_weight

    # ── Weight API ──────────────────────────────────────────────────────────

    def set_weights(self, left: float, right: float) -> None:
        """Set both weights at once."""
        self.left_weight  = left
        self.right_weight = right

    def add_weight(self, side: str, amount: float) -> None:
        """Add weight to 'left' or 'right' side (used by cascade propagation)."""
        if side == "left":
            self.left_weight  = max(0.0, self.left_weight  + amount)
        elif side == "right":
            self.right_weight = max(0.0, self.right_weight + amount)
        else:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    # ── Physics ─────────────────────────────────────────────────────────────

    def compute_torque(self) -> float:
        """
        Returns signed net torque.
        Positive  → left arm is heavier (tilts left).
        Negative  → right arm is heavier (tilts right).
        """
        t_left  = self.left_weight  * self.left_arm_length
        t_right = self.right_weight * self.right_arm_length
        return (t_left - t_right) + self.fulcrum_bias

    def evaluate(self) -> LeverState:
        """Compute torque and return a LeverState snapshot."""
        net = self.compute_torque()
        if net > self.BALANCE_THRESHOLD:
            tilt = TiltDirection.LEFT
        elif net < -self.BALANCE_THRESHOLD:
            tilt = TiltDirection.RIGHT
        else:
            tilt = TiltDirection.BALANCED
        return LeverState(
            lever_id       = self.lever_id,
            net_torque     = net,
            tilt           = tilt,
            tilt_magnitude = abs(net),
        )

    # ── Serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "lever_id":         self.lever_id,
            "left_arm_length":  self.left_arm_length,
            "right_arm_length": self.right_arm_length,
            "fulcrum_bias":     self.fulcrum_bias,
            "left_weight":      self.left_weight,
            "right_weight":     self.right_weight,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Lever:
        return cls(**d)

    def __repr__(self) -> str:
        return (
            f"Lever(id={self.lever_id}, "
            f"L={self.left_weight}×{self.left_arm_length}, "
            f"R={self.right_weight}×{self.right_arm_length}, "
            f"bias={self.fulcrum_bias:+.3f})"
        )


# ---------------------------------------------------------------------------
# Three-Bar System
# ---------------------------------------------------------------------------

class ThreeBarSystem:
    """
    A cascade of three levers:
        Lever 0 → Lever 1 → Lever 2 (Balancer)

    Linkage Matrix (3×3):
        L[i][j] = how much of Lever i's tilt_magnitude is forwarded
                  onto Lever j.

        Positive value → adds weight to the LEFT side of lever j.
        Negative value → adds weight to the RIGHT side of lever j.
        Zero           → no coupling.

        Cascade runs left-to-right (0 → 1 → 2).
        L[2] (Balancer row) only affects an override, not further levers.

    Balancer Override Logic:
        If Lever 2's tilt_magnitude exceeds `balancer_threshold`, the
        Balancer is considered to have "damped" the system: the final
        tilt is forced to BALANCED and override_active is set True.
        This models the safety weight sliding out to absorb instability.
    """

    BALANCER_OVERRIDE_THRESHOLD: float = 5.0

    def __init__(
        self,
        levers:           Optional[list[Lever]]   = None,
        linkage_matrix:   Optional[list[list[float]]] = None,
        balancer_threshold: float = BALANCER_OVERRIDE_THRESHOLD,
    ):
        self.levers = levers if levers is not None else [
            Lever(lever_id=0),
            Lever(lever_id=1),
            Lever(lever_id=2),
        ]
        # Default: no coupling between levers
        self.linkage_matrix: list[list[float]] = linkage_matrix if linkage_matrix else [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
        self.balancer_threshold = balancer_threshold

        if len(self.levers) != 3:
            raise ValueError("ThreeBarSystem requires exactly 3 levers.")

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_defaults(cls) -> ThreeBarSystem:
        """
        A sensible starting configuration:
          - Lever 0: equal arms, no bias (pure input lever)
          - Lever 1: right arm slightly longer (mildly conservative bias)
          - Lever 2: Balancer — long arms for high sensitivity, slight left bias
                     so it fires early on instability
        Coupling: Lever 0 → Lever 1 at 50% strength
        """
        levers = [
            Lever(lever_id=0, left_arm_length=1.0, right_arm_length=1.0, fulcrum_bias=0.0),
            Lever(lever_id=1, left_arm_length=1.0, right_arm_length=1.2, fulcrum_bias=0.0),
            Lever(lever_id=2, left_arm_length=2.0, right_arm_length=2.0, fulcrum_bias=0.5),
        ]
        linkage_matrix = [
            [0.0, 0.5, 0.0],   # Lever 0 feeds 50% of its magnitude into Lever 1
            [0.0, 0.0, 0.3],   # Lever 1 feeds 30% of its magnitude into Lever 2
            [0.0, 0.0, 0.0],   # Lever 2 (Balancer) does not feed forward
        ]
        return cls(levers=levers, linkage_matrix=linkage_matrix, balancer_threshold=5.0)

    # ── Simulation ───────────────────────────────────────────────────────────

    def simulate(self) -> EquilibriumResult:
        """
        Run the full 3-bar cascade simulation.

        Steps:
            1. Evaluate Lever 0, propagate its tilt into downstream levers.
            2. Evaluate Lever 1, propagate its tilt into Lever 2.
            3. Evaluate Lever 2 (Balancer).
            4. Check Balancer threshold — apply override if breached.
            5. Compute confidence from final lever's tilt_magnitude.
        """
        states: list[LeverState]  = []
        # Work on temporary weight copies so simulate() is non-destructive
        working = copy.deepcopy(self.levers)

        for i, lever in enumerate(working):
            state = lever.evaluate()
            states.append(state)

            # Propagate into downstream levers via linkage matrix
            for j in range(i + 1, 3):
                coupling = self.linkage_matrix[i][j]
                if coupling == 0.0 or state.is_locked:
                    continue
                propagated_weight = state.tilt_magnitude * abs(coupling)
                # Sign of coupling determines which side gains weight
                if coupling > 0:
                    target_side = "left"  if state.tilt == TiltDirection.LEFT  else "right"
                else:
                    target_side = "right" if state.tilt == TiltDirection.LEFT  else "left"
                working[j].add_weight(target_side, propagated_weight)

        # ── Balancer override check ──────────────────────────────────────────
        balancer_state   = states[2]
        override_active  = balancer_state.tilt_magnitude > self.balancer_threshold

        if override_active:
            final_tilt = TiltDirection.BALANCED
        else:
            final_tilt = balancer_state.tilt

        # ── Confidence: how decisively did the Balancer tilt? ───────────────
        # Normalised with a soft sigmoid so extreme torques asymptote to 1.0
        confidence = self._sigmoid_confidence(balancer_state.tilt_magnitude)

        return EquilibriumResult(
            states          = states,
            final_tilt      = final_tilt,
            override_active = override_active,
            confidence      = confidence,
        )

    @staticmethod
    def _sigmoid_confidence(magnitude: float, scale: float = 3.0) -> float:
        """Map tilt magnitude to a 0–1 confidence score via sigmoid."""
        return 1.0 / (1.0 + math.exp(-magnitude / scale))

    # ── Snapshot (State Saving) ───────────────────────────────────────────────

    def snapshot(self) -> dict:
        """
        Serialise the current system state into a KineticSnapshot dict.

        Corresponds to the blueprint's:
            S = [ W⃗ (weights), F⃗ (fulcrum biases), L (linkage matrix) ]
        """
        return {
            "W": [
                {"left": lev.left_weight, "right": lev.right_weight}
                for lev in self.levers
            ],
            "F": [lev.fulcrum_bias for lev in self.levers],
            "L": copy.deepcopy(self.linkage_matrix),
            "arm_lengths": [
                {"left": lev.left_arm_length, "right": lev.right_arm_length}
                for lev in self.levers
            ],
            "balancer_threshold": self.balancer_threshold,
        }

    def hydrate(self, s: dict) -> None:
        """
        Restore system state from a KineticSnapshot dict.
        Instantly snaps levers to their historical optimal orientation.
        """
        for i, lever in enumerate(self.levers):
            lever.left_weight      = s["W"][i]["left"]
            lever.right_weight     = s["W"][i]["right"]
            lever.fulcrum_bias     = s["F"][i]
            lever.left_arm_length  = s["arm_lengths"][i]["left"]
            lever.right_arm_length = s["arm_lengths"][i]["right"]
        self.linkage_matrix    = copy.deepcopy(s["L"])
        self.balancer_threshold = s.get("balancer_threshold", self.BALANCER_OVERRIDE_THRESHOLD)

    def save_snapshot(self, path: str) -> None:
        """Write snapshot JSON to disk."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.snapshot(), f, indent=2)

    @classmethod
    def load_snapshot(cls, path: str) -> ThreeBarSystem:
        """Reconstruct a ThreeBarSystem from a saved JSON snapshot."""
        with open(path, encoding="utf-8") as f:
            s = json.load(f)
        system = cls()
        system.hydrate(s)
        return system

    def __repr__(self) -> str:
        return (
            "ThreeBarSystem(\n"
            + "\n".join(f"  {lev}" for lev in self.levers)
            + f"\n  linkage={self.linkage_matrix}\n)"
        )
