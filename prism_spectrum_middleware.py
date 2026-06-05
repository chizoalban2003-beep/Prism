"""
PRISM Spectrum Middleware
Loads the VEAX configuration vector from prism_config.toml and exposes
four decision gates that the chain uses at runtime.

V — Verification:  score threshold a logic result must meet before being
                   accepted into accumulated context (0.0 = accept all,
                   1.0 = require high evaluator score)
E — Evolution:     plasticity of memory writes — 0.0 = only write if the
                   node is new, 1.0 = always overwrite existing nodes
A — Autonomy:      below 0.3 the chain asks user confirmation before any
                   irreversible organ; above 0.7 executes without gating
X — Explanation:   LogicPolicy trace verbosity injected into accumulated
                   context (0.0 = suppress, 1.0 = full structured trace)

The four values are floats in [0.0, 1.0].  If the [spectrum] section is
absent from config, safe defaults (0.5) are used so existing behaviour
is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from decision_spectrum import (
    AdaptiveFulcrum,
    DecisionBeam,
    DecisionNetwork,
    DecisionPlank,
    Factor,
)

_CONFIG_PATH = Path(__file__).parent / "prism_config.toml"
_DEFAULTS    = {"V": 0.5, "E": 0.5, "A": 0.5, "X": 0.5}


def _load_spectrum_config(config: dict | None = None) -> dict[str, float]:
    """
    Read [spectrum] section from config dict or prism_config.toml.
    Returns dict with keys V, E, A, X as floats in [0.0, 1.0].
    """
    if config is not None:
        raw = config.get("spectrum", {})
    else:
        try:
            import tomllib
            raw = tomllib.loads(_CONFIG_PATH.read_text()).get("spectrum", {})
        except Exception:
            raw = {}
    result = dict(_DEFAULTS)
    for k in ("V", "E", "A", "X"):
        if k in raw:
            result[k] = float(max(0.0, min(1.0, raw[k])))
    return result


@dataclass
class SpectrumGates:
    """
    Computed gate thresholds derived from the VEAX vector.
    All methods are pure (no side effects) so they are safe to call
    from multiple threads.
    """
    V: float  # Verification  0..1
    E: float  # Evolution     0..1
    A: float  # Autonomy      0..1
    X: float  # Explanation   0..1

    # ── V gate ────────────────────────────────────────────────────────────────

    def verification_threshold(self) -> int:
        """
        Minimum evaluator score (1-5) a logic result must achieve before
        being accepted into chain accumulated context.

        V=0.0 → threshold=1 (accept everything)
        V=1.0 → threshold=5 (only perfect results)
        """
        return max(1, round(1 + self.V * 4))

    def accepts_result(self, eval_score: int) -> bool:
        return eval_score >= self.verification_threshold()

    # ── E gate ────────────────────────────────────────────────────────────────

    def should_overwrite_node(self, node_exists: bool) -> bool:
        """
        E=0.0 → never overwrite existing nodes (anchor bias)
        E=1.0 → always overwrite (high plasticity)
        E=0.5 → overwrite only if node does not yet exist (safe default)
        """
        if not node_exists:
            return True
        return self.E > 0.5

    # ── A gate ────────────────────────────────────────────────────────────────

    def requires_approval(self, irreversible: bool) -> bool:
        """
        Returns True if the chain should pause and ask the user before
        running an irreversible organ.

        A < 0.3 → always ask
        A > 0.7 → never ask (full autonomy)
        A 0.3–0.7 → ask only when organ is flagged irreversible
        """
        if self.A > 0.7:
            return False
        if self.A < 0.3:
            return True
        return irreversible

    # ── X gate ────────────────────────────────────────────────────────────────

    def logicpolicy_verbosity(self) -> str:
        """
        Returns one of: 'none' | 'summary' | 'full'

        X < 0.3 → 'none'   (execution mode — suppress trace)
        X < 0.7 → 'summary' (one-line risk/caps/L1)
        X ≥ 0.7 → 'full'   (full structured JSON trace)
        """
        if self.X < 0.3:
            return "none"
        if self.X < 0.7:
            return "summary"
        return "full"

    def format_logicpolicy(self, lp_summary: str, lp_meta: dict) -> str:
        """Format the LogicPolicy trace according to X verbosity level."""
        level = self.logicpolicy_verbosity()
        if level == "none" or not lp_summary:
            return ""
        if level == "summary":
            return lp_summary
        # full: structured detail
        lines = [lp_summary]
        if lp_meta:
            caps = lp_meta.get("capabilities", [])
            if caps:
                lines.append(f"  caps={caps}")
            irrev = lp_meta.get("irreversible", False)
            if irrev:
                lines.append("  irreversible=true")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, float]:
        return {"V": self.V, "E": self.E, "A": self.A, "X": self.X}


# ── DecisionNetwork builder ───────────────────────────────────────────────────

def build_spectrum_network(gates: SpectrumGates) -> DecisionNetwork:
    """
    Construct a four-beam DecisionNetwork representing the VEAX state.
    Each beam has two planks (low/high) plus the fulcrum anchored at
    the configured value.  Beams are connected so high autonomy reduces
    verification strictness (real-world trade-off).
    """
    net = DecisionNetwork()

    def _beam(name: str, val: float, desc_low: str, desc_high: str) -> DecisionBeam:
        fulcrum = AdaptiveFulcrum(
            factors=[Factor(name, val, weight=1.0, target=val, description=desc_low)]
        )
        beam = DecisionBeam(name=name, bandwidth=0.3, fulcrum=fulcrum)
        beam.add_plank(DecisionPlank(
            name=f"{name}_low",  position=0.0,
            payoff=1.0 - val,    cost=val,       risk=0.2,
            metadata={"description": desc_low},
        ))
        beam.add_plank(DecisionPlank(
            name=f"{name}_high", position=1.0,
            payoff=val,          cost=1.0 - val, risk=0.2,
            metadata={"description": desc_high},
        ))
        return beam

    v_beam = _beam("V", gates.V, "optimistic ingestion",  "strict verification")
    e_beam = _beam("E", gates.E, "anchor bias",           "high plasticity")
    a_beam = _beam("A", gates.A, "human oversight",       "autonomous execution")
    x_beam = _beam("X", gates.X, "execution mode",        "deep audit mode")

    for b in (v_beam, e_beam, a_beam, x_beam):
        net.add_beam(b)

    # High autonomy reduces verification burden (users who trust the agent
    # don't need every result scored before acceptance)
    net.add_dependency("A", "V", strength=-0.15, factor_name="_autonomy_relaxes_verification")

    return net


# ── Public factory ────────────────────────────────────────────────────────────

def load_spectrum(config: dict | None = None) -> tuple[SpectrumGates, DecisionNetwork]:
    """
    Load spectrum config and return (SpectrumGates, DecisionNetwork).
    Safe to call with config=None — falls back to prism_config.toml,
    then to 0.5 defaults.
    """
    vals  = _load_spectrum_config(config)
    gates = SpectrumGates(**vals)
    net   = build_spectrum_network(gates)
    return gates, net


def observe_outcome(
    network: DecisionNetwork,
    logic:   str,
    actual_payoff:    float,
    predicted_payoff: float,
    chosen_position:  float,
) -> None:
    """
    Feed a real outcome back into the network's AdaptiveFulcrums.
    Called after each chain step to let the network self-calibrate.
    """
    for beam in network._beams.values():
        if isinstance(beam.fulcrum, AdaptiveFulcrum):
            beam.fulcrum.observe(actual_payoff, predicted_payoff, chosen_position)


def spectrum_summary(gates: SpectrumGates, network: DecisionNetwork) -> dict[str, Any]:
    """Return a human-readable summary of current spectrum state."""
    try:
        diags = network.solve()
        beam_positions = {n: round(d.fulcrum_position, 3) for n, d in diags.items()}
        primary_planks = {n: d.primary_plank.name for n, d in diags.items()}
    except Exception:
        beam_positions = {}
        primary_planks = {}
    return {
        "veax":            gates.to_dict(),
        "verification_threshold": gates.verification_threshold(),
        "logicpolicy_verbosity":  gates.logicpolicy_verbosity(),
        "requires_approval_at_low_autonomy": gates.requires_approval(False),
        "beam_positions":  beam_positions,
        "primary_planks":  primary_planks,
    }
