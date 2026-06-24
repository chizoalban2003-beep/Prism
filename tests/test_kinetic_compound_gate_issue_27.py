"""Compound-signal gate for the kinetic engine — issue #27 bug 5.

Live-test surfaced that a single strong signal fired all three pre-built
levers (intervene_now, defer_decision, proactive_assist) at the same time,
producing three near-identical warnings in chat. They were meant to fire on
*distinct* multi-domain patterns, but the engine treated every signal as
equally relevant to every lever.

The fix adds two pieces:

  1. `DecisionLever.gate_domains` — a lever only accepts torque from these
     signal domains. Empty tuple keeps legacy (ungated) behaviour.
  2. A compound-readiness check — a gated lever can only fire once every
     gate domain has contributed within `gate_window_sec`. Crisis (Z≥8)
     bypasses the gate so safety-critical alerts still cut through.

These tests pin both pieces so the bug can't silently regress.
"""
from __future__ import annotations

from prism_kinetic_engine import (
    DecisionLever,
    KineticEngine,
    PersonalSignal,
)


def _sig(domain: str, raw: float = 5.0, mu: float = 0.0, sigma: float = 1.0,
         signal_type: str = "test_signal") -> PersonalSignal:
    return PersonalSignal(
        domain=domain, signal_type=signal_type,
        raw_value=raw, mu=mu, sigma=sigma,
        impact=1.0, confidence=1.0,
    )


class TestGateDomains:
    def test_for_prism_levers_have_distinct_gate_domains(self):
        eng = KineticEngine.for_prism()
        gates = {lid: set(lever.gate_domains)
                 for lid, lever in eng._levers.items()}
        assert gates["intervene_now"]   == {"health", "temporal", "cognitive"}
        assert gates["defer_decision"]  == {"energy", "cognitive"}
        assert gates["proactive_assist"] == {"health", "temporal"}

    def test_lever_default_gate_is_empty(self):
        lv = DecisionLever("plain", "Plain", "no gate")
        assert lv.gate_domains == ()


class TestSingleSignalDoesNotFireAllLevers:
    """The headline bug: one signal must not fire every lever at once."""

    def test_single_health_signal_does_not_fire_defer_decision(self):
        eng = KineticEngine.for_prism()
        # Sub-crisis signal — strong enough to spike but below Z=8 black-swan.
        windows = eng.ingest(_sig("health", raw=4.0))
        fired = {w.lever_id for w in windows}
        assert "defer_decision" not in fired, (
            "defer_decision is gated to (energy, cognitive) — a pure health "
            "signal must not trigger it"
        )

    def test_single_energy_signal_does_not_fire_intervene_now(self):
        eng = KineticEngine.for_prism()
        windows = eng.ingest(_sig("energy", raw=4.0))
        fired = {w.lever_id for w in windows}
        assert "intervene_now" not in fired


class TestCompoundReadiness:
    """A gated lever must wait until every gate domain has contributed."""

    def test_defer_decision_does_not_fire_on_cognitive_alone(self):
        eng = KineticEngine.for_prism()
        # Drive cognitive past the activate threshold, but never feed energy.
        for _ in range(10):
            eng.ingest(_sig("cognitive", raw=5.0))
        active = {w.lever_id for w in eng.active_windows(max_age_seconds=3600)}
        assert "defer_decision" not in active, (
            "defer_decision requires both energy AND cognitive — cognitive "
            "alone must not fire it"
        )

    def test_defer_decision_fires_once_both_gates_present(self):
        eng = KineticEngine.for_prism()
        # Both gate domains hit the lever within the gate window.
        for _ in range(8):
            eng.ingest(_sig("energy", raw=5.0))
            eng.ingest(_sig("cognitive", raw=5.0))
        active = {w.lever_id for w in eng.active_windows(max_age_seconds=3600)}
        assert "defer_decision" in active


class TestCrisisBypass:
    """Z≥8 must still alert — the gate cannot block a true crisis."""

    def test_crisis_signal_bypasses_compound_gate(self):
        eng = KineticEngine.for_prism()
        # Z = (90 - 0) / 10 = 9 → crisis. Only health has fired (no temporal
        # or cognitive), but intervene_now should still fire on the bypass.
        sig = PersonalSignal(
            "health", "hrv_drop",
            raw_value=90, mu=0, sigma=10,
            impact=1.0, confidence=1.0,
        )
        windows = eng.ingest(sig)
        fired = {w.lever_id for w in windows}
        assert "intervene_now" in fired
        assert any(w.is_crisis for w in windows)

    def test_crisis_signal_does_not_fire_off_domain_lever(self):
        """Even in crisis, defer_decision (gated to energy/cognitive)
        must not fire on a health crisis — that's the whole point of
        gating, the lever's semantic intent doesn't apply."""
        eng = KineticEngine.for_prism()
        sig = PersonalSignal(
            "health", "hrv_drop",
            raw_value=90, mu=0, sigma=10,
        )
        windows = eng.ingest(sig)
        fired = {w.lever_id for w in windows}
        assert "defer_decision" not in fired
