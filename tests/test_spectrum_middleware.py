"""Tests for prism_spectrum_middleware — SpectrumGates, DecisionNetwork, chain wiring."""
import pytest

from decision_spectrum import AdaptiveFulcrum, DecisionNetwork
from prism_spectrum_middleware import (
    SpectrumGates,
    _load_spectrum_config,
    build_spectrum_network,
    load_spectrum,
    observe_outcome,
    spectrum_summary,
)

# ── Config loading ────────────────────────────────────────────────────────────

class TestLoadSpectrumConfig:
    def test_defaults_when_no_config(self):
        cfg = _load_spectrum_config(config={})
        assert cfg == {"V": 0.5, "E": 0.5, "A": 0.5, "X": 0.5}

    def test_reads_spectrum_section(self):
        cfg = _load_spectrum_config(config={"spectrum": {"V": 0.8, "A": 0.2}})
        assert cfg["V"] == pytest.approx(0.8)
        assert cfg["A"] == pytest.approx(0.2)
        assert cfg["E"] == pytest.approx(0.5)  # default
        assert cfg["X"] == pytest.approx(0.5)  # default

    def test_clamps_values_above_1(self):
        cfg = _load_spectrum_config(config={"spectrum": {"V": 1.5}})
        assert cfg["V"] == pytest.approx(1.0)

    def test_clamps_values_below_0(self):
        cfg = _load_spectrum_config(config={"spectrum": {"V": -0.3}})
        assert cfg["V"] == pytest.approx(0.0)

    def test_all_four_axes_loaded(self):
        cfg = _load_spectrum_config(
            config={"spectrum": {"V": 0.1, "E": 0.2, "A": 0.3, "X": 0.4}}
        )
        assert cfg == {"V": 0.1, "E": 0.2, "A": 0.3, "X": 0.4}


# ── SpectrumGates: V axis ─────────────────────────────────────────────────────

class TestVerificationGate:
    def test_v0_threshold_is_1(self):
        g = SpectrumGates(V=0.0, E=0.5, A=0.5, X=0.5)
        assert g.verification_threshold() == 1

    def test_v1_threshold_is_5(self):
        g = SpectrumGates(V=1.0, E=0.5, A=0.5, X=0.5)
        assert g.verification_threshold() == 5

    def test_v0_accepts_any_score(self):
        g = SpectrumGates(V=0.0, E=0.5, A=0.5, X=0.5)
        assert g.accepts_result(1)
        assert g.accepts_result(5)

    def test_v1_accepts_only_5(self):
        g = SpectrumGates(V=1.0, E=0.5, A=0.5, X=0.5)
        assert not g.accepts_result(4)
        assert g.accepts_result(5)

    def test_v_mid_threshold_is_3(self):
        g = SpectrumGates(V=0.5, E=0.5, A=0.5, X=0.5)
        assert g.verification_threshold() == 3
        assert not g.accepts_result(2)
        assert g.accepts_result(3)


# ── SpectrumGates: E axis ─────────────────────────────────────────────────────

class TestEvolutionGate:
    def test_always_writes_new_nodes(self):
        for e in (0.0, 0.5, 1.0):
            g = SpectrumGates(V=0.5, E=e, A=0.5, X=0.5)
            assert g.should_overwrite_node(node_exists=False)

    def test_low_e_does_not_overwrite_existing(self):
        g = SpectrumGates(V=0.5, E=0.0, A=0.5, X=0.5)
        assert not g.should_overwrite_node(node_exists=True)

    def test_high_e_overwrites_existing(self):
        g = SpectrumGates(V=0.5, E=1.0, A=0.5, X=0.5)
        assert g.should_overwrite_node(node_exists=True)

    def test_mid_e_does_not_overwrite_existing(self):
        g = SpectrumGates(V=0.5, E=0.5, A=0.5, X=0.5)
        assert not g.should_overwrite_node(node_exists=True)


# ── SpectrumGates: A axis ─────────────────────────────────────────────────────

class TestAutonomyGate:
    def test_high_autonomy_never_requires_approval(self):
        g = SpectrumGates(V=0.5, E=0.5, A=0.9, X=0.5)
        assert not g.requires_approval(irreversible=True)
        assert not g.requires_approval(irreversible=False)

    def test_low_autonomy_always_requires_approval(self):
        g = SpectrumGates(V=0.5, E=0.5, A=0.1, X=0.5)
        assert g.requires_approval(irreversible=False)
        assert g.requires_approval(irreversible=True)

    def test_mid_autonomy_only_gates_irreversible(self):
        g = SpectrumGates(V=0.5, E=0.5, A=0.5, X=0.5)
        assert not g.requires_approval(irreversible=False)
        assert g.requires_approval(irreversible=True)


# ── SpectrumGates: X axis ─────────────────────────────────────────────────────

class TestExplanationGate:
    def test_low_x_gives_none_verbosity(self):
        g = SpectrumGates(V=0.5, E=0.5, A=0.5, X=0.1)
        assert g.logicpolicy_verbosity() == "none"

    def test_mid_x_gives_summary_verbosity(self):
        g = SpectrumGates(V=0.5, E=0.5, A=0.5, X=0.5)
        assert g.logicpolicy_verbosity() == "summary"

    def test_high_x_gives_full_verbosity(self):
        g = SpectrumGates(V=0.5, E=0.5, A=0.5, X=0.9)
        assert g.logicpolicy_verbosity() == "full"

    def test_format_none_returns_empty(self):
        g = SpectrumGates(V=0.5, E=0.5, A=0.5, X=0.1)
        assert g.format_logicpolicy("risk=low L1=allowed", {"risk_level": "low"}) == ""

    def test_format_summary_returns_summary_line(self):
        g = SpectrumGates(V=0.5, E=0.5, A=0.5, X=0.5)
        result = g.format_logicpolicy("risk=low L1=allowed", {})
        assert result == "risk=low L1=allowed"

    def test_format_full_includes_caps(self):
        g = SpectrumGates(V=0.5, E=0.5, A=0.5, X=0.9)
        result = g.format_logicpolicy("risk=low", {"capabilities": ["internet_read"]})
        assert "internet_read" in result

    def test_format_full_includes_irreversible_flag(self):
        g = SpectrumGates(V=0.5, E=0.5, A=0.5, X=0.9)
        result = g.format_logicpolicy("risk=high", {"irreversible": True})
        assert "irreversible=true" in result

    def test_format_empty_summary_returns_empty(self):
        g = SpectrumGates(V=0.5, E=0.5, A=0.5, X=0.9)
        assert g.format_logicpolicy("", {}) == ""


# ── DecisionNetwork construction ──────────────────────────────────────────────

class TestBuildSpectrumNetwork:
    def test_returns_decision_network(self):
        g = SpectrumGates(V=0.5, E=0.5, A=0.5, X=0.5)
        net = build_spectrum_network(g)
        assert isinstance(net, DecisionNetwork)

    def test_has_four_beams(self):
        g = SpectrumGates(V=0.5, E=0.5, A=0.5, X=0.5)
        net = build_spectrum_network(g)
        assert set(net._beams.keys()) == {"V", "E", "A", "X"}

    def test_each_beam_has_two_planks(self):
        g = SpectrumGates(V=0.3, E=0.7, A=0.1, X=0.9)
        net = build_spectrum_network(g)
        for beam in net._beams.values():
            assert len(beam.planks) == 2

    def test_network_solve_returns_four_diagnoses(self):
        g = SpectrumGates(V=0.5, E=0.5, A=0.5, X=0.5)
        net = build_spectrum_network(g)
        diags = net.solve()
        assert set(diags.keys()) == {"V", "E", "A", "X"}

    def test_autonomy_dependency_added(self):
        g = SpectrumGates(V=0.5, E=0.5, A=0.9, X=0.5)
        net = build_spectrum_network(g)
        # High autonomy should propagate a negative influence onto V beam
        assert any(src == "A" and tgt == "V" for src, tgt, *_ in net._deps)

    def test_adaptive_fulcrums_used(self):
        g = SpectrumGates(V=0.5, E=0.5, A=0.5, X=0.5)
        net = build_spectrum_network(g)
        for beam in net._beams.values():
            assert isinstance(beam.fulcrum, AdaptiveFulcrum)


# ── load_spectrum factory ─────────────────────────────────────────────────────

class TestLoadSpectrum:
    def test_returns_gates_and_network(self):
        gates, net = load_spectrum(config={})
        assert isinstance(gates, SpectrumGates)
        assert isinstance(net, DecisionNetwork)

    def test_gates_reflect_config(self):
        gates, _ = load_spectrum(config={"spectrum": {"V": 0.9, "A": 0.1}})
        assert gates.V == pytest.approx(0.9)
        assert gates.A == pytest.approx(0.1)

    def test_to_dict_returns_all_axes(self):
        gates, _ = load_spectrum(config={})
        d = gates.to_dict()
        assert set(d.keys()) == {"V", "E", "A", "X"}


# ── observe_outcome ───────────────────────────────────────────────────────────

class TestObserveOutcome:
    def test_observe_does_not_raise(self):
        _, net = load_spectrum(config={})
        observe_outcome(net, "web_search", 1.0, 0.8, 0.6)

    def test_observe_updates_fulcrum_weights(self):
        _, net = load_spectrum(config={})
        beam = net._beams["V"]
        assert isinstance(beam.fulcrum, AdaptiveFulcrum)
        before = {f.name: f.weight for f in beam.fulcrum.factors}
        observe_outcome(net, "web_search", 1.0, 0.0, 0.5)
        after = {f.name: f.weight for f in beam.fulcrum.factors}
        # At least one weight should have changed
        changed = any(after[k] != before[k] for k in before)
        assert changed


# ── spectrum_summary ──────────────────────────────────────────────────────────

class TestSpectrumSummary:
    def test_summary_structure(self):
        gates, net = load_spectrum(config={})
        s = spectrum_summary(gates, net)
        assert "veax" in s
        assert "verification_threshold" in s
        assert "logicpolicy_verbosity" in s
        assert "beam_positions" in s

    def test_summary_veax_matches_gates(self):
        gates, net = load_spectrum(config={"spectrum": {"V": 0.2, "X": 0.8}})
        s = spectrum_summary(gates, net)
        assert s["veax"]["V"] == pytest.approx(0.2)
        assert s["veax"]["X"] == pytest.approx(0.8)


# ── PrismChain integration ────────────────────────────────────────────────────

class TestChainSpectrumIntegration:
    def test_chain_loads_spectrum_gates_on_init(self):
        from prism_chain import PrismChain
        chain = PrismChain(config={"spectrum": {"V": 0.9, "A": 0.1}})
        assert chain._spectrum_gates.V == pytest.approx(0.9)
        assert chain._spectrum_gates.A == pytest.approx(0.1)

    def test_chain_defaults_when_no_config(self):
        from prism_chain import PrismChain
        chain = PrismChain()
        assert 0.0 <= chain._spectrum_gates.V <= 1.0

    def test_chain_has_spectrum_network(self):
        from prism_chain import PrismChain
        chain = PrismChain()
        assert isinstance(chain._spectrum_network, DecisionNetwork)

    def test_v_gate_accepts_above_threshold(self):
        from prism_chain import PrismChain
        chain = PrismChain(config={"spectrum": {"V": 0.0}})  # threshold=1
        assert chain._spectrum_gates.accepts_result(1)

    def test_a_gate_high_autonomy_no_approval(self):
        from prism_chain import PrismChain
        chain = PrismChain(config={"spectrum": {"A": 0.9}})
        assert not chain._spectrum_gates.requires_approval(True)

    def test_x_gate_silent_mode(self):
        from prism_chain import PrismChain
        chain = PrismChain(config={"spectrum": {"X": 0.0}})
        formatted = chain._spectrum_gates.format_logicpolicy("risk=low caps=[a]", {})
        assert formatted == ""

    def test_x_gate_full_mode_includes_trace(self):
        from prism_chain import PrismChain
        chain = PrismChain(config={"spectrum": {"X": 1.0}})
        meta = {"capabilities": ["internet_read"], "irreversible": False}
        formatted = chain._spectrum_gates.format_logicpolicy("risk=low", meta)
        assert "internet_read" in formatted
