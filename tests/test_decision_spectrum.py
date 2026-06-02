from __future__ import annotations

import pytest

from decision_spectrum import (
    AdaptiveFulcrum,
    DecisionBeam,
    DecisionNetwork,
    DecisionPlank,
    Factor,
    SpectrumFulcrum,
)


def test_fulcrum_weighted_centroid():
    factors = [
        Factor("a", 0.8, 2.0, 0.2),
        Factor("b", 0.5, 3.0, 0.9),
    ]
    fulcrum = SpectrumFulcrum(factors)
    expected = (
        (2.0 * 0.8 * 0.2) + (3.0 * 0.5 * 0.9)
    ) / ((2.0 * 0.8) + (3.0 * 0.5))
    assert fulcrum.position() == pytest.approx(expected)


def test_fulcrum_no_factors_returns_half():
    assert SpectrumFulcrum().position() == pytest.approx(0.5)


def test_activations_sum_to_one():
    beam = DecisionBeam("beam", bandwidth=0.2)
    beam.add_plank(DecisionPlank("safe", 0.0, 10, 1, 2))
    beam.add_plank(DecisionPlank("mid", 0.5, 20, 2, 3))
    beam.add_plank(DecisionPlank("high", 1.0, 30, 3, 4))
    acts = beam.evaluate().activations
    assert sum(a.activation for a in acts) == pytest.approx(1.0, abs=1e-9)


def test_primary_plank_is_maximum():
    beam = DecisionBeam("beam", bandwidth=0.2)
    beam.add_plank(DecisionPlank("left", 0.0, 10, 1, 2))
    beam.add_plank(DecisionPlank("center", 0.5, 20, 2, 3))
    beam.add_plank(DecisionPlank("right", 1.0, 30, 3, 4))
    diag = beam.evaluate()
    assert diag.activations[0].activation == max(a.activation for a in diag.activations)


def test_expected_net_formula():
    beam = DecisionBeam("single", bandwidth=0.2)
    beam.add_plank(DecisionPlank("only", 0.5, 100, 10, 5, probability=0.8))
    diag = beam.evaluate()
    assert diag.expected_net == pytest.approx((100 * 0.8) - 10)


def test_adaptive_observe_changes_weight():
    fulcrum = AdaptiveFulcrum(
        [
            Factor("a", 0.8, 1.0, 0.7),
            Factor("b", 0.4, 1.5, 0.2),
        ]
    )
    before = {f.name: f.weight for f in fulcrum.factors}
    fulcrum.observe(actual_payoff=120.0, predicted_payoff=80.0, chosen_position=0.65)
    after = {f.name: f.weight for f in fulcrum.factors}
    assert any(after[name] != before[name] for name in before)


def test_beam_no_planks_raises():
    with pytest.raises(ValueError):
        DecisionBeam("empty").evaluate()


def test_decision_network_converges():
    beam_a = DecisionBeam("a", fulcrum=SpectrumFulcrum([Factor("fa", 0.6, 1.0, 0.7)]))
    beam_b = DecisionBeam("b", fulcrum=SpectrumFulcrum([Factor("fb", 0.4, 1.0, 0.3)]))
    for beam in (beam_a, beam_b):
        beam.add_plank(DecisionPlank("safe", 0.0, 10, 1, 2))
        beam.add_plank(DecisionPlank("attack", 1.0, 30, 5, 8))
    network = DecisionNetwork(max_iterations=10, tolerance=1e-6)
    network.add_beam(beam_a)
    network.add_beam(beam_b)
    network.add_dependency("a", "b", 0.5)
    network.add_dependency("b", "a", 0.5)
    result = network.solve()
    assert set(result) == {"a", "b"}
