from __future__ import annotations

from decision_spectrum import DecisionBeam
from sport_spectrum import ALL_SPORTS, DuelModel, PossessionChain, SportDecisionModel


def test_all_sports_present():
    assert len(ALL_SPORTS) == 6


def test_duel_model_converges():
    model = SportDecisionModel(ALL_SPORTS["Football"])
    duel = DuelModel(model)
    outcome = duel.simulate("Striker", "Goalkeeper")
    assert outcome.iterations < 20


def test_duel_advantage_attacker_high_base():
    model = SportDecisionModel(ALL_SPORTS["Football"])
    duel = DuelModel(model)
    outcome = duel.simulate("Striker", "Goalkeeper")
    assert outcome.advantage == "attacker"


def test_possession_chain_advances_zone():
    model = SportDecisionModel(ALL_SPORTS["Football"])
    chain = PossessionChain(model)
    outcome = chain.simulate(
        [("A", "Striker"), ("B", "Winger"), ("C", "Att. mid")],
        initial_context={"pitch_zone": 0.3, "xg": 0.1},
    )
    assert outcome.links[-1].context_out["pitch_zone"] > 0.3


def test_sport_decision_model_make_beam():
    model = SportDecisionModel(ALL_SPORTS["Football"])
    beam = model.make_beam("Striker", {"pitch_zone": 0.9, "xg": 0.6})
    assert isinstance(beam, DecisionBeam)
    assert beam.planks
