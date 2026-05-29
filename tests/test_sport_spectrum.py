from __future__ import annotations

from decision_spectrum import DecisionBeam
from sport_spectrum import ALL_SPORTS, DuelModel, PossessionChain, SportDecisionModel


def test_all_sports_present():
    assert len(ALL_SPORTS) == 6


def test_duel_converges():
    model = SportDecisionModel(ALL_SPORTS["FOOTBALL"])
    duel = DuelModel(model)
    outcome = duel.simulate("Striker", "Goalkeeper")
    assert outcome.iterations < 20


def test_striker_beats_goalkeeper():
    model = SportDecisionModel(ALL_SPORTS["FOOTBALL"])
    duel = DuelModel(model)
    outcome = duel.simulate(
        "Striker",
        "Goalkeeper",
        {"pitch_zone": 0.9, "xg": 0.9},
        {"pitch_zone": 0.1, "press": 0.2},
    )
    assert outcome.advantage == "attacker"


def test_contested_midfield():
    model = SportDecisionModel(ALL_SPORTS["FOOTBALL"])
    duel = DuelModel(model)
    outcome = duel.simulate(
        "Def_mid",
        "Att_mid",
        {"pitch_zone": 0.6, "press": 0.0, "support": 0.9, "xg": 0.1, "fatigue": 0.2},
        {"pitch_zone": 0.4, "press": 1.0, "support": 0.9, "xg": 0.1, "fatigue": 0.2},
    )
    assert outcome.advantage == "contested"


def test_possession_chain_advances_zone():
    model = SportDecisionModel(ALL_SPORTS["FOOTBALL"])
    chain = PossessionChain(model)
    outcome = chain.simulate(
        [("A", "Striker"), ("B", "Winger"), ("C", "Att_mid")],
        initial_context={"pitch_zone": 0.3, "xg": 0.1},
    )
    assert outcome.links[-1].context_out["pitch_zone"] > 0.3


def test_make_beam_has_planks():
    model = SportDecisionModel(ALL_SPORTS["FOOTBALL"])
    beam = model.make_beam("Striker", {"pitch_zone": 0.9, "xg": 0.6})
    assert isinstance(beam, DecisionBeam)
    assert beam.planks
