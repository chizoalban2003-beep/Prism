from __future__ import annotations

from duel_analyzer import DuelAnalyzer


def _duel_event(
    *,
    player: str = "Attacker",
    defender: str = "Defender",
    x: float = 105.0,
    y: float = 18.0,
    outcome: str = "Won",
) -> dict:
    return {
        "type": {"name": "Duel"},
        "player": {"name": player},
        "team": {"name": "Home"},
        "location": [x, y],
        "timestamp": "00:00:05.000",
        "duel": {
            "type": {"name": "ground"},
            "outcome": {"name": outcome},
            "counterpart": {
                "name": defender,
                "team": {"name": "Away"},
            },
        },
    }


def test_enrich_sets_zone_label():
    analyzer = DuelAnalyzer()
    records = analyzer.process_match([_duel_event(x=100.0, y=12.0)], "match-1")

    assert records[0].zone_label == "box"


def test_enrich_sets_prediction():
    analyzer = DuelAnalyzer()
    records = analyzer.process_match([_duel_event()], "match-1")

    assert records[0].predicted_winner in ("attacker", "defender", "contested")


def test_model_correct_is_bool():
    analyzer = DuelAnalyzer()
    records = analyzer.process_match([_duel_event(), _duel_event(player="A2", defender="D2")], "match-1")

    assert isinstance(records[0].model_correct, bool)
    assert isinstance(records[1].model_correct, bool)


def test_expected_outcome_with_profile():
    analyzer = DuelAnalyzer()

    probability = analyzer.expected_outcome(
        "Attacker",
        "Defender",
        location_x=100.0,
        attacker_profile="Striker",
        defender_profile="Centre back",
    )

    assert 0.0 <= probability <= 1.0


def test_backward_compat_win_rate():
    analyzer = DuelAnalyzer()
    analyzer.process_match([_duel_event(outcome="Won")], "match-1")

    assert analyzer.network.win_rate("Attacker", "Defender") == 1.0
