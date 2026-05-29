from __future__ import annotations

from sports_pro import DailyPlan, DailyTask
from prediction_engine import MatchPrediction
from prism_responses import CardType, plan_card, prediction_card, text_card


def test_text_card_type():
    assert text_card("hello").card_type == CardType.TEXT


def test_to_json_has_type():
    card = text_card("hello")
    assert "type" in card.to_json()


def test_plan_card_has_tasks():
    plan = DailyPlan(
        primary_focus="Recovery",
        activation=0.42,
        fulcrum=0.58,
        tasks=[DailyTask(time_slot="09:00", duration_min=45, category="recovery", title="Mobility")],
        warnings=["Watch soreness"],
        rationale="Light day",
    )
    card = plan_card(plan)
    assert isinstance(card.card_data["tasks"], list)
    assert card.card_data["tasks"][0]["title"] == "Mobility"


def test_prediction_card_probabilities():
    pred = MatchPrediction(
        subject="Arsenal vs City",
        prediction="Arsenal win",
        confidence=0.8,
        distribution={"home": 0.5, "draw": 0.3, "away": 0.2},
        expected_value=1.0,
        risk=0.2,
        risk_adj=0.8,
        fulcrum=0.5,
        key_factors=[("form", 0.7, "positive")],
        home_team="Arsenal",
        away_team="City",
        p_home_win=0.5,
        p_draw=0.3,
        p_away_win=0.2,
    )
    card = prediction_card(pred)
    total = card.card_data["p_home"] + card.card_data["p_draw"] + card.card_data["p_away"]
    assert total == 1.0
