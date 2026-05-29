from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CardType(str, Enum):
    TEXT = "text"
    PLAN = "plan"
    PREDICTION = "prediction"
    MOMENT = "moment"
    RISK = "risk"
    SQUAD = "squad"
    DOMAIN = "domain"
    IDENTITY = "identity"
    ARTIFACTS = "artifacts"
    ERROR = "error"
    THINKING = "thinking"


@dataclass
class PrismCard:
    card_type: CardType
    title: str
    body: str
    card_data: dict
    actions: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "type": self.card_type.value,
            "title": self.title,
            "body": self.body,
            "data": self.card_data,
            "actions": self.actions,
        }


def text_card(body: str, title: str = "") -> PrismCard:
    return PrismCard(CardType.TEXT, title, body, {})


def plan_card(plan) -> PrismCard:
    data = {
        "primary_focus": getattr(plan, "primary_focus", ""),
        "activation": round(float(getattr(plan, "activation", 0.0)), 3),
        "warnings": list(getattr(plan, "warnings", []) or []),
        "tasks": [
            {
                "time": task.time_slot,
                "category": task.category,
                "title": task.title,
                "duration": task.duration_min,
            }
            for task in (getattr(plan, "tasks", []) or [])
        ],
    }
    return PrismCard(CardType.PLAN, "Daily Plan", "", data)


def prediction_card(pred) -> PrismCard:
    data = {
        "home": getattr(pred, "home_team", ""),
        "away": getattr(pred, "away_team", ""),
        "p_home": round(float(getattr(pred, "p_home_win", 0.0)), 3),
        "p_draw": round(float(getattr(pred, "p_draw", 0.0)), 3),
        "p_away": round(float(getattr(pred, "p_away_win", 0.0)), 3),
        "predicted": getattr(pred, "prediction", ""),
        "confidence": round(float(getattr(pred, "confidence", 0.0)), 3),
        "key_factors": [
            (name, round(float(contribution), 3), direction)
            for name, contribution, direction in (getattr(pred, "key_factors", []) or [])[:3]
        ],
    }
    return PrismCard(CardType.PREDICTION, "Match Prediction", "", data)


def risk_card(risk) -> PrismCard:
    data = {
        "athlete": getattr(risk, "athlete_name", ""),
        "risk_level": getattr(risk, "risk_level", ""),
        "prediction": getattr(risk, "prediction", ""),
        "confidence": round(float(getattr(risk, "confidence", 0.0)), 3),
        "recommendations": list(getattr(risk, "recommendations", []) or []),
    }
    return PrismCard(CardType.RISK, "Injury Risk", "", data)


def squad_card(risks: list) -> PrismCard:
    data = {
        "players": [
            {
                "name": getattr(risk, "athlete_name", risk.get("athlete_name") if isinstance(risk, dict) else ""),
                "risk_level": getattr(risk, "risk_level", risk.get("risk_level") if isinstance(risk, dict) else ""),
                "confidence": round(
                    float(getattr(risk, "confidence", risk.get("confidence") if isinstance(risk, dict) else 0.0)),
                    3,
                ),
            }
            for risk in (risks or [])
        ]
    }
    return PrismCard(CardType.SQUAD, "Squad Risk Overview", "", data)


def domain_card(domain_name: str, diag) -> PrismCard:
    data = {
        "domain": domain_name,
        "recommended": diag.primary_plank.name,
        "confidence": round(diag.activations[0].activation, 3),
        "fulcrum": round(diag.fulcrum_position, 3),
        "options": [
            {
                "name": activation.plank.name,
                "activation": round(activation.activation, 3),
                "position": activation.plank.position,
            }
            for activation in diag.activations
        ],
    }
    return PrismCard(CardType.DOMAIN, f"{domain_name} Recommendation", "", data)


def moment_card(result) -> PrismCard:
    data = {
        "sport": result.moment.sport,
        "moment_type": result.moment.moment_type,
        "recommended": result.recommended,
        "activation": round(result.activations[0][1] if result.activations else 0, 3),
        "xg": result.xg_contextual,
        "time_pressure": round(result.time_pressure, 3),
        "options": [
            {"name": name, "activation": round(activation, 3), "ev": round(ev, 1)}
            for name, activation, ev in result.activations
        ],
    }
    return PrismCard(CardType.MOMENT, "Moment Analysis", "", data)


def identity_card(identity_data: dict) -> PrismCard:
    return PrismCard(CardType.IDENTITY, "Your Decision Profile", "", identity_data)
