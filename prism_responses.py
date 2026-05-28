from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CardType(str, Enum):
    TEXT = "text"
    PLAN = "plan"
    PREDICTION = "prediction"
    MOMENT = "moment"
    RISK = "risk"
    DOMAIN = "domain"
    IDENTITY = "identity"
    SQUAD = "squad"
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def text_card(body: str, title: str = "") -> PrismCard:
    return PrismCard(CardType.TEXT, title, body, {})


def plan_card(plan) -> PrismCard:
    tasks = []
    for task in getattr(plan, "tasks", []) or []:
        tasks.append(
            {
                "time": _safe_text(getattr(task, "time_slot", getattr(task, "time", ""))),
                "category": _safe_text(getattr(task, "category", "")),
                "title": _safe_text(getattr(task, "title", "")),
                "duration": int(getattr(task, "duration_min", getattr(task, "duration", 0)) or 0),
            }
        )
    focus = _safe_text(getattr(plan, "primary_focus", "Today"))
    activation = _safe_float(getattr(plan, "activation", 0.0))
    warnings = list(getattr(plan, "warnings", []) or [])
    rationale = _safe_text(getattr(plan, "rationale", f"Primary focus: {focus}"))
    return PrismCard(
        CardType.PLAN,
        "Daily plan",
        rationale,
        {
            "primary_focus": focus,
            "activation": activation,
            "warnings": warnings,
            "tasks": tasks,
        },
        actions=["Show recovery guidance", "Adjust my workload"],
    )


def prediction_card(pred) -> PrismCard:
    p_home = _safe_float(getattr(pred, "p_home_win", None), _safe_float(getattr(pred, "p_home", 0.0)))
    p_draw = _safe_float(getattr(pred, "p_draw", 0.0))
    p_away = _safe_float(getattr(pred, "p_away_win", None), _safe_float(getattr(pred, "p_away", 0.0)))
    total = p_home + p_draw + p_away
    if total > 0:
        p_home, p_draw, p_away = p_home / total, p_draw / total, p_away / total
    body = _safe_text(getattr(pred, "prediction", "Prediction ready"))
    factors = []
    for item in getattr(pred, "key_factors", []) or []:
        if isinstance(item, (tuple, list)) and len(item) >= 3:
            factors.append({"name": _safe_text(item[0]), "weight": _safe_float(item[1]), "direction": _safe_text(item[2])})
        else:
            factors.append({"name": _safe_text(item), "weight": 0.0, "direction": ""})
    return PrismCard(
        CardType.PREDICTION,
        "Match prediction",
        body,
        {
            "home": _safe_text(getattr(pred, "home_team", getattr(pred, "home", "Home"))),
            "away": _safe_text(getattr(pred, "away_team", getattr(pred, "away", "Away"))),
            "p_home": round(p_home, 6),
            "p_draw": round(p_draw, 6),
            "p_away": round(p_away, 6),
            "predicted": body,
            "confidence": _safe_float(getattr(pred, "confidence", 0.0)),
            "key_factors": factors,
        },
        actions=["Why this forecast?", "Compare another fixture"],
    )


def moment_card(result) -> PrismCard:
    moment = getattr(result, "moment", None)
    options = []
    for item in getattr(result, "activations", []) or []:
        if isinstance(item, (tuple, list)) and len(item) >= 3:
            name, activation, ev = item[:3]
        else:
            name, activation, ev = item, 0.0, 0.0
        options.append(
            {
                "name": _safe_text(name),
                "activation": _safe_float(activation),
                "ev": _safe_float(ev),
            }
        )
    return PrismCard(
        CardType.MOMENT,
        "Moment analysis",
        _safe_text(getattr(result, "recommended", "Moment evaluated")),
        {
            "sport": _safe_text(getattr(moment, "sport", "")),
            "moment_type": _safe_text(getattr(moment, "moment_type", "")),
            "recommended": _safe_text(getattr(result, "recommended", "")),
            "activation": _safe_float(options[0]["activation"] if options else 0.0),
            "xg": _safe_float(getattr(result, "xg_contextual", 0.0)),
            "time_pressure": _safe_float(getattr(result, "time_pressure", 0.0)),
            "options": options,
        },
        actions=["Explain the best option", "Record the outcome"],
    )


def risk_card(risk) -> PrismCard:
    return PrismCard(
        CardType.RISK,
        "Injury risk",
        _safe_text(getattr(risk, "prediction", "Risk assessment ready")),
        {
            "athlete": _safe_text(getattr(risk, "athlete_name", getattr(risk, "subject", "Athlete"))),
            "risk_level": _safe_text(getattr(risk, "risk_level", "unknown")),
            "prediction": _safe_text(getattr(risk, "prediction", "")),
            "recommendations": list(getattr(risk, "recommendations", []) or []),
            "fulcrum": _safe_float(getattr(risk, "fulcrum", 0.0)),
        },
        actions=["Reduce today's load", "Show recovery plan"],
    )


def domain_card(domain: str, diagnosis) -> PrismCard:
    options = []
    activations = list(getattr(diagnosis, "activations", []) or [])
    for item in activations:
        plank = getattr(item, "plank", None)
        options.append(
            {
                "name": _safe_text(getattr(plank, "name", "")),
                "activation": _safe_float(getattr(item, "activation", 0.0)),
                "position": _safe_float(getattr(plank, "position", 0.0)),
            }
        )
    confidence = _safe_float(getattr(activations[0], "activation", 0.0) if activations else 0.0)
    recommended = _safe_text(getattr(getattr(diagnosis, "primary_plank", None), "name", ""))
    return PrismCard(
        CardType.DOMAIN,
        f"{domain} decision",
        recommended or f"{domain} evaluation ready",
        {
            "domain": domain,
            "recommended": recommended,
            "confidence": confidence,
            "fulcrum": _safe_float(getattr(diagnosis, "fulcrum_position", 0.0)),
            "options": options,
        },
        actions=["Explain the trade-offs", "Compare another profile"],
    )


def squad_card(risks: list) -> PrismCard:
    players = []
    for risk in risks or []:
        if isinstance(risk, dict):
            players.append(
                {
                    "name": _safe_text(risk.get("name") or risk.get("athlete") or risk.get("athlete_name")),
                    "risk_level": _safe_text(risk.get("risk_level", "unknown")),
                    "confidence": _safe_float(risk.get("confidence", 0.0)),
                }
            )
        else:
            players.append(
                {
                    "name": _safe_text(getattr(risk, "athlete_name", getattr(risk, "subject", "Player"))),
                    "risk_level": _safe_text(getattr(risk, "risk_level", "unknown")),
                    "confidence": _safe_float(getattr(risk, "confidence", 0.0)),
                }
            )
    return PrismCard(
        CardType.SQUAD,
        "Squad overview",
        f"{len(players)} player risk profile(s)",
        {"players": players},
        actions=["Show highest risks", "Suggest load changes"],
    )


def identity_card(identity: dict) -> PrismCard:
    identity = identity or {}
    total_ratings = int(identity.get("total_ratings", 0) or 0)
    total_plans = int(identity.get("total_plans", 0) or 0)
    n_decisions = total_ratings + total_plans
    fulcrum = _safe_float(identity.get("fixed_fulcrum", 0.5), 0.5)
    avg_day_rating = identity.get("avg_day_rating")
    domains = [
        {
            "label": "Fulcrum",
            "value": fulcrum,
            "crystallised": n_decisions >= 3,
        },
        {
            "label": "Recovery focus",
            "value": max(0.0, min(1.0, 1.0 - fulcrum)),
            "crystallised": n_decisions >= 3,
        },
    ]
    if avg_day_rating is not None:
        domains.append(
            {
                "label": "Day rating",
                "value": max(0.0, min(1.0, _safe_float(avg_day_rating) / 5.0)),
                "crystallised": total_ratings >= 5,
            }
        )
    insight = _safe_text(
        identity.get("fulcrum_trend"),
        f"{_safe_text(identity.get('profile', 'User'))} is building a decision profile.",
    )
    confidence = min(1.0, 0.25 + (n_decisions / 20.0))
    return PrismCard(
        CardType.IDENTITY,
        "Identity profile",
        insight.replace("_", " "),
        {
            "domains": domains,
            "insight": insight.replace("_", " "),
            "confidence": confidence,
            "n_decisions": n_decisions,
        },
        actions=["How did I evolve?", "Show my plan history"],
    )
