"""
prism_routes_predict.py
=======================
FastAPI router for prediction endpoints.

Routes:
  GET /predict/match
  GET /predict/injury
  GET /predict/performance
  GET /predict/transfer
  GET /predict/brief
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from prism_state import _safe_dict, _state

router = APIRouter(prefix="/predict")


@router.get("/match")
async def predict_match(
    home: str = "Home Team",
    away: str = "Away Team",
    sport: str = "football",
    home_form: float = 0.5,
    away_form: float = 0.5,
):
    platform = _state.get("platform")
    if platform is None:
        return JSONResponse({"error": "platform not ready", "status": 503}, status_code=503)
    pred = platform.match.predict(home, away, sport, home_form=home_form, away_form=away_form)
    return _safe_dict(pred)


@router.get("/injury")
async def predict_injury(
    name: str = "Athlete",
    recovery: float = 0.7,
    load: float = 0.5,
    soreness: float = 0.3,
):
    platform = _state.get("platform")
    if platform is None:
        return JSONResponse({"error": "platform not ready", "status": 503}, status_code=503)
    pred = platform.injury.predict(
        name,
        recovery_score=recovery,
        load_7d=load,
        muscle_soreness=soreness,
    )
    return _safe_dict(pred)


@router.get("/performance")
async def predict_performance(
    name: str = "Athlete",
    form: float = 0.6,
    fitness: float = 0.7,
):
    platform = _state.get("platform")
    if platform is None:
        return JSONResponse({"error": "platform not ready", "status": 503}, status_code=503)
    pred = platform.performance.predict(name, recent_form=form, fitness_level=fitness)
    return _safe_dict(pred)


@router.get("/transfer")
async def predict_transfer(
    name: str = "Athlete",
    performance: float = 0.6,
    age: int = 24,
):
    platform = _state.get("platform")
    if platform is None:
        return JSONResponse({"error": "platform not ready", "status": 503}, status_code=503)
    pred = platform.transfer.predict(name, performance_score=performance, age=age)
    return _safe_dict(pred)


@router.get("/brief")
async def predict_brief(
    home: str = "Home Team",
    away: str = "Away Team",
    sport: str = "football",
):
    platform = _state.get("platform")
    if platform is None:
        return JSONResponse({"error": "platform not ready", "status": 503}, status_code=503)
    data = platform.pre_match_brief(home, away, sport)
    return {
        "match_prediction":  _safe_dict(data["match_prediction"]),
        "tactical_analysis": _safe_dict(data["tactical_analysis"]),
        "squad_risk":        [_safe_dict(r) for r in data["squad_risk"]],
        "squad_performance": [_safe_dict(p) for p in data["squad_performance"]],
        "generated_at":      data["generated_at"],
    }
