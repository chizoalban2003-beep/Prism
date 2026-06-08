"""
prism_routes_analytics.py
=========================
FastAPI router for domain, moment, duel, and LLM cost analytics endpoints.

Routes:
  GET  /domain/list
  GET  /domain/profiles
  GET  /domain/evaluate
  GET  /domain/sensitivity
  POST /domain/validate
  GET  /moment/configs
  GET  /moment/analyze
  GET  /moment/history
  GET  /moment/player_stats
  POST /moment/calibrate
  POST /moment/live_frame
  GET  /duel/network
  GET  /duel/player
  GET  /duel/summary
  POST /duel/add_match
  GET  /analytics/tokens
  GET  /analytics/tokens/daily
  GET  /analytics/tokens/by-model
  GET  /analytics/tokens/by-source
  POST /analytics/tokens/record
  DELETE /analytics/tokens
"""
from __future__ import annotations

import dataclasses
import math
import uuid
from collections import defaultdict
from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from prism_state import _state

router = APIRouter()

# Module-level moment history: player -> list[MomentResult]
_moment_history: dict = defaultdict(list)


# ---------------------------------------------------------------------------
# Domain endpoints
# ---------------------------------------------------------------------------

@router.get("/domain/list")
async def domain_list():
    try:
        from domain_configs import ALL_DOMAINS
    except ImportError as exc:
        return JSONResponse({"error": str(exc), "status": 503}, status_code=503)
    return {
        "domains": [
            {
                "name":       name,
                "domain":     config.domain,
                "n_planks":   len(config.planks),
                "n_profiles": len(config.profiles),
            }
            for name, config in ALL_DOMAINS.items()
        ]
    }


@router.get("/domain/profiles")
async def domain_profiles(domain: str = ""):
    domain_models = _state.get("domain_models", {})
    model = domain_models.get(domain)
    if model is None:
        return JSONResponse({"error": f"Unknown domain: {domain}", "status": 404}, status_code=404)
    profiles = [
        {
            "name":          profile.name,
            "fixed_fulcrum": profile.fixed_fulcrum,
            "description":   profile.description,
        }
        for profile in model.config.profiles
    ]
    return {"domain": domain, "profiles": profiles}


@router.get("/domain/evaluate")
async def domain_evaluate(request: Request, domain: str = "Medical", profile: str = ""):
    try:
        from domain_configs import ALL_DOMAINS, DomainDecisionModel
    except ImportError as exc:
        return JSONResponse({"error": str(exc), "status": 503}, status_code=503)

    config = ALL_DOMAINS.get(domain)
    if config is None:
        return JSONResponse({"error": f"Unknown: {domain}", "status": 404}, status_code=404)

    model = DomainDecisionModel(config)
    if not profile:
        profile = config.profiles[0].name

    qp = dict(request.query_params)
    factor_values = {
        factor.id: float(qp.get(factor.id, 0.5))
        for factor in config.factors
    }

    diagnosis = model.evaluate(profile, factor_values)
    return {
        "recommended": diagnosis.primary_plank.name,
        "fulcrum":     round(diagnosis.fulcrum_position, 3),
        "confidence":  round(diagnosis.activations[0].activation, 3),
        "options": [
            {
                "name":       activation.plank.name,
                "activation": round(activation.activation, 3),
            }
            for activation in diagnosis.activations
        ],
    }


@router.get("/domain/sensitivity")
async def domain_sensitivity(
    domain: str = "",
    profile: str = "",
    factor: str = "",
    steps: int = 5,
):
    domain_models = _state.get("domain_models", {})
    model = domain_models.get(domain)
    if model is None:
        return JSONResponse({"error": f"Unknown domain: {domain}", "status": 404}, status_code=404)
    if not profile or not factor:
        return JSONResponse(
            {"error": "profile and factor are required", "status": 400}, status_code=400
        )

    sweep = model.sensitivity_sweep(profile, factor, steps=steps)
    values = [i / (steps - 1) for i in range(steps)] if steps > 1 else [0.0]
    return {
        "domain":  domain,
        "profile": profile,
        "factor":  factor,
        "sweep": [
            {
                "value":       value,
                "recommended": diagnosis.primary_plank.name,
                "fulcrum":     diagnosis.fulcrum_position,
                "confidence":  diagnosis.activations[0].activation,
            }
            for value, diagnosis in zip(values, sweep)
        ],
    }


@router.post("/domain/validate")
async def domain_validate(request: Request):
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    try:
        from domain_validator import DomainValidator, LabeledDecision
    except ImportError as exc:
        return JSONResponse({"error": str(exc), "status": 503}, status_code=503)

    domain = body.get("domain")
    domain_models = _state.get("domain_models", {})
    if domain not in domain_models:
        return JSONResponse(
            {"error": f"Unknown domain: {domain}", "status": 404}, status_code=404
        )

    cases = [
        LabeledDecision(
            case_id=item.get("case_id", str(index)),
            domain=domain,
            profile=item.get("profile", ""),
            factor_values=dict(item.get("factor_values", {})),
            expert_choice=item.get("expert_choice", ""),
            outcome=item.get("outcome", ""),
            notes=item.get("notes", ""),
        )
        for index, item in enumerate(body.get("cases", []), start=1)
    ]
    result = DomainValidator(domain).validate(cases)
    return dataclasses.asdict(result)


# ---------------------------------------------------------------------------
# Moment endpoints
# ---------------------------------------------------------------------------

@router.get("/moment/configs")
async def moment_configs():
    try:
        from moment_analyzer import ALL_MOMENT_CONFIGS
        from moment_configs_ext import EXTENDED_CONFIGS
        all_keys = set(ALL_MOMENT_CONFIGS.keys()) | set(EXTENDED_CONFIGS.keys())
    except ImportError:
        try:
            from moment_analyzer import ALL_MOMENT_CONFIGS
            all_keys = set(ALL_MOMENT_CONFIGS.keys())
        except ImportError:
            all_keys = set()
    configs = [
        {"sport": s, "moment_type": mt}
        for s, mt in sorted(all_keys)
    ]
    return {"configs": configs}


@router.get("/moment/analyze")
async def moment_analyze(request: Request):
    qs = dict(request.query_params)
    sport       = qs.get("sport")
    moment_type = qs.get("moment_type")
    player      = qs.get("player")

    if not sport or not moment_type or not player:
        return JSONResponse(
            {"error": "sport, moment_type, player are required", "status": 400},
            status_code=400,
        )

    moment_analyzer = _state.get("moment_analyzer")
    if moment_analyzer is None:
        return JSONResponse(
            {"error": "moment_analyzer not ready", "status": 503}, status_code=503
        )

    try:
        from moment_analyzer import Moment, NearbyPlayer
    except ImportError as exc:
        return JSONResponse({"error": str(exc), "status": 503}, status_code=503)

    # Primary opponent (goalkeeper)
    primary_opp = None
    gk_name = qs.get("gk_name")
    if gk_name:
        gk_dist = float(qs.get("gk_distance", 6.0))
        primary_opp = NearbyPlayer(
            name=gk_name,
            team="",
            distance=gk_dist,
            arrival_time=gk_dist / 7.5,
            is_goalkeeper=True,
        )

    # Secondary opponents
    secondary = []
    for i in (1, 2, 3):
        dname = qs.get(f"defender{i}")
        if dname:
            arr = float(qs.get(f"defender{i}_arrival", 3.0))
            secondary.append(
                NearbyPlayer(
                    name=dname,
                    team="",
                    distance=arr * 7.5,
                    arrival_time=arr,
                )
            )

    # Teammates
    teammates = []
    for i in (1, 2, 3):
        tname = qs.get(f"teammate{i}")
        if tname:
            tdist = float(qs.get(f"teammate{i}_distance", 10.0))
            teammates.append(
                NearbyPlayer(
                    name=tname,
                    team="",
                    distance=tdist,
                    arrival_time=tdist / 7.5,
                )
            )

    moment = Moment(
        moment_id           = str(uuid.uuid4()),
        match_id            = "api",
        sport               = sport,
        moment_type         = moment_type,
        timestamp           = 0.0,
        focal_player        = player,
        focal_profile       = qs.get("profile", "Forward"),
        focal_team          = qs.get("team", ""),
        focal_base          = float(qs.get("base", 0.5)),
        pitch_x             = float(qs.get("pitch_x", 0.5)),
        pitch_y             = float(qs.get("pitch_y", 0.5)),
        primary_opponent    = primary_opp,
        secondary_opponents = secondary,
        teammates           = teammates,
        fatigue             = float(qs.get("fatigue", 0.0)),
        confidence          = float(qs.get("confidence", 0.5)),
        score_pressure      = float(qs.get("score_pressure", 0.0)),
        xg_raw              = float(qs.get("xg_raw", 0.0)),
    )

    result = moment_analyzer.analyze(moment)

    # Store in history
    _moment_history[player].append(result)

    # Build response
    bw    = result.config.bandwidth
    focal = result.focal_position
    options_list = []
    for opt in result.config.options:
        kernel = math.exp(-(opt.position - focal) ** 2 / (2.0 * bw ** 2))
        options_list.append({
            "name":       opt.name,
            "activation": round(kernel, 4),
            "ev":         round(result.option_scores[opt.name], 4),
        })

    # time_pressure: based on nearest opponent arrival_time
    if moment.primary_opponent is not None:
        time_pressure = round(
            max(0.0, min(1.0, 1.0 / (1.0 + moment.primary_opponent.arrival_time))), 4
        )
    elif secondary:
        min_arr = min(o.arrival_time for o in secondary)
        time_pressure = round(max(0.0, min(1.0, 1.0 / (1.0 + min_arr))), 4)
    else:
        time_pressure = 0.0

    return {
        "recommended":   result.recommended,
        "activation":    round(result.focal_position, 4),
        "xg_contextual": round(result.xg_contextual, 4),
        "time_pressure": time_pressure,
        "fulcrum":       round(result.focal_position, 4),
        "options":       options_list,
    }


@router.get("/moment/history")
async def moment_history(player: str = "", limit: int = 20):
    history_raw = _moment_history.get(player, [])[-limit:]
    moments_out = []
    for r in history_raw:
        try:
            moments_out.append(dataclasses.asdict(r))
        except Exception:
            moments_out.append(str(r))
    return {"player": player, "moments": moments_out}


@router.get("/moment/player_stats")
async def moment_player_stats(player: str = ""):
    if not player:
        return JSONResponse(
            {"error": "player is required", "status": 400}, status_code=400
        )
    moment_analyzer = _state.get("moment_analyzer")
    if moment_analyzer is None:
        return JSONResponse(
            {"error": "moment_analyzer not ready", "status": 503}, status_code=503
        )
    stats = moment_analyzer.player_stats(player)
    return {"player": player, **stats}


@router.post("/moment/calibrate")
async def moment_calibrate(request: Request):
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    moment_analyzer = _state.get("moment_analyzer")
    if moment_analyzer is None:
        return JSONResponse(
            {"error": "moment_analyzer not ready", "status": 503}, status_code=503
        )

    try:
        from moment_analyzer import ActionOutcome, Moment
    except ImportError as exc:
        return JSONResponse({"error": str(exc), "status": 503}, status_code=503)

    moment_id    = body.get("moment_id", "")
    action_taken = body.get("action_taken", "")
    success      = bool(body.get("success", False))
    xg_realized  = float(body.get("xg_realized", 0.0))
    notes        = body.get("notes", "")

    # Find the moment in history
    target_moment = None
    for results in _moment_history.values():
        for r in results:
            if r.moment.moment_id == moment_id:
                target_moment = r.moment
                break
        if target_moment is not None:
            break

    if target_moment is None:
        # Create a minimal placeholder moment for calibration
        target_moment = Moment(
            moment_id    = moment_id,
            match_id     = "calibration",
            sport        = body.get("sport", "Football"),
            moment_type  = body.get("moment_type", "1v1_keeper"),
            timestamp    = 0.0,
            focal_player = body.get("player", "Unknown"),
            focal_profile= "Forward",
            focal_team   = "",
            focal_base   = 0.5,
            pitch_x      = 0.5,
            pitch_y      = 0.5,
        )

    outcome = ActionOutcome(
        action_taken = action_taken,
        success      = success,
        xg_delta     = xg_realized,
        notes        = notes,
    )
    moment_analyzer.calibrate(target_moment, outcome)
    return {"status": "calibrated"}


@router.post("/moment/live_frame")
async def moment_live_frame(request: Request):
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    live_pipeline = _state.get("live_pipeline")
    if live_pipeline is None:
        return JSONResponse(
            {"error": "live_pipeline not ready", "status": 503}, status_code=503
        )
    result = live_pipeline.feed_frame(body)
    if result is None:
        return {"moment": None}
    try:
        return {"moment": dataclasses.asdict(result)}
    except Exception:
        return {"moment": str(result)}


# ---------------------------------------------------------------------------
# Duel endpoints
# ---------------------------------------------------------------------------

@router.get("/duel/network")
async def duel_network(match_id: str = ""):
    duel_analyzer = _state.get("duel_analyzer")
    if duel_analyzer is None:
        return JSONResponse(
            {"error": "duel_analyzer not ready", "status": 503}, status_code=503
        )
    edges = []
    for (att, dfn), data in duel_analyzer.network._edges.items():
        edges.append({
            "attacker": att,
            "defender": dfn,
            "total":    data["total"],
            "won":      data["won"],
            "win_rate": data["won"] / data["total"] if data["total"] else 0.5,
        })
    return {"match_id": match_id, "edges": edges}


@router.get("/duel/player")
async def duel_player(player: str = ""):
    if not player:
        return JSONResponse(
            {"error": "player is required", "status": 400}, status_code=400
        )
    duel_analyzer = _state.get("duel_analyzer")
    if duel_analyzer is None:
        return JSONResponse(
            {"error": "duel_analyzer not ready", "status": 503}, status_code=503
        )
    profile = duel_analyzer.network.player_attack_stats(player)
    return profile


@router.get("/duel/summary")
async def duel_summary():
    duel_analyzer = _state.get("duel_analyzer")
    if duel_analyzer is None:
        return JSONResponse(
            {"error": "duel_analyzer not ready", "status": 503}, status_code=503
        )
    network     = duel_analyzer.network
    total_duels = sum(e["total"] for e in network._edges.values())
    total_won   = sum(e["won"]   for e in network._edges.values())
    return {
        "total_duels": total_duels,
        "total_won":   total_won,
        "win_rate":    total_won / total_duels if total_duels else 0.5,
        "n_matchups":  len(network._edges),
    }


@router.post("/duel/add_match")
async def duel_add_match(request: Request):
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    duel_analyzer = _state.get("duel_analyzer")
    if duel_analyzer is None:
        return JSONResponse(
            {"error": "duel_analyzer not ready", "status": 503}, status_code=503
        )

    match_id = body.get("match_id", str(uuid.uuid4()))
    events   = body.get("events", [])

    records = duel_analyzer.process_match(events, match_id)
    return {
        "match_id": match_id,
        "n_duels":  len(records),
        "accuracy": (
            sum(1 for r in records if r.attacker_won) / len(records)
            if records else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# /analytics/tokens — LLM cost ledger
# ---------------------------------------------------------------------------

def _get_ledger():
    try:
        from prism_llm_ledger import get_ledger
        return get_ledger()
    except Exception:
        return None


@router.get("/analytics/tokens")
async def analytics_tokens_summary(days: int = 30):
    """Overall LLM cost summary + daily + by-model breakdowns."""
    ledger = _get_ledger()
    if ledger is None:
        return JSONResponse({"error": "ledger not available"}, status_code=503)
    import time as _t
    since = _t.time() - days * 86400
    return {
        "summary":  ledger.summary(since_ts=since),
        "by_model": ledger.by_model(days=days),
        "by_source": ledger.by_source(days=days),
        "days":     days,
    }


@router.get("/analytics/tokens/daily")
async def analytics_tokens_daily(days: int = 30):
    """Daily LLM token and cost totals for the last N days."""
    ledger = _get_ledger()
    if ledger is None:
        return JSONResponse({"error": "ledger not available"}, status_code=503)
    return {"daily": ledger.by_day(days=days), "days": days}


@router.get("/analytics/tokens/by-model")
async def analytics_tokens_by_model(days: int = 30):
    """Per-model token and cost breakdown for the last N days."""
    ledger = _get_ledger()
    if ledger is None:
        return JSONResponse({"error": "ledger not available"}, status_code=503)
    return {"by_model": ledger.by_model(days=days), "days": days}


@router.get("/analytics/tokens/by-source")
async def analytics_tokens_by_source(days: int = 30):
    """Per-caller-source (chain/agent/organ/…) breakdown for the last N days."""
    ledger = _get_ledger()
    if ledger is None:
        return JSONResponse({"error": "ledger not available"}, status_code=503)
    return {"by_source": ledger.by_source(days=days), "days": days}


@router.post("/analytics/tokens/record")
async def analytics_tokens_record(request: Request):
    """
    Manually record an LLM call. Useful for callers that bypass LLMRouter.
    Body: {provider, model, input_tokens, output_tokens, latency_ms, source?, session_id?}
    """
    ledger = _get_ledger()
    if ledger is None:
        return JSONResponse({"error": "ledger not available"}, status_code=503)
    try:
        body: dict = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    provider = body.get("provider", "")
    model    = body.get("model", "")
    if not provider or not model:
        return JSONResponse({"error": "'provider' and 'model' are required"}, status_code=400)
    rec = ledger.record_call(
        provider=provider,
        model=model,
        input_tokens=int(body.get("input_tokens", 0)),
        output_tokens=int(body.get("output_tokens", 0)),
        latency_ms=float(body.get("latency_ms", 0.0)),
        source=body.get("source", "api"),
        session_id=body.get("session_id", ""),
    )
    return {
        "ok":       True,
        "call_id":  rec.call_id,
        "cost_usd": rec.cost_usd,
    }


@router.delete("/analytics/tokens")
async def analytics_tokens_clear():
    """Delete all ledger records. Returns count removed."""
    ledger = _get_ledger()
    if ledger is None:
        return JSONResponse({"error": "ledger not available"}, status_code=503)
    count = ledger.clear()
    return {"ok": True, "deleted": count}
