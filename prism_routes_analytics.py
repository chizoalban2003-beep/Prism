"""
prism_routes_analytics.py
=========================
FastAPI router for domain and LLM cost analytics endpoints.

Routes:
  GET  /domain/list
  GET  /domain/profiles
  GET  /domain/evaluate
  GET  /domain/sensitivity
  POST /domain/validate
  GET  /analytics/tokens
  GET  /analytics/tokens/daily
  GET  /analytics/tokens/by-model
  GET  /analytics/tokens/by-source
  POST /analytics/tokens/record
  DELETE /analytics/tokens
"""
from __future__ import annotations

import dataclasses
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from prism_state import _state

router = APIRouter()



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
    body: dict[str, Any] = {}
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
