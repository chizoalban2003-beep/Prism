"""
prism_routes_agent.py
=====================
FastAPI router for agent-level endpoints.

Routes:
  GET  /status
  GET  /plan
  POST /plan
  GET  /reflect
  GET  /history
  GET  /artifacts
  POST /artifacts/rate
  GET  /identity
  GET  /identity/domains
  POST /identity/observe
  POST /identity/reset
  GET  /context
  GET  /outcomes/stats
  GET  /reflection
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict as _asdict
from typing import Optional, Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from prism_state import _get_agent, _state

router = APIRouter()




# ---------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------

@router.get("/status")
async def agent_status():
    agent = _get_agent()
    if agent is None:
        return JSONResponse({"error": "agent not ready", "status": 503}, status_code=503)

    # Check ollama in a background thread (blocking urlopen)
    async def _check_ollama():
        import json as _j
        import urllib.request as _ur
        try:
            r = await asyncio.to_thread(_ur.urlopen, "http://localhost:11434/api/tags", None, 2)
            tags = _j.loads(r.read())
            ok    = True
            model = (
                tags.get("models", [{}])[0].get("name", "")
                if tags.get("models") else ""
            )
        except Exception:
            ok    = False
            model = ""
        return ok, model

    ollama_ok, ollama_model = await _check_ollama()
    status = agent.status()
    status["ollama"]       = ollama_ok
    status["ollama_model"] = ollama_model
    return status


# ---------------------------------------------------------------------------
# /plan
# ---------------------------------------------------------------------------

@router.get("/plan")
async def plan_get():
    agent = _get_agent()
    if agent is None:
        return JSONResponse({"error": "agent not ready", "status": 503}, status_code=503)
    brief = agent.morning_briefing()
    try:
        data = _asdict(brief)
    except TypeError:
        data = str(brief)
    return data


@router.post("/plan")
async def plan_post(request: Request):
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    try:
        from prism_planner import PrismPlanner
        from prism_responses import plan_of_action_card
    except ImportError as exc:
        return JSONResponse({"error": str(exc), "status": 503}, status_code=503)

    planner = _state.get("planner")
    if not planner:
        planner = PrismPlanner()

    plan = planner.plan(
        task_description = body.get("task", ""),
        user_context     = body.get("context", {}),
        n_plans          = body.get("n_plans", 4),
    )
    return plan_of_action_card(plan).to_json()


# ---------------------------------------------------------------------------
# /reflect
# ---------------------------------------------------------------------------

@router.get("/reflect")
async def reflect():
    agent = _get_agent()
    if agent is None:
        return JSONResponse({"error": "agent not ready", "status": 503}, status_code=503)
    return agent.reflect()


# ---------------------------------------------------------------------------
# /history
# ---------------------------------------------------------------------------

@router.get("/history")
async def history(days: int = 14):
    agent = _get_agent()
    if agent is None:
        return JSONResponse({"error": "agent not ready", "status": 503}, status_code=503)
    hist = agent._assistant.history(agent._profile.name, days=days)
    return {"history": hist}


# ---------------------------------------------------------------------------
# /artifacts
# ---------------------------------------------------------------------------

@router.get("/artifacts")
async def artifacts(domain: Optional[str] = None, n: int = 10):
    agent = _get_agent()
    if agent is None:
        return JSONResponse({"error": "agent not ready", "status": 503}, status_code=503)
    return {"artifacts": agent.recent_artifacts(domain=domain, n=n)}


@router.post("/artifacts/rate")
async def artifacts_rate(request: Request):
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    agent = _get_agent()
    if agent is None:
        return JSONResponse({"error": "agent not ready", "status": 503}, status_code=503)

    artifact_id = body.get("artifact_id")
    if not artifact_id:
        return JSONResponse(
            {"error": "'artifact_id' field required", "status": 400}, status_code=400
        )
    rating = float(body.get("rating", 0.0))
    return agent.rate_artifact(artifact_id, rating)


# ---------------------------------------------------------------------------
# /identity
# ---------------------------------------------------------------------------

@router.get("/identity")
async def identity():
    agent = _get_agent()
    if agent is None:
        return JSONResponse({"error": "agent not ready", "status": 503}, status_code=503)
    return agent.identity()


@router.get("/identity/domains")
async def identity_domains():
    agent = _get_agent()
    if agent is None:
        return JSONResponse({"error": "agent not ready", "status": 503}, status_code=503)
    return {"domains": agent.identity_domains()}


@router.post("/identity/observe")
async def identity_observe(request: Request):
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    agent = _get_agent()
    if agent is None:
        return JSONResponse({"error": "agent not ready", "status": 503}, status_code=503)

    domain = body.get("domain")
    if not domain:
        return JSONResponse(
            {"error": "'domain' field required", "status": 400}, status_code=400
        )
    identity = agent.observe_identity(
        domain  = domain,
        fulcrum = float(body.get("fulcrum", 0.5)),
        rating  = float(body.get("rating", 0.5)),
        context = body.get("context") or {},
    )
    return identity


@router.post("/identity/reset")
async def identity_reset(request: Request):
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    agent = _get_agent()
    if agent is None:
        return JSONResponse({"error": "agent not ready", "status": 503}, status_code=503)

    domain = body.get("domain")
    if not domain:
        return JSONResponse(
            {"error": "'domain' field required", "status": 400}, status_code=400
        )
    return agent.reset_identity_domain(domain)


# ---------------------------------------------------------------------------
# /context
# ---------------------------------------------------------------------------

@router.get("/context")
async def context():
    agent = _get_agent()
    cm = getattr(agent, "_context_manager", None) if agent else None
    if cm is None:
        return {"active": "default", "profiles": []}
    return {
        "active":   cm.active_id,
        "profiles": [p.to_dict() for p in cm.list_profiles()],
    }


# ---------------------------------------------------------------------------
# /outcomes/stats
# ---------------------------------------------------------------------------

@router.get("/outcomes/stats")
async def outcomes_stats(days: int = 30):
    agent   = _get_agent()
    tracker = getattr(agent, "_outcome_tracker", None) if agent else None
    if tracker is None:
        return {"available": False}
    return {**tracker.stats(days=days), "available": True}


# ---------------------------------------------------------------------------
# /reflection
# ---------------------------------------------------------------------------

@router.get("/reflection")
async def reflection():
    agent = _get_agent()
    refl  = getattr(agent, "_reflection", None) if agent else None
    if refl is None:
        return {"available": False}
    try:
        report = refl.run()
        return {
            "available":        True,
            "summary":          report.summary,
            "patterns":         report.patterns,
            "belief_proposals": report.belief_proposals,
            "unresolved_goals": report.unresolved_goals,
            "applied":          report.applied,
            "ran_at":           report.ran_at,
        }
    except Exception as exc:
        return {"available": True, "error": str(exc)}
