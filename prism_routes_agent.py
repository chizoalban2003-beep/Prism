"""
prism_routes_agent.py
=====================
FastAPI router for agent-level endpoints.

Routes:
  GET  /status
  GET  /plan
  POST /plan
  GET  /context
  GET  /outcomes/stats
  GET  /reflection

Identity endpoints live in `prism_routes_identity.py`
(`/identity/dashboard`, `/identity/onboard`, `/identity/ui`, etc.).
Conversation history is persisted by `prism_session_manager` and
exposed at `GET /sessions/{id}/history`.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict as _asdict
from typing import Any

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
    if hasattr(agent, "status"):
        status = agent.status()
    else:
        chain = getattr(agent, "_chain", None)
        phase_engine = None
        try:
            import prism_phase as _pp
            phase_engine = _pp.get_engine()
        except Exception:
            pass
        status = {
            "agent": type(agent).__name__,
            "chain_ready": chain is not None,
            "soul_seeded": bool(
                getattr(getattr(agent, "_soul", None), "has_seed", lambda: False)()
            ),
            "phase": str(getattr(phase_engine, "current_phase", "UNKNOWN")) if phase_engine else "UNKNOWN",
        }
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
    body: dict[str, Any] = {}
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
