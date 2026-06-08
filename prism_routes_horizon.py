"""
prism_routes_horizon.py
=======================
FastAPI router for horizon planner and push notification endpoints.

Routes:
  GET  /horizon/goals
  GET  /horizon/status
  POST /horizon/goal
  POST /horizon/goal/{goal_id}/complete
  POST /horizon/goal/{goal_id}/abandon
  POST /horizon/goal/{goal_id}/context
  GET  /push/status
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from prism_state import _get_agent

router = APIRouter()




def _get_horizon(agent):
    return getattr(agent, "_horizon", None) if agent else None


# ---------------------------------------------------------------------------
# /horizon/goals
# ---------------------------------------------------------------------------

@router.get("/horizon/goals")
async def horizon_goals(status: str = None):
    agent = _get_agent()
    h     = _get_horizon(agent)
    if h is None:
        return {"goals": [], "total": 0}

    try:
        from prism_horizon import HorizonGoalStatus
        sf = HorizonGoalStatus(status) if status else None
    except (ImportError, ValueError):
        sf = None

    goals = h.list_goals(status=sf)
    return {
        "total": len(goals),
        "goals": [
            {
                "goal_id":              g.goal_id,
                "intent":               g.intent,
                "trigger_condition":    g.trigger_condition,
                "completion_condition": g.completion_condition,
                "status":               g.status.value,
                "session_count":        g.session_count,
                "completed_steps":      g.completed_steps,
                "accumulated_context":  g.accumulated_context,
                "created_at":           g.created_at,
                "triggered_at":         g.triggered_at,
                "expires_at":           g.expires_at,
                "notes":                g.notes,
            }
            for g in goals
        ],
    }


# ---------------------------------------------------------------------------
# /horizon/status
# ---------------------------------------------------------------------------

@router.get("/horizon/status")
async def horizon_status():
    agent = _get_agent()
    h     = _get_horizon(agent)
    if h is None:
        return {"available": False}
    return {**h.status(), "available": True}


# ---------------------------------------------------------------------------
# POST /horizon/goal
# ---------------------------------------------------------------------------

@router.post("/horizon/goal")
async def horizon_goal_create(request: Request):
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    agent = _get_agent()
    h     = _get_horizon(agent)
    if h is None:
        return JSONResponse(
            {"error": "HorizonPlanner not available", "status": 503}, status_code=503
        )

    intent   = body.get("intent", "")
    trigger  = body.get("trigger_condition", "")
    complete = body.get("completion_condition", "")
    expires  = body.get("expires_in_days")

    if not intent or not trigger:
        return JSONResponse(
            {"error": "'intent' and 'trigger_condition' are required", "status": 400},
            status_code=400,
        )

    gid = h.add(
        intent               = intent,
        trigger_condition    = trigger,
        completion_condition = complete,
        expires_in_days      = float(expires) if expires else None,
    )
    return {"goal_id": gid, "status": "watching"}


# ---------------------------------------------------------------------------
# POST /horizon/goal/{goal_id}/complete
# ---------------------------------------------------------------------------

@router.post("/horizon/goal/{goal_id}/complete")
async def horizon_goal_complete(goal_id: str, request: Request):
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    agent = _get_agent()
    h     = _get_horizon(agent)
    if h is None:
        return JSONResponse(
            {"error": "HorizonPlanner not available", "status": 503}, status_code=503
        )

    notes = body.get("notes", "")
    ok    = h.complete(goal_id, notes=notes)
    return {"ok": ok, "goal_id": goal_id}


# ---------------------------------------------------------------------------
# POST /horizon/goal/{goal_id}/abandon
# ---------------------------------------------------------------------------

@router.post("/horizon/goal/{goal_id}/abandon")
async def horizon_goal_abandon(goal_id: str, request: Request):
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    agent = _get_agent()
    h     = _get_horizon(agent)
    if h is None:
        return JSONResponse(
            {"error": "HorizonPlanner not available", "status": 503}, status_code=503
        )

    reason = body.get("reason", "")
    ok     = h.abandon(goal_id, reason=reason)
    return {"ok": ok, "goal_id": goal_id}


# ---------------------------------------------------------------------------
# POST /horizon/goal/{goal_id}/context
# ---------------------------------------------------------------------------

@router.post("/horizon/goal/{goal_id}/context")
async def horizon_goal_context(goal_id: str, request: Request):
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    agent = _get_agent()
    h     = _get_horizon(agent)
    if h is None:
        return JSONResponse(
            {"error": "HorizonPlanner not available", "status": 503}, status_code=503
        )

    facts = {k: v for k, v in body.items() if k != "goal_id"}
    h.update_context(goal_id, **facts)
    return {"ok": True, "goal_id": goal_id, "context_keys": list(facts.keys())}


# ---------------------------------------------------------------------------
# /push/status
# ---------------------------------------------------------------------------

@router.get("/push/status")
async def push_status():
    agent = _get_agent()
    if agent and hasattr(agent, "_push"):
        return agent._push.status_summary()
    return {"configured": False}
