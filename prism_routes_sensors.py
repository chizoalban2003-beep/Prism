"""
prism_routes_sensors.py
=======================
FastAPI router for memory, perception, proactive, and smarthome endpoints.

Routes:
  GET    /memory/search
  POST   /memory/ingest
  GET    /perception/status
  POST   /perception/ingest
  POST   /perception/enable
  GET    /proactive
  GET    /proactive/pending
  GET    /proactive/triggers
  POST   /proactive/triggers
  DELETE /proactive/triggers/{trigger_id}
  POST   /proactive/triggers/{trigger_id}/pause
  POST   /proactive/triggers/{trigger_id}/resume
  POST   /proactive/deliver/{trigger_id}
  GET    /smarthome/status
  POST   /smarthome
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from prism_state import _get_agent, _state

router = APIRouter()




# ---------------------------------------------------------------------------
# /memory
# ---------------------------------------------------------------------------

@router.get("/memory/search")
async def memory_search(q: str = "", n: int = 5, source: Optional[str] = None):
    agent = _get_agent()
    mem   = getattr(agent, "_memory", None) if agent else None
    if mem is None:
        return {"results": [], "note": "memory not initialised"}
    if not q:
        return JSONResponse(
            {"error": "Query parameter 'q' is required", "status": 400}, status_code=400
        )
    results = mem.search(q, top_n=n, source_filter=source)
    return {
        "results": [
            {
                "entry_id":  r.entry.entry_id,
                "title":     r.entry.title,
                "source":    r.entry.source,
                "score":     round(r.score, 4),
                "excerpt":   r.excerpt,
                "tags":      r.entry.tags,
                "timestamp": r.entry.timestamp,
            }
            for r in results
        ],
        "count": len(results),
    }


@router.post("/memory/ingest")
async def memory_ingest(request: Request):
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    agent = _get_agent()
    mem   = getattr(agent, "_memory", None) if agent else None
    if mem is None:
        return JSONResponse(
            {"error": "Memory not initialised", "status": 503}, status_code=503
        )
    content = body.get("content", "")
    if not content:
        return JSONResponse(
            {"error": "'content' field required", "status": 400}, status_code=400
        )
    entry_id = mem.ingest(
        content = content,
        source  = body.get("source", "note"),
        title   = body.get("title", ""),
        tags    = body.get("tags"),
    )
    return {"ok": True, "entry_id": entry_id}


# ---------------------------------------------------------------------------
# /perception
# ---------------------------------------------------------------------------

@router.get("/perception/status")
async def perception_status():
    agent = _get_agent()
    perception = getattr(agent, "_perception", None) if agent else None
    if perception:
        return perception.status()
    return {"active_channels": [], "factor_count": 0, "summary": "perception not initialised"}


@router.post("/perception/ingest")
async def perception_ingest(request: Request):
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    agent      = _get_agent()
    perception = getattr(agent, "_perception", None) if agent else None
    if perception:
        perception.ingest_biometrics(**body)
        return {"ok": True}
    return JSONResponse(
        {"error": "Perception not initialised", "status": 503}, status_code=503
    )


@router.post("/perception/enable")
async def perception_enable(request: Request):
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    channel = body.get("channel", "")
    enabled = body.get("enabled", True)
    agent   = _get_agent()
    perception = getattr(agent, "_perception", None) if agent else None
    if perception:
        for ch in perception._channels:
            if ch.NAME == channel:
                ch.resume() if enabled else ch.pause()
        return {"channel": channel, "enabled": enabled}
    return JSONResponse(
        {"error": "Perception not initialised", "status": 503}, status_code=503
    )


# ---------------------------------------------------------------------------
# /proactive
# ---------------------------------------------------------------------------

@router.get("/proactive")
async def proactive(n: int = 5):
    p = _state.get("proactive")
    if p is None:
        # Also try to find on agent
        agent = _get_agent()
        p = getattr(agent, "_proactive", None) if agent else None
    if p is None:
        return {"events": [], "note": "proactive not initialised"}
    events = p.pending_events(n)
    return {
        "events": [
            {"trigger_id": e.trigger_id, "message": e.message, "timestamp": e.timestamp}
            for e in events
        ],
        "count": len(events),
    }


@router.get("/proactive/pending")
async def proactive_pending():
    agent  = _get_agent()
    events = getattr(agent, "_proactive_buffer", []) if agent else []
    result = {
        "events": [
            {"trigger_id": e.trigger_id, "message": e.message, "timestamp": e.timestamp}
            for e in events[-5:]
        ]
    }
    if agent:
        agent._proactive_buffer = []
    return result


def _get_proactive():
    p = _state.get("proactive")
    if p is None:
        agent = _get_agent()
        p = getattr(agent, "_proactive", None) if agent else None
    return p


def _trigger_to_dict(t) -> dict:
    return {
        "trigger_id":  t.trigger_id,
        "name":        t.name,
        "type":        "condition",
        "enabled":     t.enabled,
        "check_every": t.check_every,
        "cooldown":    t.cooldown,
        "last_fired":  t.last_fired,
    }


def _scheduled_to_dict(s) -> dict:
    return {
        "trigger_id": s.trigger_id,
        "name":       s.name,
        "type":       "scheduled",
        "fire_at":    s.fire_at,
        "message":    s.message,
        "fired":      s.fired,
    }


@router.get("/proactive/triggers")
async def proactive_triggers_list():
    """List all registered condition triggers and scheduled reminders."""
    p = _get_proactive()
    if p is None:
        return {"triggers": [], "scheduled": [], "note": "proactive not initialised"}
    return {
        "triggers":  [_trigger_to_dict(t) for t in p._triggers],
        "scheduled": [_scheduled_to_dict(s) for s in p._scheduled],
        "count":     len(p._triggers) + len(p._scheduled),
    }


@router.post("/proactive/triggers")
async def proactive_triggers_create(request: Request):
    """
    Create a trigger. Two types:

    Scheduled (one-shot):
      {"type": "scheduled", "message": "...", "fire_at": <unix_ts>}
      {"type": "scheduled", "message": "...", "in_seconds": 300}

    Simple condition (polls agent attribute):
      {"type": "condition", "trigger_id": "...", "name": "...",
       "check_every": 60, "cooldown": 3600,
       "condition_attr": "some_flag",   # attribute checked on agent object
       "message": "static message text"}
    """
    p = _get_proactive()
    if p is None:
        return JSONResponse({"error": "proactive not initialised"}, status_code=503)

    try:
        body: dict = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    kind = body.get("type", "scheduled")

    if kind == "scheduled":
        msg = body.get("message", "")
        if not msg:
            return JSONResponse({"error": "'message' is required"}, status_code=400)
        fire_at = body.get("fire_at")
        in_seconds = body.get("in_seconds")
        if fire_at is None and in_seconds is None:
            return JSONResponse({"error": "provide 'fire_at' or 'in_seconds'"}, status_code=400)
        if in_seconds is not None:
            fire_at = time.time() + float(in_seconds)
        assert fire_at is not None
        tid = p.schedule(msg, float(fire_at), trigger_id=body.get("trigger_id"))
        return {"ok": True, "trigger_id": tid, "fire_at": fire_at, "type": "scheduled"}

    if kind == "condition":
        from prism_proactive import ProactiveTrigger
        tid  = body.get("trigger_id") or f"custom_{int(time.time())}"
        name = body.get("name", tid)
        msg  = body.get("message", "Proactive trigger fired.")
        attr = body.get("condition_attr", "")
        agent = _get_agent()

        def _condition() -> bool:
            return bool(getattr(agent, attr, False)) if attr and agent else False

        def _message() -> str:
            return msg

        t = ProactiveTrigger(
            trigger_id  = tid,
            name        = name,
            check_every = int(body.get("check_every", 60)),
            condition   = _condition,
            message     = _message,
            enabled     = bool(body.get("enabled", True)),
            cooldown    = int(body.get("cooldown", 3600)),
        )
        p.register(t)
        return {"ok": True, "trigger_id": tid, "type": "condition"}

    return JSONResponse({"error": f"unknown type: {kind!r}"}, status_code=400)


@router.delete("/proactive/triggers/{trigger_id}")
async def proactive_triggers_delete(trigger_id: str):
    """Remove a condition trigger or scheduled reminder by ID."""
    p = _get_proactive()
    if p is None:
        return JSONResponse({"error": "proactive not initialised"}, status_code=503)

    before = len(p._triggers) + len(p._scheduled)
    p._triggers  = [t for t in p._triggers  if t.trigger_id != trigger_id]
    p._scheduled = [s for s in p._scheduled if s.trigger_id != trigger_id]
    after = len(p._triggers) + len(p._scheduled)

    if before == after:
        return JSONResponse({"error": f"trigger '{trigger_id}' not found"}, status_code=404)
    return {"ok": True, "trigger_id": trigger_id, "removed": before - after}


@router.post("/proactive/triggers/{trigger_id}/pause")
async def proactive_triggers_pause(trigger_id: str):
    """Disable a condition trigger (scheduled reminders cannot be paused)."""
    p = _get_proactive()
    if p is None:
        return JSONResponse({"error": "proactive not initialised"}, status_code=503)
    for t in p._triggers:
        if t.trigger_id == trigger_id:
            t.enabled = False
            return {"ok": True, "trigger_id": trigger_id, "enabled": False}
    return JSONResponse({"error": f"trigger '{trigger_id}' not found"}, status_code=404)


@router.post("/proactive/triggers/{trigger_id}/resume")
async def proactive_triggers_resume(trigger_id: str):
    """Re-enable a paused condition trigger."""
    p = _get_proactive()
    if p is None:
        return JSONResponse({"error": "proactive not initialised"}, status_code=503)
    for t in p._triggers:
        if t.trigger_id == trigger_id:
            t.enabled = True
            return {"ok": True, "trigger_id": trigger_id, "enabled": True}
    return JSONResponse({"error": f"trigger '{trigger_id}' not found"}, status_code=404)


@router.post("/proactive/deliver/{trigger_id}")
async def proactive_deliver(trigger_id: str):
    """Mark a pending proactive event as delivered."""
    p = _get_proactive()
    if p is None:
        return JSONResponse({"error": "proactive not initialised"}, status_code=503)
    p.mark_delivered(trigger_id)
    return {"ok": True, "trigger_id": trigger_id}


# ---------------------------------------------------------------------------
# /smarthome
# ---------------------------------------------------------------------------

@router.get("/smarthome/status")
async def smarthome_status():
    agent = _get_agent()
    if agent and hasattr(agent, "_smarthome"):
        return agent._smarthome.status_summary()
    return {"configured": False}


@router.post("/smarthome")
async def smarthome(request: Request):
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    try:
        from prism_smart_home import PrismSmartHome
    except ImportError as exc:
        return JSONResponse({"error": str(exc), "status": 503}, status_code=503)

    agent = _get_agent()
    sh    = getattr(agent, "_smarthome", None) if agent else None
    if sh is None:
        sh = PrismSmartHome(
            ha_url = body.get("ha_url", "http://homeassistant.local:8123"),
            token  = body.get("token", ""),
        )

    action    = body.get("action", "")
    entity_id = body.get("entity_id", "")

    if action == "turn_on":
        result = sh.turn_on(entity_id)
        return {"ok": result.success, "error": result.error}
    elif action == "turn_off":
        result = sh.turn_off(entity_id)
        return {"ok": result.success, "error": result.error}
    elif action == "toggle":
        result = sh.toggle(entity_id)
        return {"ok": result.success, "error": result.error}
    elif action == "list":
        devices = sh.list_devices(domain=body.get("domain", ""))
        return {
            "devices": [
                {
                    "entity_id":    d.entity_id,
                    "state":        d.state,
                    "friendly_name": d.friendly_name,
                }
                for d in devices
            ]
        }
    else:
        return JSONResponse(
            {"error": f"Unknown smarthome action: {action}", "status": 400}, status_code=400
        )
