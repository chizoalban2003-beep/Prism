"""
prism_routes_sensors.py
=======================
FastAPI router for memory, perception, proactive, and smarthome endpoints.

Routes:
  GET  /memory/search
  POST /memory/ingest
  GET  /perception/status
  POST /perception/ingest
  POST /perception/enable
  GET  /proactive
  GET  /proactive/pending
  GET  /smarthome/status
  POST /smarthome
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from prism_state import _state

router = APIRouter()


def _get_agent():
    return _state.get("agent")


# ---------------------------------------------------------------------------
# /memory
# ---------------------------------------------------------------------------

@router.get("/memory/search")
async def memory_search(q: str = "", n: int = 5, source: str = None):
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
