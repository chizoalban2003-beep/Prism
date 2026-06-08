"""
prism_routes_core.py
====================
FastAPI router for core agent interaction endpoints.

Routes:
  POST /ask
  POST /chat
  POST /rate
  POST /session
  GET  /devices
  POST /device/sync
  GET  /device/capabilities
  POST /device/approve
  POST /device/execute
  GET  /search
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict as _asdict
from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from prism_state import _state

router = APIRouter()


def _get_agent():
    return _state.get("agent")


# ---------------------------------------------------------------------------
# POST /ask
# ---------------------------------------------------------------------------

@router.post("/ask")
async def ask(request: Request):
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    agent = _get_agent()
    if agent is None:
        return JSONResponse({"error": "agent not ready", "status": 503}, status_code=503)

    prompt = body.get("prompt", "")
    if not prompt:
        return JSONResponse(
            {"error": "'prompt' field required", "status": 400}, status_code=400
        )

    result = await asyncio.to_thread(agent.ask, prompt)
    return {
        "task":       result.task,
        "method":     result.method,
        "success":    result.success,
        "elapsed_ms": result.elapsed_ms,
        "output":     result.output,
    }


# ---------------------------------------------------------------------------
# POST /chat
# ---------------------------------------------------------------------------

@router.post("/chat")
async def chat(request: Request):
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    agent = _get_agent()
    if agent is None:
        # Fall back to creating a new PrismAgent instance
        try:
            from prism_agent import PrismAgent
            agent = PrismAgent()
        except ImportError as exc:
            return JSONResponse({"error": str(exc), "status": 503}, status_code=503)

    message = body.get("message", "")
    card = await asyncio.to_thread(agent.chat, message, body.get("context", {}))

    # Persist to named session if one is active
    session_id = body.get("session_id") or _state.get("active_session_id")
    if session_id and message:
        try:
            from prism_session_manager import get_session_manager
            sm = get_session_manager()
            sm.add_message(session_id, "user", message)
            sm.add_message(session_id, "assistant", card.body if hasattr(card, "body") else str(card))
        except Exception:
            pass

    return card.to_json()


# ---------------------------------------------------------------------------
# POST /rate
# ---------------------------------------------------------------------------

@router.post("/rate")
async def rate(request: Request):
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    agent = _get_agent()
    if agent is None:
        return JSONResponse({"error": "agent not ready", "status": 503}, status_code=503)

    date_str = body.get("date", "")
    rating   = float(body.get("rating", 0))
    notes    = body.get("notes", "")
    if not date_str:
        from datetime import date
        date_str = date.today().isoformat()
    agent._assistant.rate_day(agent._profile.name, date_str, rating, notes)
    return {"ok": True, "date": date_str, "rating": rating}


# ---------------------------------------------------------------------------
# POST /session
# ---------------------------------------------------------------------------

@router.post("/session")
async def session(request: Request):
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    agent = _get_agent()
    if agent is None:
        return JSONResponse({"error": "agent not ready", "status": 503}, status_code=503)

    rpe          = int(body.get("rpe", 5))
    session_type = body.get("session_type", "training")
    notes        = body.get("notes", "")
    video_folder = body.get("video_folder")
    gps_file     = body.get("gps_file")

    log = agent.log_session(
        rpe          = rpe,
        session_type = session_type,
        notes        = notes,
        video_folder = video_folder,
        gps_file     = gps_file,
    )
    return _asdict(log)


# ---------------------------------------------------------------------------
# GET /devices
# ---------------------------------------------------------------------------

@router.get("/devices")
async def devices():
    agent = _get_agent()
    if agent is None:
        return JSONResponse({"error": "agent not ready", "status": 503}, status_code=503)
    devices_list = [
        {
            "name":        d.name,
            "device_type": d.device_type.value,
            "enabled":     d.enabled,
            "last_sync":   d.last_sync,
        }
        for d in agent._hub.list_devices()
    ]
    return {"devices": devices_list}


# ---------------------------------------------------------------------------
# POST /device/sync
# ---------------------------------------------------------------------------

@router.post("/device/sync")
async def device_sync():
    agent = _get_agent()
    if agent is None:
        return JSONResponse({"error": "agent not ready", "status": 503}, status_code=503)
    result = agent.sync_devices()
    return {"synced": result}


# ---------------------------------------------------------------------------
# GET /device/capabilities
# ---------------------------------------------------------------------------

@router.get("/device/capabilities")
async def device_capabilities():
    try:
        from prism_device_agent import DeviceCapabilityScanner
        caps = DeviceCapabilityScanner().scan()
        return {
            "platform":    caps.platform,
            "has_browser": caps.has_browser,
            "categories":  {k: v for k, v in caps.cli_tools.items()},
            "py_packages": caps.py_packages,
            "summary":     caps.summary(),
        }
    except ImportError as exc:
        return JSONResponse({"error": str(exc), "status": 503}, status_code=503)


# ---------------------------------------------------------------------------
# POST /device/approve
# ---------------------------------------------------------------------------

@router.post("/device/approve")
async def device_approve(request: Request):
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    approved = body.get("approved", False)
    task     = body.get("task", "")
    params   = body.get("params", {})

    if not approved:
        try:
            from prism_responses import text_card
            return text_card("Action cancelled.").to_json()
        except ImportError:
            return {"message": "Action cancelled."}

    try:
        from prism_device_agent import PrismDeviceAgent
        from prism_responses import device_result_card
    except ImportError as exc:
        return JSONResponse({"error": str(exc), "status": 503}, status_code=503)

    device_agent = _state.get("device_agent")
    if not device_agent:
        device_agent = PrismDeviceAgent.setup()

    result = device_agent.execute(task, params=params, approval_override=True)
    return device_result_card(result, task).to_json()


# ---------------------------------------------------------------------------
# POST /device/execute
# ---------------------------------------------------------------------------

@router.post("/device/execute")
async def device_execute(request: Request):
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    try:
        from prism_device_agent import PrismDeviceAgent
    except ImportError as exc:
        return JSONResponse({"error": str(exc), "status": 503}, status_code=503)

    device_agent = _state.get("device_agent")
    if not device_agent:
        device_agent = PrismDeviceAgent.setup()

    dry_run = body.get("dry_run", False)
    result  = device_agent.execute(
        body.get("task", ""),
        params  = body.get("params", {}),
        dry_run = dry_run,
    )
    return {
        "success":        result.success,
        "output":         result.output[:2000],
        "tool_used":      result.tool_used,
        "elapsed_ms":     round(result.elapsed_ms, 1),
        "files_created":  result.files_created,
        "error":          result.error,
        "undo_available": bool(result.undo_command),
    }


# ---------------------------------------------------------------------------
# GET /search
# ---------------------------------------------------------------------------

@router.get("/search")
async def search(q: str = "", n: int = 8):
    agent = _get_agent()
    if agent and hasattr(agent, "_search") and q:
        results = agent._search.search(q, n=n)
        return {
            "query":   q,
            "results": [
                {"title": r.title, "url": r.url, "snippet": r.snippet}
                for r in results
            ],
        }
    return {"query": q, "results": []}
