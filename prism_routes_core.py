"""
prism_routes_core.py
====================
FastAPI router for core agent interaction endpoints.

Routes:
  POST /chat
  GET  /device/capabilities
  POST /device/approve
  POST /device/execute
  GET  /search
"""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from prism_state import _get_agent, _state

router = APIRouter()




# ---------------------------------------------------------------------------
# POST /chat
# ---------------------------------------------------------------------------

@router.post("/chat")
async def chat(request: Request):
    body: dict[str, Any] = {}
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
    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    approved     = body.get("approved", False)
    task         = body.get("task", "")
    params       = body.get("params", {})
    instructions = body.get("instructions", "")

    if not approved:
        # Record the denial (and any reason the user gave in the textarea)
        # so the agent can learn the boundary and avoid re-asking. Best-effort:
        # missing infra is silent because denial UX shouldn't break.
        try:
            agent = _get_agent()
            if agent and hasattr(agent, "record_denial"):
                agent.record_denial(task, params, instructions)
        except Exception:
            pass

        try:
            from prism_responses import text_card
            note = ""
            if instructions and instructions.strip():
                safe = instructions.strip()[:200]
                note = f"\n\nNoted: \"{safe}\" — I'll remember this when similar requests come up."
            if task == "_synthesize_organ":
                return text_card(
                    "Cancelled. I won't build that organ. "
                    "Rephrase or add more context if you'd like a different approach." + note,
                    "Synthesis cancelled").to_json()
            return text_card("Action cancelled." + note).to_json()
        except ImportError:
            return {"message": "Action cancelled."}

    # Synthesis approval — route to the agent's synthesis handler so the
    # new organ is generated, AST-validated, persisted, and executed with
    # the user's optional refinement instructions folded into the prompt.
    if task == "_synthesize_organ":
        agent = _get_agent()
        if agent and hasattr(agent, "handle_synthesis_approval"):
            try:
                card = agent.handle_synthesis_approval(params, instructions)
                return card.to_json()
            except Exception as exc:
                return JSONResponse(
                    {"error": f"Synthesis handler failed: {exc}", "status": 500},
                    status_code=500)
        return JSONResponse(
            {"error": "Agent not available for synthesis approval", "status": 503},
            status_code=503)

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
    body: dict[str, Any] = {}
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
