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
# GET /settings/schema?section=email
# ---------------------------------------------------------------------------

@router.get("/settings/schema")
async def settings_schema(section: str = ""):
    """Return the setup-form schema for a section, with current DB values."""
    from prism_settings_store import SETTINGS_SCHEMA, get_settings_store
    if not section:
        return {"sections": list(SETTINGS_SCHEMA.keys())}
    schema = SETTINGS_SCHEMA.get(section)
    if schema is None:
        return JSONResponse({"error": f"unknown section: {section}", "status": 404}, status_code=404)
    store = get_settings_store()
    current = store.get_section(section)
    # Strip secrets — never expose existing secret values back to the client
    safe_current = {}
    for f in schema["fields"]:
        if not f.get("secret"):
            safe_current[f["name"]] = current.get(f["name"], f.get("default", ""))
        elif f["name"] in current:
            safe_current[f["name"]] = "__set__"  # sentinel: client knows it's set
    return {
        "section":        section,
        "label":          schema.get("label", section),
        "why":            schema.get("why", ""),
        "docs_url":       schema.get("docs_url", ""),
        "fields":         schema["fields"],
        "current_values": safe_current,
    }


# ---------------------------------------------------------------------------
# POST /settings/save
# ---------------------------------------------------------------------------

@router.post("/settings/save")
async def settings_save(request: Request):
    """Persist setup-form values to settings.db and hot-rebuild the service."""
    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    section = (body.get("section") or "").strip()
    values = body.get("values") or {}
    if not section:
        return JSONResponse({"error": "'section' required", "status": 400}, status_code=400)

    from prism_settings_store import get_settings_store, validate_values
    ok, err, coerced = validate_values(section, values)
    if not ok:
        return JSONResponse({"error": err, "status": 400}, status_code=400)

    # Don't overwrite a secret with the placeholder sentinel — preserve it.
    store = get_settings_store()
    existing = store.get_section(section)
    final: dict[str, Any] = {}
    for k, v in coerced.items():
        if v == "__set__" and k in existing:
            final[k] = existing[k]
        else:
            final[k] = v
    store.set_section(section, final)

    # Hot-rebuild the affected service if the agent is up
    result = {"ok": True, "section": section, "configured": False}
    try:
        agent = _get_agent()
        if agent and hasattr(agent, "apply_settings_change"):
            r = await asyncio.to_thread(agent.apply_settings_change, section)
            result.update(r)
    except Exception as exc:
        result["warning"] = f"saved but reload failed: {exc}"

    try:
        from prism_responses import text_card
        msg = "Configured." if result.get("configured") else "Saved. Try again to verify the connection."
        return {
            "ok":         result.get("ok", True),
            "section":    section,
            "configured": result.get("configured", False),
            "card":       text_card(msg, f"{section.capitalize()} saved").to_json(),
        }
    except ImportError:
        return result


# ---------------------------------------------------------------------------
# GET /narrative/proactive
# ---------------------------------------------------------------------------

@router.get("/narrative/proactive")
async def narrative_proactive():
    """Return a fresh narrative card (if any) and mark it as shown."""
    agent = _get_agent()
    narrative = getattr(agent, "_narrative", None) if agent else None
    if narrative is None or not hasattr(narrative, "peek_fresh"):
        return {"card": None}
    try:
        entry = await asyncio.to_thread(narrative.peek_fresh)
    except Exception:
        return {"card": None}
    if entry is None:
        return {"card": None}
    try:
        from prism_responses import narrative_card
        card = narrative_card(entry.content, entry.period, entry.generated_at)
        narrative.mark_shown()
        return {"card": card.to_json()}
    except Exception:
        return {"card": None}


# ---------------------------------------------------------------------------
# POST /plan/replan
# ---------------------------------------------------------------------------

@router.post("/plan/replan")
async def plan_replan(request: Request):
    """Re-run plan generation with user refinement instructions and optional
    edited task pins. Returns a fresh plan card."""
    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    agent = _get_agent()
    if agent is None or not hasattr(agent, "replan"):
        return JSONResponse({"error": "Re-plan unavailable", "status": 503}, status_code=503)

    instructions = (body.get("instructions") or "").strip()
    tasks = body.get("tasks") or []
    card = await asyncio.to_thread(agent.replan, instructions, tasks)
    return card.to_json()


# ---------------------------------------------------------------------------
# Plan telemetry — per-step status (M12d)
# ---------------------------------------------------------------------------

def _telemetry():
    try:
        from prism_plan_telemetry import get_plan_telemetry
        return get_plan_telemetry()
    except Exception:
        return None


@router.get("/plan/latest")
async def plan_latest():
    """Return the most recent plan + per-step status, or 404."""
    pt = _telemetry()
    if pt is None:
        return JSONResponse({"error": "Plan telemetry unavailable", "status": 503}, status_code=503)
    plan = await asyncio.to_thread(pt.latest_plan)
    if plan is None:
        return JSONResponse({"error": "No plan recorded", "status": 404}, status_code=404)
    return plan


@router.get("/plan/{plan_id}")
async def plan_get(plan_id: str):
    pt = _telemetry()
    if pt is None:
        return JSONResponse({"error": "Plan telemetry unavailable", "status": 503}, status_code=503)
    plan = await asyncio.to_thread(pt.get_plan, plan_id)
    if plan is None:
        return JSONResponse({"error": "Plan not found", "status": 404}, status_code=404)
    return plan


@router.post("/plan/{plan_id}/step/{step_index}")
async def plan_step_mark(plan_id: str, step_index: int, request: Request):
    """Set a step's status. Body: {"status": "done|abandoned|skipped|in_progress|pending",
    "outcome_record_id": "optional", "notes": "optional"}."""
    pt = _telemetry()
    if pt is None:
        return JSONResponse({"error": "Plan telemetry unavailable", "status": 503}, status_code=503)
    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass
    status = str(body.get("status", "") or "").strip()
    if not status:
        return JSONResponse({"error": "status field required", "status": 400}, status_code=400)
    try:
        ok = await asyncio.to_thread(
            pt.mark_step,
            plan_id,
            int(step_index),
            status,
            str(body.get("outcome_record_id", "") or ""),
            str(body.get("notes", "") or ""),
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc), "status": 400}, status_code=400)
    if not ok:
        return JSONResponse({"error": "step not found", "status": 404}, status_code=404)
    return {"ok": True, "plan_id": plan_id, "step_index": step_index, "status": status}


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
