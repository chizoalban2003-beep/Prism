"""
prism_routes_infra.py
=====================
FastAPI router for infrastructure / platform endpoints.

Routes:
  GET    /llm/status
  GET    /tasks
  GET    /tasks/{task_id}
  GET    /metrics
  GET    /policy
  GET    /policy/spend
  POST   /policy/set
  POST   /policy/update_from_chat
  GET    /tools/find
  GET    /settings/llm
  POST   /settings/llm
  POST   /settings/llm/test
  GET    /organs
  GET    /organs/{name}
  POST   /organs/{name}/enable
  POST   /organs/{name}/disable
  DELETE /organs/{name}
  POST   /organs/reload
  POST   /organs/synthesize
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from prism_state import _safe_dict, _state

router = APIRouter()


def _get_policy_engine():
    return _state.get("policy_engine")


def _get_organ_loader():
    agent = _state.get("agent")
    if agent is None:
        return None
    return getattr(agent, "_organ_loader", None) or _state.get("organ_loader")


def _get_llm_router():
    return _state.get("llm_router")


def _get_task_queue():
    return _state.get("task_queue")


# ---------------------------------------------------------------------------
# /llm/status
# ---------------------------------------------------------------------------

@router.get("/llm/status")
async def llm_status():
    llm_router = _get_llm_router()
    if llm_router is None:
        return {"available": False, "note": "LLM router not initialised"}
    return llm_router.status_summary()


# ---------------------------------------------------------------------------
# /tasks
# ---------------------------------------------------------------------------

@router.get("/tasks")
async def tasks(n: int = 10):
    task_queue = _get_task_queue()
    if task_queue is None:
        return {"tasks": [], "count": 0, "note": "Task queue not initialised"}
    tasks_list = task_queue.list_recent(n)
    items = [
        {
            "task_id":      t.task_id,
            "title":        t.title,
            "status":       t.status if isinstance(t.status, str) else t.status.value,
            "progress":     t.progress,
            "current_step": t.current_step,
            "steps_done":   t.steps_done,
            "steps_total":  t.steps_total,
            "error":        t.error,
        }
        for t in tasks_list
    ]
    return {"tasks": items, "count": len(items)}


@router.get("/tasks/{task_id}")
async def task_by_id(task_id: str):
    task_queue = _get_task_queue()
    if task_queue is None:
        return JSONResponse(
            {"error": "Task queue not initialised", "status": 503}, status_code=503
        )
    if not task_id:
        return JSONResponse(
            {"error": "task_id is required", "status": 400}, status_code=400
        )
    progress = task_queue.get(task_id)
    if progress is None:
        return JSONResponse(
            {"error": f"Task '{task_id}' not found", "status": 404}, status_code=404
        )
    return {
        "task_id":      progress.task_id,
        "title":        progress.title,
        "status":       progress.status if isinstance(progress.status, str) else progress.status.value,
        "progress":     progress.progress,
        "current_step": progress.current_step,
        "steps_done":   progress.steps_done,
        "steps_total":  progress.steps_total,
        "result":       progress.result,
        "error":        progress.error,
        "started_at":   progress.started_at,
        "completed_at": progress.completed_at,
    }


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------

@router.get("/metrics")
async def metrics(window_s: float = 300):
    try:
        from prism_metrics import metrics as _m
    except ImportError as exc:
        return JSONResponse({"error": str(exc), "status": 503}, status_code=503)
    return _m.report(window_s=window_s)


# ---------------------------------------------------------------------------
# /policy
# ---------------------------------------------------------------------------

@router.get("/policy")
async def policy(user: str = ""):
    if not user:
        return JSONResponse(
            {"error": 'Query parameter "user" is required', "status": 400}, status_code=400
        )
    policy_engine = _get_policy_engine()
    if policy_engine is None:
        return JSONResponse(
            {"error": "Policy engine not initialised", "status": 503}, status_code=503
        )
    return _safe_dict(policy_engine.get_policy(user))


@router.get("/policy/spend")
async def policy_spend(user: str = "", category: str = "", days: int = 30):
    if not user or not category:
        return JSONResponse(
            {"error": "user and category are required", "status": 400}, status_code=400
        )
    policy_engine = _get_policy_engine()
    if policy_engine is None:
        return JSONResponse(
            {"error": "Policy engine not initialised", "status": 503}, status_code=503
        )
    return policy_engine.spend_summary(user, category, days)


@router.post("/policy/set")
async def policy_set(request: Request):
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    policy_engine = _get_policy_engine()
    if policy_engine is None:
        return JSONResponse(
            {"error": "Policy engine not initialised", "status": 503}, status_code=503
        )

    user     = body.get("user", "")
    category = body.get("category", "")
    if not user or not category:
        return JSONResponse(
            {"error": "'user' and 'category' fields required", "status": 400}, status_code=400
        )

    try:
        from prism_policy import ResourceAllocation
    except ImportError as exc:
        return JSONResponse({"error": str(exc), "status": 503}, status_code=503)

    policy     = policy_engine.get_policy(user)
    allocation = policy.allocations.get(category, ResourceAllocation(name=category))
    for field_name in (
        "currency",
        "total_budget",
        "per_action_limit",
        "monthly_limit",
        "auto_approve_below",
        "preferred_providers",
        "blacklisted",
        "time_window",
        "notifications",
        "notes",
    ):
        if field_name in body:
            setattr(allocation, field_name, body[field_name])
    policy_engine.set_allocation(user, category, allocation)
    return {"ok": True, "allocation": _safe_dict(allocation)}


@router.post("/policy/update_from_chat")
async def policy_update_from_chat(request: Request):
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    policy_engine = _get_policy_engine()
    if policy_engine is None:
        return JSONResponse(
            {"error": "Policy engine not initialised", "status": 503}, status_code=503
        )

    user    = body.get("user", "")
    message = body.get("message", "")
    if not user or not message:
        return JSONResponse(
            {"error": "'user' and 'message' fields required", "status": 400}, status_code=400
        )
    result = policy_engine.parse_policy_update(message, user)
    return {"result": result}


# ---------------------------------------------------------------------------
# /tools/find
# ---------------------------------------------------------------------------

@router.get("/tools/find")
async def tools_find(
    task: str = "",
    provider: str = "",
    urgency: float = 0.5,
    cost_tolerance: float = 0.5,
    prefers_auto: float = 0.5,
    budget_left: float = 1.0,
):
    if not task:
        return JSONResponse(
            {"error": "task is required", "status": 400}, status_code=400
        )
    tool_finder = _state.get("tool_finder")
    if tool_finder is None:
        return JSONResponse(
            {"error": "Tool finder not initialised", "status": 503}, status_code=503
        )
    result = tool_finder.find(
        task          = task,
        provider_name = provider or task,
        urgency       = urgency,
        cost_tolerance= cost_tolerance,
        prefers_auto  = prefers_auto,
        budget_left   = budget_left,
    )
    return _safe_dict(result)


# ---------------------------------------------------------------------------
# /settings/llm
# ---------------------------------------------------------------------------

@router.get("/settings/llm", response_class=HTMLResponse)
async def settings_llm_get():
    try:
        from prism_settings_llm import get_llm_settings_html
        return HTMLResponse(content=get_llm_settings_html(), status_code=200)
    except ImportError as exc:
        return HTMLResponse(content=f"<p>Error: {exc}</p>", status_code=503)


@router.post("/settings/llm")
async def settings_llm_post(request: Request):
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    try:
        from prism_settings_llm import write_llm_config
    except ImportError as exc:
        return JSONResponse({"error": str(exc), "status": 503}, status_code=503)

    p     = body.get("provider", "")
    key   = body.get("key", "")
    host  = body.get("host", "")
    model = body.get("model", "")
    updates: dict = {}

    if p == "ollama":
        updates = {
            "ollama_host":  host or "http://localhost:11434",
            "ollama_model": model or "mistral",
            "preferred":    f"ollama/{model or 'mistral'}",
        }
    elif p == "claude":
        updates = {"claude_api_key": key, "preferred": "claude"}
    elif p == "openai":
        updates = {
            "openai_api_key": key,
            "openai_host":    "https://api.openai.com",
            "preferred":      "openai",
        }
    elif p == "openai_compat":
        updates = {
            "openai_api_key": key,
            "openai_host":    host,
            "preferred":      "openai_compat",
        }

    if updates:
        write_llm_config(updates)
        llm_router = _get_llm_router()
        if llm_router:
            llm_router._config.update(updates)
            llm_router._preferred   = updates.get("preferred", "")
            llm_router._discovered  = False
        return {
            "ok":        True,
            "message":   f"{p} config saved. Restart for full effect.",
            "preferred": updates.get("preferred", ""),
        }
    else:
        return JSONResponse(
            {"error": "Unknown provider or missing fields", "status": 400}, status_code=400
        )


@router.post("/settings/llm/test")
async def settings_llm_test(request: Request):
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    try:
        from prism_settings_llm import test_provider as _tp
    except ImportError as exc:
        return JSONResponse({"error": str(exc), "status": 503}, status_code=503)

    result = _tp(
        provider = body.get("provider", ""),
        key      = body.get("key", ""),
        host     = body.get("host", ""),
        model    = body.get("model", ""),
    )
    return result


# ---------------------------------------------------------------------------
# /organs — plugin manager
# ---------------------------------------------------------------------------

@router.get("/organs")
async def organs_list(source: str = "", enabled_only: bool = False):
    """List all loaded organs with metadata."""
    loader = _get_organ_loader()
    if loader is None:
        return {"organs": [], "count": 0, "note": "organ_loader not initialised"}
    items = loader.list_organ_details()
    if source:
        items = [o for o in items if o["source"] == source]
    if enabled_only:
        items = [o for o in items if o["enabled"]]
    return {"organs": items, "count": len(items)}


@router.post("/organs/reload")
async def organs_reload():
    """Re-scan bundled and user organ directories."""
    loader = _get_organ_loader()
    if loader is None:
        return JSONResponse({"error": "organ_loader not initialised"}, status_code=503)
    count = loader.reload()
    return {"ok": True, "loaded": count}


@router.post("/organs/synthesize")
async def organs_synthesize(request: Request):
    """Synthesize a new organ via LLM. Body: {intent, message}"""
    loader = _get_organ_loader()
    if loader is None:
        return JSONResponse({"error": "organ_loader not initialised"}, status_code=503)
    try:
        body: dict = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    intent  = body.get("intent", "").strip()
    message = body.get("message", "").strip()
    if not intent:
        return JSONResponse({"error": "'intent' is required"}, status_code=400)
    if not message:
        message = intent
    ok = loader.synthesize(intent, message)
    if not ok:
        return JSONResponse(
            {"error": f"synthesis failed for '{intent}' — check LLM router and logs"},
            status_code=503,
        )
    return {"ok": True, "intent": intent, "details": loader.organ_details(intent)}


@router.get("/organs/{name}")
async def organs_get(name: str):
    """Get details for a single organ by intent name."""
    loader = _get_organ_loader()
    if loader is None:
        return JSONResponse({"error": "organ_loader not initialised"}, status_code=503)
    details = loader.organ_details(name)
    if details is None:
        return JSONResponse({"error": f"organ '{name}' not found"}, status_code=404)
    return details


@router.post("/organs/{name}/enable")
async def organs_enable(name: str):
    """Re-enable a disabled organ."""
    loader = _get_organ_loader()
    if loader is None:
        return JSONResponse({"error": "organ_loader not initialised"}, status_code=503)
    if name not in loader._organs:
        return JSONResponse({"error": f"organ '{name}' not found"}, status_code=404)
    loader.enable(name)
    return {"ok": True, "intent": name, "enabled": True}


@router.post("/organs/{name}/disable")
async def organs_disable(name: str):
    """Disable an organ without removing it from disk."""
    loader = _get_organ_loader()
    if loader is None:
        return JSONResponse({"error": "organ_loader not initialised"}, status_code=503)
    ok = loader.disable(name)
    if not ok:
        return JSONResponse({"error": f"organ '{name}' not found"}, status_code=404)
    return {"ok": True, "intent": name, "enabled": False}


@router.delete("/organs/{name}")
async def organs_delete(name: str):
    """Delete a user-synthesized organ. Bundled organs cannot be deleted."""
    loader = _get_organ_loader()
    if loader is None:
        return JSONResponse({"error": "organ_loader not initialised"}, status_code=503)
    if name not in loader._organs:
        return JSONResponse({"error": f"organ '{name}' not found"}, status_code=404)
    ok = loader.delete_user_organ(name)
    if not ok:
        return JSONResponse(
            {"error": f"'{name}' is a bundled organ and cannot be deleted"},
            status_code=403,
        )
    return {"ok": True, "intent": name, "deleted": True}
