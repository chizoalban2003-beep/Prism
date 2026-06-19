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

from typing import Any

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
    body: dict[str, Any] = {}
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
    body: dict[str, Any] = {}
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
    body: dict[str, Any] = {}
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
    body: dict[str, Any] = {}
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


@router.post("/organs/compose")
async def organs_compose(request: Request):
    """Compose a wire diagram for a set of organ intents.

    Body: ``{"intents": ["weather_check", "translate_text", ...]}``

    Returns ``{nodes, arrows, orphans, roots, leaves, has_cycle}``. Arrows
    are drawn by matching organ outputs to organ inputs declared in
    ORGAN_META — the PowerBI-style composition primitive.
    """
    loader = _get_organ_loader()
    if loader is None:
        return JSONResponse({"error": "organ_loader not initialised"}, status_code=503)
    try:
        body: dict = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    intents = body.get("intents") or []
    if not isinstance(intents, list):
        return JSONResponse({"error": "'intents' must be a list"}, status_code=400)
    intents = [str(i).strip() for i in intents if str(i).strip()]
    try:
        from prism_organ_planner import compose, has_cycle
        plan = compose(loader, intents)
        plan["has_cycle"] = has_cycle(plan)
    except Exception as exc:
        return JSONResponse({"error": f"compose failed: {exc}"}, status_code=500)
    return plan


@router.post("/organs/execute")
async def organs_execute(request: Request):
    """Execute a wire diagram in topological order.

    Body:
        ``{"intents": ["fetch_quote", "format_card"],
           "message": "AAPL",
           "ctx": {...},
           "max_steps": 50}``

    Returns ``{order, outputs, errors, skipped, executed}``. Each organ
    is run with ctx["_upstream"][producer_intent] = producer_return so
    downstream organs that declared matching input types can read the
    structured upstream data.
    """
    loader = _get_organ_loader()
    if loader is None:
        return JSONResponse({"error": "organ_loader not initialised"}, status_code=503)
    try:
        body: dict = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    intents = body.get("intents") or []
    if not isinstance(intents, list):
        return JSONResponse({"error": "'intents' must be a list"}, status_code=400)
    intents = [str(i).strip() for i in intents if str(i).strip()]
    message = str(body.get("message", "")).strip()
    raw_ctx = body.get("ctx", {})
    ctx = raw_ctx if isinstance(raw_ctx, dict) else {}
    try:
        max_steps = max(1, min(int(body.get("max_steps", 50)), 200))
    except (TypeError, ValueError):
        max_steps = 50
    try:
        from prism_organ_planner import compose, execute_plan, has_cycle
        plan = compose(loader, intents)
        if has_cycle(plan):
            return JSONResponse(
                {"error": "plan contains a cycle — refuse to execute",
                 "plan": plan},
                status_code=400,
            )
        result = execute_plan(loader, plan, message=message,
                              initial_ctx=ctx, max_steps=max_steps)
    except Exception as exc:
        return JSONResponse({"error": f"execute failed: {exc}"}, status_code=500)

    def _serialize(v):
        if hasattr(v, "model_dump"):
            try:
                return v.model_dump()
            except Exception:
                pass
        if hasattr(v, "__dict__"):
            try:
                return {k: _serialize(x) for k, x in vars(v).items() if not k.startswith("_")}
            except Exception:
                return str(v)
        try:
            import json as _json
            _json.dumps(v)
            return v
        except Exception:
            return str(v)

    result["outputs"] = {k: _serialize(v) for k, v in result["outputs"].items()}
    return result


@router.post("/organs/auto_plan")
async def organs_auto_plan(request: Request):
    """Pick organs for a user message, wire them, and run.

    Body:
        ``{"message": "what's the weather in Lagos and convert to F",
           "ctx": {...},
           "max_organs": 4,
           "execute": true}``

    Returns ``{"intents": [...], "plan": {...}, "execution": {...}}``.
    When ``execute`` is ``false`` only the picked intents and wire diagram
    are returned — useful for previewing the auto-pick in a UI before
    spending budget on running it.
    """
    loader = _get_organ_loader()
    if loader is None:
        return JSONResponse(
            {"error": "organ_loader not initialised"}, status_code=503
        )
    llm_router = _get_llm_router()
    if llm_router is None:
        return JSONResponse(
            {"error": "llm_router not initialised — auto-pick needs an LLM"},
            status_code=503,
        )
    try:
        body: dict = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    message = str(body.get("message", "")).strip()
    if not message:
        return JSONResponse(
            {"error": "'message' is required"}, status_code=400
        )
    raw_ctx = body.get("ctx", {})
    ctx = raw_ctx if isinstance(raw_ctx, dict) else {}
    try:
        max_organs = max(1, min(int(body.get("max_organs", 4)), 8))
    except (TypeError, ValueError):
        max_organs = 4
    execute_flag = bool(body.get("execute", True))

    from prism_organ_planner import (
        auto_select_organs,
        compose,
        execute_plan,
        has_cycle,
    )

    intents = auto_select_organs(loader, message, llm_router, max_organs=max_organs)
    if not intents:
        return {
            "intents":   [],
            "plan":      {"nodes": [], "arrows": [], "roots": [],
                          "leaves": [], "orphans": []},
            "execution": None,
            "note":      "auto-picker returned no organs — fall back to "
                         "single-organ routing or synthesis",
        }

    try:
        plan = compose(loader, intents)
    except Exception as exc:
        return JSONResponse(
            {"error": f"compose failed: {exc}", "intents": intents},
            status_code=500,
        )

    if has_cycle(plan):
        return JSONResponse(
            {"error": "plan contains a cycle — refuse to execute",
             "intents": intents, "plan": plan},
            status_code=400,
        )

    if not execute_flag:
        return {"intents": intents, "plan": plan, "execution": None}

    try:
        result = execute_plan(loader, plan, message=message, initial_ctx=ctx)
    except Exception as exc:
        return JSONResponse(
            {"error": f"execute failed: {exc}",
             "intents": intents, "plan": plan},
            status_code=500,
        )

    def _serialize(v):
        if hasattr(v, "model_dump"):
            try:
                return v.model_dump()
            except Exception:
                pass
        if hasattr(v, "__dict__"):
            try:
                return {k: _serialize(x) for k, x in vars(v).items()
                        if not k.startswith("_")}
            except Exception:
                return str(v)
        try:
            import json as _json
            _json.dumps(v)
            return v
        except Exception:
            return str(v)

    result["outputs"] = {k: _serialize(v) for k, v in result["outputs"].items()}
    return {"intents": intents, "plan": plan, "execution": result}


@router.get("/organs/bundles/index")
async def organs_bundle_index():
    """Return the curated list of installable organ bundles.

    Reads ``~/.prism/bundles/manifest.json`` — a CEO-maintained
    registry, *not* a remote fetch. The manifest is a list of objects:

        [
          {"intent": "stock_quote", "description": "...",
           "sha256": "<hex>", "code": "...",
           "version": "1.0",
           "capabilities": ["internet_read"],
           "source_url": "https://..."},
          ...
        ]

    The endpoint returns the manifest along with which entries are
    *already installed* (intent already known to the loader) so the
    UI can show install/installed/update buttons.
    """
    loader = _get_organ_loader()
    installed_intents: set[str] = set()
    if loader is not None:
        try:
            installed_intents = set(loader.list_organs())
        except Exception:
            installed_intents = set()

    import json
    from pathlib import Path
    manifest_path = Path("~/.prism/bundles/manifest.json").expanduser()
    if not manifest_path.exists():
        return {"bundles": [], "count": 0, "note": "no manifest at ~/.prism/bundles/manifest.json"}
    try:
        raw = manifest_path.read_text()
        data = json.loads(raw)
    except Exception as exc:
        return JSONResponse(
            {"error": f"manifest read failed: {exc}", "path": str(manifest_path)},
            status_code=500,
        )
    if not isinstance(data, list):
        return JSONResponse(
            {"error": "manifest must be a JSON array of bundle objects"},
            status_code=400,
        )

    items: list[dict] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        intent = str(entry.get("intent", "")).strip()
        if not intent:
            continue
        items.append({
            "intent":       intent,
            "description":  str(entry.get("description", "")),
            "version":      str(entry.get("version", "1.0")),
            "sha256":       str(entry.get("sha256", "")),
            "capabilities": list(entry.get("capabilities", []) or []),
            "source_url":   str(entry.get("source_url", "")),
            "installed":    intent in installed_intents,
        })
    return {"bundles": items, "count": len(items), "path": str(manifest_path)}


@router.post("/organs/install")
async def organs_install(request: Request):
    """Install a third-party organ bundle (CEO-controlled plug-in).

    Body:
        ``{"intent": "stock_quote",
           "code": "ORGAN_META = {...}\\ndef execute(...): ...",
           "sha256": "<hex digest of code>"}``

    SHA256 is verified before any code touches disk. AST safety runs in
    strict mode (same checks as synthesis). The bundle is saved to
    ``~/.prism/organs/<intent>.py`` and hot-registered.
    """
    loader = _get_organ_loader()
    if loader is None:
        return JSONResponse({"error": "organ_loader not initialised"}, status_code=503)
    try:
        body: dict = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    intent = str(body.get("intent", "")).strip()
    code   = body.get("code", "")
    digest = str(body.get("sha256", "")).strip().lower()
    if not intent or not isinstance(code, str) or not code:
        return JSONResponse(
            {"error": "'intent' and non-empty 'code' string are required"},
            status_code=400,
        )
    if not digest:
        return JSONResponse({"error": "'sha256' digest is required"}, status_code=400)
    import hashlib
    actual = hashlib.sha256(code.encode("utf-8")).hexdigest()
    if actual != digest:
        return JSONResponse(
            {"error": "sha256 mismatch — bundle integrity check failed",
             "expected": digest, "actual": actual},
            status_code=400,
        )
    try:
        ok = loader.install_bundle(intent, code)
    except Exception as exc:
        return JSONResponse({"error": f"install failed: {exc}"}, status_code=500)
    if not ok:
        return JSONResponse(
            {"error": f"install rejected for '{intent}' — see daemon logs (likely safety or interface violation)"},
            status_code=400,
        )
    return {"ok": True, "intent": intent, "details": loader.organ_details(intent)}


@router.post("/organs/pack/export")
async def organs_pack_export(request: Request):
    """Bundle one or more installed organs into a portable, hash-verified pack.

    Body:
        ``{"intents": ["hacker_news", "currency_convert"],
           "name": "research-tools", "version": "1.0",
           "description": "...", "author": "alice",
           "preview": false}``

    Returns the full pack (``prism.organ-pack/v1``) ready to share, or a
    code-free summary when ``preview=true``.
    """
    loader = _get_organ_loader()
    if loader is None:
        return JSONResponse({"error": "organ_loader not initialised"}, status_code=503)
    try:
        body: dict = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    intents = body.get("intents") or []
    if not isinstance(intents, list) or not intents:
        return JSONResponse(
            {"error": "'intents' must be a non-empty array"}, status_code=400
        )
    import prism_organ_pack as _pack
    try:
        pack = _pack.build_pack(
            loader,
            [str(i) for i in intents],
            name=str(body.get("name", "")),
            version=str(body.get("version", "1.0")),
            description=str(body.get("description", "")),
            author=str(body.get("author", "")),
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": f"export failed: {exc}"}, status_code=500)

    if bool(body.get("preview", False)):
        return _pack.pack_summary(pack)
    return pack


@router.post("/organs/pack/import")
async def organs_pack_import(request: Request):
    """Install every organ in a shared pack through the safe install path.

    Body is either the pack object directly, or
    ``{"pack": {...}, "overwrite": false}``. Each organ is sha256-verified and
    installed via the strict AST/capability-audited loader path. Returns an
    install report ``{ok, installed, skipped, failed}``.
    """
    loader = _get_organ_loader()
    if loader is None:
        return JSONResponse({"error": "organ_loader not initialised"}, status_code=503)
    try:
        body: dict = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    pack = body.get("pack") if isinstance(body, dict) and "pack" in body else body
    overwrite = bool(body.get("overwrite", False)) if isinstance(body, dict) else False
    if not isinstance(pack, dict):
        return JSONResponse({"error": "missing pack object"}, status_code=400)

    import prism_organ_pack as _pack
    ok, reason = _pack.verify_pack(pack)
    if not ok:
        return JSONResponse(
            {"error": f"pack verification failed: {reason}"}, status_code=400
        )
    try:
        report = _pack.import_pack(loader, pack, overwrite=overwrite)
    except Exception as exc:
        return JSONResponse({"error": f"import failed: {exc}"}, status_code=500)
    status = 200 if report.get("ok") else 207
    return JSONResponse(report, status_code=status)


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
