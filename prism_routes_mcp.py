"""
prism_routes_mcp.py
===================
FastAPI APIRouter for Model Context Protocol (MCP) client management.

Routes
------
GET  /mcp/status          — per-server connection status
GET  /mcp/servers         — configured server names
GET  /mcp/tools           — all tools across connected servers
POST /mcp/connect         — connect one server ({"name": ...}) or all
POST /mcp/call            — call a tool ({"server", "tool", "arguments"})
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from prism_state import _state

router = APIRouter()


def _manager():
    mgr = _state.get("mcp")
    if mgr is not None:
        return mgr
    try:
        import prism_mcp
        return prism_mcp.get_manager()
    except Exception:
        return None


@router.get("/mcp/status")
async def mcp_status():
    mgr = _manager()
    if mgr is None:
        return {"enabled": False, "servers": []}
    return {"enabled": bool(mgr.server_names()), "servers": mgr.status()}


@router.get("/mcp/servers")
async def mcp_servers():
    mgr = _manager()
    return {"servers": mgr.server_names() if mgr else []}


@router.get("/mcp/tools")
async def mcp_tools():
    mgr = _manager()
    tools = mgr.list_tools() if mgr else []
    return {"tools": tools, "count": len(tools)}


@router.post("/mcp/connect")
async def mcp_connect(request: Request):
    mgr = _manager()
    if mgr is None:
        return JSONResponse({"error": "MCP not available"}, status_code=503)
    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass
    name = str(body.get("name", "")).strip()
    try:
        import asyncio
        if name:
            result = await asyncio.to_thread(mgr.connect, name)
        else:
            result = await asyncio.to_thread(mgr.connect_all)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    # Re-register organs so newly-discovered tools become chat-routable.
    try:
        import prism_mcp
        from prism_state import _get_agent
        agent = _get_agent()
        loader = getattr(agent, "_organ_loader", None) if agent else None
        if loader is not None:
            prism_mcp.register_mcp_organs(loader, mgr)
    except Exception:
        pass
    return {"ok": True, "result": result}


@router.post("/mcp/call")
async def mcp_call(request: Request):
    mgr = _manager()
    if mgr is None:
        return JSONResponse({"error": "MCP not available"}, status_code=503)
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    server = str(body.get("server", "")).strip()
    tool = str(body.get("tool", "")).strip()
    arguments = body.get("arguments") or {}
    if not server or not tool:
        return JSONResponse(
            {"error": "'server' and 'tool' are required"}, status_code=400
        )
    if not isinstance(arguments, dict):
        return JSONResponse({"error": "'arguments' must be an object"}, status_code=400)
    try:
        import asyncio

        import prism_mcp
        result = await asyncio.to_thread(mgr.call_tool, server, tool, arguments)
        return {"ok": True, "result": result, "text": prism_mcp.extract_text(result)}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
