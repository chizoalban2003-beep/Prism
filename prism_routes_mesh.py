"""
prism_routes_mesh.py
====================
FastAPI router for device mesh / multi-device orchestration.

Routes:
  GET  /mesh/peers
  POST /mesh/register
  POST /mesh/remove
  POST /mesh/refresh
  POST /mesh/forward
"""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from prism_mesh import get_mesh

router = APIRouter()


def _peer_dict(p) -> dict:
    return {
        "peer_id":      p.peer_id,
        "name":         p.name,
        "host":         p.host,
        "port":         p.port,
        "capabilities": p.capabilities,
        "last_seen":    p.last_seen,
        "added_at":     p.added_at,
        "online":       (p.last_seen > 0),
    }


@router.get("/mesh/peers")
async def mesh_peers():
    mesh = get_mesh()
    return {"peers": [_peer_dict(p) for p in mesh.list_peers()]}


@router.post("/mesh/register")
async def mesh_register(request: Request):
    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    name  = (body.get("name") or "").strip()
    host  = (body.get("host") or "").strip()
    port  = int(body.get("port") or 8742)
    token = (body.get("token") or "").strip()

    if not host:
        return JSONResponse({"error": "'host' required", "status": 400}, status_code=400)

    mesh = get_mesh()
    peer = await asyncio.to_thread(mesh.register_peer, name, host, port, token)
    return {"ok": True, "peer": _peer_dict(peer)}


@router.post("/mesh/remove")
async def mesh_remove(request: Request):
    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    peer_id = (body.get("peer_id") or "").strip()
    if not peer_id:
        return JSONResponse({"error": "'peer_id' required", "status": 400}, status_code=400)
    ok = get_mesh().remove_peer(peer_id)
    return {"ok": ok}


@router.post("/mesh/refresh")
async def mesh_refresh(request: Request):
    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    peer_id = (body.get("peer_id") or "").strip()
    mesh = get_mesh()
    if peer_id:
        caps = await asyncio.to_thread(mesh.refresh_capabilities, peer_id)
        return {"ok": True, "peer_id": peer_id, "capabilities": caps}
    refreshed = []
    for p in mesh.list_peers():
        caps = await asyncio.to_thread(mesh.refresh_capabilities, p.peer_id)
        refreshed.append({"peer_id": p.peer_id, "name": p.name, "capabilities": caps})
    return {"ok": True, "refreshed": refreshed}


@router.post("/mesh/forward")
async def mesh_forward(request: Request):
    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    peer_id = (body.get("peer_id") or "").strip()
    task    = (body.get("task") or "").strip()
    params  = body.get("params") or {}
    dry_run = bool(body.get("dry_run", False))
    message = (body.get("message") or "").strip()

    if not peer_id:
        return JSONResponse({"error": "'peer_id' required", "status": 400}, status_code=400)

    mesh = get_mesh()
    if message and not task:
        result = await asyncio.to_thread(mesh.forward_chat, peer_id, message)
        return {"ok": True, "via": peer_id, "result": result}
    if not task:
        return JSONResponse({"error": "'task' or 'message' required", "status": 400}, status_code=400)
    result = await asyncio.to_thread(mesh.forward_task, peer_id, task, params, dry_run)
    return {"ok": True, "via": peer_id, "result": result}
