"""
prism_routes_federation.py
==========================
FastAPI APIRouter for the Federated Mesh endpoints.

Routes
------
POST   /federation/announce         — register this node with a peer
GET    /federation/peers             — list known peers
DELETE /federation/peers/{peer_id}  — remove a peer
GET    /federation/sync             — get local state snapshot
POST   /federation/sync             — receive + merge peer state
GET    /federation/status           — sync status (pending peers, vector clock)

All routes return 503 when FederationManager is not available.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from prism_state import _state

router = APIRouter()

_503 = JSONResponse(
    {"error": "FederationManager not available", "status": 503},
    status_code=503,
)


def _fed():
    return _state.get("federation")


# ---------------------------------------------------------------------------
# POST /federation/announce
# ---------------------------------------------------------------------------


@router.post("/federation/announce")
async def federation_announce(request: Request):
    """Register this node with a peer.

    Body: ``{url: str, name: str}``
    Returns: ``{node_id: str, peers: list}``
    """
    fm = _fed()
    if fm is None:
        return _503

    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    url: str = body.get("url", "")
    name: str = body.get("name", "")

    if not url:
        return JSONResponse(
            {"error": "'url' is required", "status": 400}, status_code=400
        )

    node_id = fm.announce(url)

    # If a remote name/url pair is supplied, record them as a peer too
    if name and url:
        # The peer_id defaults to the remote node_id if provided, else a
        # placeholder derived from the URL.
        remote_peer_id: str = body.get("peer_id") or url
        fm.add_peer(remote_peer_id, name, url)

    peers = [
        {
            "peer_id": p.peer_id,
            "name": p.name,
            "url": p.url,
            "last_seen": p.last_seen,
            "sync_version": p.sync_version,
        }
        for p in fm.list_peers()
    ]
    return {"node_id": node_id, "peers": peers}


# ---------------------------------------------------------------------------
# GET /federation/peers
# ---------------------------------------------------------------------------


@router.get("/federation/peers")
async def federation_peers():
    """List all known federation peers."""
    fm = _fed()
    if fm is None:
        return _503

    peers = fm.list_peers()
    return {
        "node_id": fm.node_id,
        "peers": [
            {
                "peer_id": p.peer_id,
                "name": p.name,
                "url": p.url,
                "last_seen": p.last_seen,
                "sync_version": p.sync_version,
            }
            for p in peers
        ],
        "total": len(peers),
    }


# ---------------------------------------------------------------------------
# DELETE /federation/peers/{peer_id}
# ---------------------------------------------------------------------------


@router.delete("/federation/peers/{peer_id}")
async def federation_remove_peer(peer_id: str):
    """Remove a federation peer."""
    fm = _fed()
    if fm is None:
        return _503

    ok = fm.remove_peer(peer_id)
    if not ok:
        return JSONResponse(
            {"error": f"Peer {peer_id!r} not found", "status": 404},
            status_code=404,
        )
    return {"ok": True, "peer_id": peer_id}


# ---------------------------------------------------------------------------
# GET /federation/sync
# ---------------------------------------------------------------------------


@router.get("/federation/sync")
async def federation_sync_get():
    """Return the local state snapshot to send to peers."""
    fm = _fed()
    if fm is None:
        return _503

    return fm.get_sync_payload()


# ---------------------------------------------------------------------------
# POST /federation/sync
# ---------------------------------------------------------------------------


@router.post("/federation/sync")
async def federation_sync_post(request: Request):
    """Receive and merge peer state.

    Body: ``{peer_id: str, payload: dict}``
    Returns: ``{merged_count, conflicts_resolved, peer_version}``
    """
    fm = _fed()
    if fm is None:
        return _503

    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    peer_id: str = body.get("peer_id", "")
    payload: dict = body.get("payload", {})

    if not peer_id:
        return JSONResponse(
            {"error": "'peer_id' is required", "status": 400}, status_code=400
        )
    if not payload:
        return JSONResponse(
            {"error": "'payload' is required", "status": 400}, status_code=400
        )

    result = fm.merge_peer_state(peer_id, payload)
    return result


# ---------------------------------------------------------------------------
# GET /federation/status
# ---------------------------------------------------------------------------


@router.get("/federation/status")
async def federation_status():
    """Return sync status: pending peers, last sync times, vector clock."""
    fm = _fed()
    if fm is None:
        return _503

    return fm.status()
