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

import os
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from prism_state import _state

router = APIRouter()

_503 = JSONResponse(
    {"error": "FederationManager not available", "status": 503},
    status_code=503,
)

_401 = JSONResponse(
    {"error": "Unauthorized", "status": 401},
    status_code=401,
)


def _fed():
    return _state.get("federation")


def _require_federation_auth(request: Request) -> bool:
    """Return True if the request is authorized (or auth is not configured)."""
    token = os.environ.get("PRISM_FEDERATION_TOKEN", "")
    if not token:
        return True  # no token configured — allow unauthenticated
    auth_header = request.headers.get("Authorization", "")
    if auth_header == f"Bearer {token}":
        return True
    return False


# ---------------------------------------------------------------------------
# POST /federation/announce
# ---------------------------------------------------------------------------


@router.post("/federation/announce")
async def federation_announce(request: Request):
    """Register this node with a peer.

    Body: ``{url: str, name: str}``
    Returns: ``{node_id: str, peers: list}``
    """
    if not _require_federation_auth(request):
        return _401
    fm = _fed()
    if fm is None:
        return _503

    body: dict[str, Any] = {}
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
    if not _require_federation_auth(request):
        return _401
    fm = _fed()
    if fm is None:
        return _503

    body: dict[str, Any] = {}
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


# ---------------------------------------------------------------------------
# GET /federation/identity — export identity payload for cross-device sync
# ---------------------------------------------------------------------------


@router.get("/federation/identity")
async def federation_identity_get():
    """
    Return a portable identity payload (soul + persona) for syncing to a peer.

    Peers call this endpoint then POST the result to their own
    ``/federation/identity/merge``.
    """
    import time as _time

    from prism_state import _get_agent as _ga
    agent = _ga()
    soul = getattr(agent, "_soul", None) if agent else None
    persona = getattr(agent, "_persona", None) if agent else None
    fm = _fed()

    payload: dict[str, Any] = {
        "node_id": fm.node_id if fm else None,
        "timestamp": _time.time(),
    }

    if soul is not None:
        try:
            payload["soul"] = soul.export_json()
        except Exception:
            payload["soul"] = None
    else:
        payload["soul"] = None

    if persona is not None:
        try:
            traits = persona.list_traits()
            payload["persona"] = {
                "traits": [
                    {
                        "name": t.name,
                        "value": t.value,
                        "confidence": t.confidence,
                        "source": t.source,
                        "observation_count": t.observation_count,
                    }
                    for t in traits
                ],
            }
        except Exception:
            payload["persona"] = None
    else:
        payload["persona"] = None

    return payload


# ---------------------------------------------------------------------------
# POST /federation/identity/merge — merge peer identity payload
# ---------------------------------------------------------------------------


@router.post("/federation/identity/merge")
async def federation_identity_merge(request: Request):
    """
    Merge an identity payload received from a peer.

    Body: the dict returned by the peer's ``GET /federation/identity``.

    Soul beliefs are merged belief-by-belief (higher confidence wins).
    Persona traits are upserted (higher confidence wins, source set to
    ``"federated"``).
    """
    if not _require_federation_auth(request):
        return _401
    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    if not body:
        return JSONResponse({"error": "empty payload"}, status_code=400)

    from prism_state import _get_agent as _ga
    agent = _ga()
    soul = getattr(agent, "_soul", None) if agent else None
    persona = getattr(agent, "_persona", None) if agent else None

    merged_beliefs = 0
    merged_traits = 0

    # Soul merge — belief-by-belief, higher confidence wins
    soul_payload = body.get("soul") or {}
    if soul is not None and soul_payload:
        remote_beliefs = soul_payload.get("beliefs", [])
        try:
            existing = {b.text.lower(): b for b in soul.list_beliefs()}
        except Exception:
            existing = {}

        for rb in remote_beliefs:
            text = rb.get("text", "").strip()
            if not text:
                continue
            try:
                match = existing.get(text.lower())
                if match:
                    if rb.get("confidence", 0) > match.confidence:
                        soul.update_belief(
                            match.node_id,
                            rb["confidence"],
                            notes="federated from peer",
                        )
                        merged_beliefs += 1
                else:
                    soul.add_belief(
                        text,
                        belief_type=rb.get("belief_type", "value"),
                        source="federated",
                        confidence=rb.get("confidence", 0.5),
                    )
                    merged_beliefs += 1
            except Exception:
                pass

    # Persona merge — trait upsert, higher confidence wins
    persona_payload = body.get("persona") or {}
    if persona is not None and persona_payload:
        for rt in persona_payload.get("traits", []):
            name = rt.get("name", "").strip()
            if not name:
                continue
            try:
                existing_trait = persona.get_trait(name)
                if existing_trait is None or rt.get("confidence", 0) > existing_trait.confidence:
                    persona.update_trait(
                        name=name,
                        value=rt.get("value", ""),
                        confidence=rt.get("confidence", 0.5),
                        source="federated",
                    )
                    merged_traits += 1
            except Exception:
                pass

    return {
        "ok": True,
        "peer_node_id": body.get("node_id"),
        "merged_beliefs": merged_beliefs,
        "merged_traits": merged_traits,
    }
