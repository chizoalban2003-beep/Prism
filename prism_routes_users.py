"""
prism_routes_users.py
=====================
FastAPI router for multi-user home hub and household bus endpoints.

Routes
------
GET  /users                    — list all registered users
POST /users                    — register a new user {user_id, name, role}
DELETE /users/{user_id}        — remove a user
POST /users/{user_id}/activate — switch active user context in _state["agent"]
GET  /household/signals        — recent household bus signals (last N)
POST /household/broadcast      — emit signal to all users {signal_type, payload}
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from prism_state import _state

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _registry():
    return _state.get("user_registry")


def _household_bus():
    return _state.get("household_bus")


def _no_registry() -> JSONResponse:
    return JSONResponse(
        {"error": "UserRegistry not available — daemon not started", "status": 503},
        status_code=503,
    )


# ---------------------------------------------------------------------------
# GET /users
# ---------------------------------------------------------------------------


@router.get("/users")
async def list_users():
    reg = _registry()
    if reg is None:
        return _no_registry()

    profiles = reg.list_users()
    return {
        "total": len(profiles),
        "users": [p.to_dict() for p in profiles],
    }


# ---------------------------------------------------------------------------
# POST /users
# ---------------------------------------------------------------------------


@router.post("/users")
async def register_user(request: Request):
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    reg = _registry()
    if reg is None:
        return _no_registry()

    user_id = body.get("user_id", "").strip()
    name = body.get("name", "").strip()
    role = body.get("role", "member").strip()

    if not user_id or not name:
        return JSONResponse(
            {"error": "'user_id' and 'name' are required", "status": 400},
            status_code=400,
        )

    try:
        profile = reg.register(user_id=user_id, name=name, role=role)
    except ValueError as exc:
        return JSONResponse(
            {"error": str(exc), "status": 409},
            status_code=409,
        )

    return {"ok": True, "user": profile.to_dict()}


# ---------------------------------------------------------------------------
# DELETE /users/{user_id}
# ---------------------------------------------------------------------------


@router.delete("/users/{user_id}")
async def remove_user(user_id: str):
    reg = _registry()
    if reg is None:
        return _no_registry()

    removed = reg.remove(user_id)
    if not removed:
        return JSONResponse(
            {"error": f"User {user_id!r} not found", "status": 404},
            status_code=404,
        )

    return {"ok": True, "user_id": user_id}


# ---------------------------------------------------------------------------
# POST /users/{user_id}/activate
# ---------------------------------------------------------------------------


@router.post("/users/{user_id}/activate")
async def activate_user(user_id: str):
    reg = _registry()
    if reg is None:
        return _no_registry()

    profile = reg.get(user_id)
    if profile is None:
        return JSONResponse(
            {"error": f"User {user_id!r} not found", "status": 404},
            status_code=404,
        )

    # Switch the active user context on the agent when available
    agent = _state.get("agent")
    if agent is not None:
        try:
            agent._active_user = user_id
        except Exception:
            pass

    # Store in state so other routers can see who is active
    _state["active_user_id"] = user_id

    reg.touch(user_id)
    return {"ok": True, "active_user_id": user_id, "profile": profile.to_dict()}


# ---------------------------------------------------------------------------
# GET /household/signals
# ---------------------------------------------------------------------------


@router.get("/household/signals")
async def household_signals(n: int = 20):
    bus = _household_bus()

    # Fallback to OrganBus history when HouseholdBus not yet wired
    if bus is None:
        reg = _registry()
        if reg is None:
            return _no_registry()
        return {"signals": [], "total": 0}

    signals = bus.signal_history(n=n)
    return {"signals": signals, "total": len(signals)}


# ---------------------------------------------------------------------------
# POST /household/broadcast
# ---------------------------------------------------------------------------


@router.post("/household/broadcast")
async def household_broadcast(request: Request):
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    reg = _registry()
    if reg is None:
        return _no_registry()

    signal_type = body.get("signal_type", "").strip()
    payload = body.get("payload", {})
    source = body.get("source", "household_api")

    if not signal_type:
        return JSONResponse(
            {"error": "'signal_type' is required", "status": 400},
            status_code=400,
        )

    if not isinstance(payload, dict):
        payload = {"value": payload}

    from prism_organ_bus import OrganSignal  # noqa: PLC0415

    signal = OrganSignal(
        source=source,
        signal_type=signal_type,
        payload=payload,
    )

    bus = _household_bus()
    if bus is None:
        # No HouseholdBus wired — record that we would have broadcast
        return {
            "ok": True,
            "signal_id": signal.signal_id,
            "signal_type": signal_type,
            "note": "HouseholdBus not wired — signal not delivered",
            "results": {},
        }

    results = bus.broadcast(signal)
    return {
        "ok": True,
        "signal_id": signal.signal_id,
        "signal_type": signal_type,
        "results": results,
    }
