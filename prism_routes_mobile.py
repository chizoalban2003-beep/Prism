"""
prism_routes_mobile.py
======================
FastAPI APIRouter for mobile client sync endpoints.

Routes:
  POST /mobile/register        — register a mobile client; returns {sync_token}
  GET  /mobile/sync            — get sync state (X-Device-ID + X-Sync-Token required)
  POST /mobile/health_data     — ingest health metrics
  POST /mobile/push_token      — register FCM/APNS push token
  GET  /mobile/notifications   — get pending push payloads for a device

All routes resolve MobileSyncManager from _state["mobile_sync"].
Protected routes require X-Device-ID and X-Sync-Token headers.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse

from prism_state import _state

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_manager():
    """Return the MobileSyncManager from shared state, or None."""
    return _state.get("mobile_sync")


def _unavailable():
    return JSONResponse(
        {"error": "Mobile sync not initialised", "status": 503},
        status_code=503,
    )


def _unauthorised(detail: str = "Invalid or missing credentials"):
    return JSONResponse({"error": detail, "status": 401}, status_code=401)


def _auth(
    mgr,
    x_device_id: str | None,
    x_sync_token: str | None,
) -> bool:
    """Return True if token is valid for the given device_id."""
    if not x_device_id or not x_sync_token:
        return False
    return mgr.verify_token(x_device_id, x_sync_token)


# ---------------------------------------------------------------------------
# POST /mobile/register
# ---------------------------------------------------------------------------


@router.post("/mobile/register")
async def mobile_register(request: Request):
    """Register a new mobile client. Body: {device_id, name, platform}."""
    mgr = _get_manager()
    if mgr is None:
        return _unavailable()

    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    device_id = body.get("device_id", "")
    name      = body.get("name", "")
    platform  = body.get("platform", "")

    if not device_id:
        return JSONResponse(
            {"error": "'device_id' is required", "status": 400}, status_code=400
        )

    token = mgr.register_client(device_id, name, platform)
    return {"sync_token": token, "device_id": device_id}


# ---------------------------------------------------------------------------
# GET /mobile/sync
# ---------------------------------------------------------------------------


@router.get("/mobile/sync")
async def mobile_sync(
    x_device_id: str | None = Header(default=None, alias="X-Device-ID"),
    x_sync_token: str | None = Header(default=None, alias="X-Sync-Token"),
):
    """Return sync state for authenticated device."""
    mgr = _get_manager()
    if mgr is None:
        return _unavailable()

    if not _auth(mgr, x_device_id, x_sync_token):
        return _unauthorised()

    return mgr.sync_state(x_device_id)


# ---------------------------------------------------------------------------
# POST /mobile/health_data
# ---------------------------------------------------------------------------


@router.post("/mobile/health_data")
async def mobile_health_data(
    request: Request,
    x_device_id: str | None = Header(default=None, alias="X-Device-ID"),
    x_sync_token: str | None = Header(default=None, alias="X-Sync-Token"),
):
    """
    Ingest health metrics.

    Body: {device_id, metrics: [{metric, value, unit, timestamp}]}
    Authentication: X-Device-ID + X-Sync-Token headers (preferred), or
    device_id in body with valid token.
    """
    mgr = _get_manager()
    if mgr is None:
        return _unavailable()

    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    # Resolve device_id: header takes precedence over body
    device_id = x_device_id or body.get("device_id", "")
    token     = x_sync_token

    if not device_id:
        return JSONResponse(
            {"error": "'device_id' is required (header or body)", "status": 400},
            status_code=400,
        )

    if token and not mgr.verify_token(device_id, token):
        return _unauthorised()

    metrics = body.get("metrics", [])
    if not isinstance(metrics, list):
        return JSONResponse(
            {"error": "'metrics' must be a list", "status": 400}, status_code=400
        )

    inserted = mgr.ingest_health_data(device_id, metrics)
    return {"inserted": inserted, "device_id": device_id}


# ---------------------------------------------------------------------------
# POST /mobile/push_token
# ---------------------------------------------------------------------------


@router.post("/mobile/push_token")
async def mobile_push_token(
    request: Request,
    x_device_id: str | None = Header(default=None, alias="X-Device-ID"),
    x_sync_token: str | None = Header(default=None, alias="X-Sync-Token"),
):
    """Register an FCM or APNS push token. Body: {device_id, push_token}."""
    mgr = _get_manager()
    if mgr is None:
        return _unavailable()

    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    device_id  = x_device_id or body.get("device_id", "")
    push_token = body.get("push_token", "")
    token      = x_sync_token

    if not device_id:
        return JSONResponse(
            {"error": "'device_id' is required", "status": 400}, status_code=400
        )
    if not push_token:
        return JSONResponse(
            {"error": "'push_token' is required", "status": 400}, status_code=400
        )

    if token and not mgr.verify_token(device_id, token):
        return _unauthorised()

    mgr.register_push_token(device_id, push_token)
    return {"ok": True, "device_id": device_id}


# ---------------------------------------------------------------------------
# GET /mobile/notifications
# ---------------------------------------------------------------------------


@router.get("/mobile/notifications")
async def mobile_notifications(
    x_device_id: str | None = Header(default=None, alias="X-Device-ID"),
    x_sync_token: str | None = Header(default=None, alias="X-Sync-Token"),
):
    """Return pending push payloads for an authenticated device."""
    mgr = _get_manager()
    if mgr is None:
        return _unavailable()

    if not _auth(mgr, x_device_id, x_sync_token):
        return _unauthorised()

    notifications = mgr.get_pending_notifications(x_device_id)
    return {"notifications": notifications, "count": len(notifications)}
