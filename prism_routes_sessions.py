"""
prism_routes_sessions.py
========================
FastAPI router for named conversation session endpoints.

Routes:
  GET    /sessions                         list all sessions
  POST   /sessions                         create a session
  GET    /sessions/active                  get active session id
  POST   /sessions/active                  set active session
  GET    /sessions/{session_id}            get session metadata
  PATCH  /sessions/{session_id}            update name/description/tags
  DELETE /sessions/{session_id}            delete session + all messages
  GET    /sessions/{session_id}/history    get messages (query: n=50)
  POST   /sessions/{session_id}/messages   add a message
  DELETE /sessions/{session_id}/history    clear messages
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from prism_state import _state

router = APIRouter()


def _get_session_manager():
    try:
        from prism_session_manager import get_session_manager
        return get_session_manager()
    except Exception:
        return None


def _session_to_dict(s) -> dict:
    return {
        "session_id":    s.session_id,
        "name":          s.name,
        "description":   s.description,
        "tags":          s.tags,
        "created_at":    s.created_at,
        "updated_at":    s.updated_at,
        "message_count": s.message_count,
    }


def _message_to_dict(m) -> dict:
    return {
        "message_id": m.message_id,
        "session_id": m.session_id,
        "role":       m.role,
        "content":    m.content,
        "timestamp":  m.timestamp,
    }


# ---------------------------------------------------------------------------
# GET /sessions
# ---------------------------------------------------------------------------

@router.get("/sessions")
async def list_sessions(limit: int = 50, offset: int = 0):
    sm = _get_session_manager()
    if sm is None:
        return JSONResponse({"error": "session manager unavailable", "status": 503}, status_code=503)
    sessions = sm.list_sessions(limit=limit, offset=offset)
    return {"sessions": [_session_to_dict(s) for s in sessions], "total": len(sessions)}


# ---------------------------------------------------------------------------
# POST /sessions
# ---------------------------------------------------------------------------

@router.post("/sessions")
async def create_session(request: Request):
    sm = _get_session_manager()
    if sm is None:
        return JSONResponse({"error": "session manager unavailable", "status": 503}, status_code=503)

    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    name = body.get("name", "")
    if not name:
        return JSONResponse({"error": "'name' field required", "status": 400}, status_code=400)

    session = sm.create_session(
        name=name,
        description=body.get("description", ""),
        tags=body.get("tags", []),
    )
    return _session_to_dict(session)


# ---------------------------------------------------------------------------
# GET /sessions/active  (must be before GET /sessions/{session_id})
# ---------------------------------------------------------------------------

@router.get("/sessions/active")
async def get_active_session():
    session_id = _state.get("active_session_id")
    if not session_id:
        return {"active_session_id": None}

    sm = _get_session_manager()
    if sm is None:
        return {"active_session_id": session_id}

    session = sm.get_session(session_id)
    return {
        "active_session_id": session_id,
        "session": _session_to_dict(session) if session else None,
    }


# ---------------------------------------------------------------------------
# POST /sessions/active  (must be before GET /sessions/{session_id})
# ---------------------------------------------------------------------------

@router.post("/sessions/active")
async def set_active_session(request: Request):
    sm = _get_session_manager()
    if sm is None:
        return JSONResponse({"error": "session manager unavailable", "status": 503}, status_code=503)

    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    session_id: Optional[str] = body.get("session_id")
    if not session_id:
        return JSONResponse({"error": "'session_id' field required", "status": 400}, status_code=400)

    session = sm.get_session(session_id)
    if session is None:
        return JSONResponse({"error": "session not found", "status": 404}, status_code=404)

    _state["active_session_id"] = session_id
    return {"active_session_id": session_id, "session": _session_to_dict(session)}


# ---------------------------------------------------------------------------
# GET /sessions/{session_id}
# ---------------------------------------------------------------------------

@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    sm = _get_session_manager()
    if sm is None:
        return JSONResponse({"error": "session manager unavailable", "status": 503}, status_code=503)

    session = sm.get_session(session_id)
    if session is None:
        return JSONResponse({"error": "session not found", "status": 404}, status_code=404)

    return _session_to_dict(session)


# ---------------------------------------------------------------------------
# PATCH /sessions/{session_id}
# ---------------------------------------------------------------------------

@router.patch("/sessions/{session_id}")
async def update_session(session_id: str, request: Request):
    sm = _get_session_manager()
    if sm is None:
        return JSONResponse({"error": "session manager unavailable", "status": 503}, status_code=503)

    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    session = sm.update_session(
        session_id=session_id,
        name=body.get("name"),
        description=body.get("description"),
        tags=body.get("tags"),
    )
    if session is None:
        return JSONResponse({"error": "session not found", "status": 404}, status_code=404)

    return _session_to_dict(session)


# ---------------------------------------------------------------------------
# DELETE /sessions/{session_id}
# ---------------------------------------------------------------------------

@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    sm = _get_session_manager()
    if sm is None:
        return JSONResponse({"error": "session manager unavailable", "status": 503}, status_code=503)

    deleted = sm.delete_session(session_id)
    if not deleted:
        return JSONResponse({"error": "session not found", "status": 404}, status_code=404)

    # Clear active session if it was the deleted one
    if _state.get("active_session_id") == session_id:
        _state.pop("active_session_id", None)

    return {"deleted": True, "session_id": session_id}


# ---------------------------------------------------------------------------
# GET /sessions/{session_id}/history
# ---------------------------------------------------------------------------

@router.get("/sessions/{session_id}/history")
async def get_history(session_id: str, n: int = 50, offset: int = 0):
    sm = _get_session_manager()
    if sm is None:
        return JSONResponse({"error": "session manager unavailable", "status": 503}, status_code=503)

    session = sm.get_session(session_id)
    if session is None:
        return JSONResponse({"error": "session not found", "status": 404}, status_code=404)

    messages = sm.get_history(session_id, n=n, offset=offset)
    return {"messages": [_message_to_dict(m) for m in messages], "total": len(messages)}


# ---------------------------------------------------------------------------
# POST /sessions/{session_id}/messages
# ---------------------------------------------------------------------------

@router.post("/sessions/{session_id}/messages")
async def add_message(session_id: str, request: Request):
    sm = _get_session_manager()
    if sm is None:
        return JSONResponse({"error": "session manager unavailable", "status": 503}, status_code=503)

    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    role = body.get("role", "user")
    content = body.get("content", "")
    if not content:
        return JSONResponse({"error": "'content' field required", "status": 400}, status_code=400)

    record = sm.add_message(session_id, role=role, content=content)
    if record is None:
        return JSONResponse({"error": "session not found", "status": 404}, status_code=404)

    return _message_to_dict(record)


# ---------------------------------------------------------------------------
# DELETE /sessions/{session_id}/history
# ---------------------------------------------------------------------------

@router.delete("/sessions/{session_id}/history")
async def clear_history(session_id: str):
    sm = _get_session_manager()
    if sm is None:
        return JSONResponse({"error": "session manager unavailable", "status": 503}, status_code=503)

    session = sm.get_session(session_id)
    if session is None:
        return JSONResponse({"error": "session not found", "status": 404}, status_code=404)

    count = sm.clear_history(session_id)
    return {"deleted": count, "session_id": session_id}
