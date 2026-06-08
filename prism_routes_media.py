"""
prism_routes_media.py
=====================
FastAPI router for voice and TTS endpoints.

Routes:
  POST /voice/transcribe
  POST /voice/status
  POST /tts
  POST /tts/speak
"""
from __future__ import annotations

import os
import tempfile
from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from prism_state import _state

router = APIRouter()


def _get_agent():
    return _state.get("agent")


# ---------------------------------------------------------------------------
# POST /voice/transcribe
# ---------------------------------------------------------------------------

@router.post("/voice/transcribe")
async def voice_transcribe(request: Request):
    # Determine if body is JSON or raw audio bytes
    content_type = request.headers.get("content-type", "")

    agent = _get_agent()
    if agent is None:
        try:
            from prism_agent import PrismAgent
            agent = PrismAgent()
        except ImportError as exc:
            return JSONResponse({"error": str(exc), "status": 503}, status_code=503)

    audio_path = None
    raw_bytes  = None

    if "application/json" in content_type:
        body: Dict[str, Any] = {}
        try:
            body = await request.json()
        except Exception:
            pass
        audio_path = body.get("path", "")
        if not audio_path:
            return JSONResponse(
                {"error": "Provide {'path': '/path/to/audio'} or raw audio bytes", "status": 400},
                status_code=400,
            )
    else:
        # Raw audio bytes
        raw_bytes = await request.body()
        ext = {
            "audio/mpeg": ".mp3",
            "audio/flac": ".flac",
            "audio/ogg":  ".ogg",
        }.get(content_type, ".wav")
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        tmp.write(raw_bytes if raw_bytes else b"")
        tmp.close()
        audio_path = tmp.name

    if not audio_path:
        return JSONResponse(
            {"error": "Provide {'path': '/path/to/audio'} or raw audio bytes", "status": 400},
            status_code=400,
        )

    try:
        text = agent._voice.transcribe(audio_path)
        return {"transcript": text, "path": audio_path}
    finally:
        if raw_bytes is not None:
            try:
                os.unlink(audio_path)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# POST /voice/status
# ---------------------------------------------------------------------------

@router.post("/voice/status")
async def voice_status():
    agent = _get_agent()
    if agent is None:
        try:
            from prism_agent import PrismAgent
            agent = PrismAgent()
        except ImportError as exc:
            return JSONResponse({"error": str(exc), "status": 503}, status_code=503)

    v = agent._voice
    return {
        "enabled":    v._enabled,
        "available":  v.available,
        "can_record": v.can_record,
        "backend":    v._backend or "none",
        "record_lib": v._record_lib or "none",
        "model":      v._model_size,
        "language":   v._language,
    }


# ---------------------------------------------------------------------------
# POST /tts
# ---------------------------------------------------------------------------

@router.post("/tts")
async def tts(request: Request):
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    agent = _get_agent()
    tts_  = getattr(agent, "_tts", None) if agent else None
    action = body.get("action", "speak")

    if action == "toggle":
        if tts_:
            enabled = tts_.toggle()
        else:
            enabled = False
        return {"enabled": enabled}
    elif action == "speak":
        text = body.get("text", "")
        if tts_ and text:
            tts_.speak(text)
        return {"ok": True}
    else:
        return JSONResponse(
            {"error": f"Unknown TTS action: {action}", "status": 400}, status_code=400
        )


# ---------------------------------------------------------------------------
# POST /tts/speak
# ---------------------------------------------------------------------------

@router.post("/tts/speak")
async def tts_speak(request: Request):
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    agent = _get_agent()
    tts_  = getattr(agent, "_tts", None) if agent else None
    text  = body.get("text", "")
    if tts_ and text:
        tts_.speak(text)
    return {"ok": True}
