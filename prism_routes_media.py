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
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from prism_state import _get_agent

router = APIRouter()


# Sandbox for client-supplied audio paths. /voice/transcribe accepts a `path`
# field from JSON requests; without this guard a caller could read arbitrary
# files (e.g. /etc/passwd) via the voice subsystem.
_AUDIO_SANDBOX = Path(
    os.environ.get("PRISM_AUDIO_SANDBOX")
    or (Path.home() / ".prism" / "uploads")
).expanduser().resolve()


def _validate_audio_path(raw: str) -> Path | None:
    """Resolve `raw` and confirm it lives under the audio sandbox.

    Returns the resolved Path on success, or None if the path escapes the
    sandbox or cannot be resolved.
    """
    try:
        resolved = Path(raw).expanduser().resolve(strict=False)
    except (OSError, ValueError):
        return None
    try:
        resolved.relative_to(_AUDIO_SANDBOX)
    except ValueError:
        return None
    return resolved




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
        body: dict[str, Any] = {}
        try:
            body = await request.json()
        except Exception:
            pass
        audio_path_raw = body.get("path", "")
        if not audio_path_raw:
            return JSONResponse(
                {"error": "Provide {'path': '/path/to/audio'} or raw audio bytes", "status": 400},
                status_code=400,
            )
        resolved = _validate_audio_path(audio_path_raw)
        if resolved is None:
            return JSONResponse(
                {"error": f"audio path must be under {_AUDIO_SANDBOX}", "status": 400},
                status_code=400,
            )
        audio_path = str(resolved)
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
    body: dict[str, Any] = {}
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
    body: dict[str, Any] = {}
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
