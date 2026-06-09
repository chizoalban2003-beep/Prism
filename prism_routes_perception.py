"""
prism_routes_perception.py
==========================
FastAPI APIRouter for visual and audio perception endpoints.

Routes
------
POST /perception/visual          Analyse a base64-encoded image; returns SceneAnalysis.
POST /perception/audio           Analyse base64-encoded PCM audio; returns AudioFeatures.
GET  /perception/history         Recent scene analyses (last N=20).
POST /perception/visual/file     Multipart image upload; returns SceneAnalysis.

All routes resolve the VisualPerception instance from _state.get("visual_perception")
and return HTTP 503 if it is not configured.
"""
from __future__ import annotations

import base64
import logging
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse

from prism_state import _get_agent, _state

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_vp():
    return _state.get("visual_perception")


def _get_aa():
    aa = _state.get("audio_analyzer")
    if aa is None:
        try:
            from prism_visual_perception import AudioAnalyzer
            aa = AudioAnalyzer()
        except ImportError:
            return None
    return aa


def _503(msg: str = "visual_perception not configured") -> JSONResponse:
    return JSONResponse({"error": msg, "status": 503}, status_code=503)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/perception/visual")
async def perception_visual(request: Request):
    """Analyse a base64-encoded image. Body: {image_b64, source?}"""
    vp = _get_vp()
    if vp is None:
        return _503()
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        body = {}
    image_b64: str = body.get("image_b64", "")
    source: str    = body.get("source", "camera")
    if not image_b64:
        return JSONResponse({"error": "'image_b64' is required"}, status_code=400)
    scene = vp.analyse_frame_base64(image_b64, source=source)
    return asdict(scene)


@router.post("/perception/audio")
async def perception_audio(request: Request):
    """Analyse base64-encoded 16-bit mono PCM audio. Body: {audio_b64}"""
    vp = _get_vp()
    if vp is None:
        return _503()
    aa = _get_aa()
    if aa is None:
        return _503("audio_analyzer not available")
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        body = {}
    audio_b64: str = body.get("audio_b64", "")
    if not audio_b64:
        return JSONResponse({"error": "'audio_b64' is required"}, status_code=400)
    try:
        raw = base64.b64decode(audio_b64)
    except Exception as exc:
        return JSONResponse({"error": f"base64 decode failed: {exc}"}, status_code=400)
    features = aa.extract_features(audio_bytes=raw)
    return asdict(features)


@router.get("/perception/history")
async def perception_history(n: int = 20):
    """Return the last n SceneAnalysis records (default 20)."""
    vp = _get_vp()
    if vp is None:
        return _503()
    scenes = vp.recent_history(n=n)
    return {"scenes": [asdict(s) for s in scenes], "count": len(scenes)}


@router.post("/perception/visual/file")
async def perception_visual_file(file: UploadFile = File(...), source: str = "image_file"):
    """Accept a multipart image upload and return a SceneAnalysis."""
    vp = _get_vp()
    if vp is None:
        return _503()
    raw = await file.read()
    if not raw:
        return JSONResponse({"error": "uploaded file is empty"}, status_code=400)
    scene = vp.analyse_image(image_bytes=raw, source=source)
    return asdict(scene)


@router.post("/perception/visual/reason")
async def perception_visual_reason(request: Request):
    """Ask the LLM a question about a base64-encoded image.

    Body: {"image_b64": "...", "question": "..."}
    Returns: {"answer": "...", "question": "...", "model_used": "..."}
    """
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        body = {}

    image_b64: str = body.get("image_b64", "")
    question: str = body.get("question", "What is in this image?")

    if not image_b64:
        return JSONResponse({"error": "'image_b64' is required"}, status_code=400)

    agent = _get_agent()
    if agent is None:
        return JSONResponse({"error": "agent not ready"}, status_code=503)

    router = getattr(agent, "_router", None)
    if router is None:
        return JSONResponse({"error": "LLM router not available"}, status_code=503)

    try:
        answer, model_used = router.call(
            prompt=question,
            images=[image_b64],
            min_capability=2,
            max_tokens=800,
            system="You are a visual assistant. Answer questions about images clearly and concisely.",
        )
    except Exception as exc:
        logger.warning("perception_visual_reason LLM call failed: %s", exc)
        return JSONResponse({"error": f"LLM call failed: {exc}"}, status_code=500)

    return {"answer": answer, "question": question, "model_used": model_used}
