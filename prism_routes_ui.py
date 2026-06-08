"""
prism_routes_ui.py
==================
FastAPI router for web UI / PWA asset endpoints.

Routes:
  GET /          (chat HTML)
  GET /app
  GET /index.html
  GET /mobile
  GET /manifest.json
  GET /sw.js
  GET /icon.svg
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse, Response

router = APIRouter()


# ---------------------------------------------------------------------------
# Chat HTML (/, /app, /index.html)
# ---------------------------------------------------------------------------

@router.get("/", response_class=HTMLResponse)
async def root():
    try:
        from prism_chat import get_chat_html
        return HTMLResponse(content=get_chat_html(), status_code=200)
    except ImportError as exc:
        return HTMLResponse(
            content=f"<html><body><p>prism_chat not available: {exc}</p></body></html>",
            status_code=503,
        )


@router.get("/app", response_class=HTMLResponse)
async def app_page():
    try:
        from prism_chat import get_chat_html
        return HTMLResponse(content=get_chat_html(), status_code=200)
    except ImportError as exc:
        return HTMLResponse(
            content=f"<html><body><p>prism_chat not available: {exc}</p></body></html>",
            status_code=503,
        )


@router.get("/index.html", response_class=HTMLResponse)
async def index_html():
    try:
        from prism_chat import get_chat_html
        return HTMLResponse(content=get_chat_html(), status_code=200)
    except ImportError as exc:
        return HTMLResponse(
            content=f"<html><body><p>prism_chat not available: {exc}</p></body></html>",
            status_code=503,
        )


# ---------------------------------------------------------------------------
# PWA mobile companion
# ---------------------------------------------------------------------------

@router.get("/mobile", response_class=HTMLResponse)
async def mobile():
    try:
        from prism_pwa import get_mobile_html
        return HTMLResponse(content=get_mobile_html(), status_code=200)
    except ImportError as exc:
        return HTMLResponse(
            content=f"<html><body><p>prism_pwa not available: {exc}</p></body></html>",
            status_code=503,
        )


# ---------------------------------------------------------------------------
# Web App Manifest
# ---------------------------------------------------------------------------

@router.get("/manifest.json")
async def manifest():
    try:
        from prism_pwa import get_manifest
        return Response(
            content=get_manifest(),
            media_type="application/manifest+json",
            status_code=200,
        )
    except ImportError as exc:
        return JSONResponse({"error": str(exc), "status": 503}, status_code=503)


# ---------------------------------------------------------------------------
# Service Worker
# ---------------------------------------------------------------------------

@router.get("/sw.js")
async def service_worker():
    try:
        from prism_pwa import get_service_worker
        return Response(
            content=get_service_worker(),
            media_type="application/javascript",
            status_code=200,
        )
    except ImportError as exc:
        return Response(
            content=f"/* prism_pwa not available: {exc} */",
            media_type="application/javascript",
            status_code=503,
        )


# ---------------------------------------------------------------------------
# Icon SVG
# ---------------------------------------------------------------------------

@router.get("/icon.svg")
async def icon_svg():
    try:
        from prism_pwa import get_icon_svg
        return Response(
            content=get_icon_svg(),
            media_type="image/svg+xml",
            status_code=200,
        )
    except ImportError as exc:
        return Response(
            content=f"<!-- prism_pwa not available: {exc} -->",
            media_type="image/svg+xml",
            status_code=503,
        )
