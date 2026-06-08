"""
prism_asgi.py
=============
FastAPI/ASGI application for PRISM — sole HTTP server on port 8742.

Replaces kde_server.py (retired after Phase 7 migration).

SECURITY: Always bound to 127.0.0.1 — never expose to 0.0.0.0.
"""
from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    FastAPI = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Module-level state — wired by prism_daemon via _set_state()
# Re-exported here so prism_daemon can do: from prism_asgi import _set_state
# ---------------------------------------------------------------------------

from prism_state import _state  # noqa: E402  re-export


def _get_agent():
    return _state.get("agent")


def _get_chain():
    agent = _get_agent()
    return getattr(agent, "_chain", None)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

if _FASTAPI_AVAILABLE:
    app = FastAPI(title="Prism ASGI", version="0.2")

    # CORS — allow all origins for local dashboard access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Include all routers ───────────────────────────────────────────────

    from prism_routes_agent import router as agent_router
    from prism_routes_analytics import router as analytics_router
    from prism_routes_causality import router as causality_router
    from prism_routes_chain import router as chain_router
    from prism_routes_core import router as core_router
    from prism_routes_federation import router as federation_router
    from prism_routes_horizon import router as horizon_router
    from prism_routes_infra import router as infra_router
    from prism_routes_integrations import router as integrations_router
    from prism_routes_media import router as media_router
    from prism_routes_mobile import router as mobile_router
    from prism_routes_perception import router as perception_router
    from prism_routes_predict import router as predict_router
    from prism_routes_sensors import router as sensors_router
    from prism_routes_ui import router as ui_router
    from prism_routes_users import router as users_router

    # UI routes first so "/" doesn't get shadowed
    app.include_router(ui_router)
    app.include_router(predict_router)
    app.include_router(analytics_router)
    app.include_router(agent_router)
    app.include_router(infra_router)
    app.include_router(sensors_router)
    app.include_router(integrations_router)
    app.include_router(chain_router)
    app.include_router(horizon_router)
    app.include_router(core_router)
    app.include_router(media_router)
    app.include_router(mobile_router)
    app.include_router(users_router)
    app.include_router(federation_router)
    app.include_router(perception_router)
    app.include_router(causality_router)

    # ── Built-in routes (kept from Phase 1) ──────────────────────────────

    @app.get("/_health")
    async def health():
        return {"ok": True, "server": "prism-asgi"}

    # ── Async SSE streaming chat ──────────────────────────────────────────

    @app.get("/stream/chat")
    async def stream_chat(request: Request, message: str = "", q: str = ""):
        """
        True async SSE endpoint. Runs the chain in a background thread and
        bridges events into the asyncio event loop via an asyncio.Queue.
        True async — does not block other requests while the chain runs.
        """
        msg = message or q
        if not msg:
            return JSONResponse({"error": "'message' query parameter required"}, status_code=400)

        chain = _get_chain()
        agent = _get_agent()
        if chain is None or agent is None:
            return JSONResponse({"error": "agent not ready"}, status_code=503)

        async def event_generator():
            async for evt in chain.run_streaming_async(msg, agent._execute, {"source": "sse"}):
                if await request.is_disconnected():
                    break
                yield f"data: {json.dumps(evt, default=str)}\n\n"
            yield f"data: {json.dumps({'event': 'close'})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Access-Control-Allow-Origin": "*",
            },
        )

else:
    # Stub so imports don't blow up when fastapi isn't installed
    class _StubApp:  # type: ignore[no-redef]
        def get(self, *a, **kw):
            def _dec(fn):
                return fn
            return _dec

        def post(self, *a, **kw):
            def _dec(fn):
                return fn
            return _dec

        def include_router(self, *a, **kw):
            pass

        def add_middleware(self, *a, **kw):
            pass

    app = _StubApp()  # type: ignore[assignment]
    logger.warning("FastAPI not installed — prism_asgi running in stub mode. "
                   "Install with: pip install 'prism-platform[full]'")


# ---------------------------------------------------------------------------
# Standalone entry point (for development)
# ---------------------------------------------------------------------------

def serve(host: str = "127.0.0.1", port: int = 8743, log_level: str = "info") -> None:
    """Run the ASGI server directly (used by prism_daemon shadow thread)."""
    assert host == "127.0.0.1", (
        "SECURITY: prism_asgi must only bind to 127.0.0.1. "
        "Use a reverse proxy with authentication for remote access."
    )
    try:
        import uvicorn
        uvicorn.run(app, host=host, port=port, log_level=log_level)
    except ImportError:
        logger.error("uvicorn not installed — cannot start ASGI server. "
                     "Install with: pip install 'uvicorn[standard]'")


if __name__ == "__main__":
    serve()
