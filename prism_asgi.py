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
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    FastAPI = None

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

    # ── Messaging gateway lifespan ────────────────────────────────────────
    # To wire messaging gateway startup/shutdown into a FastAPI lifespan:
    #
    #   from contextlib import asynccontextmanager
    #   from prism_messaging_gateway import start_all_gateways, stop_all_gateways
    #
    #   @asynccontextmanager
    #   async def lifespan(app: FastAPI):
    #       config = _state.get("messaging_config", {})
    #       await start_all_gateways(config)
    #       yield
    #       await stop_all_gateways()
    #
    #   app = FastAPI(title="Prism ASGI", version="0.2", lifespan=lifespan)
    #
    # The daemon can also call start_all_gateways() directly after agent init:
    #   from prism_messaging_gateway import start_all_gateways
    #   await start_all_gateways(messaging_config)

    # ── Include all routers ───────────────────────────────────────────────

    from prism_routes_agent import router as agent_router
    from prism_routes_analytics import router as analytics_router
    from prism_routes_causality import router as causality_router
    from prism_routes_chain import router as chain_router
    from prism_routes_core import router as core_router
    from prism_routes_federation import router as federation_router
    from prism_routes_horizon import router as horizon_router
    from prism_routes_ide import router as ide_router
    from prism_routes_identity import router as identity_router
    from prism_routes_infra import router as infra_router
    from prism_routes_integrations import router as integrations_router
    from prism_routes_kinetic import router as kinetic_router
    from prism_routes_media import router as media_router
    from prism_routes_mobile import router as mobile_router
    from prism_routes_perception import router as perception_router
    from prism_routes_predict import router as predict_router
    from prism_routes_sensors import router as sensors_router
    from prism_routes_sessions import router as sessions_router
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
    app.include_router(ide_router)
    app.include_router(core_router)
    app.include_router(media_router)
    app.include_router(mobile_router)
    app.include_router(sessions_router)
    app.include_router(users_router)
    app.include_router(federation_router)
    app.include_router(identity_router)
    app.include_router(perception_router)
    app.include_router(causality_router)
    app.include_router(kinetic_router)

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

    # ── WebSocket bidirectional chat ──────────────────────────────────────

    @app.websocket("/ws/chat")
    async def ws_chat(websocket: WebSocket):
        """
        Bidirectional WebSocket chat endpoint.

        Client sends:  {"message": "...", "session_id": "optional-name"}
        Server streams: {"event": "step", "step": N, ...}
                        {"event": "done", "answer": "...", ...}
                        {"event": "error", "message": "..."}

        The connection stays open for multiple turns — send another message
        after receiving the "done" event for the previous turn.
        """
        await websocket.accept()

        chain = _get_chain()
        agent = _get_agent()
        if chain is None or agent is None:
            await websocket.send_json({"event": "error", "message": "agent not ready"})
            await websocket.close(1011)
            return

        try:
            while True:
                data = await websocket.receive_json()
                msg = data.get("message") or data.get("q", "")
                if not msg:
                    await websocket.send_json({"event": "error", "message": "'message' required"})
                    continue

                session_id = data.get("session_id") or _state.get("active_session_id")
                answer = None

                async for evt in chain.run_streaming_async(msg, agent._execute, {"source": "ws"}):
                    await websocket.send_json(evt)
                    if evt.get("event") == "done":
                        answer = evt.get("answer")

                if session_id and answer:
                    try:
                        from prism_session_manager import get_session_manager
                        sm = get_session_manager()
                        sm.add_message(session_id, "user", msg)
                        sm.add_message(session_id, "assistant", answer)
                    except Exception:
                        pass

        except WebSocketDisconnect:
            pass

else:
    # Stub so imports don't blow up when fastapi isn't installed
    class _StubApp:  # noqa: F811
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

        def websocket(self, *a, **kw):
            def _dec(fn):
                return fn
            return _dec

    app = _StubApp()
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
