"""
prism_asgi.py
=============
FastAPI/ASGI application for PRISM — sole HTTP server on port 8742.

Replaces kde_server.py (retired after Phase 7 migration).

SECURITY: Always bound to 127.0.0.1 — never expose to 0.0.0.0.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import time

logger = logging.getLogger(__name__)


# ── Chat rate limit ─────────────────────────────────────────────────────────
# Even though the daemon is loopback-only and bearer-authenticated, a leaked
# token would otherwise allow unbounded LLM spend. Token-bucket per client
# host: sustained refill rate per second, burst capped at bucket size.
# Defaults match a generous human cadence (~30/min sustained, 60 burst).

_CHAT_BUCKET_SIZE = float(os.environ.get("PRISM_CHAT_RATE_BUCKET", "60"))
_CHAT_REFILL_PER_SEC = float(os.environ.get("PRISM_CHAT_RATE_REFILL", "0.5"))
_chat_buckets: dict[str, tuple[float, float]] = {}


def _chat_rate_allow(key: str) -> bool:
    """Token-bucket gate. True if the request fits within the budget for *key*."""
    now = time.monotonic()
    tokens, last = _chat_buckets.get(key, (_CHAT_BUCKET_SIZE, now))
    tokens = min(_CHAT_BUCKET_SIZE, tokens + (now - last) * _CHAT_REFILL_PER_SEC)
    if tokens < 1.0:
        _chat_buckets[key] = (tokens, now)
        return False
    _chat_buckets[key] = (tokens - 1.0, now)
    return True

try:
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
    from starlette.middleware.base import BaseHTTPMiddleware
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    FastAPI = None

from prism_auth import get_token as _get_auth_token  # noqa: E402

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

    # ── Bearer auth ───────────────────────────────────────────────────────
    # When prism_auth.get_token() returns a token (set via env var or the
    # ~/.prism/auth_token file), every HTTP request must carry
    #   Authorization: Bearer <token>
    # Disabled (no-op pass-through) when no token is configured — see
    # prism_auth for the resolution order. WebSocket routes do their own
    # check via the `token` query parameter; BaseHTTPMiddleware only sees
    # HTTP traffic.

    # Liveness probes must remain reachable so orchestrators (systemd
    # WatchdogSec, k8s livenessProbe, docker HEALTHCHECK, uptime monitors)
    # can distinguish "process down" from "auth misconfigured". The route
    # reveals only {"ok": True, "server": "prism-asgi"} — no internal state.
    _AUTH_EXEMPT_PATHS = frozenset({"/_health"})

    _AUTH_COOKIE = "prism_auth"

    class BearerAuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            token = _get_auth_token()
            if token is None:
                return await call_next(request)
            if request.method == "OPTIONS":
                # CORS preflights never carry Authorization.
                return await call_next(request)
            if request.url.path in _AUTH_EXEMPT_PATHS:
                return await call_next(request)

            header = request.headers.get("authorization", "")
            scheme, _, supplied = header.partition(" ")
            if scheme.lower() == "bearer" and supplied:
                if hmac.compare_digest(supplied.strip(), token):
                    return await call_next(request)

            # Cookie path: once a successful `?token=` exchange has set the
            # prism_auth cookie, the browser auto-attaches it on subsequent
            # fetches and EventSource connections (which can't set headers).
            cookie_token = request.cookies.get(_AUTH_COOKIE, "")
            if cookie_token and hmac.compare_digest(cookie_token.strip(), token):
                return await call_next(request)

            # Browser fallback: `?token=<token>` query parameter. Mirrors the
            # WebSocket auth path. Trades a small log-leak risk (query strings
            # appear in access logs and Referer headers) for the ability to
            # poke at the daemon from a tab without a header-injecting
            # extension. On success, set the cookie so the rest of the session
            # works without keeping the token in the URL.
            query_token = request.query_params.get("token", "")
            if query_token and hmac.compare_digest(query_token.strip(), token):
                response = await call_next(request)
                response.set_cookie(
                    key      = _AUTH_COOKIE,
                    value    = token,
                    max_age  = 86400,
                    httponly = True,
                    samesite = "strict",
                    secure   = request.url.scheme == "https",
                    path     = "/",
                )
                return response

            return JSONResponse(
                {"error": "unauthorized"},
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="prism"'},
            )

    app.add_middleware(BearerAuthMiddleware)

    if _get_auth_token() is None:
        logger.warning(
            "prism_asgi: auth disabled — no token in env (PRISM_AUTH_TOKEN) "
            "or file (~/.prism/auth_token). Call prism_auth.ensure_token() "
            "or set PRISM_AUTH_TOKEN before exposing the server."
        )

    # CORS — restrict to local dashboard origins only.
    # The server is always bound to 127.0.0.1, but without origin restriction
    # any webpage in the user's browser could drive the agent via CSRF.
    import os as _os
    _extra_origins = [o.strip() for o in _os.environ.get("PRISM_CORS_ORIGINS", "").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1",
            "http://localhost",
            "tauri://localhost",  # tray webview
        ] + _extra_origins,
        allow_origin_regex=r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$",
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
    from prism_routes_mesh import router as mesh_router
    from prism_routes_ml import router as ml_router
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
    app.include_router(ml_router)
    app.include_router(mesh_router)

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
        client_host = request.client.host if request.client else "anon"
        if not _chat_rate_allow(client_host):
            return JSONResponse(
                {"error": "rate limit exceeded", "status": 429},
                status_code=429,
                headers={"Retry-After": "2"},
            )

        msg = message or q
        if not msg:
            return JSONResponse({"error": "'message' query parameter required"}, status_code=400)

        agent = _get_agent()
        if agent is None:
            return JSONResponse({"error": "agent not ready"}, status_code=503)

        async def event_generator():
            # Emit a single thinking step so the UI shows progress while
            # agent.chat() runs. Routing through agent.chat() (instead of
            # chain.run_streaming_async) ensures intent matching, ORGAN_POLICY
            # approval gates, and per-organ execution all fire — the bare
            # streaming chain only ran LLM reasoning steps without invoking
            # action organs, which produced "I would do X" text rather than
            # actually doing X.
            yield "data: " + json.dumps({
                "event": "step", "step": 1, "logic": "agent",
                "result": "Routing...", "policy": "",
                "score": 0, "risk": "low", "caps": [],
                "constitution": "allowed",
            }) + "\n\n"

            try:
                card = await asyncio.to_thread(agent.chat, msg, {"source": "sse"})
            except Exception as exc:
                yield "data: " + json.dumps({"event": "error", "message": str(exc)}) + "\n\n"
                yield "data: " + json.dumps({"event": "close"}) + "\n\n"
                return

            if await request.is_disconnected():
                return

            cj = card.to_json() if hasattr(card, "to_json") else {
                "type": "text", "title": "", "body": str(card),
                "data": {}, "actions": [],
            }
            yield "data: " + json.dumps({
                "event":        "done",
                "answer":       cj.get("body", ""),
                "chain_id":     "",
                "card_type":    cj.get("type", "text"),
                "card_title":   cj.get("title", ""),
                "card_data":    cj.get("data", {}),
                "card_actions": cj.get("actions", []),
            }, default=str) + "\n\n"
            yield "data: " + json.dumps({"event": "close"}) + "\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
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

        Auth: when an auth token is configured, the client must connect with
        `?token=<token>` (browsers cannot set Authorization on WS handshakes).
        """
        expected = _get_auth_token()
        if expected is not None:
            supplied = websocket.query_params.get("token", "")
            if not supplied or not hmac.compare_digest(supplied, expected):
                # 1008 = policy violation; close before accept to avoid
                # leaking that the endpoint exists.
                await websocket.close(code=1008)
                return
        await websocket.accept()

        chain = _get_chain()
        agent = _get_agent()
        if chain is None or agent is None:
            await websocket.send_json({"event": "error", "message": "agent not ready"})
            await websocket.close(1011)
            return

        client_host = websocket.client.host if websocket.client else "anon"
        try:
            while True:
                data = await websocket.receive_json()
                msg = data.get("message") or data.get("q", "")
                if not msg:
                    await websocket.send_json({"event": "error", "message": "'message' required"})
                    continue
                if not _chat_rate_allow(client_host):
                    await websocket.send_json({"event": "error", "message": "rate limit exceeded"})
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
