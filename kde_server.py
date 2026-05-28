"""
kde_server.py
=============
KDE Sports Platform — Local REST API

A lightweight HTTP server using only Python's stdlib http.server.
Exposes the KDE agent as a JSON API on localhost:8742.

SECURITY: This server binds to 127.0.0.1 ONLY by design.
          Never change host to 0.0.0.0 without adding authentication.
          All data is local. No external connections are made by this server.

Routes:
  GET  /status                 → agent.status()
  GET  /plan                   → today's daily plan (JSON)
  POST /ask                    → body: {"prompt":"..."} → TaskResult
  GET  /predict/match          → MatchPrediction JSON
  GET  /predict/injury         → InjuryRiskPrediction JSON
  GET  /predict/performance    → PerformancePrediction JSON
  GET  /predict/transfer       → TransferPrediction JSON
  GET  /predict/brief          → full pre-match brief JSON
  GET  /history                → plan history JSON
  POST /rate                   → {"date":"...","rating":4.0,"notes":"..."}
  POST /session                → {rpe, session_type, notes, video_folder}
  GET  /reflect                → learned state JSON
  GET  /devices                → connected devices list
  POST /device/sync            → trigger device sync

All responses: Content-Type: application/json
Error format:  {"error": "message", "status": 4xx}
CORS headers included for local web dashboard access.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, fields
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)


def _safe_dict(obj) -> dict:
    """Convert a dataclass (or any object) to a dict safely."""
    try:
        return asdict(obj)
    except TypeError:
        pass
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return {}

DEFAULT_PORT = 8742   # KDE port
DEFAULT_HOST = "127.0.0.1"


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class KDEHandler(BaseHTTPRequestHandler):
    """
    HTTP request handler for the KDE local API.

    ``agent`` and ``platform`` are injected as class attributes by KDEServer.
    """

    agent:    object   # KDEAgent
    platform: object   # PredictionPlatform

    # ── common ────────────────────────────────────────────────────────────

    def _json_response(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message: str, status: int = 400) -> None:
        self._json_response({"error": message, "status": status}, status)

    def log_message(self, format, *args) -> None:  # noqa: A002
        # Suppress routine HTTP logs unless debug-level logging is active
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("%s - %s", self.address_string(), format % args)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _qs(self, parsed) -> dict:
        """Return query string as flat dict of first values."""
        return {k: v[0] for k, v in parse_qs(parsed.query).items()}

    # ── GET ───────────────────────────────────────────────────────────────

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/")
        qs     = self._qs(parsed)

        try:
            if path == "/status":
                self._json_response(self.agent.status())

            elif path == "/plan":
                brief = self.agent.morning_briefing()
                from dataclasses import asdict as _asdict
                try:
                    data = _asdict(brief)
                except TypeError:
                    data = str(brief)
                self._json_response(data)

            elif path == "/reflect":
                self._json_response(self.agent.reflect())

            elif path == "/history":
                days    = int(qs.get("days", 14))
                history = self.agent._assistant.history(
                    self.agent._profile.name, days=days
                )
                self._json_response({"history": history})

            elif path == "/devices":
                devices = [
                    {"name":        d.name,
                     "device_type": d.device_type.value,
                     "enabled":     d.enabled,
                     "last_sync":   d.last_sync}
                    for d in self.agent._hub.list_devices()
                ]
                self._json_response({"devices": devices})

            elif path == "/predict/match":
                home      = qs.get("home", "Home Team")
                away      = qs.get("away", "Away Team")
                sport     = qs.get("sport", "football")
                home_form = float(qs.get("home_form", 0.5))
                away_form = float(qs.get("away_form", 0.5))
                pred = self.platform.match.predict(
                    home, away, sport,
                    home_form=home_form, away_form=away_form,
                )
                self._json_response(_safe_dict(pred))

            elif path == "/predict/injury":
                name     = qs.get("name", "Athlete")
                recovery = float(qs.get("recovery", 0.7))
                load     = float(qs.get("load", 0.5))
                soreness = float(qs.get("soreness", 0.3))
                pred = self.platform.injury.predict(
                    name,
                    recovery_score  = recovery,
                    load_7d         = load,
                    muscle_soreness = soreness,
                )
                self._json_response(_safe_dict(pred))

            elif path == "/predict/performance":
                name    = qs.get("name", "Athlete")
                form    = float(qs.get("form", 0.6))
                fitness = float(qs.get("fitness", 0.7))
                pred = self.platform.performance.predict(
                    name,
                    recent_form   = form,
                    fitness_level = fitness,
                )
                self._json_response(_safe_dict(pred))

            elif path == "/predict/transfer":
                name        = qs.get("name", "Athlete")
                performance = float(qs.get("performance", 0.6))
                age         = int(qs.get("age", 24))
                pred = self.platform.transfer.predict(
                    name,
                    performance_score = performance,
                    age               = age,
                )
                self._json_response(_safe_dict(pred))

            elif path == "/predict/brief":
                home  = qs.get("home",  "Home Team")
                away  = qs.get("away",  "Away Team")
                sport = qs.get("sport", "football")
                brief_data = self.platform.pre_match_brief(home, away, sport)
                # Serialize dataclasses
                serialised = {
                    "match_prediction":  _safe_dict(brief_data["match_prediction"]),
                    "tactical_analysis": _safe_dict(brief_data["tactical_analysis"]),
                    "squad_risk":        [_safe_dict(r) for r in brief_data["squad_risk"]],
                    "squad_performance": [_safe_dict(p) for p in brief_data["squad_performance"]],
                    "generated_at":      brief_data["generated_at"],
                }
                self._json_response(serialised)

            else:
                self._error(f"Unknown route: {path}", 404)

        except Exception as exc:
            logger.exception("GET %s failed", path)
            self._error(str(exc), 500)

    # ── POST ──────────────────────────────────────────────────────────────

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/")
        body   = self._read_body()

        try:
            if path == "/ask":
                prompt = body.get("prompt", "")
                if not prompt:
                    self._error("'prompt' field required", 400)
                    return
                result = self.agent.ask(prompt)
                self._json_response({
                    "task":       result.task,
                    "method":     result.method,
                    "success":    result.success,
                    "elapsed_ms": result.elapsed_ms,
                    "output":     result.output,
                })

            elif path == "/rate":
                date_str   = body.get("date", "")
                rating     = float(body.get("rating", 0))
                notes      = body.get("notes", "")
                if not date_str:
                    from datetime import date
                    date_str = date.today().isoformat()
                self.agent._assistant.rate_day(
                    self.agent._profile.name, date_str, rating, notes
                )
                self._json_response({"ok": True, "date": date_str, "rating": rating})

            elif path == "/session":
                rpe          = int(body.get("rpe", 5))
                session_type = body.get("session_type", "training")
                notes        = body.get("notes", "")
                video_folder = body.get("video_folder")
                gps_file     = body.get("gps_file")
                log = self.agent.log_session(
                    rpe          = rpe,
                    session_type = session_type,
                    notes        = notes,
                    video_folder = video_folder,
                    gps_file     = gps_file,
                )
                from dataclasses import asdict as _asdict
                self._json_response(_asdict(log))

            elif path == "/device/sync":
                result = self.agent.sync_devices()
                self._json_response({"synced": result})

            else:
                self._error(f"Unknown route: {path}", 404)

        except Exception as exc:
            logger.exception("POST %s failed", path)
            self._error(str(exc), 500)

    # ── OPTIONS (CORS preflight) ───────────────────────────────────────────

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


# ---------------------------------------------------------------------------
# KDEServer
# ---------------------------------------------------------------------------

class KDEServer:
    """
    Local REST API server for the KDE agent.

    Usage:
        server = KDEServer(agent, port=8742)
        server.start()            # non-blocking (daemon thread)
        server.start(blocking=True)  # blocks until Ctrl-C
        server.stop()
    """

    def __init__(
        self,
        agent,
        port:    int  = DEFAULT_PORT,
        host:    str  = DEFAULT_HOST,   # localhost only
        verbose: bool = False,
        platform=None,
    ) -> None:
        # SECURITY: enforce localhost-only binding
        if host != DEFAULT_HOST:
            logger.warning(
                "KDEServer: host '%s' overridden to '%s' for security.",
                host, DEFAULT_HOST,
            )
            host = DEFAULT_HOST

        self._agent    = agent
        self._port     = port
        self._host     = host
        self._verbose  = verbose
        self._platform = platform or _get_or_create_platform(agent)
        self._server:  Optional[HTTPServer] = None
        self._thread:  Optional[threading.Thread] = None

        if verbose:
            logging.getLogger(__name__).setLevel(logging.DEBUG)

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}"

    def start(self, blocking: bool = False) -> None:
        """Start the HTTP server. If blocking=False, run in a daemon thread."""
        # Build a handler class with agent + platform injected
        agent    = self._agent
        platform = self._platform

        class _Handler(KDEHandler):
            pass

        _Handler.agent    = agent
        _Handler.platform = platform

        self._server = HTTPServer((self._host, self._port), _Handler)
        logger.warning("KDE server running on %s", self.url)
        print(f"KDE server running on {self.url}")

        if blocking:
            try:
                self._server.serve_forever()
            except KeyboardInterrupt:
                self.stop()
        else:
            self._thread = threading.Thread(
                target = self._server.serve_forever,
                daemon = True,
                name   = "kde-server",
            )
            self._thread.start()

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None
        self._thread = None
        logger.warning("KDE server stopped.")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_or_create_platform(agent) -> object:
    """Extract or create a PredictionPlatform from an agent."""
    try:
        # KDEAgent may expose _platform after prompt-3 integration
        return agent._platform
    except AttributeError:
        pass
    try:
        from prediction_engine import PredictionPlatform
        return PredictionPlatform()
    except ImportError:
        return None
