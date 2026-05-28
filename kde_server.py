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

import dataclasses
import json
import logging
import threading
import uuid
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

    agent:            object   # KDEAgent
    platform:         object   # PredictionPlatform
    moment_analyzer:  object   # MomentAnalyzer
    duel_analyzer:    object   # DuelAnalyzer
    domain_models:    dict     # domain name -> DomainDecisionModel
    live_pipeline:    object   # LiveMomentPipeline
    _moment_history:  dict     # player → list[MomentResult]

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

            # ── Domain endpoints ─────────────────────────────────────────

            elif path == "/domain/list":
                from domain_configs import ALL_DOMAINS

                domains = [
                    {
                        "name": config.name,
                        "domain": config.domain,
                        "n_planks": len(config.planks),
                        "n_profiles": len(config.profiles),
                        "calibrated": config.calibrated,
                    }
                    for config in ALL_DOMAINS.values()
                ]
                self._json_response({"domains": domains})

            elif path == "/domain/profiles":
                domain = qs.get("domain")
                model = self.domain_models.get(domain)
                if model is None:
                    self._error(f"Unknown domain: {domain}", 404)
                    return

                profiles = [
                    {
                        "name": profile.name,
                        "fixed_fulcrum": profile.fixed_fulcrum,
                        "description": profile.description,
                    }
                    for profile in model.config.profiles
                ]
                self._json_response({"domain": domain, "profiles": profiles})

            elif path == "/domain/evaluate":
                domain = qs.get("domain")
                profile = qs.get("profile")
                model = self.domain_models.get(domain)
                if model is None:
                    self._error(f"Unknown domain: {domain}", 404)
                    return
                if not profile:
                    self._error("profile is required", 400)
                    return

                factor_values = {
                    factor.id: float(qs.get(factor.id, 0.5))
                    for factor in model.config.factors
                }
                beam = model.make_beam(profile, factor_values)
                diagnosis = beam.evaluate()
                labels = {factor.id: factor.label for factor in model.config.factors}
                key_factors = sorted(
                    [
                        {
                            "name": labels.get(factor.name, factor.name),
                            "contribution": factor.contribution(),
                        }
                        for factor in beam.fulcrum.factors
                        if factor.name != "_base"
                    ],
                    key=lambda item: abs(item["contribution"]),
                    reverse=True,
                )
                self._json_response({
                    "domain": domain,
                    "profile": profile,
                    "recommended": diagnosis.primary_plank.name,
                    "confidence": diagnosis.activations[0].activation,
                    "fulcrum": diagnosis.fulcrum_position,
                    "options": [
                        {
                            "name": activation.plank.name,
                            "activation": activation.activation,
                            "position": activation.plank.position,
                        }
                        for activation in diagnosis.activations
                    ],
                    "key_factors": key_factors[:5],
                })

            elif path == "/domain/sensitivity":
                domain = qs.get("domain")
                profile = qs.get("profile")
                factor_id = qs.get("factor")
                steps = int(qs.get("steps", 5))
                model = self.domain_models.get(domain)
                if model is None:
                    self._error(f"Unknown domain: {domain}", 404)
                    return
                if not profile or not factor_id:
                    self._error("profile and factor are required", 400)
                    return

                sweep = model.sensitivity_sweep(profile, factor_id, steps=steps)
                values = [i / (steps - 1) for i in range(steps)] if steps > 1 else [0.0]
                self._json_response({
                    "domain": domain,
                    "profile": profile,
                    "factor": factor_id,
                    "sweep": [
                        {
                            "value": value,
                            "recommended": diagnosis.primary_plank.name,
                            "fulcrum": diagnosis.fulcrum_position,
                            "confidence": diagnosis.activations[0].activation,
                        }
                        for value, diagnosis in zip(values, sweep)
                    ],
                })

            # ── Moment endpoints ─────────────────────────────────────────

            elif path == "/moment/configs":
                from moment_analyzer import ALL_MOMENT_CONFIGS
                from moment_configs_ext import EXTENDED_CONFIGS
                all_keys = set(ALL_MOMENT_CONFIGS.keys()) | set(EXTENDED_CONFIGS.keys())
                configs = [
                    {"sport": s, "moment_type": mt}
                    for s, mt in sorted(all_keys)
                ]
                self._json_response({"configs": configs})

            elif path == "/moment/analyze":
                sport       = qs.get("sport")
                moment_type = qs.get("moment_type")
                player      = qs.get("player")
                if not sport or not moment_type or not player:
                    self._error("sport, moment_type, player are required", 400)
                    return

                from moment_analyzer import Moment, NearbyPlayer

                # Primary opponent (goalkeeper)
                primary_opp = None
                gk_name = qs.get("gk_name")
                if gk_name:
                    gk_dist = float(qs.get("gk_distance", 6.0))
                    primary_opp = NearbyPlayer(
                        name=gk_name, team="", distance=gk_dist,
                        arrival_time=gk_dist / 7.5, is_goalkeeper=True,
                    )

                # Secondary opponents
                secondary = []
                for i in (1, 2, 3):
                    dname = qs.get(f"defender{i}")
                    if dname:
                        arr = float(qs.get(f"defender{i}_arrival", 3.0))
                        secondary.append(NearbyPlayer(
                            name=dname, team="", distance=arr * 7.5,
                            arrival_time=arr,
                        ))

                # Teammates
                teammates = []
                for i in (1, 2, 3):
                    tname = qs.get(f"teammate{i}")
                    if tname:
                        tdist = float(qs.get(f"teammate{i}_distance", 10.0))
                        teammates.append(NearbyPlayer(
                            name=tname, team="", distance=tdist,
                            arrival_time=tdist / 7.5,
                        ))

                moment = Moment(
                    moment_id           = str(uuid.uuid4()),
                    match_id            = "api",
                    sport               = sport,
                    moment_type         = moment_type,
                    timestamp           = 0.0,
                    focal_player        = player,
                    focal_profile       = qs.get("profile", "Forward"),
                    focal_team          = qs.get("team", ""),
                    focal_base          = float(qs.get("base", 0.5)),
                    pitch_x             = float(qs.get("pitch_x", 0.5)),
                    pitch_y             = float(qs.get("pitch_y", 0.5)),
                    primary_opponent    = primary_opp,
                    secondary_opponents = secondary,
                    teammates           = teammates,
                    fatigue             = float(qs.get("fatigue", 0.0)),
                    confidence          = float(qs.get("confidence", 0.5)),
                    score_pressure      = float(qs.get("score_pressure", 0.0)),
                    xg_raw              = float(qs.get("xg_raw", 0.0)),
                )

                result = self.moment_analyzer.analyze(moment)

                # Store in history
                self._moment_history.setdefault(player, []).append(result)

                # Build response
                import math
                bw = result.config.bandwidth
                focal = result.focal_position
                options_list = []
                for opt in result.config.options:
                    kernel = math.exp(
                        -(opt.position - focal) ** 2 / (2.0 * bw ** 2)
                    )
                    options_list.append({
                        "name":       opt.name,
                        "activation": round(kernel, 4),
                        "ev":         round(result.option_scores[opt.name], 4),
                    })

                # time_pressure: based on nearest opponent arrival_time
                if moment.primary_opponent is not None:
                    time_pressure = round(
                        max(0.0, min(1.0, 1.0 / (1.0 + moment.primary_opponent.arrival_time))), 4
                    )
                elif secondary:
                    min_arr = min(o.arrival_time for o in secondary)
                    time_pressure = round(max(0.0, min(1.0, 1.0 / (1.0 + min_arr))), 4)
                else:
                    time_pressure = 0.0

                self._json_response({
                    "recommended":   result.recommended,
                    "activation":    round(result.focal_position, 4),
                    "xg_contextual": round(result.xg_contextual, 4),
                    "time_pressure": time_pressure,
                    "fulcrum":       round(result.focal_position, 4),
                    "options":       options_list,
                })

            elif path == "/moment/history":
                player = qs.get("player", "")
                limit  = int(qs.get("limit", 20))
                history_raw = self._moment_history.get(player, [])[-limit:]
                moments_out = []
                for r in history_raw:
                    try:
                        moments_out.append(dataclasses.asdict(r))
                    except Exception:
                        moments_out.append(str(r))
                self._json_response({"player": player, "moments": moments_out})

            elif path == "/moment/player_stats":
                player = qs.get("player", "")
                if not player:
                    self._error("player is required", 400)
                    return
                stats = self.moment_analyzer.player_stats(player)
                self._json_response({"player": player, **stats})

            # ── Duel endpoints ────────────────────────────────────────────

            elif path == "/duel/network":
                match_id = qs.get("match_id", "")
                edges = []
                for (att, dfn), data in self.duel_analyzer.network._edges.items():
                    if not match_id or True:   # no per-match filtering; return all
                        edges.append({
                            "attacker": att,
                            "defender": dfn,
                            "total":    data["total"],
                            "won":      data["won"],
                            "win_rate": data["won"] / data["total"] if data["total"] else 0.5,
                        })
                self._json_response({"match_id": match_id, "edges": edges})

            elif path == "/duel/player":
                player = qs.get("player", "")
                if not player:
                    self._error("player is required", 400)
                    return
                profile = self.duel_analyzer.network.player_attack_stats(player)
                self._json_response(profile)

            elif path == "/duel/summary":
                network = self.duel_analyzer.network
                total_duels = sum(e["total"] for e in network._edges.values())
                total_won   = sum(e["won"]   for e in network._edges.values())
                self._json_response({
                    "total_duels": total_duels,
                    "total_won":   total_won,
                    "win_rate":    total_won / total_duels if total_duels else 0.5,
                    "n_matchups":  len(network._edges),
                })

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

            # ── Moment POST endpoints ─────────────────────────────────────

            elif path == "/moment/calibrate":
                moment_id    = body.get("moment_id", "")
                action_taken = body.get("action_taken", "")
                success      = bool(body.get("success", False))
                xg_realized  = float(body.get("xg_realized", 0.0))
                notes        = body.get("notes", "")

                from moment_analyzer import ActionOutcome, Moment

                # Find the moment in history
                target_moment = None
                for results in self._moment_history.values():
                    for r in results:
                        if r.moment.moment_id == moment_id:
                            target_moment = r.moment
                            break
                    if target_moment is not None:
                        break

                if target_moment is None:
                    # Create a minimal placeholder moment for calibration
                    target_moment = Moment(
                        moment_id  = moment_id,
                        match_id   = "calibration",
                        sport      = body.get("sport", "Football"),
                        moment_type= body.get("moment_type", "1v1_keeper"),
                        timestamp  = 0.0,
                        focal_player  = body.get("player", "Unknown"),
                        focal_profile = "Forward",
                        focal_team    = "",
                        focal_base    = 0.5,
                        pitch_x       = 0.5,
                        pitch_y       = 0.5,
                    )

                outcome = ActionOutcome(
                    action_taken = action_taken,
                    success      = success,
                    xg_delta     = xg_realized,
                    notes        = notes,
                )
                self.moment_analyzer.calibrate(target_moment, outcome)
                self._json_response({"status": "calibrated"})

            elif path == "/moment/live_frame":
                result = self.live_pipeline.feed_frame(body)
                if result is None:
                    self._json_response({"moment": None})
                else:
                    try:
                        self._json_response({"moment": dataclasses.asdict(result)})
                    except Exception:
                        self._json_response({"moment": str(result)})

            # ── Duel POST endpoints ───────────────────────────────────────

            elif path == "/duel/add_match":
                match_id    = body.get("match_id", str(uuid.uuid4()))
                events      = body.get("events", [])
                freeze_frames = body.get("freeze_frames", {})
                profile_map = body.get("profile_map", {})

                records = self.duel_analyzer.process_match(events, match_id)
                self._json_response({
                    "match_id": match_id,
                    "n_duels":  len(records),
                    "accuracy": (
                        sum(1 for r in records if r.attacker_won) / len(records)
                        if records else 0.0
                    ),
                })

            elif path == "/domain/validate":
                from domain_validator import DomainValidator, LabeledDecision

                domain = body.get("domain")
                if domain not in self.domain_models:
                    self._error(f"Unknown domain: {domain}", 404)
                    return
                cases = [
                    LabeledDecision(
                        case_id=item.get("case_id", str(index)),
                        domain=domain,
                        profile=item.get("profile", ""),
                        factor_values=dict(item.get("factor_values", {})),
                        expert_choice=item.get("expert_choice", ""),
                        outcome=item.get("outcome", ""),
                        notes=item.get("notes", ""),
                    )
                    for index, item in enumerate(body.get("cases", []), start=1)
                ]
                result = DomainValidator(domain).validate(cases, domain=domain)
                self._json_response(dataclasses.asdict(result))

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
        moment_analyzer=None,
        duel_analyzer=None,
        domain_models=None,
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

        # Moment + Duel analyzers (lazy-import to keep startup fast)
        if moment_analyzer is None:
            try:
                from moment_analyzer import MomentAnalyzer as _MA
                moment_analyzer = _MA()
            except ImportError:
                moment_analyzer = None

        if duel_analyzer is None:
            try:
                from duel_analyzer import DuelAnalyzer as _DA
                duel_analyzer = _DA()
            except ImportError:
                duel_analyzer = None

        self._moment_analyzer = moment_analyzer
        self._duel_analyzer   = duel_analyzer
        if domain_models is None:
            try:
                from domain_configs import ALL_DOMAINS, DomainDecisionModel

                domain_models = {
                    domain: DomainDecisionModel(config)
                    for domain, config in ALL_DOMAINS.items()
                }
            except ImportError:
                domain_models = {}
        self._domain_models = domain_models

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
        agent           = self._agent
        platform        = self._platform
        moment_analyzer = self._moment_analyzer
        duel_analyzer   = self._duel_analyzer
        domain_models   = self._domain_models

        # Build live pipeline from the moment analyzer
        live_pipeline = None
        if moment_analyzer is not None:
            try:
                from moment_pipeline import LiveMomentPipeline
                live_pipeline = LiveMomentPipeline(moment_analyzer)
            except ImportError:
                pass

        class _Handler(KDEHandler):
            pass

        _Handler.agent           = agent
        _Handler.platform        = platform
        _Handler.moment_analyzer = moment_analyzer
        _Handler.duel_analyzer   = duel_analyzer
        _Handler.domain_models   = domain_models
        _Handler.live_pipeline   = live_pipeline
        _Handler._moment_history = {}  # player → list[MomentResult]

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
