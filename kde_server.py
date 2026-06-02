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
  GET  /policy                 → policy JSON for a user
  GET  /plan                   → today's daily plan (JSON)
  GET  /tools/find             → tool discovery JSON
  GET  /policy/spend           → policy spend summary JSON
  POST /ask                    → body: {"prompt":"..."} → TaskResult
  POST /policy/set             → set one policy allocation
  POST /policy/update_from_chat→ parse/update policy from chat text
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
  GET  /llm/status             → LLM router status (available models, best, cost data)
  POST /llm/set                → set preferred LLM {"preferred":"provider/model"}
  GET  /tasks                  → recent background tasks list (?n=10)
  GET  /tasks/<id>             → single task progress by task_id

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
from dataclasses import asdict
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
    policy_engine:    object   # PolicyEngine
    tool_finder:      object   # ToolFinder
    llm_router:       object   # LLMRouter
    task_queue:       object   # TaskQueue
    _moment_history:  dict     # player → list[MomentResult]

    # ── common ────────────────────────────────────────────────────────────

    def _json_response(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, default=str).encode()
        self._respond(body, status, "application/json")

    def _respond(self, body: bytes, status: int = 200, content_type: str = "application/octet-stream") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
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
        path   = parsed.path.rstrip("/") or "/"
        qs     = self._qs(parsed)

        try:
            if path in ('/', '/app', '/chat', '/index.html'):
                from prism_chat import get_chat_html

                self._respond(get_chat_html().encode('utf-8'), 200, 'text/html; charset=utf-8')
                return

            elif path == "/status":
                import urllib.request as _ur
                import json as _j
                ollama_ok = False; ollama_model = ""
                try:
                    r = _ur.urlopen("http://localhost:11434/api/tags", timeout=2)
                    tags = _j.loads(r.read())
                    ollama_ok    = True
                    ollama_model = (tags.get("models", [{}])[0].get("name", "")
                                    if tags.get("models") else "")
                except Exception:
                    pass
                status = self.agent.status()
                status["ollama"]       = ollama_ok
                status["ollama_model"] = ollama_model
                self._json_response(status)

            elif path == "/plan":
                brief = self.agent.morning_briefing()
                from dataclasses import asdict as _asdict
                try:
                    data = _asdict(brief)
                except TypeError:
                    data = str(brief)
                self._json_response(data)

            elif path == "/policy":
                user = qs.get("user", "")
                if not user:
                    self._error('Query parameter "user" is required', 400)
                    return
                self._json_response(_safe_dict(self.policy_engine.get_policy(user)))

            elif path == "/tools/find":
                task = qs.get("task", "")
                if not task:
                    self._error("task is required", 400)
                    return
                provider = qs.get("provider", task)
                result = self.tool_finder.find(
                    task=task,
                    provider_name=provider,
                    urgency=float(qs.get("urgency", 0.5)),
                    cost_tolerance=float(qs.get("cost_tolerance", 0.5)),
                    prefers_auto=float(qs.get("prefers_auto", 0.5)),
                    budget_left=float(qs.get("budget_left", 1.0)),
                )
                self._json_response(_safe_dict(result))

            elif path == "/policy/spend":
                user = qs.get("user", "")
                category = qs.get("category", "")
                if not user or not category:
                    self._error("user and category are required", 400)
                    return
                days = int(qs.get("days", 30))
                self._json_response(self.policy_engine.spend_summary(user, category, days))

            elif path == "/reflect":
                self._json_response(self.agent.reflect())

            elif path == "/identity":
                self._json_response(self.agent.identity())

            elif path == "/identity/domains":
                self._json_response({"domains": self.agent.identity_domains()})

            elif path == "/artifacts":
                domain = qs.get("domain")
                n = int(qs.get("n", 10))
                self._json_response({"artifacts": self.agent.recent_artifacts(domain=domain, n=n)})

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

                self._json_response({
                    "domains": [
                        {
                            "name": name,
                            "domain": config.domain,
                            "n_planks": len(config.planks),
                            "n_profiles": len(config.profiles),
                        }
                        for name, config in ALL_DOMAINS.items()
                    ]
                })

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
                from domain_configs import ALL_DOMAINS, DomainDecisionModel

                domain = qs.get("domain", "Medical")
                profile = qs.get("profile")
                config = ALL_DOMAINS.get(domain)
                if config is None:
                    self._error(f"Unknown: {domain}", 404)
                    return
                model = DomainDecisionModel(config)
                if not profile:
                    profile = config.profiles[0].name
                factor_values = {
                    factor.id: float(qs.get(factor.id, 0.5))
                    for factor in config.factors
                }
                diagnosis = model.evaluate(profile, factor_values)
                self._json_response({
                    "recommended": diagnosis.primary_plank.name,
                    "fulcrum": round(diagnosis.fulcrum_position, 3),
                    "confidence": round(diagnosis.activations[0].activation, 3),
                    "options": [
                        {
                            "name": activation.plank.name,
                            "activation": round(activation.activation, 3),
                        }
                        for activation in diagnosis.activations
                    ],
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

            elif path == '/device/capabilities':
                from prism_device_agent import DeviceCapabilityScanner
                caps = DeviceCapabilityScanner().scan()
                self._json_response({
                    "platform":     caps.platform,
                    "has_browser":  caps.has_browser,
                    "categories":   {k: v for k, v in caps.cli_tools.items()},
                    "py_packages":  caps.py_packages,
                    "summary":      caps.summary(),
                })
                return

            elif path == "/llm/status":
                if self.llm_router is None:
                    self._json_response({"available": False, "note": "LLM router not initialised"})
                    return
                from prism_llm_router import PROVIDER_COSTS
                status = self.llm_router.status_summary()
                for opt_data in status.get("available", []):
                    model = opt_data.get("model", "")
                    costs = PROVIDER_COSTS.get(model, (0, 0))
                    opt_data["cost_per_query_usd"] = round(
                        (500 * costs[0] / 1000 + 500 * costs[1] / 1000), 5)
                    opt_data["free"] = costs[0] == 0
                self._json_response(status)

            elif path == "/tasks":
                if self.task_queue is None:
                    self._json_response({"tasks": [], "count": 0, "note": "Task queue not initialised"})
                    return
                raw_n = qs.get("n", 10)
                try:
                    n = int(raw_n)
                except (ValueError, TypeError):
                    self._error(f"Invalid n parameter: '{raw_n}' must be an integer", 400)
                    return
                tasks = self.task_queue.list_recent(n)
                items = [
                    {
                        "task_id":      t.task_id,
                        "title":        t.title,
                        "status":       t.status if isinstance(t.status, str) else t.status.value,
                        "progress":     t.progress,
                        "current_step": t.current_step,
                        "steps_done":   t.steps_done,
                        "steps_total":  t.steps_total,
                        "error":        t.error,
                    }
                    for t in tasks
                ]
                self._json_response({"tasks": items, "count": len(items)})

            elif path.startswith("/tasks/"):
                if self.task_queue is None:
                    self._error("Task queue not initialised", 503)
                    return
                task_id = path[len("/tasks/"):]
                if not task_id:
                    self._error("task_id is required", 400)
                    return
                progress = self.task_queue.get(task_id)
                if progress is None:
                    self._error(f"Task '{task_id}' not found", 404)
                    return
                self._json_response({
                    "task_id":      progress.task_id,
                    "title":        progress.title,
                    "status":       progress.status if isinstance(progress.status, str) else progress.status.value,
                    "progress":     progress.progress,
                    "current_step": progress.current_step,
                    "steps_done":   progress.steps_done,
                    "steps_total":  progress.steps_total,
                    "result":       progress.result,
                    "error":        progress.error,
                    "started_at":   progress.started_at,
                    "completed_at": progress.completed_at,
                })

            elif path == '/perception/status':
                agent = getattr(self.server, 'prism_agent', None)
                if agent and hasattr(agent, '_perception') and agent._perception:
                    self._json_response(agent._perception.status())
                else:
                    self._json_response({"active_channels": [], "factor_count": 0,
                                         "summary": "perception not initialised"})

            elif path == '/memory/search':
                agent = getattr(self.server, 'prism_agent', None)
                mem   = agent._memory if agent and hasattr(agent, '_memory') else None
                if mem is None:
                    self._json_response({"results": [], "note": "memory not initialised"})
                    return
                query = qs.get("q", "")
                if not query:
                    self._error("Query parameter 'q' is required", 400)
                    return
                top_n  = int(qs.get("n", 5))
                source = qs.get("source")
                results = mem.search(query, top_n=top_n, source_filter=source)
                self._json_response({
                    "results": [
                        {
                            "entry_id": r.entry.entry_id,
                            "title":    r.entry.title,
                            "source":   r.entry.source,
                            "score":    round(r.score, 4),
                            "excerpt":  r.excerpt,
                            "tags":     r.entry.tags,
                            "timestamp": r.entry.timestamp,
                        }
                        for r in results
                    ],
                    "count": len(results),
                })

            elif path == '/proactive':
                p = getattr(self.server, 'prism_proactive', None)
                if p is None:
                    self._json_response({"events": [], "note": "proactive not initialised"})
                    return
                n      = int(qs.get("n", 5))
                events = p.pending_events(n)
                self._json_response({
                    "events": [
                        {"trigger_id": e.trigger_id, "message": e.message,
                         "timestamp": e.timestamp}
                        for e in events
                    ],
                    "count": len(events),
                })

            elif path == '/proactive/pending':
                agent  = getattr(self.server, 'prism_agent', None)
                events = getattr(agent, '_proactive_buffer', []) if agent else []
                self._json_response({"events": [
                    {"trigger_id": e.trigger_id, "message": e.message,
                     "timestamp": e.timestamp}
                    for e in events[-5:]
                ]})
                if agent:
                    agent._proactive_buffer = []

            elif path == '/smarthome/status':
                agent = getattr(self.server, 'prism_agent', None)
                if agent and hasattr(agent, '_smarthome'):
                    self._json_response(agent._smarthome.status_summary())
                else:
                    self._json_response({"configured": False})

            elif path == '/email/status':
                from prism_email import PrismEmail
                agent = getattr(self.server, 'prism_agent', None)
                em = getattr(agent, '_email', None) if agent else None
                if em is None:
                    em = PrismEmail()
                self._json_response(em.status_summary())

            elif path == '/email/inbox':
                from prism_email import PrismEmail
                agent = getattr(self.server, 'prism_agent', None)
                em = getattr(agent, '_email', None) if agent else None
                if em is None:
                    em = PrismEmail()
                if not em.configured:
                    self._error("Email not configured", 503)
                    return
                n = int(qs.get("n", 20))
                folder = qs.get("folder", "INBOX")
                unread_only = qs.get("unread", "true").lower() != "false"
                msgs = em.fetch_unread(folder=folder, n=n) if unread_only else em.fetch_recent(n=n)
                self._json_response({
                    "count": len(msgs),
                    "messages": [
                        {
                            "msg_id":  m.msg_id,
                            "subject": m.subject,
                            "sender":  m.sender,
                            "date":    m.date,
                            "unread":  m.unread,
                            "snippet": m.body[:200],
                        }
                        for m in msgs
                    ],
                })

            elif path == '/email/unread':
                agent = getattr(self.server, 'prism_agent', None)
                if agent and hasattr(agent, '_email') and agent._email.configured:
                    msgs = agent._email.fetch_unread(n=10)
                    self._json_response({"count": len(msgs),
                        "messages": [{"from": m.sender, "subject": m.subject,
                                      "date": m.date, "body": m.body[:500]} for m in msgs]})
                else:
                    self._error("Email not configured", 503)

            elif path == '/calendar/status':
                agent = getattr(self.server, 'prism_agent', None)
                if agent and hasattr(agent, '_calendar'):
                    self._json_response(agent._calendar.status_summary())
                else:
                    self._json_response({"configured": False})

            elif path == '/calendar/today':
                agent = getattr(self.server, 'prism_agent', None)
                if agent and hasattr(agent, '_calendar') and agent._calendar.configured:
                    events = agent._calendar.today()
                    self._json_response({"count": len(events),
                        "events": [{"title": e.title,
                                    "start": e.start.isoformat(),
                                    "end": e.end.isoformat(),
                                    "location": e.location} for e in events]})
                else:
                    self._error("Calendar not configured", 503)

            elif path == '/browser/status':
                agent = getattr(self.server, 'prism_agent', None)
                if agent and hasattr(agent, '_browser'):
                    self._json_response(agent._browser.status())
                else:
                    self._json_response({"available": False})

            elif path == '/instructions':
                agent = getattr(self.server, 'prism_agent', None)
                if agent and hasattr(agent, '_instructions'):
                    instrs = agent._instructions.all_active()
                    self._json_response({"count": len(instrs),
                        "instructions": [{"id": i.instr_id, "text": i.text,
                                          "trigger": i.trigger,
                                          "use_count": i.use_count}
                                         for i in instrs]})
                else:
                    self._json_response({"count": 0, "instructions": []})

            elif path == '/discovery/services':
                agent = getattr(self.server, 'prism_agent', None)
                if agent and hasattr(agent, '_discovery'):
                    services = agent._discovery.list_all()
                    self._json_response({"count": len(services),
                        "services": [{"name": s.name, "category": s.category,
                                      "method": s.access_method,
                                      "configured": s.configured}
                                     for s in services]})
                else:
                    self._json_response({"count": 0, "services": []})

            elif path.startswith('/documents'):
                agent = getattr(self.server, 'prism_agent', None)
                if not agent or not hasattr(agent, '_docs'):
                    self._json_response({"results": []})
                    return
                if 'q' in qs:
                    docs = agent._docs.search(qs['q'], n=10)
                else:
                    docs = agent._docs.recent(n=10)
                self._json_response({"results": [
                    {"id": d.doc_id, "title": d.title, "url": d.url,
                     "provider": d.provider, "modified": d.modified}
                    for d in docs]})

            elif path == '/calls/status':
                from prism_calls import PrismCalls
                agent = getattr(self.server, 'prism_agent', None)
                calls = getattr(agent, '_calls', None) if agent else None
                if calls is None:
                    calls = PrismCalls()
                self._json_response(calls.status_summary())

            elif path == '/messages/status':
                from prism_messaging import PrismMessaging
                agent = getattr(self.server, 'prism_agent', None)
                msgs = getattr(agent, '_messages', None) if agent else None
                if msgs is None:
                    msgs = PrismMessaging()
                self._json_response(msgs.status_summary())

            elif path.startswith('/search'):
                agent  = getattr(self.server, 'prism_agent', None)
                q      = qs.get('q', '')
                if agent and hasattr(agent, '_search') and q:
                    results = agent._search.search(q, n=8)
                    self._json_response({"query": q, "results": [
                        {"title": r.title, "url": r.url, "snippet": r.snippet}
                        for r in results]})
                else:
                    self._json_response({"query": q, "results": []})

            elif path == '/push/status':
                agent = getattr(self.server, 'prism_agent', None)
                if agent and hasattr(agent, '_push'):
                    self._json_response(agent._push.status_summary())
                else:
                    self._json_response({"configured": False})

            else:
                self._error(f"Unknown route: {path}", 404)

        except Exception as exc:
            logger.exception("GET %s failed", path)
            self._error(str(exc), 500)

    # ── POST ──────────────────────────────────────────────────────────────

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/") or "/"
        body   = self._read_body()

        try:
            if path == '/plan':
                from prism_planner import PrismPlanner
                planner = getattr(self.server, 'prism_planner', None)
                if not planner:
                    planner = PrismPlanner()
                    self.server.prism_planner = planner
                plan = planner.plan(
                    task_description = body.get('task', ''),
                    user_context     = body.get('context', {}),
                    n_plans          = body.get('n_plans', 4),
                )
                from prism_responses import plan_of_action_card
                self._json_response(plan_of_action_card(plan).to_json())
                return

            if path == '/chat':
                from prism_agent import PrismAgent

                agent = getattr(self.server, 'prism_agent', None) or PrismAgent()
                card = agent.chat(body.get('message', ''), body.get('context', {}))
                self._json_response(card.to_json())
                return

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

            elif path == "/policy/set":
                from prism_policy import ResourceAllocation

                user = body.get("user", "")
                category = body.get("category", "")
                if not user or not category:
                    self._error("'user' and 'category' fields required", 400)
                    return
                policy = self.policy_engine.get_policy(user)
                allocation = policy.allocations.get(category, ResourceAllocation(name=category))
                for field_name in (
                    "currency",
                    "total_budget",
                    "per_action_limit",
                    "monthly_limit",
                    "auto_approve_below",
                    "preferred_providers",
                    "blacklisted",
                    "time_window",
                    "notifications",
                    "notes",
                ):
                    if field_name in body:
                        setattr(allocation, field_name, body[field_name])
                self.policy_engine.set_allocation(user, category, allocation)
                self._json_response({"ok": True, "allocation": _safe_dict(allocation)})

            elif path == "/policy/update_from_chat":
                user = body.get("user", "")
                message = body.get("message", "")
                if not user or not message:
                    self._error("'user' and 'message' fields required", 400)
                    return
                result = self.policy_engine.parse_policy_update(message, user)
                self._json_response({"result": result})

            elif path == "/identity/observe":
                domain = body.get("domain")
                if not domain:
                    self._error("'domain' field required", 400)
                    return
                identity = self.agent.observe_identity(
                    domain=domain,
                    fulcrum=float(body.get("fulcrum", 0.5)),
                    rating=float(body.get("rating", 0.5)),
                    context=body.get("context") or {},
                )
                self._json_response(identity)

            elif path == "/identity/reset":
                domain = body.get("domain")
                if not domain:
                    self._error("'domain' field required", 400)
                    return
                self._json_response(self.agent.reset_identity_domain(domain))

            elif path == "/artifacts/rate":
                artifact_id = body.get("artifact_id")
                if not artifact_id:
                    self._error("'artifact_id' field required", 400)
                    return
                rating = float(body.get("rating", 0.0))
                self._json_response(self.agent.rate_artifact(artifact_id, rating))

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
                result = DomainValidator(domain).validate(cases)
                self._json_response(dataclasses.asdict(result))

            elif path == '/device/approve':
                approved = body.get('approved', False)
                task     = body.get('task', '')
                params   = body.get('params', {})
                if not approved:
                    from prism_responses import text_card
                    self._json_response(text_card("Action cancelled.").to_json())
                    return
                agent = getattr(self.server, 'device_agent', None)
                if not agent:
                    from prism_device_agent import PrismDeviceAgent
                    agent = PrismDeviceAgent.setup()
                    self.server.device_agent = agent
                result = agent.execute(task, params=params, approval_override=True)
                from prism_responses import device_result_card
                self._json_response(device_result_card(result, task).to_json())
                return

            elif path == '/device/execute':
                agent  = getattr(self.server, 'device_agent', None)
                if not agent:
                    from prism_device_agent import PrismDeviceAgent
                    agent = PrismDeviceAgent.setup()
                    self.server.device_agent = agent
                dry_run = body.get('dry_run', False)
                result  = agent.execute(
                    body.get('task', ''),
                    params  = body.get('params', {}),
                    dry_run = dry_run,
                )
                self._json_response({
                    "success":        result.success,
                    "output":         result.output[:2000],
                    "tool_used":      result.tool_used,
                    "elapsed_ms":     round(result.elapsed_ms, 1),
                    "files_created":  result.files_created,
                    "error":          result.error,
                    "undo_available": bool(result.undo_command),
                })
                return

            elif path == '/perception/ingest':
                agent = getattr(self.server, 'prism_agent', None)
                if agent and hasattr(agent, '_perception') and agent._perception:
                    agent._perception.ingest_biometrics(**body)
                    self._json_response({"ok": True})
                else:
                    self._error("Perception not initialised", 503)

            elif path == '/memory/ingest':
                agent = getattr(self.server, 'prism_agent', None)
                mem   = agent._memory if agent and hasattr(agent, '_memory') else None
                if mem is None:
                    self._error("Memory not initialised", 503)
                    return
                content = body.get("content", "")
                if not content:
                    self._error("'content' field required", 400)
                    return
                entry_id = mem.ingest(
                    content = content,
                    source  = body.get("source", "note"),
                    title   = body.get("title", ""),
                    tags    = body.get("tags"),
                )
                self._json_response({"ok": True, "entry_id": entry_id})

            elif path == '/tts':
                agent = getattr(self.server, 'prism_agent', None)
                tts   = agent._tts if agent and hasattr(agent, '_tts') else None
                action = body.get("action", "speak")
                if action == "toggle":
                    if tts:
                        enabled = tts.toggle()
                    else:
                        enabled = False
                    self._json_response({"enabled": enabled})
                elif action == "speak":
                    text = body.get("text", "")
                    if tts and text:
                        tts.speak(text)
                    self._json_response({"ok": True})
                else:
                    self._error(f"Unknown TTS action: {action}", 400)

            elif path == '/tts/speak':
                agent = getattr(self.server, 'prism_agent', None)
                tts   = agent._tts if agent and hasattr(agent, '_tts') else None
                text  = body.get("text", "")
                if tts and text:
                    tts.speak(text)
                self._json_response({"ok": True})

            elif path == '/smarthome':
                from prism_smart_home import PrismSmartHome
                sh = getattr(self.server, 'prism_smart_home', None)
                if sh is None:
                    sh = PrismSmartHome(
                        ha_url = body.get("ha_url", "http://homeassistant.local:8123"),
                        token  = body.get("token", ""),
                    )
                    self.server.prism_smart_home = sh
                action    = body.get("action", "")
                entity_id = body.get("entity_id", "")
                if action == "turn_on":
                    result = sh.turn_on(entity_id)
                    self._json_response({"ok": result.success, "error": result.error})
                elif action == "turn_off":
                    result = sh.turn_off(entity_id)
                    self._json_response({"ok": result.success, "error": result.error})
                elif action == "toggle":
                    result = sh.toggle(entity_id)
                    self._json_response({"ok": result.success, "error": result.error})
                elif action == "list":
                    devices = sh.list_devices(domain=body.get("domain", ""))
                    self._json_response({
                        "devices": [
                            {"entity_id": d.entity_id, "state": d.state,
                             "friendly_name": d.friendly_name}
                            for d in devices
                        ]
                    })
                else:
                    self._error(f"Unknown smarthome action: {action}", 400)

            elif path == '/perception/enable':
                channel = body.get("channel", "")
                enabled = body.get("enabled", True)
                agent   = getattr(self.server, 'prism_agent', None)
                if agent and hasattr(agent, '_perception') and agent._perception:
                    for ch in agent._perception._channels:
                        if ch.NAME == channel:
                            ch.resume() if enabled else ch.pause()
                    self._json_response({"channel": channel, "enabled": enabled})
                else:
                    self._error("Perception not initialised", 503)

            elif path == '/email/send':
                from prism_email import PrismEmail
                agent = getattr(self.server, 'prism_agent', None)
                em = getattr(agent, '_email', None) if agent else None
                if em is None:
                    em = PrismEmail()
                if not em.configured:
                    self._error("Email not configured", 503)
                    return
                to      = body.get("to", "")
                subject = body.get("subject", "")
                text    = body.get("body", "")
                if not to or not subject or not text:
                    self._error("'to', 'subject' and 'body' fields required", 400)
                    return
                ok = em.send(to=to, subject=subject, body=text,
                             reply_to=body.get("reply_to", ""))
                self._json_response({"ok": ok})

            elif path == '/instructions':
                agent = getattr(self.server, 'prism_agent', None)
                if agent and hasattr(agent, '_instructions'):
                    instr = agent._instructions.add(
                        body.get('text', ''), body.get('trigger', 'always'))
                    self._json_response({"id": instr.instr_id, "text": instr.text})
                else:
                    self._error("Instructions not initialised", 503)

            elif path == '/discovery/build':
                agent = getattr(self.server, 'prism_agent', None)
                if agent and hasattr(agent, '_discovery'):
                    service = agent._discovery.get(body.get('service_id', ''))
                    if service:
                        ok  = agent._discovery.build_integration(
                            service, body.get('answers', {}))
                        msg = agent._discovery.confirmation_message(service)
                        self._json_response({"success": ok, "message": msg})
                    else:
                        self._error("Service not found", 404)
                else:
                    self._error("Discovery not initialised", 503)

            elif path == '/llm/set':
                router = getattr(self.server, 'llm_router', None)
                if router:
                    provider_model = body.get('preferred', '')
                    router.set_preferred(provider_model)
                    self._json_response({"ok": True, "preferred": provider_model})
                else:
                    self._error("LLM router not initialised", 503)
                return

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
        policy_engine=None,
        tool_finder=None,
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
        try:
            from prism_agent import PrismAgent

            self._prism_agent = PrismAgent(kde_agent=agent)
        except Exception:
            self._prism_agent = None
        self._policy_engine = policy_engine
        self._tool_finder = tool_finder
        if self._policy_engine is None:
            try:
                from prism_policy import PolicyEngine

                self._policy_engine = PolicyEngine()
            except Exception:
                self._policy_engine = None
        if self._tool_finder is None:
            try:
                from prism_collaborator import PrismCollaborator
                from prism_tool_finder import ToolFinder

                self._tool_finder = ToolFinder(collaborator=PrismCollaborator())
            except Exception:
                self._tool_finder = None

        try:
            from prism_llm_router import LLMRouter
            self._llm_router = LLMRouter.from_config()
        except Exception:
            self._llm_router = None
        try:
            from prism_task_queue import TaskQueue
            self._task_queue = TaskQueue()
        except Exception:
            self._task_queue = None

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
        policy_engine   = self._policy_engine
        tool_finder     = self._tool_finder
        llm_router      = self._llm_router
        task_queue      = self._task_queue

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
        _Handler.policy_engine   = policy_engine
        _Handler.tool_finder     = tool_finder
        _Handler.llm_router      = llm_router
        _Handler.task_queue      = task_queue
        _Handler._moment_history = {}  # player → list[MomentResult]

        self._server = HTTPServer((self._host, self._port), _Handler)
        self._server.prism_agent = self._prism_agent
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
