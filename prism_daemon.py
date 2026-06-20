"""
prism_daemon.py
===============
PRISM continuous-operation daemon.

Runs PRISM as a persistent background process — not session-based, but
always-on.  Handles graceful shutdown, health monitoring, OrganBus batch
flushing, and horizon goal evaluation on a schedule.

Usage
-----
    # Direct run (foreground, logs to stdout):
    python3 prism_daemon.py

    # Run identity ceremony first (first-time setup):
    python3 prism_daemon.py --ceremony

    # systemd service unit (place in ~/.config/systemd/user/prism.service):
    #
    # [Unit]
    # Description=PRISM Personal Intelligence Daemon
    # After=network.target
    #
    # [Service]
    # ExecStart=/usr/bin/python3 /path/to/prism_daemon.py
    # Restart=on-failure
    # RestartSec=10
    # StandardOutput=journal
    # StandardError=journal
    #
    # [Install]
    # WantedBy=default.target

Environment
-----------
PRISM_LOG_LEVEL   DEBUG / INFO / WARNING  (default INFO)
PRISM_PORT        HTTP server port         (default 8742)
PRISM_HOST        HTTP server host         (default 127.0.0.1)
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

logging.basicConfig(
    level=os.environ.get("PRISM_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("prism.daemon")

_SHUTDOWN = threading.Event()


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

def _handle_signal(signum, _frame):
    sig_name = signal.Signals(signum).name
    logger.info("Daemon received %s — initiating graceful shutdown", sig_name)
    _SHUTDOWN.set()


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT,  _handle_signal)


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------

def _bus_flush_worker(agent, interval: int = 60):
    """Flush LOW-priority OrganBus signals every `interval` seconds."""
    while not _SHUTDOWN.wait(timeout=interval):
        ob = getattr(agent, '_organ_bus', None)
        if ob is not None:
            try:
                records = ob.flush_batch()
                if records:
                    logger.debug("OrganBus: flushed %d batched signal(s)", len(records))
            except Exception as exc:
                logger.warning("OrganBus flush error: %s", exc, exc_info=True)


def _horizon_worker(agent, interval: int = 300):
    """Evaluate horizon goals every `interval` seconds (default 5 min)."""
    while not _SHUTDOWN.wait(timeout=interval):
        h = getattr(agent, '_horizon', None)
        if h is None:
            continue
        # Phase gate: defer evaluation in LIQUID (system under thermal stress)
        try:
            import prism_phase as _pp
            _engine = _pp.get_engine()
            if _engine.history and _engine.current_phase.value == "LIQUID":
                logger.debug("[horizon] LIQUID phase — deferring goal evaluation")
                continue
        except Exception:
            pass
        try:
            triggered = h.on_session_start()
            if triggered:
                logger.info("HorizonPlanner: %d goal(s) triggered in background", len(triggered))
        except Exception as exc:
            logger.warning("HorizonPlanner check error: %s", exc, exc_info=True)


def _reflection_worker(agent, interval: int = 604800):
    """Run weekly reflection every `interval` seconds (default 7 days)."""
    while not _SHUTDOWN.wait(timeout=interval):
        refl = getattr(agent, '_reflection', None)
        if refl is None:
            continue
        try:
            report = refl.run()
            logger.info(
                "Reflection: %d pattern(s), %d belief proposal(s), %d stale goal(s)",
                len(report.patterns),
                len(report.belief_proposals),
                len(report.unresolved_goals),
            )
        except Exception as exc:
            logger.warning("Reflection error: %s", exc, exc_info=True)


def _outcome_feed_worker(agent, interval: int = 3600):
    """Feed outcome deltas into soul and horizon every hour."""
    while not _SHUTDOWN.wait(timeout=interval):
        tracker = getattr(agent, '_outcome_tracker', None)
        if tracker is None:
            continue
        try:
            soul    = getattr(agent, '_soul', None)
            horizon = getattr(agent, '_horizon', None)
            if soul:
                n = tracker.feed_soul(soul)
                if n:
                    logger.debug("OutcomeTracker: fed %d soul update(s)", n)
            if horizon:
                tracker.feed_horizon(horizon)
        except Exception as exc:
            logger.warning("OutcomeTracker feed error: %s", exc, exc_info=True)


def _surprise_reflection_worker(agent, interval: int = 3600):
    """
    Fire an immediate reflection when the 7-day completion rate drops 15+
    points vs. the prior 7-day baseline — indicating a sudden performance dip.
    """
    while not _SHUTDOWN.wait(timeout=interval):
        try:
            tracker    = getattr(agent, "_outcome_tracker", None)
            reflection = getattr(agent, "_reflection", None)
            if tracker is None or reflection is None:
                continue
            recent = tracker.stats(days=7)
            full   = tracker.stats(days=14)
            recent_total = recent.get("total", 0)
            full_total   = full.get("total", 0)
            prior_total  = full_total - recent_total
            if prior_total >= 5 and recent_total >= 5:
                prior_done  = (full.get("done", 0) or 0) - (recent.get("done", 0) or 0)
                prior_rate  = prior_done / prior_total
                recent_rate = recent.get("completion_rate", 0)
                delta       = recent_rate - prior_rate
                if delta < -0.15:
                    logger.warning(
                        "[daemon] surprise reflection: rate %.0f%% → %.0f%% (Δ=%.2f)",
                        prior_rate * 100, recent_rate * 100, delta,
                    )
                    reflection.run()
        except Exception as exc:
            logger.warning("Surprise reflection error: %s", exc, exc_info=True)


def _crystalliser_worker(agent, interval: int = 3600):
    """Hourly deep analysis of recent interactions."""
    while not _SHUTDOWN.wait(timeout=interval):
        crystalliser = getattr(agent, '_crystalliser', None)
        if crystalliser is None:
            continue
        try:
            n = crystalliser.deep_analyse(lookback_hours=2)
            if n > 0:
                logger.info("[crystalliser] Updated %d persona signals", n)
        except Exception as exc:
            logger.warning("[crystalliser] Error: %s", exc, exc_info=True)


def _narrative_worker(agent, interval: int = 604800):
    """Weekly narrative generation stored to memory."""
    while not _SHUTDOWN.wait(timeout=interval):
        narrative = getattr(agent, '_narrative', None)
        if narrative is None:
            continue
        try:
            _ = narrative.weekly()
            logger.info("[narrative] Weekly narrative generated")
        except Exception as exc:
            logger.warning("[narrative] Error: %s", exc, exc_info=True)


def _phase_ticker_worker(agent, interval: int = 10):
    """Recompute Φ_melt every 10 s so phase stays current for all consumers."""
    try:
        import prism_phase as _pp
        engine = _pp.get_engine()
    except Exception:
        return  # phase module unavailable; skip silently
    while not _SHUTDOWN.wait(timeout=interval):
        try:
            soul    = getattr(agent, '_soul', None)
            kinetic = getattr(agent, '_kinetic', None)
            engine.compute(soul=soul, kinetic=kinetic)
        except Exception as exc:
            logger.debug("[phase-ticker] Error: %s", exc)


def _lora_weekly_worker(agent, interval: int = 604800):
    """Weekly LoRA training — skipped when system is under thermal/RAM pressure."""
    while not _SHUTDOWN.wait(timeout=interval):
        trainer = getattr(agent, '_lora_trainer', None)
        if trainer is None:
            continue
        # Phase gate: defer training if VISCOUS or LIQUID
        try:
            import prism_phase as _pp
            _engine = _pp.get_engine()
            if _engine.history:
                _phase = _engine.current_phase
                if _phase.value in ("VISCOUS", "LIQUID"):
                    logger.info("[lora-weekly] phase=%s — deferring training", _phase.value)
                    continue
        except Exception:
            pass
        # RAM gate: require ≥ 4 GB free
        try:
            import psutil as _ps
            if _ps.virtual_memory().available < 4 * 1024 ** 3:
                logger.info("[lora-weekly] <4 GB RAM free — deferring training")
                continue
        except Exception:
            pass
        # Concurrency guard: skip if a job is already running
        try:
            if any(j.status == "running" for j in trainer.list_jobs()):
                logger.debug("[lora-weekly] training job already running — skipping")
                continue
        except Exception:
            pass
        try:
            trainer.start_training()
            logger.info("[lora-weekly] Training started")
        except Exception as exc:
            logger.debug("[lora-weekly] Error: %s", exc)


def _federation_push_worker(agent, interval: int = 300):
    """Push pending federation state — skipped when system is in LIQUID phase."""
    while not _SHUTDOWN.wait(timeout=interval):
        # Phase gate: skip in LIQUID (thermal throttling — avoid extra network I/O)
        try:
            import prism_phase as _pp
            _engine = _pp.get_engine()
            if _engine.history and _engine.current_phase.value == "LIQUID":
                logger.debug("[federation-push] LIQUID phase — skipping push")
                continue
        except Exception:
            pass
        fed = getattr(agent, '_federation', None)
        if fed is not None and hasattr(fed, 'push_pending'):
            try:
                result = fed.push_pending()
                if result and result.get("pushed"):
                    logger.debug("[federation-push] pushed=%d failed=%d",
                                 result.get("pushed", 0), result.get("failed", 0))
            except Exception as exc:
                logger.debug("[federation-push] Error: %s", exc)


def _health_worker(agent, interval: int = 120):
    """Log a brief health line every `interval` seconds."""
    while not _SHUTDOWN.wait(timeout=interval):
        try:
            status = agent.status() if hasattr(agent, 'status') else {}
            soul   = getattr(agent, '_soul', None)
            chains = Path("~/.prism/chains.db").expanduser()
            soul_beliefs = len(soul.list_beliefs()) if soul else 0
            logger.info(
                "[health] uptime=%.0fs beliefs=%d horizon=%s chains_db=%s",
                time.time() - _START_TIME,
                soul_beliefs,
                status.get("horizon_goals", "?"),
                "ok" if chains.exists() else "missing",
            )
        except Exception as exc:
            logger.debug("Health check error: %s", exc)


_START_TIME = time.time()


# ---------------------------------------------------------------------------
# Identity ceremony (first-time setup)
# ---------------------------------------------------------------------------

def _run_ceremony(agent):
    """Interactive identity ceremony via stdin/stdout."""
    print("\n" + "=" * 70)
    print("  PRISM — Identity Ceremony")
    print("  Setting up your digital soul. This takes about 3 minutes.")
    print("  Your answers are stored locally. Nothing leaves your machine.")
    print("=" * 70 + "\n")

    try:
        from prism_identity_ceremony import IdentityCeremony
        soul = getattr(agent, '_soul', None)
        if soul is None:
            from prism_soul import PrismSoul
            soul = PrismSoul(llm_router=getattr(agent, '_router', None))

        ceremony = IdentityCeremony(
            soul       = soul,
            llm_router = getattr(agent, '_router', None),
        )

        answers = {}
        for i, (key, question) in enumerate(
            __import__('prism_identity_ceremony').CEREMONY_QUESTIONS.items(), 1
        ):
            print(f"  [{i}/7] {question}\n")
            answer = input("  > ").strip()
            if answer:
                answers[key] = answer
            print()

        seed = ceremony.run_from_answers(answers)
        print("\n  Soul seed created.")
        print(f"  Values: {', '.join(seed.stated_values[:4])}")
        print(f"  Goals:  {', '.join(seed.stated_goals[:2])}")
        print("\n  Saved to ~/.prism/soul.md — you can edit this at any time.")
        print("=" * 70 + "\n")
        return soul

    except KeyboardInterrupt:
        print("\n  Ceremony interrupted — you can run it again with --ceremony")
        return None
    except Exception as exc:
        logger.warning("Ceremony failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# HTTP server threads
# ---------------------------------------------------------------------------

def _build_asgi_state(agent) -> dict:
    """Build the full dependency dict for prism_asgi._set_state()."""
    state: dict = {"agent": agent, "active_session_id": None}

    try:
        from prediction_engine import PredictionPlatform
        state["platform"] = _get_or_build(agent, "_platform", PredictionPlatform)
    except Exception:
        pass

    try:
        from moment_analyzer import MomentAnalyzer
        state["moment_analyzer"] = MomentAnalyzer()
    except Exception:
        pass

    try:
        from duel_analyzer import DuelAnalyzer
        state["duel_analyzer"] = DuelAnalyzer()
    except Exception:
        pass

    try:
        from domain_configs import ALL_DOMAINS, DomainDecisionModel
        state["domain_models"] = {
            name: DomainDecisionModel(cfg) for name, cfg in ALL_DOMAINS.items()
        }
    except Exception:
        state["domain_models"] = {}

    try:
        from moment_pipeline import LiveMomentPipeline
        ma = state.get("moment_analyzer")
        if ma is not None:
            state["live_pipeline"] = LiveMomentPipeline(ma)
    except Exception:
        pass

    try:
        from prism_policy import PolicyEngine
        state["policy_engine"] = PolicyEngine()
    except Exception:
        pass

    try:
        from prism_collaborator import PrismCollaborator
        from prism_tool_finder import ToolFinder
        state["tool_finder"] = ToolFinder(collaborator=PrismCollaborator())
    except Exception:
        pass

    try:
        from prism_llm_router import LLMRouter
        state["llm_router"] = LLMRouter.from_config()
    except Exception:
        pass

    try:
        from prism_task_queue import TaskQueue
        state["task_queue"] = TaskQueue()
    except Exception:
        pass

    # Federation: lazily construct one process-wide FederationManager and pin
    # it both into _state (where prism_routes_federation looks it up) and onto
    # the agent (where _federation_push_worker looks). Without this wiring,
    # /federation/* routes always 503 and the push worker is a no-op.
    try:
        from prism_federation import FederationManager
        fed = getattr(agent, "_federation", None) or FederationManager()
        agent._federation = fed
        state["federation"] = fed
    except Exception as _exc:
        logger.warning("FederationManager wire failed: %s", _exc)

    # Causal reasoner (belief DAG + counterfactuals) — without this the
    # /causality/* routes always answered "causal_reasoner not configured".
    try:
        from prism_causality import CausalGraph, CausalReasoner
        graph = CausalGraph()
        state["causal_reasoner"] = CausalReasoner(
            graph,
            soul=getattr(agent, "_soul", None),
            llm_router=getattr(agent, "_router", None),
        )
    except Exception as _exc:
        logger.warning("CausalReasoner wire failed: %s", _exc)

    # Multi-user registry + household bus — without these the /users and
    # /household routes answered 503 "UserRegistry not available" even though
    # the daemon was running.
    try:
        from prism_multi_user import HouseholdBus, UserRegistry
        reg = getattr(agent, "_user_registry", None) or UserRegistry()
        agent._user_registry = reg
        state["user_registry"] = reg
        state["household_bus"] = HouseholdBus(
            reg, organ_bus=getattr(agent, "_organ_bus", None)
        )
    except Exception as _exc:
        logger.warning("UserRegistry wire failed: %s", _exc)

    # Mobile sync manager — needed by the /mobile/* routes.
    try:
        from prism_mobile_sync import MobileSyncManager
        _mobile_secret = os.environ.get("PRISM_MOBILE_SECRET", "")
        if not _mobile_secret:
            logger.warning(
                "Mobile sync using the built-in default HMAC secret. "
                "Set PRISM_MOBILE_SECRET=<secret> to secure device tokens."
            )
        state["mobile_sync"] = MobileSyncManager(secret_key=_mobile_secret)
    except Exception as _exc:
        logger.warning("MobileSyncManager wire failed: %s", _exc)

    # OutcomeTracker — surfaced for routes that read it (e.g. ml nightly sweep).
    state.setdefault("outcome_tracker", getattr(agent, "_outcome_tracker", None))

    state["ml_assembler"] = getattr(agent, "_ml_assembler", None)

    try:
        from prism_vision_ml_bridge import VisionMLBridge, get_or_set_bridge
        asm = state.get("ml_assembler")
        if asm is not None:
            bridge = get_or_set_bridge()
            if bridge is None:
                bridge = VisionMLBridge(assembler=asm)
                get_or_set_bridge(bridge)
            state["vision_ml_bridge"] = bridge
    except Exception:
        pass

    return state


def _get_or_build(agent, attr: str, factory):
    val = getattr(agent, attr, None)
    return val if val is not None else factory()


# ---------------------------------------------------------------------------
# Durability stack — WAL-backed graph memory (hot → WAL → cold)
# ---------------------------------------------------------------------------

def _start_durability(agent):
    """Bring the WAL-backed memory-graph durability stack online.

    On startup we (1) replay any uncommitted WAL entries into the hot buffer,
    (2) run the shadow pipeline that drains hot → cold on a fixed interval,
    and (3) supervise it with a watchdog that resurrects the pipeline if it
    dies while mutations are still pending. Without this wiring the graph WAL
    accumulates (Ψ > 0) and crash recovery never runs in production.

    Returns ``(graph, pipeline, watchdog)`` or ``(None, None, None)`` if the
    stack could not be started (degrades gracefully — the flat PrismMemory
    recall path is unaffected).
    """
    try:
        from prism_memory_graph import PrismMemoryGraph
        from prism_shadow_pipeline import PrismShadowPipeline
        from prism_watchdog import PrismWatchdog
    except Exception as exc:
        logger.warning("[durability] stack unavailable: %s", exc)
        return None, None, None

    try:
        graph = getattr(agent, "_memory_graph", None) or PrismMemoryGraph()
        agent._memory_graph = graph

        replayed = graph.replay_wal()
        if replayed:
            logger.info(
                "[durability] replayed %d uncommitted WAL entr%s on startup",
                replayed, "y" if replayed == 1 else "ies",
            )
        try:
            import prism_metrics as _pm
            _pm.metrics.inc("wal_replays", replayed or 0)
        except Exception:
            pass

        phase_engine = None
        try:
            import prism_phase as _pp
            phase_engine = _pp.get_engine()
        except Exception:
            pass

        pipeline = PrismShadowPipeline(
            graph,
            soul=getattr(agent, "_soul", None),
            phase_engine=phase_engine,
        )
        pipeline.start()

        watchdog = PrismWatchdog(pipeline)
        watchdog.start()

        agent._shadow_pipeline = pipeline
        agent._watchdog = watchdog
        logger.info("[durability] shadow pipeline + watchdog online")
        return graph, pipeline, watchdog
    except Exception as exc:
        logger.warning("[durability] failed to start: %s", exc)
        return None, None, None


def _asgi_server_thread(agent, host: str, port: int):
    """Start the FastAPI/ASGI server (primary HTTP server, Phase 4+)."""
    try:
        import prism_asgi
        from prism_state import _set_state
        _set_state(**_build_asgi_state(agent))
        logger.info("ASGI server starting on %s:%d", host, port)
        prism_asgi.serve(host=host, port=port, log_level="warning")
    except Exception as exc:
        logger.error("ASGI server failed: %s", exc)
        _SHUTDOWN.set()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_agent():
    """Construct and return a fully wired PrismAgent."""
    from prism_agent import PrismAgent
    agent = PrismAgent()
    return agent


def main():
    parser = argparse.ArgumentParser(description="PRISM Personal Intelligence Daemon")
    parser.add_argument("--ceremony", action="store_true",
                        help="Run the identity ceremony before starting")
    parser.add_argument("--setup-llm", action="store_true",
                        help="Run the LLM setup wizard and exit")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("PRISM_PORT", 8742)))
    parser.add_argument("--host",
                        default=os.environ.get("PRISM_HOST", "127.0.0.1"))
    parser.add_argument("--no-server", action="store_true",
                        help="Run daemon without the HTTP server")
    args = parser.parse_args()

    if args.setup_llm:
        from prism_setup_llm import run_wizard
        run_wizard()
        sys.exit(0)

    logger.info("PRISM daemon starting (pid=%d)", os.getpid())

    # Ensure the HTTP bearer auth token exists before the ASGI server boots.
    # The daemon is the only process expected to call this; tests run the
    # ASGI app directly and opt out via PRISM_AUTH_DISABLE.
    try:
        from prism_auth import TOKEN_FILE as _AUTH_TOKEN_FILE
        from prism_auth import ensure_token
        ensure_token()
        logger.info("HTTP auth token at %s (chmod 600)", _AUTH_TOKEN_FILE)
    except Exception as _exc:
        logger.warning("Failed to ensure auth token: %s", _exc)

    # Eagerly migrate all 22 SQLite databases before agent touches them.
    try:
        from prism_schema_registry import run_migrations
        _mig = run_migrations()
        _errs = {k: v for k, v in _mig.items() if v.startswith("error:")}
        if _errs:
            logger.warning("Schema migration issues: %s", _errs)
        else:
            logger.info("Schema migrations: %d db(s) ok", len(_mig))
    except Exception as _exc:
        logger.warning("Schema migration step failed: %s", _exc)

    # Build agent
    try:
        agent = build_agent()
        logger.info("PrismAgent ready")
    except Exception as exc:
        logger.critical("Failed to build PrismAgent: %s", exc)
        sys.exit(1)

    # Identity ceremony
    if args.ceremony:
        soul = _run_ceremony(agent)
        if soul and hasattr(agent, '_soul'):
            agent._soul = soul
            if hasattr(agent, '_chain'):
                agent._chain._soul = soul

    elif not getattr(getattr(agent, '_soul', None), 'has_seed', lambda: True)():
        logger.info(
            "No soul seed found. Run `python3 prism_daemon.py --ceremony` to personalise PRISM."
        )

    # Background workers
    workers = [
        threading.Thread(target=_phase_ticker_worker,        args=(agent,), daemon=True, name="phase-ticker"),
        threading.Thread(target=_bus_flush_worker,    args=(agent,), daemon=True, name="bus-flush"),
        threading.Thread(target=_horizon_worker,      args=(agent,), daemon=True, name="horizon"),
        threading.Thread(target=_health_worker,       args=(agent,), daemon=True, name="health"),
        threading.Thread(target=_reflection_worker,         args=(agent,), daemon=True, name="reflection"),
        threading.Thread(target=_outcome_feed_worker,       args=(agent,), daemon=True, name="outcome-feed"),
        threading.Thread(target=_surprise_reflection_worker,args=(agent,), daemon=True, name="surprise-refl"),
        threading.Thread(target=_crystalliser_worker,       args=(agent,), daemon=True, name="crystalliser"),
        threading.Thread(target=_narrative_worker,          args=(agent,), daemon=True, name="narrative"),
        threading.Thread(target=_lora_weekly_worker,        args=(agent,), daemon=True, name="lora-weekly"),
        threading.Thread(target=_federation_push_worker,    args=(agent,), daemon=True, name="federation-push"),
    ]
    for w in workers:
        w.start()
    logger.info("Background workers started: %s", [w.name for w in workers])

    # Durability: replay WAL, then run the shadow pipeline under the watchdog.
    _graph, _pipeline, _watchdog = _start_durability(agent)

    # Federation security advisory — quiet when config or env covers it.
    import os as _os
    _fed_cfg = (getattr(agent, "_config", {}) or {}).get("federation", {}) or {}
    _fed_strict = bool(
        _os.environ.get("PRISM_FEDERATION_REQUIRE_AUTH")
        or _fed_cfg.get("require_auth")
    )
    _fed_token = bool(
        _os.environ.get("PRISM_FEDERATION_TOKEN")
        or _fed_cfg.get("token")
    )
    if not (_fed_strict and _fed_token):
        logger.warning(
            "Federation running in legacy-permissive mode. "
            "Set [federation] require_auth=true and token=<secret> in "
            "prism_config.toml (or PRISM_FEDERATION_REQUIRE_AUTH=1 + "
            "PRISM_FEDERATION_TOKEN=<secret>) to harden multi-node deployments."
        )
    if not _os.environ.get("PRISM_FEDERATION_HMAC_SECRET"):
        logger.warning(
            "Federation payload signing disabled. "
            "Set PRISM_FEDERATION_HMAC_SECRET=<secret> to enable HMAC-SHA256 "
            "verification on incoming sync and identity-merge payloads."
        )

    # Primary ASGI server (FastAPI/uvicorn — Phase 4: sole HTTP server on main port)
    if not args.no_server:
        asgi_srv = threading.Thread(
            target=_asgi_server_thread, args=(agent, args.host, args.port),
            daemon=True, name="asgi-server",
        )
        asgi_srv.start()

    # Main loop — keep alive until shutdown signal
    logger.info("PRISM daemon running. Send SIGTERM or SIGINT to stop.")
    try:
        while not _SHUTDOWN.is_set():
            _SHUTDOWN.wait(timeout=5)
    finally:
        logger.info("Shutting down...")
        # Stop the durability stack first so a final commit drains the hot
        # buffer before the process exits (watchdog must stop before pipeline
        # so it doesn't resurrect the pipeline mid-shutdown).
        if _watchdog is not None:
            try:
                _watchdog.stop()
            except Exception as exc:
                logger.warning("Watchdog stop error: %s", exc)
        if _pipeline is not None:
            try:
                _pipeline.stop()
            except Exception as exc:
                logger.warning("Shadow pipeline stop error: %s", exc)
        if hasattr(agent, 'stop'):
            try:
                agent.stop()
                logger.info("PrismAgent stopped cleanly")
            except Exception as exc:
                logger.warning("Agent stop error: %s", exc)
        logger.info("PRISM daemon exited after %.0fs", time.time() - _START_TIME)


if __name__ == "__main__":
    main()
