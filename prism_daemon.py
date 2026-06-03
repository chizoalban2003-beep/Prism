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
                logger.debug("OrganBus flush error: %s", exc)


def _horizon_worker(agent, interval: int = 300):
    """Evaluate horizon goals every `interval` seconds (default 5 min)."""
    while not _SHUTDOWN.wait(timeout=interval):
        h = getattr(agent, '_horizon', None)
        if h is None:
            continue
        try:
            triggered = h.check_now()
            if triggered:
                logger.info("HorizonPlanner: %d goal(s) triggered in background", len(triggered))
        except Exception as exc:
            logger.debug("HorizonPlanner check error: %s", exc)


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
# HTTP server thread
# ---------------------------------------------------------------------------

def _server_thread(agent, host: str, port: int):
    """Start the KDE/PRISM HTTP server in a daemon thread."""
    try:
        from kde_server import KDEServer
        server = KDEServer(agent=agent, host=host, port=port)
        logger.info("HTTP server listening on %s:%d", host, port)
        server.serve_forever()
    except Exception as exc:
        logger.error("HTTP server failed: %s", exc)
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
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("PRISM_PORT", 8742)))
    parser.add_argument("--host",
                        default=os.environ.get("PRISM_HOST", "127.0.0.1"))
    parser.add_argument("--no-server", action="store_true",
                        help="Run daemon without the HTTP server")
    args = parser.parse_args()

    logger.info("PRISM daemon starting (pid=%d)", os.getpid())

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
        threading.Thread(target=_bus_flush_worker,  args=(agent,), daemon=True, name="bus-flush"),
        threading.Thread(target=_horizon_worker,    args=(agent,), daemon=True, name="horizon"),
        threading.Thread(target=_health_worker,     args=(agent,), daemon=True, name="health"),
    ]
    for w in workers:
        w.start()
    logger.info("Background workers started: %s", [w.name for w in workers])

    # HTTP server
    if not args.no_server:
        srv = threading.Thread(
            target=_server_thread, args=(agent, args.host, args.port),
            daemon=True, name="http-server",
        )
        srv.start()

    # Main loop — keep alive until shutdown signal
    logger.info("PRISM daemon running. Send SIGTERM or SIGINT to stop.")
    try:
        while not _SHUTDOWN.is_set():
            _SHUTDOWN.wait(timeout=5)
    finally:
        logger.info("Shutting down...")
        if hasattr(agent, 'stop'):
            try:
                agent.stop()
                logger.info("PrismAgent stopped cleanly")
            except Exception as exc:
                logger.warning("Agent stop error: %s", exc)
        logger.info("PRISM daemon exited after %.0fs", time.time() - _START_TIME)


if __name__ == "__main__":
    main()
