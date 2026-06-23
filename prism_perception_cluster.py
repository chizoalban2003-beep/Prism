"""
prism_perception_cluster.py
===========================
Factory for the perception / proactive / kinetic cluster.

Three fail-soft components that together form PRISM's awareness layer:

* **PrismPerception** — context aggregator (time, biometrics, system
  state, typing, voice, screen). Runs a background fuser thread.
* **PrismProactive**  — trigger evaluator. Registers a default trigger
  set built from perception/policy/queue, then runs as a daemon.
* **KineticEngine**   — compound-signal physics aggregator. Wires
  *into* perception's fuser (so it sees signals in real time) and
  *out to* proactive (so action windows fire as scheduled messages).

The inter-component wiring is the reason these belong together — the
agent constructor used to thread perception/proactive references back
and forth across three separate try/except blocks. The factory contains
that wiring in one named place; each component remains independently
fail-soft via :func:`safe_init`.

The agent still owns the ``proactive._push = …`` back-patch after this
factory returns: ``PrismPush`` is constructed later in the agent
bootstrap, so that wire crosses cluster boundaries and stays out here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from prism_agent_bootstrap import safe_init
from prism_perception import PrismPerception
from prism_proactive import PrismProactive, build_default_triggers


@dataclass
class PerceptionCluster:
    perception: Optional[PrismPerception]
    proactive:  Optional[PrismProactive]
    kinetic:    Optional[Any]


def build_perception_cluster(
    *,
    config:             Mapping[str, Any],
    policy:             Any,
    task_queue:         Any,
    on_voice_command:   Callable[..., Any],
    on_proactive_event: Callable[..., Any],
    logger:             logging.Logger,
) -> PerceptionCluster:
    """Build the perception/proactive/kinetic cluster. Each member is
    fail-soft and the cross-wiring between them is best-effort — a
    failure in one leaves the others intact."""

    def _build_perception():
        cfg = config.get("agent", {}) if config else {}
        p = PrismPerception.setup(
            enable_voice     = cfg.get("enable_voice", False),
            enable_screen    = cfg.get("enable_screen", False),
            enable_typing    = cfg.get("enable_typing", True),
            enable_system    = cfg.get("enable_system", True),
            enable_biometric = cfg.get("enable_biometric", True),
            on_voice_command = on_voice_command,
        )
        p.start()
        return p
    perception = safe_init("PrismPerception", _build_perception, logger=logger)

    def _build_proactive():
        proactive = PrismProactive(on_event=on_proactive_event)
        triggers = build_default_triggers(
            perception    = perception,
            policy_engine = policy,
            task_queue    = task_queue,
        )
        for t in triggers:
            proactive.register(t)
        proactive.start()
        return proactive
    proactive = safe_init("PrismProactive", _build_proactive, logger=logger)

    def _build_kinetic():
        from prism_kinetic_engine import KineticEngine
        from prism_routes_kinetic import get_or_set_engine
        engine = KineticEngine.for_prism()
        get_or_set_engine(engine)
        if proactive is not None:
            import time as _time
            def _on_kinetic_action(window: Any) -> None:
                fire_at = _time.time() + 2.0  # slight delay so context is ready
                proactive.schedule(
                    window.to_proactive_message(), fire_at,
                    trigger_id=f"kinetic_{window.window_id}")
            engine.on_action(_on_kinetic_action)
        if perception is not None:
            fuser = getattr(perception, '_fuser', None)
            if fuser is not None:
                fuser._kinetic = engine
        logger.info("KineticEngine ready (%d levers)", len(engine._levers))
        return engine
    kinetic = safe_init("KineticEngine", _build_kinetic, logger=logger)

    return PerceptionCluster(
        perception = perception,
        proactive  = proactive,
        kinetic    = kinetic,
    )
