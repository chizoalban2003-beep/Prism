"""
prism_identity_learning.py
==========================
Factory for the identity & learning cluster.

Six fail-soft components that together form PRISM's self-model and its
learning loop:

* **PrismSoul**           — living identity document (beliefs, lenses)
* **OutcomeTracker**      — closes the prediction → outcome learning loop
* **PrismPersona**        — observed-vs-stated user traits
* **PrismCrystalliser**   — nightly grid-search hyperparameter tuner
* **PrismNarrative**      — long-form self-summary engine
* **PrismReflection**     — weekly meta-learning loop

The original ``PrismAgent.__init__`` built these in an order that left
Crystalliser holding ``outcome_tracker=None`` at construction (it was
back-patched later via ``_wire_backpatches``). This factory reorders
construction so OutcomeTracker exists *before* the persona block, which
breaks that cycle for the in-cluster wiring. Cross-cluster back-patches
(chain, kinetic, ml_assembler, orchestrator) still run via
``_wire_backpatches`` on the agent — this module does not touch them.

Every component is wrapped in :func:`safe_init` from
``prism_agent_bootstrap`` so any single failure leaves the others
intact. Deferred imports inside closures preserve the existing
resilience invariant: an ImportError on one optional module never
prevents the rest of the cluster from being built.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from prism_agent_bootstrap import safe_init, safe_init_class


@dataclass
class IdentityLearningCluster:
    soul:            Optional[Any]
    outcome_tracker: Optional[Any]
    persona:         Optional[Any]
    crystalliser:    Optional[Any]
    narrative:       Optional[Any]
    reflection:      Optional[Any]


def build_identity_learning(
    *,
    router:       Any,
    organ_bus:    Any,
    chain:        Any,
    horizon:      Any,
    memory:       Any,
    calibration:  Any,
    ml_assembler: Any,
    logger:       logging.Logger,
) -> IdentityLearningCluster:
    """Build the identity & learning cluster. Each member is independent
    fail-soft; the returned dataclass holds ``None`` for any component
    that failed to construct."""

    def _build_soul():
        from prism_soul import PrismSoul
        soul = PrismSoul(llm_router=router)
        if organ_bus is not None:
            soul.register_with_bus(organ_bus)
        if chain is not None:
            chain._soul = soul
        if not soul.has_seed():
            logger.info(
                "PrismSoul: no soul seed found — run identity ceremony to personalise")
        else:
            logger.info("PrismSoul: loaded (%d beliefs, %d lenses)",
                        len(soul.list_beliefs()),
                        len(soul.list_lenses()))
        return soul
    soul = safe_init("PrismSoul", _build_soul, logger=logger)

    def _build_outcome_tracker():
        from prism_outcome_tracker import OutcomeTracker
        tracker = OutcomeTracker(soul=soul, horizon=horizon)
        if chain is not None:
            chain._outcome_tracker = tracker
        logger.info("OutcomeTracker ready")
        return tracker
    outcome_tracker = safe_init(
        "OutcomeTracker", _build_outcome_tracker, logger=logger)

    def _build_living_model():
        from prism_crystalliser import PrismCrystalliser
        from prism_narrative import PrismNarrative
        from prism_persona import PrismPersona
        persona = PrismPersona()
        crystalliser = PrismCrystalliser(
            persona         = persona,
            memory          = memory,
            outcome_tracker = outcome_tracker,
            calibration     = calibration,
            llm_router      = router,
            ml_assembler    = ml_assembler,
        )
        narrative = PrismNarrative(
            persona         = persona,
            memory          = memory,
            outcome_tracker = outcome_tracker,
            calibration     = calibration,
            soul            = soul,
            llm_router      = router,
        )
        logger.info("Living user model ready (persona, crystalliser, narrative)")
        return persona, crystalliser, narrative
    _living = safe_init("Living user model", _build_living_model, logger=logger)
    persona, crystalliser, narrative = (
        _living if _living else (None, None, None))

    reflection = safe_init_class(
        "PrismReflection", "prism_reflection", "PrismReflection",
        outcome_tracker = outcome_tracker,
        soul            = soul,
        horizon         = horizon,
        llm_router      = router,
        auto_apply      = False,
        logger          = logger,
        info_on_success = "PrismReflection ready",
    )

    return IdentityLearningCluster(
        soul            = soul,
        outcome_tracker = outcome_tracker,
        persona         = persona,
        crystalliser    = crystalliser,
        narrative       = narrative,
        reflection      = reflection,
    )
