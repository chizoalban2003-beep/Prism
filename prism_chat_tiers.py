"""
prism_chat_tiers.py
===================
Tiered routing dispatcher extracted from ``PrismAgent.chat``.

The chat path tries progressively cheaper handlers until one returns a
usable card:

* **Tier 0** — orchestrator (conditional / multi-domain / cross-session)
* **Tier 0.5** — expert chain (research / evaluation-heavy requests)
* **Tier 1** — general chain (adaptive multi-step)
* **Tier 2** — static composer (known multi-step, predictable)
* **Tier 3** — single-intent execution (always succeeds — produces *a* card)

Each tier above 3 may fail (exception) or return a "bad" card (raw dict
body or planner-noise marker). Both outcomes fall through to the next
tier; the single-intent tier is the safety net.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from prism_responses import PrismCard

logger = logging.getLogger(__name__)


class TierDispatcher:
    """Runs the tiered chat pipeline. Constructed once per agent and
    reused for every turn."""

    EXPERT_SIGNALS: tuple[str, ...] = (
        "research", "analyse", "analyze", "figure out",
        "decide", "best way", "should i", "compare",
        "comprehensive", "investigate", "evaluate",
    )

    def __init__(
        self,
        *,
        orchestrator: Any,
        chain_expert: Any,
        chain: Any,
        composer: Any,
        execute: Callable[..., PrismCard],
        route: Callable[[str], str],
        tool_loop: Optional[Callable[..., Optional[PrismCard]]] = None,
        fold_tiers: bool = True,
    ) -> None:
        self._orchestrator = orchestrator
        self._chain_expert = chain_expert
        self._chain = chain
        self._composer = composer
        self._execute = execute
        self._route = route
        # RFC step 5 (docs/rfc-agentic-loop.md): when a chain/composer
        # trigger fires, try the policied tool loop FIRST — it is the
        # same "decompose and sequence steps" job those tiers hand-roll.
        # The loop declining (offline / disabled / no belt) falls
        # through to the legacy tier unchanged, so behaviour without an
        # LLM is identical. [tool_loop].fold_tiers=false restores the
        # legacy order outright.
        self._tool_loop = tool_loop
        self._fold_tiers = fold_tiers

    def dispatch(
        self,
        message: str,
        context: dict,
        initial_card: Optional[PrismCard] = None,
    ) -> PrismCard:
        card = initial_card
        msg_ln = len((message or "").split())
        msg_lw = (message or "").lower()

        if (card is None and self._orchestrator and message
                and self._orchestrator.should_orchestrate(message)):
            card = self._safe(
                "Orchestrator",
                lambda: self._orchestrator.orchestrate(message, self._execute, context),
            )

        if (card is None and message and msg_ln > 5
                and any(s in msg_lw for s in self.EXPERT_SIGNALS)):
            card = self._safe(
                "Expert chain",
                lambda: self._chain_expert.run(message, self._execute, context),
            )

        chain_wants = (message and msg_ln > 5
                       and self._chain.should_chain(message))
        compose_wants = (message and msg_ln > 6
                         and self._composer.should_compose(message))

        if (card is None and self._fold_tiers and self._tool_loop
                and (chain_wants or compose_wants)):
            card = self._safe(
                "Tool loop (folded tier)",
                lambda: self._tool_loop(message, context, multistep=True),
            )
            if card is not None:
                logger.debug("[tiers] folded %s trigger into tool loop",
                             "chain" if chain_wants else "composer")

        if card is None and chain_wants:
            card = self._safe(
                "Chain",
                lambda: self._chain.run(message, self._execute, context),
            )

        if card is None and compose_wants:
            card = self._safe("Composer", lambda: self._compose(message, context))

        if card is None:
            intent = self._route(message or "")
            card = self._execute(intent, message or "", context)

        return card

    def _compose(self, message: str, context: dict) -> Optional[PrismCard]:
        plan = self._composer.decompose(message)
        if not plan:
            return None
        return self._composer.execute(plan, self._execute, context)

    def _safe(self, label: str, fn: Callable[[], Optional[PrismCard]]) -> Optional[PrismCard]:
        try:
            card = fn()
        except Exception as exc:
            logger.debug("%s failed: %s", label, exc)
            return None
        if _is_bad_card(card):
            return None
        return card


def _is_bad_card(card: Optional[PrismCard]) -> bool:
    """True when a chain/orchestrator returned a raw dict or planner noise."""
    if card is None:
        return False
    body = getattr(card, "body", "") or ""
    return body.startswith("{") or "replanned" in body or body.startswith("[{")
