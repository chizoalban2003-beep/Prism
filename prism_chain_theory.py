"""
prism_chain_theory.py
=====================
Theory experiment classes for the LLM→Logic+Policy→LLM alternating chain.

Experiment 1 — Recursive Sub-chains: SubChainLogic
Experiment 2 — Vertical LLMs inside Logic Nodes: SoftLogic
Experiment 3 — Policy-as-Interceptor: InterceptorPolicy + PolicyIntercept
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ── Experiment 1: Recursive Sub-chains ───────────────────────────────────────


class SubChainLogic:
    """
    A logic wrapper that internally runs a mini PrismChain (max 3 steps,
    no evaluator) over a fixed sequence of sub-logics.

    When called via __call__(goal, agent_execute_fn, ctx) it returns the
    synthesised string result of the sub-chain.  The outer chain sees only
    this final string — not the individual sub-step outputs.

    Default sub-logic sequence for the "research" use case:
        web_search → parse_result → cross_reference
    """

    def __init__(
        self,
        sub_logics: Optional[list[str]] = None,
        llm_router: Any = None,
    ):
        self._sub_logics = sub_logics or ["web_search", "parse_result", "cross_reference"]
        self._router = llm_router

    # ------------------------------------------------------------------
    # Main entry point — matches the signature used in experiments
    # ------------------------------------------------------------------

    def __call__(self, goal: str, agent_execute_fn: Callable, ctx: dict) -> str:
        """
        Run the sub-chain and return the synthesised result string.
        agent_execute_fn(intent, message, ctx) -> card  (same as outer chain)
        """
        results: list[str] = []
        accumulated = ""

        for logic in self._sub_logics:
            try:
                message = (
                    f"{goal}"
                    + (f"\n\nPrevious sub-results:\n{accumulated}" if accumulated else "")
                )
                card = agent_execute_fn(logic, message, ctx)
                result = getattr(card, "body", str(card)) or ""
                results.append(f"[{logic}]: {result[:300]}")
                accumulated = "\n".join(results)
            except Exception as exc:
                results.append(f"[{logic}]: Error — {exc}")

        # Synthesise: if we have a router, ask it to compress; else join
        if self._router and results:
            synth_prompt = (
                f"Synthesise these sub-research steps for the goal: '{goal}'\n\n"
                + "\n".join(results)
                + "\n\nWrite a concise 2-sentence summary of the findings."
            )
            try:
                text, _ = self._router.call(synth_prompt, min_capability=1, max_tokens=200)
                return text
            except Exception:
                pass

        return "\n".join(results)

    @property
    def sub_logics(self) -> list[str]:
        return list(self._sub_logics)


# ── Experiment 2: Vertical LLMs inside Logic Nodes ───────────────────────────


class SoftLogic:
    """
    Wraps any logic callable and adds an in-node LLM "softener" call.

    Flow:
        1. Call underlying logic (agent_execute_fn style or direct callable)
        2. Make a focused LLM call to extract the 3 most relevant facts
        3. Return the compressed text (never the raw logic output)

    The underlying_logic is called as: underlying_logic(goal, agent_execute_fn, ctx)
    or, if it's a string logic name, via agent_execute_fn(underlying_logic, goal, ctx).
    """

    SOFTEN_PROMPT = (
        "Extract the 3 most relevant facts from this for the goal: {goal}. "
        "Raw output: {result}. Reply in 2 sentences."
    )

    def __init__(self, underlying_logic: Any, llm_router: Any = None):
        """
        underlying_logic: either a string (logic name) or a callable
                          with signature (goal, agent_execute_fn, ctx) -> str
        llm_router: an object with .call(prompt, **kwargs) -> (str, dict)
        """
        self._logic = underlying_logic
        self._router = llm_router

    def __call__(self, goal: str, agent_execute_fn: Callable, ctx: dict) -> str:
        """Run underlying logic then apply LLM softening."""
        # Step 1: get raw result from underlying logic
        raw_result = self._call_underlying(goal, agent_execute_fn, ctx)

        # Step 2: LLM softening
        if self._router:
            prompt = self.SOFTEN_PROMPT.format(goal=goal, result=raw_result[:600])
            try:
                text, _ = self._router.call(prompt, min_capability=1, max_tokens=120)
                return text
            except Exception as exc:
                logger.debug("[SoftLogic] LLM softener failed: %s", exc)
                # Graceful degradation: return truncated raw result
                return raw_result[:400]

        # No router — graceful degradation
        return raw_result[:400]

    def _call_underlying(
        self, goal: str, agent_execute_fn: Callable, ctx: dict
    ) -> str:
        """Dispatch to underlying logic — handles string name or callable."""
        if callable(self._logic) and not isinstance(self._logic, str):
            try:
                result = self._logic(goal, agent_execute_fn, ctx)
                return str(result)
            except Exception as exc:
                return f"Error in underlying logic: {exc}"
        else:
            # String logic name — call via agent_execute_fn
            logic_name = str(self._logic)
            try:
                card = agent_execute_fn(logic_name, goal, ctx)
                return getattr(card, "body", str(card)) or ""
            except Exception as exc:
                return f"Error in {logic_name}: {exc}"

    @property
    def underlying_logic(self) -> Any:
        return self._logic


# ── Experiment 3: Policy-as-Interceptor ──────────────────────────────────────


@dataclass
class PolicyIntercept:
    """
    Returned by InterceptorPolicy.intercept() when an intercept fires.

    substitute_logic:   the logic name to run instead (or in addition)
    substitute_message: the message to pass to the substitute logic
    reason:             human-readable explanation of why the intercept fired
    """
    substitute_logic:   str
    substitute_message: str
    reason:             str


class InterceptorPolicy:
    """
    A policy layer that can actively reroute the chain by substituting a
    different logic when certain conditions are detected.

    intercept(current_logic, result, next_logic, goal) returns:
      - None            → no interception, proceed normally
      - PolicyIntercept → skip the next LLM node and run substitute logic

    Hard-coded intercept rules (for the experiment):
      1. web_search + "Error" in result  → substitute autonomous,
         "retry web search using a different query approach"
      2. email_send + "sent" not in result → substitute email_read,
         "verify email send status"
      3. autonomous + "approval" in result → substitute approve_pending,
         "auto-approve this tool execution"
    """

    def intercept(
        self,
        current_logic: str,
        result: str,
        next_logic: str,
        goal: str,
    ) -> Optional[PolicyIntercept]:
        """
        Evaluate intercept rules.

        Returns PolicyIntercept if a rule fires, else None.
        """
        result_lower = result.lower()

        # Rule 1: web_search returned an error
        if current_logic == "web_search" and "error" in result_lower:
            return PolicyIntercept(
                substitute_logic="autonomous",
                substitute_message="retry web search using a different query approach",
                reason="web_search returned an error — retrying via autonomous",
            )

        # Rule 2: email_send did not confirm delivery
        if current_logic == "email_send" and "sent" not in result_lower:
            return PolicyIntercept(
                substitute_logic="email_read",
                substitute_message="verify email send status",
                reason="email_send result did not contain 'sent' — verifying status",
            )

        # Rule 3: autonomous step requires approval
        if current_logic == "autonomous" and "approval" in result_lower:
            return PolicyIntercept(
                substitute_logic="approve_pending",
                substitute_message="auto-approve this tool execution",
                reason="autonomous result requires approval — auto-approving",
            )

        return None
