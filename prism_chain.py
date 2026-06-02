from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ChainStep:
    """One completed step in an alternating chain execution."""
    step_num:    int
    logic:       str
    message_in:  str        # what the LLM sent to the logic
    result_out:  str        # what the logic returned
    policy_note: str        # any policy annotation
    duration_ms: float
    timestamp:   float = field(default_factory=time.time)


@dataclass
class ChainState:
    """
    Live working memory of the chain.
    Passed through every LLM and logic+policy node.
    Accumulates context, history, and the evolving answer.
    """
    chain_id:     str
    original:     str           # user's original message, never changed
    goal:         str           # LLM's parsed goal, may be refined
    steps:        list[ChainStep] = field(default_factory=list)
    accumulated:  str = ""      # growing answer/context across steps
    done:         bool = False
    final_answer: str = ""


@dataclass
class LLMDecision:
    """
    What the LLM decides after seeing a logic+policy result.
    This is the output of every LLM node in the alternating chain.
    """
    done:        bool           # True = chain is complete
    next_logic:  str            # which logic to invoke next (if not done)
    next_message:str            # what to send to that logic
    reasoning:   str            # LLM's reasoning (for transparency)
    answer:      str = ""       # final answer if done=True


class PrismChain:
    """
    Alternating LLM → Logic+Policy → LLM → Logic+Policy chain.

    Architecture per iteration:
      1. LLM node: receives current state + last result
                   decides: done? or next_logic + reframed message
      2. Logic node: executes chosen logic via agent._execute()
      3. Policy node: checks result against policy engine
      4. State update: accumulates result into chain working memory
      5. Repeat from 1

    The chain is adaptive — the plan emerges from real intermediate
    results rather than being fixed upfront. Each LLM sees actual
    logic output and can change direction based on what it finds.

    Limits:
      MAX_STEPS = 8  (prevents runaway chains)
      30s per logic step (inherited from autonomous engine timeout)
    """

    MAX_STEPS = 8

    SYSTEM_PROMPT = """You are the reasoning layer of an AI personal assistant.
You orchestrate a chain of specialised logic modules to complete a task.
After each module runs, you evaluate its output and decide what to do next.

Available logics:
{registry}

Your response must ALWAYS be valid JSON:
{{
  "done": false,
  "next_logic": "<logic name from list>",
  "next_message": "<exact instruction for that logic, incorporating relevant context from prior results>",
  "reasoning": "<1-2 sentences: why this logic, what you expect it to return>"
}}

OR if the task is complete:
{{
  "done": true,
  "answer": "<final answer to give the user, synthesised from all results>",
  "reasoning": "<what you accomplished>"
}}

Rules:
- Only use logic names from the available list
- next_message must be self-contained — the logic cannot see prior steps unless you include context
- Be decisive: if you have enough information, mark done=true
- Never repeat the same logic twice with identical input
- If a logic fails, try a different approach or conclude with what you have
"""

    def __init__(self, llm_router=None, policy_engine=None,
                  push=None, autonomous=None):
        self._router     = llm_router
        self._policy     = policy_engine
        self._push       = push
        self._autonomous = autonomous

        from prism_composer import LOGIC_REGISTRY
        self._registry = LOGIC_REGISTRY
        self._registry_str = "\n".join(
            f"  {k}: {v}" for k, v in self._registry.items())

    # ── Public API ────────────────────────────────────────────────────────────

    def should_chain(self, message: str) -> bool:
        """
        Heuristic: is this request complex enough to warrant a chain?
        Avoids LLM call. Checks for ambiguity, multi-goal structure,
        conditional language, or open-ended research patterns.
        """
        msg = message.lower()
        # Multi-goal or conditional signals
        signals = [
            " if ", " depending on ", " based on what ",
            " unless ", " in case ", " as needed ",
            " figure out ", " work out ", " decide ",
            " whatever is best ", " best way ",
            "and then", "after that", "once you",
            "also check", "also find", "also send",
        ]
        if any(s in msg for s in signals):
            return True
        # Open-ended / research-style
        research = [
            "research", "find out", "investigate",
            "look into", "analyse", "summarise everything",
            "give me a full", "comprehensive",
        ]
        if any(r in msg for r in research):
            return True
        # Multiple question marks or conjunctions suggest complexity
        if msg.count("?") > 1:
            return True
        return False

    def run(self, message: str, agent_execute_fn,
             base_ctx: dict) -> "PrismCard":
        """
        Run the alternating chain to completion.

        agent_execute_fn: agent._execute(intent, message, ctx) -> PrismCard
        base_ctx: context dict from the agent's chat() call

        Returns a PrismCard with the composed final answer.
        """
        from prism_responses import text_card

        if not self._router:
            return None   # caller falls back to normal routing

        state = ChainState(
            chain_id = str(uuid.uuid4())[:8],
            original = message,
            goal     = message,
        )

        logger.info("[chain %s] Starting — '%s'", state.chain_id, message[:60])

        for step_num in range(1, self.MAX_STEPS + 1):
            # ── LLM NODE: decide what to do next ─────────────────────────────
            decision = self._llm_node(state, step_num)
            if decision is None:
                logger.warning("[chain %s] LLM node failed at step %d",
                               state.chain_id, step_num)
                break

            if decision.done:
                state.done         = True
                state.final_answer = decision.answer
                logger.info("[chain %s] Done after %d steps",
                            state.chain_id, step_num - 1)
                break

            # ── LOGIC NODE: execute chosen logic ──────────────────────────────
            t0 = time.time()
            logic_result = self._logic_node(
                decision.next_logic, decision.next_message,
                base_ctx, agent_execute_fn)
            elapsed = (time.time() - t0) * 1000

            # ── POLICY NODE: check result ─────────────────────────────────────
            policy_note = self._policy_node(
                decision.next_logic, logic_result, base_ctx)

            # ── STATE UPDATE ──────────────────────────────────────────────────
            step = ChainStep(
                step_num    = step_num,
                logic       = decision.next_logic,
                message_in  = decision.next_message[:200],
                result_out  = logic_result[:400],
                policy_note = policy_note,
                duration_ms = elapsed,
            )
            state.steps.append(step)
            # Accumulate: the growing context the next LLM node will see
            state.accumulated += (
                f"\n\n[Step {step_num} — {decision.next_logic}]\n"
                f"Asked: {decision.next_message[:150]}\n"
                f"Got: {logic_result[:350]}"
                + (f"\nPolicy: {policy_note}" if policy_note else "")
            )
            logger.info("[chain %s] Step %d (%s) done in %.0fms",
                        state.chain_id, step_num,
                        decision.next_logic, elapsed)

        # ── Build final card ──────────────────────────────────────────────────
        return self._build_card(state)

    # ── LLM node ─────────────────────────────────────────────────────────────

    def _llm_node(self, state: ChainState,
                   step_num: int) -> Optional[LLMDecision]:
        """
        The LLM reasoning layer between logic steps.
        Sees: original goal + all prior step results.
        Produces: done flag OR (next_logic + reframed message).
        """
        system = self.SYSTEM_PROMPT.format(registry=self._registry_str)

        if step_num == 1:
            user_prompt = (
                f"The user wants: \"{state.original}\"\n\n"
                f"This is step 1. Choose the first logic to invoke.")
        else:
            steps_so_far = len(state.steps)
            user_prompt = (
                f"Original goal: \"{state.original}\"\n\n"
                f"Progress so far ({steps_so_far} steps completed):"
                f"{state.accumulated}\n\n"
                f"Decide: is the task complete, or invoke another logic?\n"
                f"Step budget remaining: {self.MAX_STEPS - step_num + 1}")

        full_prompt = f"{system}\n\n{user_prompt}"

        raw, _ = self._router.call(
            full_prompt, min_capability=2, max_tokens=400, json_mode=True)

        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data  = json.loads(clean)
        except Exception as e:
            logger.debug("[chain] LLM parse failed: %s | raw: %s", e, raw[:100])
            return None

        if data.get("done"):
            return LLMDecision(
                done        = True,
                next_logic  = "",
                next_message= "",
                reasoning   = data.get("reasoning",""),
                answer      = data.get("answer",""),
            )

        logic = data.get("next_logic","")
        if logic not in self._registry:
            logic = "autonomous"

        return LLMDecision(
            done         = False,
            next_logic   = logic,
            next_message = data.get("next_message", state.original),
            reasoning    = data.get("reasoning",""),
        )

    # ── Logic node ────────────────────────────────────────────────────────────

    def _logic_node(self, logic: str, message: str,
                     ctx: dict, agent_execute_fn) -> str:
        """Execute the chosen logic, return plain text result."""
        try:
            card   = agent_execute_fn(logic, message, ctx)
            result = getattr(card, "body", str(card)) or ""
            return result[:800]
        except Exception as e:
            logger.warning("[chain] Logic node %s failed: %s", logic, e)
            return f"Error in {logic}: {e}"

    # ── Policy node ───────────────────────────────────────────────────────────

    def _policy_node(self, logic: str, result: str, ctx: dict) -> str:
        """
        Check the logic result against policy.
        Returns a note string (empty = all clear).
        High-risk logics get an extra annotation for the next LLM node.
        """
        HIGH_RISK = {"email_send", "browser_task", "device_task",
                     "send_push", "calendar_write", "autonomous"}
        if logic in HIGH_RISK:
            return f"[policy: {logic} is an action logic — verify intent before repeating]"
        if self._policy:
            try:
                # If policy engine exposes a check method, use it
                if hasattr(self._policy, "check_action"):
                    ok, note = self._policy.check_action(logic, ctx)
                    if not ok:
                        return f"[policy blocked: {note}]"
            except Exception:
                pass
        return ""

    # ── Output builder ────────────────────────────────────────────────────────

    def _build_card(self, state: ChainState) -> "PrismCard":
        from prism_responses import text_card

        n_steps  = len(state.steps)
        total_ms = sum(s.duration_ms for s in state.steps)
        chain_id = state.chain_id

        if state.final_answer:
            body = state.final_answer
        elif state.steps:
            # Chain hit MAX_STEPS without LLM saying done
            # — synthesise from accumulated context
            if self._router:
                synth_prompt = (
                    f"Task: '{state.original}'\n\n"
                    f"Work done:\n{state.accumulated}\n\n"
                    f"Write a concise final answer for the user (2-4 sentences).")
                body, _ = self._router.call(
                    synth_prompt, min_capability=1, max_tokens=250)
            else:
                body = state.accumulated
        else:
            body = "Chain produced no results."

        logics_used = " → ".join(s.logic for s in state.steps)
        title = (f"Chain {chain_id} · {n_steps} steps · "
                 f"{total_ms/1000:.1f}s"
                 + (f" · {logics_used}" if logics_used else ""))

        return text_card(body, title)
