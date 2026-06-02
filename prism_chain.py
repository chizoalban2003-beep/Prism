from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

# Evaluator prompt reused from expert chain — narrow, single-responsibility role
from prism_chain_expert import EVALUATOR_PROMPT

if TYPE_CHECKING:
    from prism_responses import PrismCard

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
    eval_score:  Optional[int] = None   # 1-5 from Evaluator node, None = not scored
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
    steps:          list[ChainStep] = field(default_factory=list)
    accumulated:    str = ""      # growing answer/context across steps
    done:           bool = False
    final_answer:   str = ""
    eval_scores:    list[int] = field(default_factory=list)  # per-step evaluator scores


@dataclass
class LLMDecision:
    """
    What the LLM decides after seeing a logic+policy result.
    This is the output of every LLM node in the alternating chain.
    """
    done:         bool           # True = chain is complete
    next_logic:   str            # single next logic (simple case)
    next_message: str
    reasoning:    str            # LLM's reasoning (for transparency)
    answer:       str = ""       # final answer if done=True
    # Branching: when ambiguous, LLM spawns multiple parallel paths
    branches:     list[dict] = field(default_factory=list)
    # branches = [{"logic": "web_search", "message": "..."}, ...]
    is_branch:    bool = False   # True when branches is populated


@dataclass
class BranchResult:
    """Results from a parallel branch execution."""
    branch_id:   str
    logic:       str
    result:      str
    success:     bool
    duration_ms: float


class PrismChain:
    """
    Alternating LLM → Logic+Policy → LLM → Logic+Policy chain.

    Architecture per iteration:
      1. LLM node: receives current state + last result
                   decides: done? or next_logic + reframed message
                   OR: branch into multiple parallel logics
      2. Logic node: executes chosen logic via agent._execute()
      3. Policy node: checks result against policy engine
      4. State update: accumulates result into chain working memory
      5. Repeat from 1

    The chain is adaptive — the plan emerges from real intermediate
    results rather than being fixed upfront. Each LLM sees actual
    logic output and can change direction based on what it finds.

    Branching: when genuinely ambiguous, the LLM can spawn up to 3
    parallel logic executions and merge their results before the next
    LLM node.  This turns the spine into a tree.

    Limits:
      MAX_STEPS = 8  (prevents runaway chains)
      30s per logic step (inherited from autonomous engine timeout)
      45s per branch (per branch thread join timeout)
    """

    MAX_STEPS = 8

    SYSTEM_PROMPT = """You are the reasoning layer of an AI personal assistant.
You orchestrate a chain of specialised logic modules to complete a task.
After each module runs, you evaluate its output and decide what to do next.

Available logics:
{registry}

Your response must ALWAYS be valid JSON.

Single logic (most common):
{{
  "done": false,
  "next_logic": "<logic name from list>",
  "next_message": "<exact instruction for that logic, incorporating relevant context from prior results>",
  "reasoning": "<1-2 sentences: why this logic, what you expect it to return>"
}}

When the task is ambiguous or two different logics might both be needed simultaneously, you may branch:
{{
  "done": false,
  "is_branch": true,
  "branches": [
    {{"logic": "<logic1>", "message": "<instruction for logic1>"}},
    {{"logic": "<logic2>", "message": "<instruction for logic2>"}}
  ],
  "reasoning": "<why you are branching>"
}}
Maximum 3 branches. Only branch when genuinely uncertain which path is better, or when two logics are truly independent and both needed.

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
                  push=None, autonomous=None, memory=None,
                  use_evaluator: bool = True,
                  interceptor_policy=None):
        self._router             = llm_router
        self._policy             = policy_engine
        self._push               = push
        self._autonomous         = autonomous
        self._memory             = memory
        self._use_evaluator      = use_evaluator
        self._interceptor_policy = interceptor_policy

        # Thread-safety note: _state_lock protects the internal results list
        # in _execute_branch. The append to state.steps happens in run() which
        # is single-threaded; only the local `results` accumulation needs a lock.
        self._state_lock = threading.Lock()

        from prism_composer import LOGIC_REGISTRY
        self._registry = LOGIC_REGISTRY
        self._registry_str = "\n".join(
            f"  {k}: {v}" for k, v in self._registry.items())

        # Chain persistence DB
        self._db = Path("~/.prism/chains.db").expanduser()
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

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

        if not self._router:
            logger.debug("PrismChain: no router, skipping chain")
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

            # ── Handle branch or single logic ─────────────────────────────────
            if decision.is_branch and decision.branches:
                branch_results = self._execute_branch(
                    decision.branches, agent_execute_fn, base_ctx)

                # Policy check on each branch; accumulate text for next LLM node
                branch_text = ""
                for br in branch_results:
                    policy_note = self._policy_node(br.logic, br.result, base_ctx)
                    branch_text += (
                        f"\n\n[Branch {br.branch_id} — {br.logic}]\n"
                        f"Got: {br.result[:300]}"
                        + (f"\nPolicy: {policy_note}" if policy_note else ""))
                    step = ChainStep(
                        step_num    = step_num,
                        logic       = f"{br.logic}[{br.branch_id}]",
                        message_in  = next((b.get("message", "") for b in decision.branches
                                            if b.get("logic") == br.logic), "")[:200],
                        result_out  = br.result[:400],
                        policy_note = policy_note,
                        duration_ms = br.duration_ms,
                    )
                    state.steps.append(step)

                state.accumulated += f"\n\n[Step {step_num} — PARALLEL BRANCH]{branch_text}"
                logger.info("[chain %s] Step %d branched into %d parallel logics",
                            state.chain_id, step_num, len(branch_results))

            else:
                # ── Single logic (existing behaviour) ─────────────────────────
                t0 = time.time()
                logic_result = self._logic_node(
                    decision.next_logic, decision.next_message,
                    base_ctx, agent_execute_fn)
                elapsed = (time.time() - t0) * 1000

                # ── INTERCEPTOR POLICY: active rerouting ──────────────────────
                if self._interceptor_policy is not None:
                    intercept = self._interceptor_policy.intercept(
                        decision.next_logic, logic_result,
                        "", state.goal)
                    if intercept is not None:
                        logger.info(
                            "[chain %s] Interceptor fired: %s → %s (%s)",
                            state.chain_id, decision.next_logic,
                            intercept.substitute_logic, intercept.reason)
                        # Record the original step (unscored)
                        orig_step = ChainStep(
                            step_num    = step_num,
                            logic       = decision.next_logic,
                            message_in  = decision.next_message[:200],
                            result_out  = logic_result[:400],
                            policy_note = f"[intercepted: {intercept.reason}]",
                            duration_ms = elapsed,
                        )
                        state.steps.append(orig_step)
                        # Run substitute logic immediately
                        t_sub = time.time()
                        sub_result = self._logic_node(
                            intercept.substitute_logic,
                            intercept.substitute_message,
                            base_ctx, agent_execute_fn)
                        sub_elapsed = (time.time() - t_sub) * 1000
                        sub_step = ChainStep(
                            step_num    = step_num,
                            logic       = intercept.substitute_logic,
                            message_in  = intercept.substitute_message[:200],
                            result_out  = sub_result[:400],
                            policy_note = f"[intercept substitute for {decision.next_logic}]",
                            duration_ms = sub_elapsed,
                        )
                        state.steps.append(sub_step)
                        state.accumulated += (
                            f"\n\n[Step {step_num} — INTERCEPTED {decision.next_logic}"
                            f" → {intercept.substitute_logic}]\n"
                            f"Reason: {intercept.reason}\n"
                            f"Substitute result: {sub_result[:300]}")
                        continue

                # ── POLICY NODE: check result ─────────────────────────────────
                policy_note = self._policy_node(
                    decision.next_logic, logic_result, base_ctx)

                # ── EVALUATOR NODE: quality gate ──────────────────────────────
                eval_score, sufficient, gap = self._evaluator_node(
                    state.original, decision.next_logic, logic_result)

                # ── STATE UPDATE ──────────────────────────────────────────────
                step = ChainStep(
                    step_num    = step_num,
                    logic       = decision.next_logic,
                    message_in  = decision.next_message[:200],
                    result_out  = logic_result[:400],
                    policy_note = policy_note,
                    duration_ms = elapsed,
                    eval_score  = eval_score,
                )
                state.steps.append(step)
                state.eval_scores.append(eval_score)
                # Accumulate: the growing context the next LLM node will see
                eval_note = f"\nEval: {eval_score}/5" + (f" — still missing: {gap}" if gap else "")
                state.accumulated += (
                    f"\n\n[Step {step_num} — {decision.next_logic}]\n"
                    f"Asked: {decision.next_message[:150]}\n"
                    f"Got: {logic_result[:350]}"
                    + (f"\nPolicy: {policy_note}" if policy_note else "")
                    + eval_note
                )
                logger.info("[chain %s] Step %d (%s) eval=%d/5 done in %.0fms",
                            state.chain_id, step_num,
                            decision.next_logic, eval_score, elapsed)

                # Early exit: evaluator says result is sufficient (score ≥ 4)
                if sufficient:
                    logger.info("[chain %s] Evaluator early exit at step %d (score %d)",
                                state.chain_id, step_num, eval_score)
                    state.done = True
                    # Let the synthesiser build the final answer from accumulated context
                    if self._router:
                        synth_prompt = (
                            f"Task: '{state.original}'\n\n"
                            f"Evidence:\n{state.accumulated}\n\n"
                            "Write a concise final answer (2-4 sentences).")
                        state.final_answer, _ = self._router.call(
                            synth_prompt, min_capability=1, max_tokens=250)
                    else:
                        state.final_answer = logic_result
                    break

        # ── Build final card ──────────────────────────────────────────────────
        card = self._build_card(state)

        # ── Persist to SQLite ─────────────────────────────────────────────────
        try:
            self._save_state(state)
        except Exception as e:
            logger.debug("[chain] Failed to persist state: %s", e)

        # ── Store chain result in memory for future retrieval ─────────────────
        if self._memory and state.final_answer:
            try:
                self._memory.ingest_conversation(
                    "assistant",
                    f"Chain {state.chain_id}: {state.original}\n"
                    f"Result: {state.final_answer[:300]}")
            except Exception:
                pass

        return card

    # ── Branch execution ──────────────────────────────────────────────────────

    def _execute_branch(self, branches: list[dict],
                         agent_execute_fn, ctx: dict) -> list[BranchResult]:
        """
        Execute multiple logic branches in parallel.
        Results are merged into a single context string for the next LLM node.

        Thread-safety note: each branch thread appends to the local `results`
        list protected by self._state_lock.  The append to state.steps happens
        in run() which is single-threaded, so no lock is needed there.
        """
        results = []
        lock    = self._state_lock

        def run_branch(b: dict, bid: str):
            logic   = b.get("logic", "autonomous")
            if logic not in self._registry:
                logic = "autonomous"
            message = b.get("message", "")
            t0      = time.time()
            try:
                card    = agent_execute_fn(logic, message, ctx)
                result  = getattr(card, "body", str(card)) or ""
                success = True
            except Exception as e:
                result  = f"Branch failed: {e}"
                success = False
            elapsed = (time.time() - t0) * 1000
            with lock:
                results.append(BranchResult(bid, logic, result[:400], success, elapsed))

        threads = []
        for i, b in enumerate(branches[:3]):   # max 3 branches
            bid = f"branch_{i+1}"
            t   = threading.Thread(target=run_branch, args=(b, bid))
            threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=45)

        return results

    # ── LLM node ─────────────────────────────────────────────────────────────

    def _llm_node(self, state: ChainState,
                   step_num: int) -> Optional[LLMDecision]:
        """
        The LLM reasoning layer between logic steps.
        Sees: original goal + all prior step results.
        Produces: done flag OR (next_logic + reframed message) OR branch.
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

        # ── Branching path ────────────────────────────────────────────────────
        if data.get("is_branch") and data.get("branches"):
            branches = data["branches"]
            # Validate each branch has logic + message
            valid = [b for b in branches
                     if isinstance(b, dict) and b.get("logic") and b.get("message")]
            if valid:
                return LLMDecision(
                    done         = False,
                    next_logic   = "",
                    next_message = "",
                    reasoning    = data.get("reasoning", ""),
                    is_branch    = True,
                    branches     = valid,
                )

        # ── Single logic path (existing behaviour) ────────────────────────────
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

    # ── Evaluator node ────────────────────────────────────────────────────────

    def _evaluator_node(self, goal: str, logic: str,
                         result: str) -> tuple[int, bool, str]:
        """
        Post-step quality gate using the Expert chain's EVALUATOR_PROMPT.
        Returns (score 1-5, sufficient bool, gap description).
        If LLM call fails, returns (3, False, "") — chain continues normally.
        """
        if not self._router or not self._use_evaluator:
            return 3, False, ""

        prompt = (
            EVALUATOR_PROMPT + "\n\n"
            f"Goal: {goal}\n"
            f"Logic that ran: {logic}\n"
            f"Output:\n{result[:500]}"
        )
        try:
            raw, _ = self._router.call(
                prompt, min_capability=1, max_tokens=120, json_mode=True)
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data  = json.loads(clean)
            score      = int(data.get("score", 3))
            sufficient = bool(data.get("sufficient", False))
            gap        = str(data.get("gap", ""))
            return score, sufficient, gap
        except Exception as e:
            logger.debug("[chain] Evaluator parse failed: %s", e)
            return 3, False, ""

    # ── Chain persistence ─────────────────────────────────────────────────────

    def _init_db(self):
        import sqlite3
        with sqlite3.connect(self._db) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS chains(
                chain_id TEXT PRIMARY KEY,
                original TEXT, goal TEXT,
                accumulated TEXT, done INTEGER,
                final_answer TEXT, n_steps INTEGER,
                avg_eval_score REAL,
                created_at REAL, updated_at REAL)""")

    def _save_state(self, state: ChainState):
        import sqlite3
        avg_score = (sum(state.eval_scores) / len(state.eval_scores)
                     if state.eval_scores else None)
        with sqlite3.connect(self._db) as c:
            c.execute("""INSERT OR REPLACE INTO chains VALUES(
                ?,?,?,?,?,?,?,?,?,?)""", (
                state.chain_id, state.original, state.goal,
                state.accumulated, int(state.done),
                state.final_answer, len(state.steps),
                avg_score,
                state.steps[0].timestamp if state.steps else time.time(),
                time.time()))

    def recent_chains(self, n: int = 5) -> list[dict]:
        import sqlite3
        with sqlite3.connect(self._db) as c:
            rows = c.execute(
                "SELECT chain_id,original,n_steps,done,final_answer,"
                "avg_eval_score,updated_at "
                "FROM chains ORDER BY updated_at DESC LIMIT ?", (n,)).fetchall()
        return [{"chain_id": r[0], "original": r[1], "n_steps": r[2],
                 "done": bool(r[3]), "summary": r[4][:80] if r[4] else "",
                 "avg_eval_score": r[5], "updated_at": r[6]} for r in rows]

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
        avg_score   = (sum(state.eval_scores) / len(state.eval_scores)
                       if state.eval_scores else None)
        score_tag   = f" · eval {avg_score:.1f}/5" if avg_score is not None else ""
        title = (f"Chain {chain_id} · {n_steps} steps · "
                 f"{total_ms/1000:.1f}s{score_tag}"
                 + (f" · {logics_used}" if logics_used else ""))

        return text_card(body, title)
