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
    # LogicPolicy metadata — fed back into the next LLM node
    organ_meta:  dict = field(default_factory=dict)
    # {"risk_level": "low", "capabilities": [...], "irreversible": False, "constitution": "allowed"}


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

    # Logics whose raw output is verbose/noisy — apply SoftLogic compression
    SOFT_LOGICS = frozenset({"web_search", "email_read", "device_task", "browser_task"})

    def __init__(self, llm_router=None, policy_engine=None,
                  push=None, autonomous=None, memory=None,
                  use_evaluator: bool = True,
                  interceptor_policy=None,
                  use_soft_logic: bool = True,
                  horizon_planner=None,
                  soul=None,
                  organ_loader=None,
                  outcome_tracker=None,
                  context_id: str = "default",
                  config: dict | None = None):
        self._router             = llm_router
        self._policy             = policy_engine
        self._push               = push
        self._autonomous         = autonomous
        self._memory             = memory
        self._use_evaluator      = use_evaluator
        self._interceptor_policy = interceptor_policy
        self._use_soft_logic     = use_soft_logic
        self._horizon            = horizon_planner
        self._soul               = soul
        self._organ_loader       = organ_loader
        self._outcome_tracker    = outcome_tracker
        self._context_id         = context_id
        self._persona            = None

        # ── Spectrum middleware (VEAX vector) ─────────────────────────────────
        from prism_spectrum_middleware import load_spectrum
        self._spectrum_gates, self._spectrum_network = load_spectrum(config)

        # Thread-safety note: _state_lock protects the internal results list
        # in _execute_branch. The append to state.steps happens in run() which
        # is single-threaded; only the local `results` accumulation needs a lock.
        self._state_lock = threading.Lock()

        from prism_chain_theory import SubChainLogic
        from prism_composer import LOGIC_REGISTRY
        self._registry = LOGIC_REGISTRY
        self._registry_str = "\n".join(
            f"  {k}: {v}" for k, v in self._registry.items())

        # Research sub-chain: web_search → parse_result → cross_reference
        self._research_logic = SubChainLogic(
            sub_logics=["web_search", "parse_result", "cross_reference"],
            llm_router=self._router,
        )

        # Chain persistence DB
        self._db = Path("~/.prism/chains.db").expanduser()
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Public API ────────────────────────────────────────────────────────────

    def _sync_spectrum(self) -> None:
        """Pick up in-session VEAX updates made by the veax_control organ."""
        from prism_spectrum_middleware import get_current_gates, get_current_network
        live = get_current_gates()
        if live is not None and live is not self._spectrum_gates:
            self._spectrum_gates = live
        live_net = get_current_network()
        if live_net is not None and live_net is not self._spectrum_network:
            self._spectrum_network = live_net

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
             base_ctx: dict,
             step_callback=None) -> "PrismCard":
        """
        Run the alternating chain to completion.

        agent_execute_fn: agent._execute(intent, message, ctx) -> PrismCard
        base_ctx: context dict from the agent's chat() call
        step_callback: optional callable(ChainStep) called after each step completes

        Returns a PrismCard with the composed final answer.
        """

        self._sync_spectrum()   # pick up any in-session VEAX updates

        if not self._router:
            logger.debug("PrismChain: no router, skipping chain")
            return None   # caller falls back to normal routing

        state = ChainState(
            chain_id = str(uuid.uuid4())[:8],
            original = message,
            goal     = message,
        )

        logger.info("[chain %s] Starting — '%s'", state.chain_id, message[:60])

        # ── Prior memory recall ───────────────────────────────────────────────────
        if self._memory is not None:
            try:
                hits = self._memory.search(message, top_n=5)
                if hits:
                    snippets = "\n".join(
                        f"- [{r.entry.source}] {r.excerpt}" for r in hits
                    )
                    state.accumulated = f"[Relevant memories]\n{snippets}\n"
                    base_ctx.setdefault("memory_context", snippets)
            except Exception as _mem_exc:
                logger.debug("[chain] Memory recall failed: %s", _mem_exc)

        # ── Persona context injection ─────────────────────────────────────────────
        _persona = self._persona or base_ctx.get("_persona_ref")
        if _persona is not None:
            try:
                persona_ctx = _persona.build_context(max_chars=400)
                if persona_ctx:
                    state.accumulated = (state.accumulated or "") + f"\n[Crystallised user profile]\n{persona_ctx}\n"
                    base_ctx.setdefault("persona_context", persona_ctx)
            except Exception as _p_exc:
                logger.debug("[chain] Persona context failed: %s", _p_exc)

        # ── Memory anchor ─────────────────────────────────────────────────────────
        _anchor_id: Optional[str] = None
        if self._horizon is not None:
            try:
                _anchor_id = self._horizon.add_triggered(
                    intent=f"chain:{state.chain_id}",
                    completion_condition=message[:200],
                    context={"original_message": message[:400], "step_count": 0},
                )
            except Exception as exc:
                logger.debug("[chain] Anchor create failed: %s", exc)

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
                    policy_note   = self._policy_node(br.logic, br.result, base_ctx)
                    _, br_lp_sum  = self._logicpolicy_meta(br.logic)
                    branch_text += (
                        f"\n\n[Branch {br.branch_id} — {br.logic}]\n"
                        f"Got: {br.result[:300]}"
                        + (f"\nLogicPolicy: {br_lp_sum}" if br_lp_sum else "")
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
                    if step_callback is not None:
                        try:
                            step_callback(step)
                        except Exception:
                            pass
                    # Checkpoint step to horizon anchor
                    if _anchor_id is not None:
                        try:
                            step_summary = (
                                f"Step {step_num} [{step.logic if hasattr(step, 'logic') else 'branch'}]: "
                                f"{(step.result_out if hasattr(step, 'result_out') else '')[:120]}"
                            )
                            self._horizon.record_step(_anchor_id, step_summary)
                            self._horizon.update_context(_anchor_id, step_count=step_num)
                        except Exception:
                            pass

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
                        # Checkpoint step to horizon anchor
                        if _anchor_id is not None:
                            try:
                                _logic = orig_step.logic if hasattr(orig_step, "logic") else "branch"
                                _out = (orig_step.result_out if hasattr(orig_step, "result_out") else "")[:120]
                                step_summary = f"Step {step_num} [{_logic}]: {_out}"
                                self._horizon.record_step(_anchor_id, step_summary)
                                self._horizon.update_context(_anchor_id, step_count=step_num)
                            except Exception:
                                pass
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
                        # Checkpoint step to horizon anchor
                        if _anchor_id is not None:
                            try:
                                step_summary = (
                                    f"Step {step_num} [{sub_step.logic if hasattr(sub_step, 'logic') else 'branch'}]: "
                                    f"{(sub_step.result_out if hasattr(sub_step, 'result_out') else '')[:120]}"
                                )
                                self._horizon.record_step(_anchor_id, step_summary)
                                self._horizon.update_context(_anchor_id, step_count=step_num)
                            except Exception:
                                pass
                        state.accumulated += (
                            f"\n\n[Step {step_num} — INTERCEPTED {decision.next_logic}"
                            f" → {intercept.substitute_logic}]\n"
                            f"Reason: {intercept.reason}\n"
                            f"Substitute result: {sub_result[:300]}")
                        continue

                # ── POLICY NODE: check result ─────────────────────────────────
                policy_note = self._policy_node(
                    decision.next_logic, logic_result, base_ctx)

                # ── LOGICPOLICY META: capabilities + risk + constitution ───────
                lp_meta, lp_summary = self._logicpolicy_meta(decision.next_logic)

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
                    organ_meta  = lp_meta,
                )
                state.steps.append(step)
                if step_callback is not None:
                    try:
                        step_callback(step)
                    except Exception:
                        pass
                # Checkpoint step to horizon anchor
                if _anchor_id is not None:
                    try:
                        step_summary = (
                            f"Step {step_num} [{step.logic if hasattr(step, 'logic') else 'branch'}]: "
                            f"{(step.result_out if hasattr(step, 'result_out') else '')[:120]}"
                        )
                        self._horizon.record_step(_anchor_id, step_summary)
                        self._horizon.update_context(_anchor_id, step_count=step_num)
                    except Exception:
                        pass
                state.eval_scores.append(eval_score)
                # Accumulate: the growing context the next LLM node will see.
                # lp_summary closes the loop: the LLM knows exactly what the
                # previous organ was capable of and whether constitution allowed it.
                eval_note = f"\nEval: {eval_score}/5" + (f" — still missing: {gap}" if gap else "")
                # ── X axis: format LogicPolicy trace at configured verbosity ──
                lp_formatted = self._spectrum_gates.format_logicpolicy(
                    lp_summary, lp_meta)

                state.accumulated += (
                    f"\n\n[Step {step_num} — {decision.next_logic}]\n"
                    f"Asked: {decision.next_message[:150]}\n"
                    f"Got: {logic_result[:350]}"
                    + (f"\nLogicPolicy: {lp_formatted}" if lp_formatted else "")
                    + (f"\nPolicy: {policy_note}" if policy_note else "")
                    + eval_note
                )
                logger.info("[chain %s] Step %d (%s) eval=%d/5 done in %.0fms",
                            state.chain_id, step_num,
                            decision.next_logic, eval_score, elapsed)

                # ── V axis: spectrum verification threshold ───────────────────
                # At V=0.5 the threshold is 3 — same as the old hard-coded ≥4
                # guard but now user-tunable. Only skip if score is below threshold.
                if not self._spectrum_gates.accepts_result(eval_score):
                    logger.debug(
                        "[chain %s] V-gate: step %d score %d below threshold %d — continuing",
                        state.chain_id, step_num, eval_score,
                        self._spectrum_gates.verification_threshold(),
                    )

                # ── A axis: approval gate for irreversible organs ─────────────
                if lp_meta.get("irreversible") and self._spectrum_gates.requires_approval(True):
                    logger.info(
                        "[chain %s] A-gate: step %d organ '%s' is irreversible — "
                        "pausing for approval (A=%.2f)",
                        state.chain_id, step_num, decision.next_logic, self._spectrum_gates.A,
                    )
                    if self._push:
                        try:
                            self._push.send(
                                "PRISM needs your approval",
                                f"Chain step {step_num} wants to run '{decision.next_logic}' "
                                f"(irreversible). Reply 'approve' to continue.",
                            )
                        except Exception:
                            pass

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

        # ── Finalise anchor ────────────────────────────────────────────────────────
        if _anchor_id is not None:
            try:
                if state.done and state.final_answer:
                    self._horizon.complete(
                        _anchor_id,
                        notes=state.final_answer[:300],
                    )
                elif not state.done:
                    # Chain hit MAX_STEPS without completing — abandon the anchor
                    self._horizon.abandon(
                        _anchor_id,
                        reason=f"chain hit MAX_STEPS ({self.MAX_STEPS}) without completing",
                    )
            except Exception as exc:
                logger.debug("[chain] Anchor finalise failed: %s", exc)

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

        # ── Record outcome for learning loop ──────────────────────────────────
        if self._outcome_tracker is not None:
            try:
                from prism_outcome_tracker import OUTCOME_ABANDONED, OUTCOME_DONE
                outcome = OUTCOME_DONE if state.done else OUTCOME_ABANDONED
                policy_flags = sum(1 for s in state.steps if s.policy_note)
                self._outcome_tracker.record(
                    chain_id    = state.chain_id,
                    goal        = state.original,
                    outcome     = outcome,
                    steps_count = len(state.steps),
                    duration_ms = sum(s.duration_ms for s in state.steps),
                    policy_flags= policy_flags,
                    final_answer= state.final_answer or "",
                    context_id  = self._context_id,
                )
            except Exception as exc:
                logger.debug("[chain] outcome_tracker record failed: %s", exc)

        return card

    def run_streaming(self, message: str, agent_execute_fn, base_ctx: dict):
        """
        Generator version of run() — yields SSE-compatible dicts as each
        chain step completes, then a final 'done' event.

        Yields:
            {"event": "step",  "step": N, "logic": "...", "result": "...", "policy": "..."}
            {"event": "done",  "answer": "...", "chain_id": "..."}
            {"event": "error", "message": "..."}

        Usage in prism_asgi SSE handler — see run_streaming_async() for async variant:
            for evt in chain.run_streaming(msg, fn, ctx):
                wfile.write(f"data: {json.dumps(evt)}\n\n".encode())
        """
        import queue
        import threading

        step_queue: queue.Queue = queue.Queue()
        result_holder: list = []
        error_holder:  list = []
        done_event = threading.Event()

        _SENTINEL = object()

        def _on_step(step):
            step_queue.put({
                "event":       "step",
                "step":        step.step_num,
                "logic":       step.logic,
                "result":      step.result_out[:200],
                "policy":      step.policy_note or "",
                "score":       step.eval_score,
                "risk":        step.organ_meta.get("risk_level", "low"),
                "caps":        step.organ_meta.get("capabilities", []),
                "constitution": step.organ_meta.get("constitution", "allowed"),
            })

        def _run():
            try:
                card = self.run(message, agent_execute_fn, base_ctx,
                                step_callback=_on_step)
                result_holder.append(card)
            except Exception as exc:
                error_holder.append(str(exc))
            finally:
                step_queue.put(_SENTINEL)
                done_event.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        while True:
            try:
                item = step_queue.get(timeout=30)
            except Exception:
                yield {"event": "error", "message": "stream timeout"}
                return
            if item is _SENTINEL:
                break
            yield item

        if error_holder:
            yield {"event": "error", "message": error_holder[0]}
            return

        card = result_holder[0] if result_holder else None
        if card is None:
            yield {"event": "error", "message": "chain returned no card"}
            return

        yield {
            "event":    "done",
            "answer":   card.body if hasattr(card, "body") else str(card),
            "chain_id": getattr(card, "source", ""),
            "card_type": (lambda ct: ct.value if hasattr(ct, "value") else str(ct))(
                         getattr(card, "card_type", "text")),
            "card_data": getattr(card, "card_data", {}),
            "card_title": getattr(card, "title", ""),
        }

    async def run_streaming_async(self, message: str, agent_execute_fn, base_ctx: dict):
        """
        Async generator version of run_streaming(). Bridges the synchronous
        run_streaming() generator into asyncio via a queue so the ASGI server
        can yield SSE events without blocking the event loop.

        Yields the same event dicts as run_streaming():
            {"event": "step",  ...}
            {"event": "done",  ...}
            {"event": "error", ...}
        """
        import asyncio
        import threading

        loop = asyncio.get_event_loop()
        q: asyncio.Queue = asyncio.Queue()
        _SENTINEL = object()

        def _produce():
            try:
                for evt in self.run_streaming(message, agent_execute_fn, base_ctx):
                    loop.call_soon_threadsafe(q.put_nowait, evt)
            except Exception as exc:
                loop.call_soon_threadsafe(
                    q.put_nowait, {"event": "error", "message": str(exc)}
                )
            finally:
                loop.call_soon_threadsafe(q.put_nowait, _SENTINEL)

        t = threading.Thread(target=_produce, daemon=True)
        t.start()

        while True:
            item = await q.get()
            if item is _SENTINEL:
                break
            yield item

    def resume(self, goal_id: str, agent_execute_fn, base_ctx: dict) -> "PrismCard":
        """
        Resume an interrupted chain from its last checkpoint.

        Looks up a PAUSED chain anchor in HorizonPlanner by goal_id,
        reconstructs ChainState from the recorded steps, and continues
        the chain from where it left off.

        Returns None if the goal is not found or not resumable.
        """
        if self._horizon is None:
            logger.warning("[chain] resume() called but no horizon_planner")
            return None

        goal = self._horizon.get(goal_id)
        if goal is None:
            logger.warning("[chain] resume: goal %s not found", goal_id)
            return None

        # Extract chain_id from intent ("chain:abc12345" → "abc12345")
        chain_id = goal.intent.split(":", 1)[-1] if ":" in goal.intent else goal.goal_id
        original = goal.accumulated_context.get("original_message", goal.completion_condition)

        # Reconstruct accumulated context from recorded steps
        accumulated = "\n\n".join(
            f"[Checkpoint {i+1}] {s}"
            for i, s in enumerate(goal.completed_steps)
        )

        state = ChainState(
            chain_id  = chain_id,
            original  = original,
            goal      = original,
            accumulated = accumulated,
        )

        # Re-mark goal as TRIGGERED so on_session_end() checkpoints it again
        # if this resume is also interrupted
        try:
            from prism_horizon import HorizonGoalStatus
            with self._horizon._lock:
                g = self._horizon._load_goal(goal_id)
                if g and g.status == HorizonGoalStatus.PAUSED:
                    g.status = HorizonGoalStatus.TRIGGERED
                    self._horizon._upsert(g)
        except Exception:
            pass

        logger.info(
            "[chain] Resuming chain %s from %d checkpoints",
            chain_id, len(goal.completed_steps),
        )

        # Continue from the next step (skip completed steps in the budget)
        steps_done = goal.accumulated_context.get("step_count", len(goal.completed_steps))
        remaining  = max(1, self.MAX_STEPS - steps_done)

        # Re-run from the next step using remaining budget
        _anchor_id = goal_id
        for step_num in range(steps_done + 1, steps_done + remaining + 1):
            decision = self._llm_node(state, step_num)
            if decision is None:
                break

            if decision.done:
                state.done         = True
                state.final_answer = decision.answer
                break

            t0           = time.time()
            logic_result = self._logic_node(
                decision.next_logic, decision.next_message,
                base_ctx, agent_execute_fn)
            elapsed      = (time.time() - t0) * 1000

            policy_note  = self._policy_node(decision.next_logic, logic_result, base_ctx)
            eval_score, sufficient, gap = self._evaluator_node(
                state.original, decision.next_logic, logic_result)

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
            state.accumulated += (
                f"\n\n[Resumed Step {step_num} — {decision.next_logic}]\n"
                f"Got: {logic_result[:300]}"
            )

            try:
                self._horizon.record_step(
                    _anchor_id,
                    f"Resumed step {step_num} [{decision.next_logic}]: {logic_result[:120]}"
                )
                self._horizon.update_context(_anchor_id, step_count=step_num)
            except Exception:
                pass

            if sufficient:
                state.done = True
                if self._router:
                    synth_prompt = (
                        f"Task: '{state.original}'\n\nEvidence:\n{state.accumulated}\n\n"
                        "Write a concise final answer (2-4 sentences).")
                    state.final_answer, _ = self._router.call(
                        synth_prompt, min_capability=1, max_tokens=250)
                else:
                    state.final_answer = logic_result
                break

        if _anchor_id:
            try:
                if state.done and state.final_answer:
                    self._horizon.complete(_anchor_id, notes=state.final_answer[:300])
                else:
                    self._horizon.abandon(
                        _anchor_id,
                        reason="resumed chain hit step limit without completing",
                    )
            except Exception:
                pass

        return self._build_card(state)

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
                logger.warning("[chain] branch logic %r not in registry, falling back to autonomous", logic)
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
        soul_ctx = ""
        if self._soul is not None:
            try:
                soul_ctx = "\n\nUser identity context:\n" + self._soul.compress_for_llm(max_chars=500)
            except Exception:
                pass
        system = self.SYSTEM_PROMPT.format(registry=self._registry_str) + soul_ctx

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
            logger.warning("[chain] LLM requested unknown logic %r, falling back to autonomous", logic)
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
        """
        Execute the chosen logic, return plain text result.

        Special handling:
          "research" → SubChainLogic (web_search → parse → cross_reference → synth)
          SOFT_LOGICS → result compressed by SoftLogic LLM before returning
        """
        # Research: internally a mini sub-chain
        if logic == "research":
            try:
                return self._research_logic(message, agent_execute_fn, ctx)
            except Exception as e:
                logger.warning("[chain] Research sub-chain failed: %s", e)
                return f"Research failed: {e}"

        try:
            card   = agent_execute_fn(logic, message, ctx)
            result = getattr(card, "body", str(card)) or ""
            result = result[:800]
        except Exception as e:
            logger.warning("[chain] Logic node %s failed: %s", logic, e)
            return f"Error in {logic}: {e}"

        # SoftLogic: compress verbose output for noisy logics
        if self._use_soft_logic and logic in self.SOFT_LOGICS and self._router:
            result = self._soften(message, logic, result)

        return result

    def _soften(self, goal: str, logic: str, result: str) -> str:
        """Apply SoftLogic LLM compression to verbose/noisy logic output."""
        prompt = (
            f"Extract the 3 most relevant facts for the goal: '{goal[:200]}'\n"
            f"Source: {logic}\nRaw output: {result[:600]}\n"
            "Reply in 2 concise sentences. No JSON."
        )
        try:
            text, _ = self._router.call(prompt, min_capability=1, max_tokens=120)
            return text.strip() or result[:400]
        except Exception:
            return result[:400]

    # ── LogicPolicy metadata ──────────────────────────────────────────────────

    def _logicpolicy_meta(self, logic: str) -> tuple[dict, str]:
        """
        Collect organ capabilities, risk level, and L1 constitution verdict.
        Returns (meta_dict, compact_summary_string) to inject into accumulated state.

        This is what closes the llm→(logic+logicpolicy)→policy→llm loop:
        the summary string is appended to state.accumulated so the next LLM
        node knows exactly what the previous action was capable of and whether
        the constitution allowed it.
        """
        meta: dict = {
            "risk_level":   "low",
            "capabilities": [],
            "irreversible": False,
            "constitution": "allowed",
        }
        if self._organ_loader is None:
            return meta, ""

        try:
            caps = self._organ_loader.get_organ_capabilities(logic)
            pol  = self._organ_loader.get_organ_policy(logic)
            meta["capabilities"] = caps
            meta["risk_level"]   = pol.get("risk_level", "low") if pol else "low"
            meta["irreversible"] = bool(pol.get("irreversible", False)) if pol else False

            try:
                from prism_constitution import get_guard
                ok, reason = get_guard().check(logic, caps)
                meta["constitution"] = "allowed" if ok else f"blocked({reason[:40]})"
            except Exception:
                pass
        except Exception:
            pass

        parts = [f"risk={meta['risk_level']}"]
        if meta["capabilities"]:
            parts.append(f"caps=[{', '.join(meta['capabilities'])}]")
        if meta["irreversible"]:
            parts.append("irreversible=true")
        parts.append(f"L1={meta['constitution']}")
        return meta, "  ".join(parts)

    # ── Policy node ───────────────────────────────────────────────────────────

    # Fallback set for organs that don't declare ORGAN_POLICY
    _LEGACY_HIGH_RISK = frozenset({
        "email_send", "browser_task", "device_task",
        "send_push", "calendar_write", "autonomous",
    })

    def _policy_node(self, logic: str, result: str, ctx: dict) -> str:
        """
        Check the logic result against policy.
        Returns a note string (empty = all clear).

        Priority order:
          1. Organ's own ORGAN_POLICY declaration (risk_level, requires_approval,
             irreversible, max_per_session)
          2. PolicyEngine.check_action() if available
          3. Legacy HIGH_RISK fallback for organs with no ORGAN_POLICY
        """
        notes: list[str] = []

        # 1. Organ-declared policy
        organ_policy: dict = {}
        if self._organ_loader is not None:
            try:
                organ_policy = self._organ_loader.get_organ_policy(logic)
            except Exception:
                pass

        if organ_policy:
            risk = organ_policy.get("risk_level", "low")
            if risk in ("high", "critical"):
                notes.append(f"[policy: {logic} risk={risk} — verify intent before repeating]")
            if organ_policy.get("irreversible"):
                notes.append(f"[policy: {logic} is irreversible — cannot be undone]")
            max_sess = organ_policy.get("max_per_session")
            if max_sess is not None:
                used = ctx.get(f"_policy_count_{logic}", 0)
                if used >= max_sess:
                    notes.append(f"[policy blocked: {logic} reached session limit of {max_sess}]")
            if organ_policy.get("requires_approval") and not ctx.get(f"_approved_{logic}"):
                notes.append(f"[policy: {logic} requires explicit user approval]")
        elif logic in self._LEGACY_HIGH_RISK:
            # No ORGAN_POLICY declared — fall back to hardcoded annotation
            notes.append(f"[policy: {logic} is an action logic — verify intent before repeating]")

        # 2. PolicyEngine
        if self._policy:
            try:
                if hasattr(self._policy, "check_action"):
                    ok, note = self._policy.check_action(logic, ctx)
                    if not ok:
                        notes.append(f"[policy blocked: {note}]")
            except Exception:
                pass

        combined = "  ".join(notes)
        if combined:
            self._write_policy_audit(logic, combined)
        return combined

    _AUDIT_DB: str = "~/.prism/policy_audit.db"

    def _write_policy_audit(self, logic: str, note: str) -> None:
        """Persist a policy flag to the audit log so policy_audit organ can surface it."""
        import sqlite3
        import time
        from pathlib import Path
        try:
            db = Path(self._AUDIT_DB).expanduser()
            db.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(db) as con:
                con.execute(
                    "CREATE TABLE IF NOT EXISTS audit_log("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "ts REAL NOT NULL, logic TEXT NOT NULL, note TEXT NOT NULL)"
                )
                con.execute(
                    "INSERT INTO audit_log(ts, logic, note) VALUES (?,?,?)",
                    (time.time(), logic, note),
                )
        except Exception:
            pass

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
