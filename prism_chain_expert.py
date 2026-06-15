from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from prism_responses import PrismCard

logger = logging.getLogger(__name__)

# ── Role definitions ──────────────────────────────────────────────────────────

ROUTER_PROMPT = """You are a ROUTER. Your only job is to select the single best logic module for the current task state.

Available logics:
{registry}

You receive: the user's goal and what has been done so far.
You output: exactly ONE logic name and ONE instruction for it.

Be decisive. Never hedge. If unsure, pick the logic most likely to make progress.

Respond ONLY with valid JSON:
{{"logic": "<name>", "message": "<precise instruction>", "reasoning": "<one sentence>"}}"""

EVALUATOR_PROMPT = """You are an EVALUATOR. Your only job is to judge whether a logic module's output sufficiently advances the task.

You receive: the original goal, the logic that ran, and its output.
You output: a quality score and whether to continue or conclude.

Scoring:
  5 = perfect, task can be concluded from this
  4 = good, useful progress, one more step may help
  3 = partial, important info missing, another logic needed
  2 = poor, barely useful, try a different approach
  1 = failed, output is an error or irrelevant

Respond ONLY with valid JSON:
{{"score": <1-5>, "sufficient": <true if score>=4 and task could be answered now>, "gap": "<what is still missing if not sufficient>", "reasoning": "<one sentence>"}}"""

BRANCH_JUDGE_PROMPT = """You are a BRANCH JUDGE. Your only job is to decide whether the current task state warrants parallel exploration or a single focused path.

Branch when: the goal has two genuinely independent sub-questions, or you are uncertain which of two specific logics would be more useful.
Do NOT branch when: a single logic clearly applies, or branching would produce redundant results.

Available logics:
{registry}

Respond ONLY with valid JSON — either:
{{"branch": false, "reasoning": "<why single path>"}}
OR
{{"branch": true, "paths": [{{"logic": "<name>", "message": "<instruction>"}}, {{"logic": "<name>", "message": "<instruction>"}}], "reasoning": "<why branch>"}}

Maximum 3 paths. Minimum 2."""

SYNTHESISER_PROMPT = """You are a SYNTHESISER. Your only job is to compose a clear, accurate final answer from the evidence collected.

You receive: the user's original goal and all intermediate results.
You do NOT add information not present in the results.
You write in plain prose, 2-5 sentences unless the task demands more.
You are specific — name actual values, dates, counts from the results.

Respond with plain text only. No JSON."""


@dataclass
class NodeTrace:
    """Records what each specialised node decided."""
    node:      str        # "router"|"evaluator"|"branch_judge"|"synthesiser"
    step_num:  int
    input_tokens: int
    output:    str
    duration_ms: float
    score:     Optional[int] = None    # evaluator score if applicable


@dataclass
class ExpertChainState:
    chain_id:    str
    original:    str
    steps:       list = field(default_factory=list)   # ChainStep-compatible
    accumulated: str  = ""
    done:        bool = False
    final_answer:str  = ""
    traces:      list[NodeTrace] = field(default_factory=list)


class PrismChainExpert:
    """
    Specialised-role variant of PrismChain.

    Each node in the alternating chain is handled by a specialised LLM role:

      BRANCH JUDGE → decides: single path or parallel branches?
           │
           ▼
      ROUTER       → selects logic + frames message (if single path)
           │
           ▼
      Logic + Policy execution
           │
           ▼
      EVALUATOR    → scores output quality (1-5), decides: done or continue?
           │
      if score >= 4: SYNTHESISER → composes final answer
      if score <  4: back to BRANCH JUDGE

    Benefits over general LLM:
    - Each role has a tightly scoped prompt → fewer tokens, clearer decisions
    - Evaluator adds explicit quality gate → fewer wasted steps
    - Branch Judge has dedicated logic for the harder ambiguity decision
    - Synthesiser never makes routing decisions → cleaner final answers
    - Each role can use a different model/temperature in future

    Costs vs general LLM:
    - 3-4 LLM calls per step instead of 1
    - More complex to debug
    - Prompt maintenance overhead
    """

    MAX_STEPS = 8

    def __init__(self, llm_router=None, policy_engine=None,
                  push=None, autonomous=None, memory=None):
        self._router     = llm_router
        self._policy     = policy_engine
        self._push       = push
        self._autonomous = autonomous
        self._memory     = memory

        from prism_composer import LOGIC_REGISTRY
        self._registry     = LOGIC_REGISTRY
        self._registry_str = "\n".join(
            f"  {k}: {v}" for k, v in self._registry.items())

        self._db = Path("~/.prism/chains_expert.db").expanduser()
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, message: str, agent_execute_fn,
             base_ctx: dict) -> Optional[PrismCard]:
        from prism_chain import ChainStep
        from prism_responses import text_card

        if not self._router:
            return None

        state = ExpertChainState(
            chain_id = str(uuid.uuid4())[:8],
            original = message,
        )
        logger.info("[expert-chain %s] Starting — '%s'",
                    state.chain_id, message[:60])

        for step_num in range(1, self.MAX_STEPS + 1):

            # ── 1. BRANCH JUDGE ───────────────────────────────────────────────
            branch_decision = self._branch_judge(state, step_num)

            if branch_decision.get("branch") and branch_decision.get("paths"):
                paths   = branch_decision["paths"][:3]
                b_results = self._execute_branches(paths, agent_execute_fn, base_ctx)
                for i, (path, result) in enumerate(zip(paths, b_results)):
                    policy_note = self._policy_node(path["logic"], result, base_ctx)
                    state.steps.append(ChainStep(
                        step_num    = step_num,
                        logic       = f"{path['logic']}[b{i+1}]",
                        message_in  = path["message"][:200],
                        result_out  = result[:400],
                        policy_note = policy_note,
                        duration_ms = 0.0,
                    ))
                branch_summary = "\n".join(
                    f"  [{p['logic']}]: {r[:200]}"
                    for p, r in zip(paths, b_results))
                state.accumulated += (
                    f"\n\n[Step {step_num} — BRANCH]\n{branch_summary}")

            else:
                # ── 2. ROUTER ─────────────────────────────────────────────────
                route = self._router_node(state, step_num)
                if not route:
                    break

                logic   = route.get("logic","autonomous")
                msg_in  = route.get("message", message)
                if logic not in self._registry:
                    logic = "autonomous"

                # ── 3. LOGIC + POLICY ─────────────────────────────────────────
                t0 = time.time()
                try:
                    card   = agent_execute_fn(logic, msg_in, base_ctx)
                    result = getattr(card, "body", str(card)) or ""
                except Exception as e:
                    result = f"Error: {e}"
                elapsed = (time.time() - t0) * 1000

                policy_note = self._policy_node(logic, result, base_ctx)
                state.steps.append(ChainStep(
                    step_num    = step_num,
                    logic       = logic,
                    message_in  = msg_in[:200],
                    result_out  = result[:400],
                    policy_note = policy_note,
                    duration_ms = elapsed,
                ))
                state.accumulated += (
                    f"\n\n[Step {step_num} — {logic}]\n"
                    f"Asked: {msg_in[:150]}\nGot: {result[:350]}"
                    + (f"\nPolicy: {policy_note}" if policy_note else ""))

                # ── 4. EVALUATOR ──────────────────────────────────────────────
                evaluation = self._evaluator_node(state, logic, result, step_num)
                score      = evaluation.get("score", 3)
                sufficient = evaluation.get("sufficient", False)
                gap        = evaluation.get("gap","")

                logger.info("[expert-chain %s] Step %d (%s) score=%d sufficient=%s",
                            state.chain_id, step_num, logic, score, sufficient)

                # Annotate accumulated with evaluator's gap for next router
                if gap and not sufficient:
                    state.accumulated += f"\n[Evaluator: still missing — {gap}]"

                if sufficient or score >= 4:
                    state.done = True
                    break

        # ── 5. SYNTHESISER ────────────────────────────────────────────────────
        state.final_answer = self._synthesiser_node(state)
        self._save_state(state)

        if self._memory and state.final_answer:
            try:
                self._memory.ingest_conversation(
                    "assistant",
                    f"ExpertChain {state.chain_id}: {state.original}\n"
                    f"Result: {state.final_answer[:300]}")
            except Exception:
                pass

        n_steps  = len(state.steps)
        total_ms = sum(getattr(s, "duration_ms", 0) for s in state.steps)
        logics   = " → ".join(
            getattr(s, "logic","?") for s in state.steps)
        n_llm    = len(state.traces)

        title = (f"Expert chain {state.chain_id} · "
                 f"{n_steps} logic steps · {n_llm} LLM calls · "
                 f"{total_ms/1000:.1f}s · {logics}")

        return text_card(state.final_answer, title)

    # ── Specialised LLM nodes ─────────────────────────────────────────────────

    def _branch_judge(self, state: ExpertChainState, step_num: int) -> dict:
        prompt = (
            BRANCH_JUDGE_PROMPT.format(registry=self._registry_str) +
            f"\n\nGoal: \"{state.original}\"\n"
            f"Steps done: {len(state.steps)}\n"
            + (f"Context so far:\n{state.accumulated[-600:]}"
               if state.accumulated else "No steps yet.")
        )
        t0 = time.time()
        raw, _ = self._router.call(prompt, min_capability=1,
                                    max_tokens=250, json_mode=True)
        elapsed = (time.time() - t0) * 1000
        state.traces.append(NodeTrace(
            "branch_judge", step_num, len(prompt.split()),
            raw[:100], elapsed))
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(clean)
        except Exception:
            return {"branch": False, "reasoning": "parse error"}

    def _router_node(self, state: ExpertChainState, step_num: int) -> Optional[dict]:
        already_used = [getattr(s,"logic","") for s in state.steps]
        prompt = (
            ROUTER_PROMPT.format(registry=self._registry_str) +
            f"\n\nGoal: \"{state.original}\"\n"
            f"Already used: {already_used}\n"
            + (f"Context:\n{state.accumulated[-800:]}"
               if state.accumulated else "No prior steps.")
        )
        t0 = time.time()
        raw, _ = self._router.call(prompt, min_capability=1,
                                    max_tokens=200, json_mode=True)
        elapsed = (time.time() - t0) * 1000
        state.traces.append(NodeTrace(
            "router", step_num, len(prompt.split()), raw[:100], elapsed))
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(clean)
        except Exception:
            return None

    def _evaluator_node(self, state: ExpertChainState,
                         logic: str, result: str, step_num: int) -> dict:
        steps_budget = self.MAX_STEPS - step_num
        prompt = (
            EVALUATOR_PROMPT +
            f"\n\nOriginal goal: \"{state.original}\"\n"
            f"Logic used: {logic}\n"
            f"Output: {result[:500]}\n"
            f"Steps remaining: {steps_budget}\n"
            f"Prior context summary: {state.accumulated[-400:]}"
        )
        t0 = time.time()
        raw, _ = self._router.call(prompt, min_capability=1,
                                    max_tokens=150, json_mode=True)
        elapsed = (time.time() - t0) * 1000
        state.traces.append(NodeTrace(
            "evaluator", step_num, len(prompt.split()),
            raw[:100], elapsed, score=None))
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data  = json.loads(clean)
            state.traces[-1].score = data.get("score")
            return data
        except Exception:
            return {"score": 3, "sufficient": False, "gap": "parse error"}

    def _synthesiser_node(self, state: ExpertChainState) -> str:
        if not state.accumulated and not state.steps:
            return "No results to synthesise."
        prompt = (
            SYNTHESISER_PROMPT +
            f"\n\nUser's original goal: \"{state.original}\"\n\n"
            f"All results collected:\n{state.accumulated}"
        )
        t0 = time.time()
        raw, _ = self._router.call(prompt, min_capability=2, max_tokens=400)
        elapsed = (time.time() - t0) * 1000
        state.traces.append(NodeTrace(
            "synthesiser", len(state.steps),
            len(prompt.split()), raw[:100], elapsed))
        return raw.strip() if raw.strip() else state.accumulated[-400:]

    # ── Branch execution ──────────────────────────────────────────────────────

    def _execute_branches(self, paths: list[dict],
                           agent_fn, ctx: dict) -> list[str]:
        results = [""] * len(paths)
        lock    = threading.Lock()

        def run(i: int, path: dict):
            logic = path.get("logic","autonomous")
            if logic not in self._registry:
                logic = "autonomous"
            msg = path.get("message","")
            try:
                card = agent_fn(logic, msg, ctx)
                r    = getattr(card, "body", str(card)) or ""
            except Exception as e:
                r = f"Error: {e}"
            with lock:
                results[i] = r[:400]

        threads = [threading.Thread(target=run, args=(i, p))
                   for i, p in enumerate(paths)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=45)
        return results

    # ── Policy node (same as PrismChain) ─────────────────────────────────────

    def _policy_node(self, logic: str, result: str, ctx: dict) -> str:
        HIGH_RISK = {"email_send","browser_task","device_task",
                     "send_push","calendar_write","autonomous"}
        if logic in HIGH_RISK:
            return f"[policy: {logic} is an action logic]"
        return ""

    # ── Persistence ───────────────────────────────────────────────────────────

    def _init_db(self):
        import sqlite3
        with sqlite3.connect(self._db, timeout=30.0) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS expert_chains(
                chain_id TEXT PRIMARY KEY, original TEXT,
                n_steps INTEGER, n_llm_calls INTEGER,
                done INTEGER, final_answer TEXT,
                avg_eval_score REAL, created_at REAL)""")

    def _save_state(self, state: ExpertChainState):
        import sqlite3
        scores = [t.score for t in state.traces
                  if t.score is not None]
        avg_score = sum(scores)/len(scores) if scores else 0.0
        with sqlite3.connect(self._db, timeout=30.0) as c:
            c.execute("INSERT OR REPLACE INTO expert_chains VALUES(?,?,?,?,?,?,?,?)", (
                state.chain_id, state.original,
                len(state.steps), len(state.traces),
                int(state.done), state.final_answer[:500],
                avg_score, time.time()))

    def trace_summary(self, state: ExpertChainState) -> str:
        """Human-readable trace for debugging and comparison."""
        lines = [f"Chain {state.chain_id} — {len(state.steps)} steps, "
                 f"{len(state.traces)} LLM calls"]
        for t in state.traces:
            score_str = f" score={t.score}" if t.score else ""
            lines.append(f"  [{t.node:14s}] step={t.step_num} "
                         f"{t.duration_ms:5.0f}ms{score_str}")
        return "\n".join(lines)
