"""
prism_orchestrator.py
=====================
ChainOrchestrator — the prefrontal cortex of PRISM.

Decomposes complex, multi-step, conditional, or cross-session tasks into a
TaskGraph, executes nodes in dependency order (parallel where safe), handles
cross-session waits via HorizonPlanner, and synthesises a final answer.

Architecture position
---------------------
    PrismAgent.chat()
        → ChainOrchestrator.orchestrate()        ← this module
            ├── node (reactive profile)  → agent_execute_fn directly
            ├── node (analytical/...)    → PrismChain.run()
            ├── parallel ready nodes     → ThreadPoolExecutor
            └── horizon_pause node      → HorizonPlanner.add() + persist
        → PrismChain.run()                        (fallback for simple tasks)

Chain profiles
--------------
Each node declares a profile controlling how its sub-chain runs:

  reactive     – fast direct organ call, no evaluator: simple lookups
  analytical   – best model, full eval+policy, speculative routing
  verification – strict policy, no shortcuts: financial/irreversible actions
  creative     – relaxed eval, broad reasoning: writing, planning
  negotiation  – cross-session, HorizonPlanner pause: conditional workflows

Usage
-----
    orch = ChainOrchestrator(chain=chain, organ_loader=loader,
                              outcome_tracker=tracker, horizon=horizon,
                              router=router, soul=soul)

    if orch.should_orchestrate(message):
        card = orch.orchestrate(message, agent._execute, context)

    # On session start — resume cross-session waits:
    for card in orch.resume_waiting(agent._execute, context):
        ...
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as _FTimeout
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from prism_chain import PrismChain
    from prism_horizon import HorizonPlanner
    from prism_organ_loader import OrganLoader
    from prism_outcome_tracker import OutcomeTracker
    from prism_responses import PrismCard

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chain profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChainProfile:
    name: str
    min_capability: int
    use_parallel: bool      # parallelise independent ready nodes
    speculative: bool       # use speculative LLM routing within sub-chain
    cross_session: bool     # may create HorizonGoal pauses
    synthesis_style: str    # "merge" | "sequential" | "conditional"


PROFILES: dict[str, ChainProfile] = {
    "reactive":     ChainProfile("reactive",     1, False, False, False, "sequential"),
    "analytical":   ChainProfile("analytical",   2, True,  True,  False, "merge"),
    "verification": ChainProfile("verification", 2, False, False, False, "sequential"),
    "creative":     ChainProfile("creative",     2, True,  True,  False, "merge"),
    "negotiation":  ChainProfile("negotiation",  3, False, False, True,  "conditional"),
}


# ---------------------------------------------------------------------------
# Node and graph data structures
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorNode:
    """One unit of work within a TaskGraph."""
    node_id:       str
    intent:        str                     # organ intent or chain logic name
    goal:          str                     # specific sub-goal text
    profile:       str  = "analytical"
    depends_on:    list[str] = field(default_factory=list)
    condition:     str  = ""               # check before running; "" = unconditional
    horizon_pause: bool = False            # pause here; resume when horizon fires
    status:        str  = "pending"        # pending|running|done|failed|waiting|skipped
    result:        str  = ""
    error:         str  = ""
    duration_ms:   float = 0.0
    started_at:    float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "node_id":       self.node_id,
            "intent":        self.intent,
            "goal":          self.goal,
            "profile":       self.profile,
            "depends_on":    self.depends_on,
            "condition":     self.condition,
            "horizon_pause": self.horizon_pause,
            "status":        self.status,
            "result":        self.result,
            "error":         self.error,
            "duration_ms":   self.duration_ms,
            "started_at":    self.started_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> OrchestratorNode:
        valid = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass
class TaskGraph:
    """Directed acyclic graph of orchestrator nodes for one user request."""
    graph_id:         str
    original:         str
    context_id:       str
    nodes:            list[OrchestratorNode]
    synthesis_hint:   str  = ""
    created_at:       float = field(default_factory=time.time)
    status:           str  = "running"     # running|completed|paused|failed
    final_answer:     str  = ""
    horizon_goal_ids: dict[str, str] = field(default_factory=dict)  # node_id → goal_id

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------

    def ready_nodes(self) -> list[OrchestratorNode]:
        """Nodes whose dependencies are all done."""
        done = {n.node_id for n in self.nodes if n.status == "done"}
        return [
            n for n in self.nodes
            if n.status == "pending"
            and all(dep in done for dep in n.depends_on)
        ]

    def is_complete(self) -> bool:
        return all(n.status in ("done", "failed", "skipped") for n in self.nodes)

    def is_paused(self) -> bool:
        return any(n.status == "waiting" for n in self.nodes)

    def get_node(self, node_id: str) -> Optional[OrchestratorNode]:
        return next((n for n in self.nodes if n.node_id == node_id), None)

    def node_results(self) -> dict[str, str]:
        return {n.node_id: n.result for n in self.nodes if n.result}

    def to_dict(self) -> dict:
        return {
            "graph_id":         self.graph_id,
            "original":         self.original,
            "context_id":       self.context_id,
            "nodes":            [n.to_dict() for n in self.nodes],
            "synthesis_hint":   self.synthesis_hint,
            "created_at":       self.created_at,
            "status":           self.status,
            "final_answer":     self.final_answer,
            "horizon_goal_ids": self.horizon_goal_ids,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TaskGraph:
        return cls(
            graph_id         = d["graph_id"],
            original         = d["original"],
            context_id       = d.get("context_id", "default"),
            nodes            = [OrchestratorNode.from_dict(n) for n in d.get("nodes", [])],
            synthesis_hint   = d.get("synthesis_hint", ""),
            created_at       = d.get("created_at", time.time()),
            status           = d.get("status", "running"),
            final_answer     = d.get("final_answer", ""),
            horizon_goal_ids = d.get("horizon_goal_ids", {}),
        )


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_DECOMPOSE_PROMPT = """\
You are PRISM's task orchestrator. Analyse this user request and decompose it
into a coordinated task graph if it genuinely requires multiple sub-tasks.

USER REQUEST:
{message}

AVAILABLE INTENTS (intent → description):
{logic_list}

SOUL CONTEXT:
{soul_context}

Respond with JSON only — no prose:
{{
  "needs_orchestration": true,
  "rationale": "one sentence why",
  "nodes": [
    {{
      "node_id": "n1",
      "intent": "exact_intent_from_the_list_above",
      "goal": "specific sub-goal for this node",
      "profile": "reactive|analytical|verification|creative|negotiation",
      "depends_on": [],
      "condition": "",
      "horizon_pause": false
    }}
  ],
  "synthesis_hint": "how to combine the results into a final answer"
}}

Profile guide:
  reactive     — fast single lookups (weather, price, time check)
  analytical   — reasoning, research, multi-step planning
  verification — financial, irreversible, or approval-required actions
  creative     — writing, brainstorming, drafting schedules
  negotiation  — conditional flows that may wait days for external events

Rules:
- node_id: unique short strings n1, n2, n3 ...
- intent: must match an entry from the available intents list above
- depends_on: list of node_ids this node must wait for before starting
- condition: a short English phrase to check against dep results, or ""
- horizon_pause: true ONLY for nodes waiting days/weeks for external events
- max 6 nodes; keep the graph minimal
- if needs_orchestration is false, set nodes to []
"""

_CONDITION_CHECK_PROMPT = """\
Does the following output satisfy the condition "{condition}"?

Output:
{output}

Reply with exactly: YES or NO
"""

_SYNTHESIS_PROMPT = """\
You are composing the final answer to a user's request from completed sub-tasks.

ORIGINAL REQUEST:
{original}

SYNTHESIS HINT:
{hint}

COMPLETED SUB-TASK RESULTS:
{results}

Write a clear, direct answer integrating all results above.
Speak directly to the user. Do not mention "nodes", "sub-tasks", or
"orchestration". Write as a single cohesive response.
"""


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

_CONDITIONAL_RE = re.compile(
    r"\bonly if\b|\bif and only if\b|\bprovided that\b|\bassuming that\b"
    r"|\bonce .{1,30} confirms?\b|\bafter .{1,30} (is |are )?done\b"
    r"|\bwhen .{1,30} happens?\b|\bfirst .{1,30} then\b",
    re.IGNORECASE,
)
_CROSS_SESSION_RE = re.compile(
    r"\bevery day\b|\bremind me when\b|\bmonitor until\b"
    r"|\bcheck (daily|weekly|periodically)\b|\bwhen (the )?price\b",
    re.IGNORECASE,
)
_MULTI_DOMAIN = [
    "flight", "hotel", "weather", "finance", "health", "calendar",
    "email", "book", "schedule", "reminder", "task", "document",
    "meeting", "research", "invoice", "payment",
]


# ---------------------------------------------------------------------------
# ChainOrchestrator
# ---------------------------------------------------------------------------

class ChainOrchestrator:
    """
    Prefrontal cortex of PRISM.

    Decomposes tasks into TaskGraphs, executes nodes in dependency order,
    handles parallel execution, cross-session waits, and final synthesis.
    """

    def __init__(
        self,
        chain:           Optional[PrismChain]       = None,
        organ_loader:    Optional[OrganLoader]      = None,
        outcome_tracker: Optional[OutcomeTracker]   = None,
        horizon:         Optional[HorizonPlanner]   = None,
        router: Optional[Any]                          = None,
        soul: Optional[Any]                          = None,
        db_path:         str                          = "~/.prism/orchestrator.db",
    ) -> None:
        self._chain   = chain
        self._loader  = organ_loader
        self._tracker = outcome_tracker
        self._horizon = horizon
        self._router  = router
        self._soul    = soul
        self._persona = None
        self._db      = Path(db_path).expanduser()
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_orchestrate(self, message: str) -> bool:
        """
        Heuristic gate — avoids LLM overhead for simple queries.
        Returns True when the message shows conditional, sequential, or
        multi-domain intent that benefits from graph-based coordination.
        """
        if not message or len(message.split()) < 6:
            return False
        if _CONDITIONAL_RE.search(message):
            return True
        if _CROSS_SESSION_RE.search(message):
            return True
        msg_lower = message.lower()
        domain_hits = sum(1 for d in _MULTI_DOMAIN if d in msg_lower)
        if domain_hits >= 2:
            return True
        # Long multi-sentence request
        sentences = [s.strip() for s in message.split(".") if len(s.strip()) > 15]
        if len(sentences) >= 3 and len(message) > 160:
            return True
        return False

    def orchestrate(
        self,
        message:          str,
        agent_execute_fn: Callable,
        base_ctx:         dict,
    ) -> PrismCard:
        """
        Main entry point. Decomposes → executes → synthesises.
        Falls back to a single chain.run() if decomposition fails or
        the LLM decides orchestration is unnecessary.
        """
        from prism_responses import text_card

        graph_id   = str(uuid.uuid4())[:8]
        context_id = base_ctx.get("context_id", "default")

        # 1. Decompose
        try:
            graph = self._decompose(message, graph_id, context_id)
        except Exception as exc:
            logger.warning("[orch] decompose failed (%s) — single chain fallback", exc)
            return self._chain_run(message, agent_execute_fn, base_ctx)

        if graph is None or not graph.nodes:
            logger.debug("[orch] no nodes — single chain fallback")
            return self._chain_run(message, agent_execute_fn, base_ctx)

        logger.info("[orch] graph %s: %d node(s) for %r", graph_id, len(graph.nodes), message[:60])
        self._persist(graph)

        # 2. Execute graph
        try:
            self._run_graph(graph, agent_execute_fn, base_ctx)
        except Exception as exc:
            logger.warning("[orch] graph execution error: %s", exc)
            graph.status = "failed"

        # 3. Synthesise
        if graph.is_complete() and not graph.final_answer:
            graph.final_answer = self._synthesise(graph)

        if not graph.is_paused():
            graph.status = "completed" if graph.is_complete() else "failed"
        self._persist(graph)

        # 4. Outcome tracking
        if self._tracker:
            from prism_outcome_tracker import OUTCOME_ABANDONED, OUTCOME_DONE
            outcome = OUTCOME_DONE if graph.status == "completed" else OUTCOME_ABANDONED
            try:
                self._tracker.record(
                    chain_id    = graph.graph_id,
                    goal        = message[:400],
                    outcome     = outcome,
                    steps_count = len(graph.nodes),
                    context_id  = context_id,
                )
            except Exception:
                pass

        if graph.is_paused():
            waiting = [n for n in graph.nodes if n.status == "waiting"]
            conditions = "; ".join(f"{n.node_id}: {n.goal[:60]}" for n in waiting)
            return text_card(
                f"Task is underway — paused while waiting for external conditions.\n\n"
                f"Waiting on: {conditions}\n\n"
                "PRISM will resume automatically when the conditions are met.",
                "Task in progress",
            )

        return text_card(
            graph.final_answer or "Task completed.",
            f"[Orchestrated] {message[:60]}",
        )

    def resume_waiting(
        self,
        agent_execute_fn: Callable,
        base_ctx: dict,
    ) -> list[PrismCard]:
        """
        Called at session start. Checks all paused graphs for horizon goals
        that have since fired, and resumes them.
        """
        cards: list[PrismCard] = []
        if self._horizon is None:
            return cards

        for graph in self._load_paused():
            resumed = False
            for node in graph.nodes:
                if node.status != "waiting":
                    continue
                goal_id = graph.horizon_goal_ids.get(node.node_id)
                if not goal_id:
                    continue
                goal = self._horizon.get(goal_id)
                if goal and goal.status.value in ("triggered", "completed"):
                    node.status = "pending"
                    node.condition = ""
                    node.horizon_pause = False
                    resumed = True
                    logger.info("[orch] resuming node %s in graph %s", node.node_id, graph.graph_id)

            if resumed:
                graph.status = "running"
                ctx = {**base_ctx, "context_id": graph.context_id, "_resumed_graph_id": graph.graph_id}
                try:
                    self._run_graph(graph, agent_execute_fn, ctx)
                    if graph.is_complete() and not graph.final_answer:
                        graph.final_answer = self._synthesise(graph)
                    graph.status = "completed" if graph.is_complete() else (
                        "paused" if graph.is_paused() else "failed"
                    )
                except Exception as exc:
                    graph.status = "failed"
                    graph.final_answer = f"Resume error: {exc}"

                self._persist(graph)

                from prism_responses import text_card
                cards.append(text_card(
                    graph.final_answer or "Resumed task completed.",
                    f"[Resumed] {graph.original[:60]}",
                ))

        return cards

    # ------------------------------------------------------------------
    # Async public API (Phase 6)
    # ------------------------------------------------------------------

    async def orchestrate_async(
        self,
        message:          str,
        agent_execute_fn: Callable,
        base_ctx:         dict,
    ) -> PrismCard:
        """
        Async entry point for orchestrate().

        Identical flow to orchestrate() but uses asyncio.gather for parallel
        node execution — non-blocking LLM calls and organ fan-out without
        spawning a raw ThreadPoolExecutor.  Sync orchestrate() is untouched.
        """
        from prism_responses import text_card

        graph_id   = str(uuid.uuid4())[:8]
        context_id = base_ctx.get("context_id", "default")

        try:
            graph = await asyncio.to_thread(self._decompose, message, graph_id, context_id)
        except Exception as exc:
            logger.warning("[orch] async decompose failed (%s) — single chain fallback", exc)
            return await asyncio.to_thread(self._chain_run, message, agent_execute_fn, base_ctx)

        if graph is None or not graph.nodes:
            logger.debug("[orch] no nodes — single chain fallback")
            return await asyncio.to_thread(self._chain_run, message, agent_execute_fn, base_ctx)

        logger.info("[orch] graph %s: %d node(s) for %r", graph_id, len(graph.nodes), message[:60])
        self._persist(graph)

        try:
            await self._run_graph_async(graph, agent_execute_fn, base_ctx)
        except Exception as exc:
            logger.warning("[orch] async graph execution error: %s", exc)
            graph.status = "failed"

        if graph.is_complete() and not graph.final_answer:
            graph.final_answer = await asyncio.to_thread(self._synthesise, graph)

        if not graph.is_paused():
            graph.status = "completed" if graph.is_complete() else "failed"
        self._persist(graph)

        if self._tracker:
            from prism_outcome_tracker import OUTCOME_ABANDONED, OUTCOME_DONE
            outcome = OUTCOME_DONE if graph.status == "completed" else OUTCOME_ABANDONED
            try:
                self._tracker.record(
                    chain_id    = graph.graph_id,
                    goal        = message[:400],
                    outcome     = outcome,
                    steps_count = len(graph.nodes),
                    context_id  = context_id,
                )
            except Exception:
                pass

        if graph.is_paused():
            waiting    = [n for n in graph.nodes if n.status == "waiting"]
            conditions = "; ".join(f"{n.node_id}: {n.goal[:60]}" for n in waiting)
            return text_card(
                f"Task is underway — paused while waiting for external conditions.\n\n"
                f"Waiting on: {conditions}\n\n"
                "PRISM will resume automatically when the conditions are met.",
                "Task in progress",
            )

        return text_card(
            graph.final_answer or "Task completed.",
            f"[Orchestrated] {message[:60]}",
        )

    async def _run_graph_async(
        self,
        graph:            TaskGraph,
        agent_execute_fn: Callable,
        base_ctx:         dict,
    ) -> None:
        """
        Async variant of _run_graph().

        Serial nodes are executed via asyncio.to_thread so the event loop
        remains unblocked.  Parallel-safe nodes are fanned out with
        asyncio.gather under a 60-second asyncio.wait_for timeout.
        _execute_node() itself is unchanged.
        """
        max_rounds = len(graph.nodes) + 2
        for _ in range(max_rounds):
            if graph.is_complete() or graph.is_paused():
                break

            ready = graph.ready_nodes()
            if not ready:
                logger.debug("[orch] no ready nodes — graph may be stuck")
                break

            profile0 = PROFILES.get(ready[0].profile, PROFILES["analytical"])

            if len(ready) == 1 or not profile0.use_parallel:
                for node in ready:
                    await asyncio.to_thread(
                        self._execute_node, node, graph, agent_execute_fn, base_ctx
                    )
                    if graph.is_paused():
                        break
            else:
                parallel_safe = [
                    n for n in ready
                    if not n.horizon_pause
                    and PROFILES.get(n.profile, PROFILES["analytical"]).use_parallel
                ]
                serial = [n for n in ready if n not in parallel_safe]

                async def _run_one(n: OrchestratorNode) -> None:
                    await asyncio.to_thread(
                        self._execute_node, n, graph, agent_execute_fn, base_ctx
                    )

                try:
                    await asyncio.wait_for(
                        asyncio.gather(*[_run_one(n) for n in parallel_safe]),
                        timeout=60.0,
                    )
                except TimeoutError:
                    for n in parallel_safe:
                        if n.status == "running":
                            n.status = "failed"
                            n.error  = "parallel timeout"

                for node in serial:
                    await asyncio.to_thread(
                        self._execute_node, node, graph, agent_execute_fn, base_ctx
                    )

    # ------------------------------------------------------------------
    # Decomposition
    # ------------------------------------------------------------------

    def _decompose(self, message: str, graph_id: str, context_id: str) -> Optional[TaskGraph]:
        """Ask the LLM to decompose message into a TaskGraph."""
        if not self._router:
            return None

        # Build logic list: organs + LOGIC_REGISTRY
        logic_lines: list[str] = []
        if self._loader:
            for intent in self._loader.list_organs():
                policy = self._loader.get_organ_policy(intent)
                desc   = self._loader.known_intents().get(intent, intent)
                risk   = policy.get("risk_level", "low")
                logic_lines.append(f"  {intent}: {desc} [risk={risk}]")
        try:
            from prism_composer import LOGIC_REGISTRY
            for intent, desc in LOGIC_REGISTRY.items():
                if intent not in (self._loader.known_intents() if self._loader else {}):
                    logic_lines.append(f"  {intent}: {desc}")
        except ImportError:
            pass

        soul_ctx = ""
        if self._soul:
            try:
                beliefs = self._soul.list_beliefs(belief_type="stated")[:5]
                soul_ctx = "; ".join(f"{b.text[:60]} (conf={b.confidence:.2f})" for b in beliefs)
            except Exception:
                pass

        persona_ctx = ""
        if self._persona is not None:
            try:
                persona_ctx = self._persona.build_context(max_chars=300)
            except Exception:
                pass

        combined_ctx = "; ".join(filter(None, [soul_ctx, persona_ctx])) or "(no user context)"

        prompt = _DECOMPOSE_PROMPT.format(
            message    = message,
            logic_list = "\n".join(logic_lines) or "  (none loaded)",
            soul_context = combined_ctx,
        )
        raw, _ = self._router.call(
            prompt,
            min_capability = 2,
            max_tokens     = 900,
            json_mode      = True,
            speculative    = False,
        )
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        data = json.loads(raw)

        if not data.get("needs_orchestration", False):
            logger.debug("[orch] LLM: no orchestration needed for %r", message[:60])
            return None

        raw_nodes = data.get("nodes", [])
        if not raw_nodes:
            return None

        nodes: list[OrchestratorNode] = []
        for nd in raw_nodes[:6]:
            nodes.append(OrchestratorNode(
                node_id       = str(nd.get("node_id", f"n{len(nodes)+1}")),
                intent        = str(nd.get("intent", "autonomous")),
                goal          = str(nd.get("goal", message))[:300],
                profile       = str(nd.get("profile", "analytical"))
                                if nd.get("profile") in PROFILES else "analytical",
                depends_on    = [str(x) for x in nd.get("depends_on", [])],
                condition     = str(nd.get("condition", "")),
                horizon_pause = bool(nd.get("horizon_pause", False)),
            ))

        return TaskGraph(
            graph_id       = graph_id,
            original       = message,
            context_id     = context_id,
            nodes          = nodes,
            synthesis_hint = str(data.get("synthesis_hint", "")),
        )

    # ------------------------------------------------------------------
    # Graph execution
    # ------------------------------------------------------------------

    @staticmethod
    def _is_liquid_phase() -> bool:
        """Return True when Φ_melt ≥ 0.70 — used to pause mid-DAG execution."""
        try:
            import prism_phase as _pp
            _engine = _pp.get_engine()
            return bool(_engine.history and _engine.current_phase.value == "LIQUID")
        except Exception:
            return False

    def _run_graph(
        self,
        graph:            TaskGraph,
        agent_execute_fn: Callable,
        base_ctx:         dict,
    ) -> None:
        """Execute graph nodes respecting dependency order and parallelism."""
        max_rounds = len(graph.nodes) + 2  # guard against cycles
        for _ in range(max_rounds):
            if graph.is_complete() or graph.is_paused():
                break

            # Phase gate: pause the DAG if system enters LIQUID phase mid-execution
            if self._is_liquid_phase():
                for n in graph.nodes:
                    if n.status == "pending":
                        n.status = "waiting"
                graph.status = "paused"
                logger.info("[orch] LIQUID phase detected mid-DAG — graph %s paused", graph.graph_id)
                break

            ready = graph.ready_nodes()
            if not ready:
                logger.debug("[orch] no ready nodes — graph may be stuck")
                break

            profile0 = PROFILES.get(ready[0].profile, PROFILES["analytical"])

            if len(ready) == 1 or not profile0.use_parallel:
                # Serial execution
                for node in ready:
                    self._execute_node(node, graph, agent_execute_fn, base_ctx)
                    if graph.is_paused():
                        break
            else:
                # Parallel execution — only safe (non-horizon, non-serial) nodes
                parallel_safe = [
                    n for n in ready
                    if not n.horizon_pause
                    and PROFILES.get(n.profile, PROFILES["analytical"]).use_parallel
                ]
                serial = [n for n in ready if n not in parallel_safe]

                with ThreadPoolExecutor(max_workers=min(len(parallel_safe), 4)) as pool:
                    futures = {
                        pool.submit(self._execute_node, n, graph, agent_execute_fn, base_ctx): n
                        for n in parallel_safe
                    }
                    try:
                        for future in as_completed(futures, timeout=60.0):
                            futures[future]  # node already mutated in-place
                    except _FTimeout:
                        for n in parallel_safe:
                            if n.status == "running":
                                n.status = "failed"
                                n.error  = "parallel timeout"

                for node in serial:
                    self._execute_node(node, graph, agent_execute_fn, base_ctx)

    def _execute_node(
        self,
        node:             OrchestratorNode,
        graph:            TaskGraph,
        agent_execute_fn: Callable,
        base_ctx:         dict,
    ) -> None:
        """Execute one node, handling conditions, horizon pauses, and errors."""
        node.status     = "running"
        node.started_at = time.time()

        # Check condition
        if node.condition and not self._check_condition(node, graph):
            logger.info("[orch] node %s condition not met — skipping", node.node_id)
            node.status = "skipped"
            return

        # Horizon pause
        if node.horizon_pause:
            self._handle_horizon_pause(node, graph)
            return

        # Build context with dependency results injected
        ctx = dict(base_ctx)
        ctx["_orch_graph_id"] = graph.graph_id
        ctx["_orch_node_id"]  = node.node_id
        ctx["_orch_profile"]  = node.profile
        for dep_id in node.depends_on:
            dep = graph.get_node(dep_id)
            if dep and dep.result:
                ctx[f"prior_output_{dep_id}"] = dep.result
                ctx["prior_output"] = dep.result  # latest dep as prior_output

        t0 = time.time()
        try:
            profile = PROFILES.get(node.profile, PROFILES["analytical"])
            result_text = self._call_node(node, profile, agent_execute_fn, ctx)
            node.result      = result_text[:2000]
            node.status      = "done"
            node.duration_ms = (time.time() - t0) * 1000
            logger.debug("[orch] node %s done in %.0fms", node.node_id, node.duration_ms)
        except Exception as exc:
            node.error       = str(exc)
            node.status      = "failed"
            node.duration_ms = (time.time() - t0) * 1000
            logger.warning("[orch] node %s failed: %s", node.node_id, exc)

    def _call_node(
        self,
        node:             OrchestratorNode,
        profile:          ChainProfile,
        agent_execute_fn: Callable,
        ctx:              dict,
    ) -> str:
        """Route node to direct organ call or full chain reasoning."""
        is_organ = (
            self._loader is not None
            and node.intent in self._loader.list_organs()
        )

        if profile.name == "reactive" and is_organ and self._loader is not None:
            # Direct organ call — fast path
            fn = self._loader.get(node.intent)
            if fn is None:
                raise ValueError(f"organ '{node.intent}' not found")
            card = fn(node.intent, node.goal, ctx)
            return card.body if hasattr(card, "body") else str(card)

        if profile.use_parallel and is_organ and self._loader is not None:
            # Parallel organ call (still single organ, but respects parallelism flag)
            results = self._loader.execute_parallel([node.intent], node.goal, ctx)
            if node.intent in results:
                r = results[node.intent]
                return r.body if hasattr(r, "body") else str(r.get("output", r))

        # Full chain reasoning for analytical / creative / verification / negotiation
        card = self._chain_run(node.goal, agent_execute_fn, ctx)
        return card.body if hasattr(card, "body") else str(card)

    def _chain_run(self, message: str, agent_execute_fn: Callable, ctx: dict) -> PrismCard:
        if self._chain is not None:
            return self._chain.run(message, agent_execute_fn, ctx)
        card = agent_execute_fn("autonomous", message, ctx)
        return card

    # ------------------------------------------------------------------
    # Condition checking
    # ------------------------------------------------------------------

    def _check_condition(self, node: OrchestratorNode, graph: TaskGraph) -> bool:
        """Ask the LLM whether dependency results satisfy node.condition."""
        if not self._router:
            return True
        dep_outputs = "\n".join(
            (lambda dn: f"{dn.node_id}: {(dn.result or '')[:300]}"
             if dn is not None else f"{dep_id}: ")(graph.get_node(dep_id))
            for dep_id in node.depends_on
        )
        if not dep_outputs.strip():
            return True
        prompt = _CONDITION_CHECK_PROMPT.format(
            condition = node.condition,
            output    = dep_outputs,
        )
        try:
            resp, _ = self._router.call(prompt, min_capability=1, max_tokens=10, speculative=False)
            return resp.strip().upper().startswith("Y")
        except Exception:
            return True

    # ------------------------------------------------------------------
    # Horizon pause handling
    # ------------------------------------------------------------------

    def _handle_horizon_pause(self, node: OrchestratorNode, graph: TaskGraph) -> None:
        if self._horizon is None:
            logger.warning("[orch] horizon_pause requested but no HorizonPlanner — skipping pause")
            node.status = "skipped"
            return
        try:
            goal_id = self._horizon.add(
                intent             = node.goal,
                trigger_condition  = node.condition or "user_confirms",
                completion_condition = "task_resumed_by_orchestrator",
                expires_in_days    = 14.0,
            )
            graph.horizon_goal_ids[node.node_id] = goal_id
            node.status = "waiting"
            logger.info("[orch] node %s paused — horizon goal %s created", node.node_id, goal_id)
        except Exception as exc:
            logger.warning("[orch] horizon_pause failed: %s", exc)
            node.status = "skipped"

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def _synthesise(self, graph: TaskGraph) -> str:
        results_text = "\n\n".join(
            f"[{n.node_id} — {n.intent}]\n{n.result}"
            for n in graph.nodes
            if n.result and n.status == "done"
        )
        if not results_text:
            return "All sub-tasks completed but no results to synthesise."

        if not self._router:
            # Fallback: concatenate
            return results_text

        prompt = _SYNTHESIS_PROMPT.format(
            original = graph.original,
            hint     = graph.synthesis_hint or "Summarise all results clearly.",
            results  = results_text[:3000],
        )
        try:
            text, _ = self._router.call(
                prompt,
                min_capability = 2,
                max_tokens     = 600,
                speculative    = True,
            )
            return text.strip()
        except Exception as exc:
            logger.warning("[orch] synthesis LLM call failed: %s", exc)
            return results_text

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self, graph: TaskGraph) -> None:
        d = graph.to_dict()
        completed_at = time.time() if graph.status in ("completed", "failed") else None
        with sqlite3.connect(self._db, timeout=30.0) as con:
            con.execute(
                """INSERT OR REPLACE INTO task_graphs VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    d["graph_id"], d["original"], d["context_id"],
                    json.dumps(d["nodes"]),
                    d["synthesis_hint"], d["status"], d["final_answer"],
                    json.dumps(d["horizon_goal_ids"]),
                    d["created_at"], completed_at,
                ),
            )

    def _load_paused(self) -> list[TaskGraph]:
        try:
            with sqlite3.connect(self._db, timeout=30.0) as con:
                rows = con.execute(
                    "SELECT * FROM task_graphs WHERE status='paused'"
                ).fetchall()
            graphs = []
            for row in rows:
                d = {
                    "graph_id":         row[0], "original": row[1],
                    "context_id":       row[2],
                    "nodes":            json.loads(row[3]),
                    "synthesis_hint":   row[4],  "status":   row[5],
                    "final_answer":     row[6],
                    "horizon_goal_ids": json.loads(row[7]),
                    "created_at":       row[8],
                }
                graphs.append(TaskGraph.from_dict(d))
            return graphs
        except Exception as exc:
            logger.debug("[orch] load_paused failed: %s", exc)
            return []

    def _init_db(self) -> None:
        with sqlite3.connect(self._db, timeout=30.0) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS task_graphs (
                    graph_id             TEXT PRIMARY KEY,
                    original             TEXT NOT NULL,
                    context_id           TEXT NOT NULL DEFAULT 'default',
                    nodes_json           TEXT NOT NULL DEFAULT '[]',
                    synthesis_hint       TEXT NOT NULL DEFAULT '',
                    status               TEXT NOT NULL DEFAULT 'running',
                    final_answer         TEXT NOT NULL DEFAULT '',
                    horizon_goal_ids_json TEXT NOT NULL DEFAULT '{}',
                    created_at           REAL NOT NULL,
                    completed_at         REAL
                )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS ix_orch_status ON task_graphs(status)")
            self._migrate(con)

    def _migrate(self, con: sqlite3.Connection) -> None:
        ver = con.execute("PRAGMA user_version").fetchone()[0]
        if ver < 1:
            con.execute("PRAGMA user_version = 1")
