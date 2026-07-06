from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from prism_responses import PrismCard

logger = logging.getLogger(__name__)


# ── Typed inter-logic communication ──────────────────────────────────────────

@dataclass
class LogicResult:
    """
    Typed output from one logic step, usable as input to the next.
    Every logic in a chain produces one of these.
    """
    step_id:    str
    logic:      str           # which logic produced this
    data:       Any           # structured result (dict, list, str, number)
    text:       str           # human-readable summary for LLM context
    success:    bool = True
    error:      str  = ""
    duration_ms:float = 0.0


@dataclass
class CompositionStep:
    """One node in a composition plan."""
    step_id:    str
    logic:      str           # intent name or "autonomous"
    description:str           # what this step does
    depends_on: list[str] = field(default_factory=list)  # step_ids this waits for
    params:     dict     = field(default_factory=dict)   # static params
    input_from: str      = ""   # step_id whose .data feeds this step's message


@dataclass
class CompositionPlan:
    """A parsed multi-step execution plan."""
    plan_id:   str
    original:  str              # original user message
    steps:     list[CompositionStep]
    parallel:  bool = False     # true if steps can run concurrently


# ── Logic registry ────────────────────────────────────────────────────────────

LOGIC_REGISTRY: dict[str, str] = {
    # intent_name → what it does / what it returns
    "plan":              "generates a daily plan or schedule",
    "universal_plan":    "creates a multi-strategy action plan for any goal",
    "domain_medical":    "triages medical situations and recommends urgency level",
    "domain_financial":  "recommends portfolio allocation and investment strategy",
    "domain_legal":      "recommends legal strategy (settle vs litigate etc.)",
    "domain_hr":         "recommends hiring/HR decisions",
    "domain_supply":     "recommends supply chain/procurement decisions",
    "domain_climate":    "recommends climate/energy policy decisions",
    "email_read":        "reads and summarises unread emails, returns email list",
    "email_send":        "drafts and sends an email",
    "calendar_read":     "reads today's calendar events, returns event list",
    "calendar_write":    "creates a new calendar event",
    "web_search":        "searches the web and returns summarised results",
    "contacts":          "looks up a contact and returns their details",
    "add_task":          "adds a task to the task manager",
    "list_tasks":        "returns list of open tasks",
    "reminder":          "schedules a timed reminder",
    "send_push":         "sends a push notification to the user's device",
    "smart_home":        "controls smart home devices",
    "browser_task":      "navigates the web or fills in forms via browser",
    "device_task":       "executes a file system or shell task on this device",
    "ksa_task":          "searches or indexes code files",
    "identity":          "returns the user's digital identity profile",
    "task_status":       "returns status of background tasks",
    "status":            "returns PRISM system status",
    "calibrate":         "records user feedback to update decision model",
    "autonomous":        "synthesises and executes a custom tool for novel tasks",
    "research":          "deep multi-step research: web search → parse → cross-reference → synthesise",
}


class PrismComposer:
    """
    Logic composition engine.

    Decomposes a complex user request into a directed acyclic graph (DAG)
    of logic steps, executes them in dependency order (parallelising
    independent steps), pipes typed outputs between steps, and returns
    a composed PrismCard.

    Example:
        "Check my emails, find anything urgent, add those as tasks,
         then send me a push notification summary"

        → Step 1: email_read       (no deps)
        → Step 2: add_task         (depends on step 1, input_from step 1)
        → Step 3: send_push        (depends on step 2, input_from step 1)
    """

    # Minimum number of detected sub-tasks to trigger composition
    # (single-step requests go straight to normal routing)
    MIN_STEPS_FOR_COMPOSITION = 2

    def __init__(self, llm_router=None, policy_engine=None,
                  push=None, task_queue=None):
        self._router = llm_router
        self._policy = policy_engine
        self._push   = push
        self._queue  = task_queue

    # ── Public API ────────────────────────────────────────────────────────────

    def should_compose(self, message: str) -> bool:
        """
        Quick heuristic: does this message describe multiple sub-tasks?
        Avoids LLM call for simple single-step requests.
        """
        connectors = [
            " and then ", " then ", " after that ", " afterwards ",
            " also ", " and also ", " followed by ", ", then ",
            " once you ", " when you ", " while you ",
            " and send ", " and add ", " and create ", " and email ",
            " and notify ", " and push ", " and schedule ",
        ]
        msg_lower = message.lower()
        return any(c in msg_lower for c in connectors)

    def decompose(self, message: str) -> Optional[CompositionPlan]:
        """
        Use LLM to decompose message into a CompositionPlan.
        Returns None if the message is actually single-step.
        """
        if not self._router:
            return None

        registry_desc = "\n".join(
            f"  {k}: {v}" for k, v in LOGIC_REGISTRY.items())

        prompt = f"""You are decomposing a user request into a sequence of logic steps for an AI assistant.

Available logics:
{registry_desc}

User request: "{message}"

Break this into 2-5 ordered steps. Each step uses one logic from the list above.
Steps can depend on previous steps' outputs.

Return ONLY valid JSON:
{{
  "steps": [
    {{
      "step_id": "s1",
      "logic": "<logic_name_from_list>",
      "description": "<what this step does>",
      "depends_on": [],
      "input_from": "",
      "params": {{}}
    }},
    {{
      "step_id": "s2",
      "logic": "<logic_name_from_list>",
      "description": "<what this step does>",
      "depends_on": ["s1"],
      "input_from": "s1",
      "params": {{}}
    }}
  ],
  "parallel": false
}}

Rules:
- Only use logic names from the available list above
- If a step needs output from a previous step, set input_from to that step's step_id
- depends_on lists step_ids that must complete before this step starts
- parallel: true only if ALL steps are fully independent (no depends_on)
- If this is actually a single task, return {{"steps": [], "parallel": false}}
"""

        raw, _ = self._router.call(prompt, min_capability=2, max_tokens=600,
                                    json_mode=True)
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data  = json.loads(clean)
        except Exception as e:
            logger.debug("Decompose parse failed: %s", e)
            return None

        steps_data = data.get("steps", [])
        if len(steps_data) < self.MIN_STEPS_FOR_COMPOSITION:
            return None

        steps = []
        for s in steps_data:
            logic = s.get("logic", "")
            if logic not in LOGIC_REGISTRY:
                logger.info("Unknown logic '%s' in decomposition plan → autonomous", logic)
                logic = "autonomous"   # unknown logic → autonomous engine
            steps.append(CompositionStep(
                step_id     = s.get("step_id", str(uuid.uuid4())[:4]),
                logic       = logic,
                description = s.get("description", ""),
                depends_on  = s.get("depends_on", []),
                input_from  = s.get("input_from", ""),
                params      = s.get("params", {}),
            ))

        return CompositionPlan(
            plan_id  = str(uuid.uuid4())[:8],
            original = message,
            steps    = steps,
            parallel = data.get("parallel", False),
        )

    def execute(self, plan: CompositionPlan,
                 agent_execute_fn,
                 base_ctx: dict) -> PrismCard:
        """
        Execute a CompositionPlan.

        agent_execute_fn: callable(intent, message, ctx) -> PrismCard
            — the agent's own _execute method, so composition reuses all
               existing logic handlers including the autonomous engine.

        Returns a single composed PrismCard summarising all step results.
        """

        results: dict[str, LogicResult] = {}
        completed = set()

        if plan.parallel:
            self._execute_parallel(plan.steps, agent_execute_fn,
                                    base_ctx, results)
        else:
            self._execute_sequential(plan.steps, agent_execute_fn,
                                      base_ctx, results, completed)

        return self._compose_output(plan, results)

    # ── Execution strategies ──────────────────────────────────────────────────

    def _execute_sequential(self, steps, agent_fn, base_ctx,
                             results, completed):
        """Execute steps in dependency order."""
        remaining = list(steps)
        max_iterations = len(steps) * 2
        iterations = 0

        while remaining and iterations < max_iterations:
            iterations += 1
            for step in list(remaining):
                if all(dep in completed for dep in step.depends_on):
                    result = self._run_step(step, agent_fn, base_ctx, results)
                    results[step.step_id] = result
                    completed.add(step.step_id)
                    remaining.remove(step)
                    break   # restart loop after each completion

    def _execute_parallel(self, steps, agent_fn, base_ctx, results):
        """Execute all steps concurrently (only when no dependencies)."""
        threads = []
        lock    = threading.Lock()

        def run(step):
            result = self._run_step(step, agent_fn, base_ctx, {})
            with lock:
                results[step.step_id] = result

        for step in steps:
            t = threading.Thread(target=run, args=(step,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=60)

    def _run_step(self, step: CompositionStep, agent_fn,
                   base_ctx: dict,
                   prior_results: dict) -> LogicResult:
        """Run one step, optionally injecting output from a prior step."""
        t0  = time.time()
        ctx = {**base_ctx, "composition_step": step.step_id}

        # Inject prior step output into message context if requested
        message = step.description
        if step.input_from and step.input_from in prior_results:
            prior = prior_results[step.input_from]
            if prior.text:
                message = f"{step.description}\n\nContext from previous step:\n{prior.text[:800]}"
                ctx["prior_output"] = prior.data

        try:
            card    = agent_fn(step.logic, message, ctx)
            body    = getattr(card, "body", str(card)) or ""
            elapsed = (time.time() - t0) * 1000
            return LogicResult(
                step_id     = step.step_id,
                logic       = step.logic,
                data        = {"body": body, "title": getattr(card, "title", "")},
                text        = body[:600],
                success     = True,
                duration_ms = elapsed,
            )
        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            logger.warning("Composition step %s (%s) failed: %s",
                           step.step_id, step.logic, e)
            return LogicResult(
                step_id     = step.step_id,
                logic       = step.logic,
                data        = {},
                text        = "",
                success     = False,
                error       = str(e),
                duration_ms = elapsed,
            )

    # ── Output composition ────────────────────────────────────────────────────

    def _compose_output(self, plan: CompositionPlan,
                         results: dict[str, LogicResult]) -> PrismCard:
        """
        Merge all step results into one coherent PrismCard.
        If an LLM router is available, synthesise a narrative summary.
        Otherwise, concatenate step outputs.
        """
        from prism_responses import text_card

        ordered = [results[s.step_id] for s in plan.steps
                   if s.step_id in results]

        successes = [r for r in ordered if r.success]
        failures  = [r for r in ordered if not r.success]

        # Try LLM narrative synthesis
        if self._router and successes:
            steps_summary = "\n".join(
                f"Step {i+1} ({r.logic}): {r.text[:300]}"
                for i, r in enumerate(successes))
            prompt = (
                f"You completed a multi-step task for the user.\n"
                f"Original request: '{plan.original}'\n\n"
                f"Results:\n{steps_summary}\n\n"
                f"Write a concise 2-4 sentence summary of what was accomplished. "
                f"Be specific about outcomes. No bullet points.")
            summary, _ = self._router.call(
                prompt, min_capability=1, max_tokens=200)
        else:
            summary = "\n\n".join(
                f"**{r.logic}**: {r.text}" for r in successes)

        # Append failure notes
        if failures:
            fail_notes = "; ".join(
                f"{r.logic} failed: {r.error[:60]}" for r in failures)
            summary += f"\n\n⚠ Some steps failed: {fail_notes}"

        n       = len(plan.steps)
        n_ok    = len(successes)
        total_ms= sum(r.duration_ms for r in ordered)

        title = (f"Done — {n_ok}/{n} steps · "
                 f"{total_ms/1000:.1f}s · plan {plan.plan_id}")

        return text_card(summary, title)
