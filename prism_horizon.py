"""
prism_horizon.py
================
PRISM Horizon Planner — Cross-Session Long-Horizon Goal Persistence

Closes the gap between single-session execution and goals that span
multiple days or sessions.  A ``HorizonGoal`` watches for a real-world
condition to become true (a price drop, an inbox event, a data threshold),
then automatically triggers execution via the TaskQueue when the condition
fires — picking up any partially-completed steps from the previous session.

Key properties that distinguish ``HorizonGoal`` from ``PrismTasks`` or the
task queue:

* **Condition-triggered** — fires when a state predicate is satisfied,
  not just on a schedule.
* **Context-accumulating** — each session can deposit new facts; the LLM
  receives the full history on the next check.
* **Step-checkpointing** — partial execution is saved so the next session
  resumes from the last completed step rather than restarting.
* **Completion-aware** — the goal knows when it is truly done, rather than
  running indefinitely.

Persistence
-----------
All state lives in ``~/.prism/horizon.db`` (SQLite), consistent with every
other PRISM subsystem.

Typical flow
------------
::

    horizon = HorizonPlanner(llm_router=router, task_queue=queue, push=push)

    gid = horizon.add(
        intent="Book a flight to Lisbon when the price drops below $300",
        trigger_condition="price drops below 300",
        completion_condition="flight is booked",
    )

    # Attach a probe that checks the real world directly
    horizon.register_probe(gid, lambda ctx: fetch_price("LIS") < 300)

    # At every session start — evaluates all WATCHING goals
    triggered = horizon.on_session_start()

    # Deposit new facts mid-session (these survive to the next session)
    horizon.update_context(gid, last_price=290, checked_at="2026-06-02")

    # Before shutdown — checkpoint any active goals
    horizon.on_session_end()
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


class HorizonGoalStatus(str, Enum):
    WATCHING   = "watching"    # condition not yet met; checked each session
    TRIGGERED  = "triggered"   # condition met this session; executing now
    PAUSED     = "paused"      # mid-execution; session ended before completion
    COMPLETED  = "completed"   # goal fully achieved
    ABANDONED  = "abandoned"   # user cancelled or deadline passed


@dataclass
class HorizonGoal:
    """Full record for one cross-session goal."""

    goal_id:              str
    intent:               str              # "Book a flight when price < $300"
    trigger_condition:    str              # "price drops below 300"
    completion_condition: str = ""         # "flight is booked"

    status:               HorizonGoalStatus = HorizonGoalStatus.WATCHING
    accumulated_context:  dict = field(default_factory=dict)
    completed_steps:      list[str] = field(default_factory=list)

    created_at:           float = field(default_factory=time.time)
    last_checked_at:      Optional[float] = None
    triggered_at:         Optional[float] = None
    completed_at:         Optional[float] = None
    session_count:        int = 0
    expires_at:           Optional[float] = None
    notes:                str = ""
    outcome_rate:         float = 0.0
    sample_size:          int = 0

    def is_expired(self) -> bool:
        return self.expires_at is not None and time.time() > self.expires_at

    def to_row(self) -> tuple:
        return (
            self.goal_id,
            self.intent,
            self.trigger_condition,
            self.completion_condition,
            self.status.value,
            json.dumps(self.accumulated_context),
            json.dumps(self.completed_steps),
            self.created_at,
            self.last_checked_at,
            self.triggered_at,
            self.completed_at,
            self.session_count,
            self.expires_at,
            self.notes,
            self.outcome_rate,
            self.sample_size,
        )

    @classmethod
    def from_row(cls, row: tuple) -> HorizonGoal:
        (
            goal_id, intent, trigger_condition, completion_condition,
            status, ctx_json, steps_json,
            created_at, last_checked_at, triggered_at, completed_at,
            session_count, expires_at, notes,
            *extra,  # outcome_rate, sample_size — absent in pre-migration rows
        ) = row
        return cls(
            goal_id=goal_id,
            intent=intent,
            trigger_condition=trigger_condition,
            completion_condition=completion_condition or "",
            status=HorizonGoalStatus(status),
            accumulated_context=json.loads(ctx_json or "{}"),
            completed_steps=json.loads(steps_json or "[]"),
            created_at=created_at or time.time(),
            last_checked_at=last_checked_at,
            triggered_at=triggered_at,
            completed_at=completed_at,
            session_count=session_count or 0,
            expires_at=expires_at,
            notes=notes or "",
            outcome_rate=extra[0] if len(extra) > 0 else 0.0,
            sample_size=int(extra[1]) if len(extra) > 1 else 0,
        )


# ---------------------------------------------------------------------------
# HorizonPlanner
# ---------------------------------------------------------------------------


class HorizonPlanner:
    """
    Cross-session goal watchdog and condition-triggered executor.

    Parameters
    ----------
    llm_router : LLMRouter or None
        Used for natural-language condition evaluation when no probe is
        registered.
    task_queue : TaskQueue or None
        Receives triggered goals as multi-step background tasks.
    push : PrismPush or None
        Desktop/mobile notifications on trigger, completion, and abandonment.
    db_path : str
        Path to the SQLite database (default: ``~/.prism/horizon.db``).
    """

    def __init__(
        self,
        llm_router=None,
        task_queue=None,
        push=None,
        db_path: str = "~/.prism/horizon.db",
    ) -> None:
        self._llm    = llm_router
        self._queue  = task_queue
        self._push   = push
        self._db     = Path(db_path).expanduser()
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._probes: dict[str, Callable[[dict], bool]] = {}
        self._lock   = threading.Lock()
        self._chain: Any = None  # optional PrismChain for richer hand-off
        self._init_db()

    # ------------------------------------------------------------------
    # Goal lifecycle
    # ------------------------------------------------------------------

    def add(
        self,
        intent:               str,
        trigger_condition:    str,
        completion_condition: str = "",
        expires_in_days:      Optional[float] = None,
        probe:                Optional[Callable[[dict], bool]] = None,
    ) -> str:
        """Register a new horizon goal and return its ID.

        Parameters
        ----------
        intent : str
            Full natural-language goal, e.g. ``"Book a flight when the price
            drops below $300."``.
        trigger_condition : str
            Predicate checked each session, e.g. ``"price below 300"``.
        completion_condition : str, optional
            What constitutes success, e.g. ``"flight is booked"``.
        expires_in_days : float, optional
            Automatically abandon the goal after this many days.
        probe : callable, optional
            ``probe(context: dict) -> bool`` checked directly; skips the
            LLM fallback when provided.

        Returns
        -------
        str
            The new goal's ID.
        """
        goal_id    = str(uuid.uuid4())[:8]
        expires_at = time.time() + expires_in_days * 86400 if expires_in_days else None
        goal = HorizonGoal(
            goal_id=goal_id,
            intent=intent,
            trigger_condition=trigger_condition,
            completion_condition=completion_condition,
            expires_at=expires_at,
        )
        with self._lock:
            self._upsert(goal)
            if probe is not None:
                self._probes[goal_id] = probe
        logger.info("HorizonPlanner: added goal %s — %r", goal_id, intent[:60])
        return goal_id

    def add_triggered(
        self,
        intent: str,
        completion_condition: str = "",
        context: Optional[dict] = None,
    ) -> str:
        """Create a goal that is immediately TRIGGERED (used for active chains).

        Unlike add(), this skips the WATCHING state — the goal is already
        executing. The caller is responsible for recording steps and completing
        or abandoning the goal.
        """
        goal_id = str(uuid.uuid4())[:8]
        goal = HorizonGoal(
            goal_id=goal_id,
            intent=intent,
            trigger_condition="immediate",
            completion_condition=completion_condition,
            status=HorizonGoalStatus.TRIGGERED,
            triggered_at=time.time(),
            accumulated_context=context or {},
        )
        with self._lock:
            self._upsert(goal)
        logger.info("HorizonPlanner: chain anchor created %s — %r", goal_id, intent[:60])
        return goal_id

    def resumable_chains(self) -> list[HorizonGoal]:
        """Return PAUSED goals whose intent starts with 'chain:'.

        These are chains that were interrupted mid-run and can be resumed
        by PrismChain.resume().
        """
        return [
            g for g in self.list_goals(status=HorizonGoalStatus.PAUSED)
            if g.intent.startswith("chain:")
        ]

    def register_probe(self, goal_id: str, probe: Callable[[dict], bool]) -> None:
        """Attach a probe function to an existing goal.

        The probe receives the goal's ``accumulated_context`` dict and
        returns ``True`` when the trigger condition is met.
        """
        if self.get(goal_id) is None:
            raise KeyError(f"No horizon goal with id={goal_id!r}")
        with self._lock:
            self._probes[goal_id] = probe

    def update_context(self, goal_id: str, **facts) -> None:
        """Deposit new facts into a goal's accumulated context.

        Facts persist between sessions and are injected into every subsequent
        trigger evaluation and execution prompt.

        Example::

            horizon.update_context(gid, last_price=290, checked_at="2026-06-02")
        """
        with self._lock:
            goal = self._load_goal(goal_id)
            if goal is None:
                raise KeyError(f"No horizon goal with id={goal_id!r}")
            goal.accumulated_context.update(facts)
            self._upsert(goal)
        logger.debug("HorizonPlanner: context updated for %s — %s",
                     goal_id, list(facts.keys()))

    def complete(self, goal_id: str, notes: str = "") -> bool:
        """Mark a goal as completed. Returns False if already terminal."""
        with self._lock:
            goal = self._load_goal(goal_id)
            if goal is None or goal.status == HorizonGoalStatus.COMPLETED:
                return False
            goal.status       = HorizonGoalStatus.COMPLETED
            goal.completed_at = time.time()
            if notes:
                goal.notes = notes
            self._upsert(goal)
        self._notify(f"Horizon goal completed: {goal.intent[:60]}")
        logger.info("HorizonPlanner: goal %s completed", goal_id)
        return True

    def abandon(self, goal_id: str, reason: str = "") -> bool:
        """Abandon a goal (user-cancelled or expired)."""
        with self._lock:
            goal = self._load_goal(goal_id)
            if goal is None or goal.status in (
                HorizonGoalStatus.COMPLETED, HorizonGoalStatus.ABANDONED
            ):
                return False
            goal.status       = HorizonGoalStatus.ABANDONED
            goal.completed_at = time.time()
            if reason:
                goal.notes = reason
            self._upsert(goal)
        logger.info("HorizonPlanner: goal %s abandoned — %s", goal_id, reason or "(no reason)")
        return True

    def record_step(self, goal_id: str, step_description: str) -> None:
        """Record a completed execution step.

        Call this after each discrete step succeeds so the next session
        can resume rather than restart.
        """
        with self._lock:
            goal = self._load_goal(goal_id)
            if goal is None:
                return
            goal.completed_steps.append(step_description)
            self._upsert(goal)

    # ------------------------------------------------------------------
    # Session lifecycle hooks
    # ------------------------------------------------------------------

    def on_session_start(self) -> list[str]:
        """Evaluate all WATCHING goals; trigger those whose conditions are met.

        Also resumes PAUSED goals that were interrupted last session.
        Call this once at PRISM startup.

        Returns
        -------
        list[str]
            IDs of goals that were triggered or resumed this session.
        """
        goals     = self.list_goals()
        activated: list[str] = []

        # Separate deterministic (fast) goals from LLM-evaluated ones
        fast_goals: list[HorizonGoal] = []
        llm_goals:  list[HorizonGoal] = []

        for goal in goals:
            if goal.status in (HorizonGoalStatus.COMPLETED, HorizonGoalStatus.ABANDONED):
                continue
            if goal.is_expired():
                self.abandon(goal.goal_id, reason="deadline passed")
                continue
            with self._lock:
                g = self._load_goal(goal.goal_id)
                if g is None:
                    continue
                g.session_count  += 1
                g.last_checked_at = time.time()
                self._upsert(g)
            if goal.status in (HorizonGoalStatus.PAUSED, HorizonGoalStatus.WATCHING):
                # Goals with a registered probe or simple deterministic trigger → fast
                _det = self._deterministic_condition(goal.trigger_condition, goal.accumulated_context)
                if goal.goal_id in self._probes or _det is not None:
                    fast_goals.append(goal)
                else:
                    llm_goals.append(goal)

        def _eval_one(goal: HorizonGoal) -> tuple[HorizonGoal, bool]:
            if goal.status == HorizonGoalStatus.PAUSED:
                return goal, True
            return goal, self._evaluate_trigger(goal)

        # Evaluate fast goals serially, LLM goals concurrently (max 4 workers)
        results: list[tuple[HorizonGoal, bool]] = []
        results.extend(_eval_one(g) for g in fast_goals)
        if llm_goals:
            with ThreadPoolExecutor(max_workers=min(len(llm_goals), 4),
                                    thread_name_prefix="horizon-eval") as pool:
                futs = {pool.submit(_eval_one, g): g for g in llm_goals}
                for fut in as_completed(futs, timeout=90.0):
                    try:
                        results.append(fut.result())
                    except Exception as exc:
                        logger.debug("HorizonPlanner: eval error for %s: %s",
                                     futs[fut].goal_id, exc)

        for goal, triggered in results:
            if goal.status == HorizonGoalStatus.PAUSED and triggered:
                logger.info("HorizonPlanner: resuming paused goal %s (%d steps done)",
                            goal.goal_id, len(goal.completed_steps))
                self._hand_off(goal, resume=True)
                activated.append(goal.goal_id)
            elif goal.status == HorizonGoalStatus.WATCHING and triggered:
                with self._lock:
                    g = self._load_goal(goal.goal_id)
                    if g is None:
                        continue
                    g.status       = HorizonGoalStatus.TRIGGERED
                    g.triggered_at = time.time()
                    self._upsert(g)
                self._notify(f"Horizon goal triggered: {goal.intent[:60]}")
                logger.info("HorizonPlanner: goal %s triggered", goal.goal_id)
                self._hand_off(goal, resume=False)
                activated.append(goal.goal_id)
            else:
                logger.debug("HorizonPlanner: goal %s still watching (session #%d)",
                             goal.goal_id, goal.session_count)

        return activated

    def on_session_end(self) -> None:
        """Checkpoint any TRIGGERED goals as PAUSED before shutdown.

        Call this when PRISM is shutting down so in-progress goals are
        preserved for the next session.
        """
        paused = 0
        for goal in self.list_goals(status=HorizonGoalStatus.TRIGGERED):
            with self._lock:
                g = self._load_goal(goal.goal_id)
                if g and g.status == HorizonGoalStatus.TRIGGERED:
                    g.status = HorizonGoalStatus.PAUSED
                    self._upsert(g)
                    paused += 1
        if paused:
            logger.info(
                "HorizonPlanner: %d goal(s) checkpointed as PAUSED", paused
            )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get(self, goal_id: str) -> Optional[HorizonGoal]:
        """Return the HorizonGoal for the given ID, or None."""
        with self._lock:
            return self._load_goal(goal_id)

    def list_goals(
        self, status: Optional[HorizonGoalStatus] = None
    ) -> list[HorizonGoal]:
        """Return all goals, optionally filtered by status."""
        with sqlite3.connect(self._db) as conn:
            if status is not None:
                rows = conn.execute(
                    "SELECT * FROM horizon_goals WHERE status = ? ORDER BY created_at DESC",
                    (status.value,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM horizon_goals ORDER BY created_at DESC"
                ).fetchall()
        return [HorizonGoal.from_row(r) for r in rows]

    def check_now(self, goal_id: str) -> bool:
        """Manually evaluate a trigger condition without changing status.

        Useful for interactive inspection without side-effects.
        """
        goal = self.get(goal_id)
        if goal is None:
            raise KeyError(f"No horizon goal with id={goal_id!r}")
        return self._evaluate_trigger(goal)

    def status(self) -> dict:
        """Return a JSON-serialisable summary of all goals."""
        goals = self.list_goals()
        counts = {s.value: 0 for s in HorizonGoalStatus}
        for g in goals:
            counts[g.status.value] += 1
        return {
            "total": len(goals),
            "counts": counts,
            "goals": [
                {
                    "goal_id":          g.goal_id,
                    "intent":           g.intent[:60],
                    "trigger_condition":g.trigger_condition[:40],
                    "status":           g.status.value,
                    "session_count":    g.session_count,
                    "completed_steps":  len(g.completed_steps),
                    "last_checked_at":  g.last_checked_at,
                    "triggered_at":     g.triggered_at,
                    "expires_at":       g.expires_at,
                }
                for g in goals
            ],
        }

    def as_proactive_trigger(self, goal_id: str):
        """Wrap a horizon goal as a ``ProactiveTrigger`` for ``PrismProactive``.

        This lets ``PrismProactive``'s background daemon re-check the
        condition every ``check_every`` seconds within a live session —
        on top of the per-session check at startup.

        Returns
        -------
        ProactiveTrigger
            Ready to pass to ``PrismProactive.register()``.
        """
        from prism_proactive import ProactiveTrigger

        goal = self.get(goal_id)
        if goal is None:
            raise KeyError(f"No horizon goal with id={goal_id!r}")

        def _condition() -> bool:
            g = self.get(goal_id)
            if g is None or g.status != HorizonGoalStatus.WATCHING:
                return False
            return self._evaluate_trigger(g)

        def _message() -> str:
            g = self.get(goal_id)
            return f"Horizon goal triggered: {g.intent[:60]}" if g else ""

        return ProactiveTrigger(
            trigger_id=f"horizon_{goal_id}",
            name=f"Horizon: {goal.intent[:40]}",
            check_every=300,   # re-check every 5 minutes within a session
            condition=_condition,
            message=_message,
            cooldown=3600,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deterministic_condition(condition: str, ctx: dict) -> bool | None:
        """
        Evaluate simple numeric/date/presence conditions without an LLM call.
        Returns True/False if the pattern is recognized, None to fall through to LLM.

        Recognized patterns (case-insensitive):
          - "<key> >= <number>"  /  "<key> > <number>"
          - "<key> <= <number>"  /  "<key> < <number>"
          - "<key> == <number>"  /  "<key> != <number>"
          - "day is <weekday>" — True if today matches
          - "<key> exists" / "<key> present" — True if key has a truthy value
        """
        import re as _re
        from datetime import datetime as _dt

        cond = condition.strip().lower()

        # Numeric comparison: "hrv >= 60" / "price < 100" etc.
        m = _re.match(r"(\w+)\s*(>=|<=|>|<|==|!=)\s*([0-9.+-]+)", cond)
        if m:
            key, op, val_str = m.group(1), m.group(2), m.group(3)
            if key in ctx:
                try:
                    lhs = float(ctx[key])
                    rhs = float(val_str)
                    return {
                        ">=": lhs >= rhs,
                        "<=": lhs <= rhs,
                        ">":  lhs >  rhs,
                        "<":  lhs <  rhs,
                        "==": lhs == rhs,
                        "!=": lhs != rhs,
                    }[op]
                except (TypeError, ValueError):
                    pass

        # Day-of-week: "day is monday"
        m = _re.match(r"day\s+is\s+(\w+)", cond)
        if m:
            target = m.group(1)[:3]
            today = _dt.now().strftime("%a").lower()
            return today == target

        # Presence check: "temperature exists" / "user_name present"
        m = _re.match(r"(\w+)\s+(exists|present)", cond)
        if m:
            return bool(ctx.get(m.group(1)))

        return None  # unrecognized — let LLM handle it

    def _evaluate_trigger(self, goal: HorizonGoal) -> bool:
        """Return True if the goal's trigger condition is currently met."""
        # 1. Probe function (highest-fidelity — checks the real world directly)
        probe = self._probes.get(goal.goal_id)
        if probe is not None:
            try:
                return bool(probe(dict(goal.accumulated_context)))
            except Exception as exc:
                logger.warning(
                    "HorizonPlanner: probe for %s raised: %s", goal.goal_id, exc
                )
                return False

        # 2. Deterministic router — zero-latency for numeric/date/presence patterns
        det = self._deterministic_condition(
            goal.trigger_condition, dict(goal.accumulated_context)
        )
        if det is not None:
            logger.debug("HorizonPlanner: deterministic eval %s → %s", goal.goal_id, det)
            return det

        # 3. LLM-based natural-language evaluation
        if self._llm is not None:
            return self._llm_evaluate(goal)

        # 4. No probe, no LLM — ask the user
        logger.debug(
            "HorizonPlanner: cannot evaluate %s — no probe or LLM", goal.goal_id
        )
        self._notify(
            f"Horizon goal needs your attention: {goal.trigger_condition[:60]}"
        )
        return False

    def _llm_evaluate(self, goal: HorizonGoal) -> bool:
        """Ask the LLM whether the trigger condition is currently satisfied."""
        ctx_str = (
            json.dumps(goal.accumulated_context, indent=2)
            if goal.accumulated_context
            else "(no context accumulated yet)"
        )
        prompt = (
            f"You are evaluating whether a monitoring condition is met.\n\n"
            f"Goal: {goal.intent}\n"
            f"Condition to check: {goal.trigger_condition}\n\n"
            f"Accumulated context from previous sessions:\n{ctx_str}\n\n"
            f"Based only on the context above, is the condition currently met? "
            f"Reply with ONLY the word 'yes' or 'no'."
        )
        try:
            response, _ = self._llm.call(prompt)
            result = (response or "").strip().lower().startswith("yes")
            logger.debug(
                "HorizonPlanner: LLM evaluated %s → %s (raw=%r)",
                goal.goal_id, result, (response or "")[:30],
            )
            return result
        except Exception as exc:
            logger.warning("HorizonPlanner: LLM evaluation failed: %s", exc)
            return False

    def _hand_off(self, goal: HorizonGoal, resume: bool) -> None:
        """Submit the goal to TaskQueue for execution."""
        if self._queue is None:
            logger.warning(
                "HorizonPlanner: no TaskQueue — goal %s cannot execute", goal.goal_id
            )
            return
        # Phase gate: defer goal execution if system is in LIQUID phase
        try:
            import prism_phase as _pp
            _engine = _pp.get_engine()
            if _engine.history and _engine.current_phase.value == "LIQUID":
                logger.info(
                    "HorizonPlanner: LIQUID phase — deferring goal %s to next session",
                    goal.goal_id,
                )
                with self._lock:
                    g = self._load_goal(goal.goal_id)
                    if g and g.status == HorizonGoalStatus.TRIGGERED:
                        g.status = HorizonGoalStatus.PAUSED
                        self._upsert(g)
                return
        except Exception:
            pass

        description = self._build_execution_prompt(goal, resume)

        def _execute_step(params: dict) -> str:
            # Prefer PrismChain for richer execution; fall back to bare LLM
            goal_prompt = params.get("prompt", description)
            if self._chain is not None:
                try:
                    card = self._chain.run(
                        goal_prompt,
                        agent_execute_fn=lambda intent, msg, ctx: None,
                        base_ctx={},
                    )
                    step_text = (getattr(card, "body", None) or "").strip()[:200]
                    if step_text:
                        self.record_step(goal.goal_id, step_text)
                        return step_text
                except Exception as exc:
                    logger.debug("HorizonPlanner: chain hand-off failed, falling back to LLM: %s", exc)
            if self._llm is None:
                return "No LLM available — manual execution required."
            response, _ = self._llm.call(goal_prompt)
            step_text = (response or "").strip()[:200]
            self.record_step(goal.goal_id, step_text)
            return step_text

        steps = [
            {
                "title": f"Execute: {goal.intent[:50]}",
                "fn":    _execute_step,
                "params": {"prompt": description},
            }
        ]

        def _on_complete(progress) -> None:
            if progress.status.value == "completed":
                self.complete(goal.goal_id, notes="Executed via TaskQueue")

        self._queue.submit(
            title=f"Horizon: {goal.intent[:50]}",
            steps=steps,
            on_complete=_on_complete,
        )
        with self._lock:
            g = self._load_goal(goal.goal_id)
            if g:
                g.status = HorizonGoalStatus.TRIGGERED
                self._upsert(g)

    def _build_execution_prompt(self, goal: HorizonGoal, resume: bool) -> str:
        """Build the full execution prompt with accumulated context and prior steps."""
        parts = [goal.intent]

        if goal.accumulated_context:
            ctx_lines = "\n".join(
                f"  {k}: {v}" for k, v in goal.accumulated_context.items()
            )
            parts.append(f"\nAccumulated context:\n{ctx_lines}")

        if resume and goal.completed_steps:
            done = "\n".join(f"  - {s}" for s in goal.completed_steps)
            parts.append(
                f"\nAlready completed in previous sessions:\n{done}"
                f"\n\nContinue from where execution stopped — skip completed steps."
            )

        if goal.completion_condition:
            parts.append(f"\nSuccess criterion: {goal.completion_condition}")

        return "\n".join(parts)

    def _notify(self, message: str) -> None:
        if self._push is not None:
            try:
                self._push.send("PRISM Horizon", message)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # SQLite persistence
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with sqlite3.connect(self._db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS horizon_goals (
                    goal_id              TEXT PRIMARY KEY,
                    intent               TEXT,
                    trigger_condition    TEXT,
                    completion_condition TEXT,
                    status               TEXT,
                    context_json         TEXT,
                    steps_json           TEXT,
                    created_at           REAL,
                    last_checked_at      REAL,
                    triggered_at         REAL,
                    completed_at         REAL,
                    session_count        INTEGER,
                    expires_at           REAL,
                    notes                TEXT,
                    outcome_rate         REAL NOT NULL DEFAULT 0.0,
                    sample_size          INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_hz_status ON horizon_goals(status)"
            )
            self._migrate(conn)

    def _migrate(self, conn) -> None:
        ver = conn.execute("PRAGMA user_version").fetchone()[0]
        if ver < 1:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(horizon_goals)")}
            if "outcome_rate" not in cols:
                conn.execute("ALTER TABLE horizon_goals ADD COLUMN outcome_rate REAL NOT NULL DEFAULT 0.0")
            if "sample_size" not in cols:
                conn.execute("ALTER TABLE horizon_goals ADD COLUMN sample_size INTEGER NOT NULL DEFAULT 0")
            conn.execute("PRAGMA user_version = 1")

    def _upsert(self, goal: HorizonGoal) -> None:
        with sqlite3.connect(self._db) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO horizon_goals VALUES
                   (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                goal.to_row(),
            )

    def _load_goal(self, goal_id: str) -> Optional[HorizonGoal]:
        with sqlite3.connect(self._db) as conn:
            row = conn.execute(
                "SELECT * FROM horizon_goals WHERE goal_id = ?", (goal_id,)
            ).fetchone()
        return HorizonGoal.from_row(row) if row else None

    def __repr__(self) -> str:  # pragma: no cover
        s = self.status()
        return (
            f"HorizonPlanner(total={s['total']}, "
            f"watching={s['counts']['watching']}, "
            f"triggered={s['counts']['triggered']}, "
            f"paused={s['counts']['paused']})"
        )
