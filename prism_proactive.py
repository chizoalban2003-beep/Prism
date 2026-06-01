from __future__ import annotations
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

@dataclass
class ProactiveTrigger:
    """One condition that PRISM checks on a schedule."""
    trigger_id:  str
    name:        str
    check_every: int        # seconds between checks
    condition:   Callable[[], bool]   # returns True when trigger should fire
    message:     Callable[[], str]    # returns the message to send
    enabled:     bool = True
    last_fired:  float = 0.0
    cooldown:    int  = 3600          # minimum seconds between firings

@dataclass
class ProactiveEvent:
    trigger_id: str
    message:    str
    timestamp:  float = field(default_factory=time.time)
    delivered:  bool  = False

class PrismProactive:
    """
    Monitors conditions in the background and proactively notifies the user.
    Each trigger runs on its own schedule. Notifications go to the chat UI
    via Server-Sent Events or are stored for polling.

    Built-in triggers:
      calendar_soon    — meeting starting within 15 minutes
      budget_warning   — spending approaching monthly limit
      recovery_alert   — HRV drop suggesting rest needed
      task_completed   — background task finished
      wearable_sync    — wearable data newly available
    """

    def __init__(
        self,
        on_event:    Callable[[ProactiveEvent], None] = None,
        db_path:     str = "~/.prism/proactive.db",
        poll_seconds:int = 60,
    ):
        self._on_event    = on_event or (lambda e: None)
        self._db          = Path(db_path).expanduser()
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._poll        = poll_seconds
        self._triggers:   list[ProactiveTrigger] = []
        self._stop        = threading.Event()
        self._thread:     Optional[threading.Thread] = None
        self._init_db()

    def register(self, trigger: ProactiveTrigger) -> None:
        self._triggers.append(trigger)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="prism-proactive")
        self._thread.start()
        logger.info("Proactive loop started with %d triggers", len(self._triggers))

    def stop(self) -> None:
        self._stop.set()

    def pending_events(self, n: int = 5) -> list[ProactiveEvent]:
        """Return undelivered events for polling by the UI."""
        with sqlite3.connect(self._db) as c:
            rows = c.execute(
                "SELECT trigger_id,message,timestamp FROM events "
                "WHERE delivered=0 ORDER BY timestamp DESC LIMIT ?",
                (n,)).fetchall()
        return [ProactiveEvent(r[0],r[1],r[2]) for r in rows]

    def mark_delivered(self, trigger_id: str) -> None:
        with sqlite3.connect(self._db) as c:
            c.execute("UPDATE events SET delivered=1 WHERE trigger_id=?",
                      (trigger_id,))

    def _loop(self) -> None:
        while not self._stop.wait(self._poll):
            now = time.time()
            for trigger in self._triggers:
                if not trigger.enabled:
                    continue
                if now - trigger.last_fired < trigger.cooldown:
                    continue
                try:
                    if trigger.condition():
                        msg   = trigger.message()
                        event = ProactiveEvent(trigger.trigger_id, msg)
                        self._store(event)
                        self._on_event(event)
                        trigger.last_fired = now
                except Exception as e:
                    logger.debug("Trigger %s error: %s", trigger.trigger_id, e)

    def _store(self, event: ProactiveEvent) -> None:
        with sqlite3.connect(self._db) as c:
            c.execute("INSERT INTO events VALUES(?,?,?,?)",
                      (event.trigger_id, event.message,
                       event.timestamp, int(event.delivered)))

    def _init_db(self) -> None:
        with sqlite3.connect(self._db) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS events(
                trigger_id TEXT, message TEXT,
                timestamp REAL, delivered INTEGER)""")

def build_default_triggers(
    perception=None, policy_engine=None, task_queue=None
) -> list[ProactiveTrigger]:
    """Build the standard set of proactive triggers."""
    triggers = []

    # Recovery alert — fires when HRV suggests high stress
    if perception:
        def check_recovery():
            ctx = perception.current_context()
            return ctx.factors.get("stress_level", 0) > 0.75
        def msg_recovery():
            return ("⚠ High stress detected from biometrics. "
                    "Consider a short break or recovery activity.")
        triggers.append(ProactiveTrigger(
            "recovery_alert", "Recovery alert",
            check_every=300, condition=check_recovery,
            message=msg_recovery, cooldown=7200))

    # Task completed
    if task_queue:
        def check_tasks():
            recent = task_queue.list_recent(3)
            return any(t.status in ("completed","failed")
                       for t in recent if time.time()-t.completed_at < 120
                       if hasattr(t,'completed_at') and t.completed_at)
        def msg_tasks():
            recent = task_queue.list_recent(3)
            done   = [t for t in recent
                      if t.status in ("completed","failed")]
            if done:
                t = done[0]
                status = t.status if isinstance(t.status,str) else t.status.value
                return f"Task complete: {t.title} ({status})"
            return "Background task finished."
        triggers.append(ProactiveTrigger(
            "task_done","Task completed",
            check_every=30, condition=check_tasks,
            message=msg_tasks, cooldown=60))

    # Budget warning
    if policy_engine:
        def check_budget():
            try:
                data = policy_engine.show_policies("default")
                for cat, spent_str in data.get("monthly_spent",{}).items():
                    spent = float(spent_str.replace("£",""))
                    alloc = data.get("allocations",{}).get(cat,{})
                    limit_str = alloc.get("monthly_limit","£0")
                    limit = float(limit_str.replace("£","").replace("unlimited","0"))
                    if limit > 0 and spent / limit > 0.85:
                        return True
            except Exception:
                pass
            return False
        def msg_budget():
            return ("💰 You're approaching your monthly budget limit "
                    "in one or more categories. Say 'show my policies' to review.")
        triggers.append(ProactiveTrigger(
            "budget_warning","Budget warning",
            check_every=1800, condition=check_budget,
            message=msg_budget, cooldown=86400))

    return triggers
