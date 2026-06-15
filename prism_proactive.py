from __future__ import annotations

import logging
import sqlite3
import threading
import time
import uuid as _uuid_mod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class ScheduledTrigger:
    """One-shot trigger that fires at a specific datetime."""
    trigger_id:  str
    name:        str
    fire_at:     float        # Unix timestamp
    message:     str          # static message (no callable needed)
    fired:       bool = False


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
        on_event: Optional[Callable[[ProactiveEvent], None]] = None,
        db_path:     str = "~/.prism/proactive.db",
        poll_seconds:int = 60,
    ):
        self._on_event    = on_event or (lambda e: None)
        self._push = None  # set by PrismAgent after init (type: PrismPush | None)
        self._db          = Path(db_path).expanduser()
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._poll        = poll_seconds
        self._triggers:   list[ProactiveTrigger] = []
        self._scheduled:  list[ScheduledTrigger] = []
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
        with sqlite3.connect(self._db, timeout=30.0) as c:
            rows = c.execute(
                "SELECT trigger_id,message,timestamp FROM events "
                "WHERE delivered=0 ORDER BY timestamp DESC LIMIT ?",
                (n,)).fetchall()
        return [ProactiveEvent(r[0],r[1],r[2]) for r in rows]

    def mark_delivered(self, trigger_id: str) -> None:
        with sqlite3.connect(self._db, timeout=30.0) as c:
            c.execute("UPDATE events SET delivered=1 WHERE trigger_id=?",
                      (trigger_id,))

    def schedule(self, message: str, fire_at: float,
                  trigger_id: Optional[str] = None) -> str:
        """
        Schedule a one-shot reminder.
        fire_at: Unix timestamp (use time.time() + seconds for relative).
        Returns trigger_id.
        """
        tid = trigger_id or str(_uuid_mod.uuid4())[:8]
        self._scheduled.append(ScheduledTrigger(
            trigger_id = tid,
            name       = f"Reminder: {message[:40]}",
            fire_at    = fire_at,
            message    = message,
        ))
        return tid

    def schedule_in(self, message: str, seconds: float) -> str:
        """Schedule a reminder N seconds from now."""
        return self.schedule(message, time.time() + seconds)

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
                        if getattr(self, '_push', None) and self._push.configured:
                            self._push.alert(event.message)
                        trigger.last_fired = now
                except Exception as e:
                    logger.debug("Trigger %s error: %s", trigger.trigger_id, e)
            # Check one-shot scheduled reminders
            for st in self._scheduled:
                if not st.fired and time.time() >= st.fire_at:
                    st.fired = True
                    event = ProactiveEvent(st.trigger_id, st.message)
                    self._store(event)
                    self._on_event(event)
                    if getattr(self, '_push', None) and self._push.configured:
                        self._push.alert(st.message)

    def _store(self, event: ProactiveEvent) -> None:
        with sqlite3.connect(self._db, timeout=30.0) as c:
            c.execute("INSERT INTO events VALUES(?,?,?,?)",
                      (event.trigger_id, event.message,
                       event.timestamp, int(event.delivered)))

    def _init_db(self) -> None:
        with sqlite3.connect(self._db, timeout=30.0) as c:
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

    # Wearable sync trigger — fires when a device agent has new data
    if perception:
        def check_wearable():
            try:
                ctx = perception.current_context()
                # Fires if wearable data freshness flag is set
                return ctx.factors.get("wearable_new_data", 0) > 0.5
            except Exception:
                return False
        def msg_wearable():
            return ("New wearable data available. "
                    "Say 'sync wearables' or 'show my recovery' to analyse it.")
        triggers.append(ProactiveTrigger(
            "wearable_sync", "Wearable data available",
            check_every=300, condition=check_wearable,
            message=msg_wearable, cooldown=3600))

    def check_calibration():
        from prism_calibration import PrismCalibration
        cal    = PrismCalibration()
        events = cal.history(n=1)
        if not events:
            return True
        return (time.time() - events[0].timestamp) > 86400 * 3

    def msg_calibration():
        return ("How are my recent recommendations feeling? "
                "Say 'that was too aggressive' or 'good call' "
                "to help me learn your preferences.")

    triggers.append(ProactiveTrigger(
        "calibration_prompt", "Calibration check",
        check_every=3600,
        condition=check_calibration,
        message=msg_calibration,
        cooldown=86400 * 3,
        enabled=True,
    ))

    return triggers


# ── Advanced / organ-driven triggers ──────────────────────────────────────────

def _run_organ(organ_loader, intent: str, message: str, ctx: dict) -> str:
    """Execute an organ and return its body string, or '' on failure."""
    try:
        fn = organ_loader.get(intent) if organ_loader else None
        if fn is None:
            return ""
        card = fn(intent, message, ctx)
        return card.body if hasattr(card, "body") else str(card)
    except Exception as exc:
        logger.debug("Proactive organ %s failed: %s", intent, exc)
        return ""


def build_advanced_triggers(
    organ_loader=None,
    router=None,
    calendar=None,
    persona=None,
    horizon=None,
    config: Optional[dict] = None,
) -> list[ProactiveTrigger]:
    """
    Build organ-driven and system-aware proactive triggers.
    Call this after PrismAgent finishes initialising all dependencies.

    Triggers added:
      reminder_fire      — polls ~/.prism/reminders.json; fires overdue reminders
      morning_brief      — at wake hour; runs weather + news, composes LLM brief
      calendar_warning   — 15 min before a calendar event
      disk_space         — warns when disk usage > 90 %
      horizon_deadline   — warns 48 h before a HorizonGoal expires
      evening_summary    — at end-of-day; summarises tasks + tomorrow
    """
    cfg = config or {}
    triggers: list[ProactiveTrigger] = []

    # ── 1. Reminder poller ────────────────────────────────────────────────────
    # Closes the loop from the reminder_set organ: reads ~/.prism/reminders.json
    # and fires any reminder whose fire_at has passed.
    def _check_reminders() -> bool:
        import datetime
        import json
        from pathlib import Path
        f = Path("~/.prism/reminders.json").expanduser()
        if not f.exists():
            return False
        try:
            items = json.loads(f.read_text(encoding="utf-8"))
            now = datetime.datetime.now()
            return any(
                item.get("status") == "pending"
                and datetime.datetime.fromisoformat(item["fire_at"]) <= now
                for item in items
            )
        except Exception:
            return False

    def _msg_reminders() -> str:
        import datetime
        import json
        from pathlib import Path
        f = Path("~/.prism/reminders.json").expanduser()
        try:
            items = json.loads(f.read_text(encoding="utf-8"))
            now = datetime.datetime.now()
            due = [
                item for item in items
                if item.get("status") == "pending"
                and datetime.datetime.fromisoformat(item["fire_at"]) <= now
            ]
            for item in due:
                item["status"] = "fired"
            f.write_text(json.dumps(items, indent=2), encoding="utf-8")
            if not due:
                return "Reminder due."
            return "\n".join(f"\u23f0 Reminder: {d['text']}" for d in due[:5])
        except Exception:
            return "Reminder due."

    triggers.append(ProactiveTrigger(
        "reminder_fire", "Reminder due",
        check_every=30,
        condition=_check_reminders,
        message=_msg_reminders,
        cooldown=1,        # can fire multiple times per minute for different reminders
        enabled=True,
    ))

    # ── 2. Morning brief ──────────────────────────────────────────────────────
    # Fires once per day at the user's wake hour (persona-aware).
    # Runs weather_check + news_headlines organs and composes an LLM brief.
    if organ_loader:
        _morning_hour_cache: dict = {}

        def _morning_hour() -> int:
            """Return the wake hour from persona peak hours or default (7)."""
            if persona:
                try:
                    peaks = persona.peak_hours()
                    if peaks:
                        return min(int(h) for h in peaks)
                except Exception:
                    pass
            return int(cfg.get("morning_hour", 7))

        def _check_morning() -> bool:
            import datetime
            now = datetime.datetime.now()
            today = now.strftime("%Y-%m-%d")
            if _morning_hour_cache.get("last_date") == today:
                return False
            return now.hour == _morning_hour()

        def _msg_morning() -> str:
            import datetime
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            _morning_hour_cache["last_date"] = today

            ctx: dict = {}
            parts: list[str] = []

            weather = _run_organ(organ_loader, "weather_check", "weather today", ctx)
            if weather:
                parts.append(weather)

            news = _run_organ(organ_loader, "news_headlines", "top headlines", ctx)
            if news:
                parts.append(news)

            cal_text = ""
            if calendar:
                try:
                    events = calendar.events_today() or []
                    if events:
                        titles = []
                        for e in events[:5]:
                            t = (getattr(e, "title", None)
                                 or (e.get("title") if hasattr(e, "get") else None)
                                 or "Event")
                            titles.append(str(t))
                        cal_text = "Today's calendar: " + ", ".join(titles)
                        parts.append(cal_text)
                except Exception:
                    pass

            if not parts:
                return "\u2600 Good morning! Have a great day."

            combined = "\n\n".join(parts)
            if router:
                try:
                    prompt = (
                        "Write a concise, friendly good-morning briefing for the user "
                        "based on this information. 3 sentences max. Be direct.\n\n"
                        f"{combined[:2000]}"
                    )
                    answer, _ = router.call(prompt)
                    return f"\u2600 Good morning!\n\n{answer.strip()}"
                except Exception:
                    pass
            return f"\u2600 Good morning!\n\n{combined[:600]}"

        triggers.append(ProactiveTrigger(
            "morning_brief", "Morning briefing",
            check_every=60,
            condition=_check_morning,
            message=_msg_morning,
            cooldown=86400,
            enabled=True,
        ))

    # ── 3. Calendar 15-minute warning ─────────────────────────────────────────
    if calendar:
        _warned_events: set = set()

        def _check_cal_warning() -> bool:
            import datetime
            try:
                events = calendar.events_today() or []
                now = datetime.datetime.now()
                for evt in events:
                    start = (getattr(evt, "start_dt", None)
                             or getattr(evt, "start", None)
                             or (evt.get("start_dt") or evt.get("start")
                                 if hasattr(evt, "get") else None))
                    if start is None:
                        continue
                    if isinstance(start, str):
                        try:
                            start = datetime.datetime.fromisoformat(start)
                        except Exception:
                            continue
                    delta = (start - now).total_seconds()
                    evt_id = str(getattr(evt, "uid", None) or id(evt))
                    if 0 < delta <= 900 and evt_id not in _warned_events:
                        return True
            except Exception:
                pass
            return False

        def _msg_cal_warning() -> str:
            import datetime
            try:
                events = calendar.events_today() or []
                now = datetime.datetime.now()
                msgs = []
                for evt in events:
                    start = (getattr(evt, "start_dt", None)
                             or getattr(evt, "start", None)
                             or (evt.get("start_dt") or evt.get("start")
                                 if hasattr(evt, "get") else None))
                    if start is None:
                        continue
                    if isinstance(start, str):
                        try:
                            start = datetime.datetime.fromisoformat(start)
                        except Exception:
                            continue
                    delta = (start - now).total_seconds()
                    evt_id = str(getattr(evt, "uid", None) or id(evt))
                    if 0 < delta <= 900 and evt_id not in _warned_events:
                        title = (getattr(evt, "title", None)
                                 or (evt.get("title") if hasattr(evt, "get") else None)
                                 or "Meeting")
                        mins = int(delta // 60) + 1
                        msgs.append(f"\U0001f4c5 {title} starts in {mins} minute(s)")
                        _warned_events.add(evt_id)
                return "\n".join(msgs) if msgs else "Meeting starting soon."
            except Exception:
                return "Meeting starting soon."

        triggers.append(ProactiveTrigger(
            "calendar_warning", "Meeting soon",
            check_every=60,
            condition=_check_cal_warning,
            message=_msg_cal_warning,
            cooldown=60,    # can fire for multiple events with short gap
            enabled=True,
        ))

    # ── 4. Disk space warning ─────────────────────────────────────────────────
    def _check_disk() -> bool:
        try:
            import psutil
            return psutil.disk_usage("/").percent > 90
        except Exception:
            return False

    def _msg_disk() -> str:
        try:
            import psutil
            usage = psutil.disk_usage("/")
            return (
                f"\U0001f4be Disk space low: {usage.percent:.0f}% used "
                f"({usage.free // (1024**3)} GB free). "
                "Consider clearing old files."
            )
        except Exception:
            return "\U0001f4be Disk space is running low."

    triggers.append(ProactiveTrigger(
        "disk_space", "Disk space warning",
        check_every=3600,
        condition=_check_disk,
        message=_msg_disk,
        cooldown=86400,
        enabled=True,
    ))

    # ── 5. Horizon goal deadline warning ─────────────────────────────────────
    if horizon:
        def _check_horizon_deadline() -> bool:
            import datetime
            try:
                goals = horizon.list_goals(status="watching") or []
                now = datetime.datetime.now()
                for g in goals:
                    exp = getattr(g, "expires_at", None)
                    if exp and (exp - now).total_seconds() < 172800:  # 48h
                        return True
            except Exception:
                pass
            return False

        def _msg_horizon_deadline() -> str:
            import datetime
            try:
                goals = horizon.list_goals(status="watching") or []
                now = datetime.datetime.now()
                msgs = []
                for g in goals:
                    exp = getattr(g, "expires_at", None)
                    if exp and (exp - now).total_seconds() < 172800:
                        hours = int((exp - now).total_seconds() // 3600)
                        intent = getattr(g, "intent", "goal")
                        msgs.append(f"\U0001f3af Goal deadline in {hours}h: \"{intent}\"")
                return "\n".join(msgs) if msgs else "A horizon goal is expiring soon."
            except Exception:
                return "A horizon goal is expiring soon."

        triggers.append(ProactiveTrigger(
            "horizon_deadline", "Goal deadline approaching",
            check_every=1800,
            condition=_check_horizon_deadline,
            message=_msg_horizon_deadline,
            cooldown=43200,
            enabled=True,
        ))

    # ── 6. Evening summary ───────────────────────────────────────────────────
    # Fires once per day at the configured evening hour (default 18:00).
    # Summarises outcomes + what's on the calendar tomorrow.
    _evening_cache: dict = {}
    _evening_hour = int(cfg.get("evening_hour", 18))

    def _check_evening() -> bool:
        import datetime
        now = datetime.datetime.now()
        today = now.strftime("%Y-%m-%d")
        if _evening_cache.get("last_date") == today:
            return False
        return now.hour == _evening_hour

    def _msg_evening() -> str:
        import datetime
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        _evening_cache["last_date"] = today

        parts: list[str] = []
        if calendar:
            try:
                tomorrow = (datetime.datetime.now()
                            + datetime.timedelta(days=1))
                events = calendar.events_on(tomorrow) or []
                if events:
                    titles = []
                    for e in events[:5]:
                        t = (getattr(e, "title", None)
                             or (e.get("title") if hasattr(e, "get") else None)
                             or "Event")
                        titles.append(str(t))
                    parts.append("Tomorrow: " + ", ".join(titles))
            except Exception:
                pass

        if not parts and not router:
            return "\U0001f307 Good evening! Day complete."

        if router and parts:
            try:
                prompt = (
                    "Write a brief, warm evening wrap-up for the user in 2 sentences. "
                    "Mention what's coming up tomorrow if relevant.\n\n"
                    + "\n".join(parts)
                )
                answer, _ = router.call(prompt)
                return f"\U0001f307 {answer.strip()}"
            except Exception:
                pass

        return "\U0001f307 Good evening!\n\n" + "\n".join(parts)

    triggers.append(ProactiveTrigger(
        "evening_summary", "Evening summary",
        check_every=60,
        condition=_check_evening,
        message=_msg_evening,
        cooldown=86400,
        enabled=True,
    ))

    return triggers
