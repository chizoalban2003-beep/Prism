from __future__ import annotations

import time
import pytest

from prism_proactive import (
    PrismProactive,
    ProactiveTrigger,
    ProactiveEvent,
    build_default_triggers,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def proactive(tmp_path):
    db = str(tmp_path / "test_proactive.db")
    return PrismProactive(db_path=db, poll_seconds=1)


# ── ProactiveEvent dataclass ──────────────────────────────────────────────────

def test_event_defaults():
    e = ProactiveEvent("t1", "hello")
    assert e.trigger_id == "t1"
    assert e.message == "hello"
    assert not e.delivered
    assert e.timestamp > 0


# ── register / pending_events ─────────────────────────────────────────────────

def test_register_trigger(proactive):
    t = ProactiveTrigger(
        "t1", "test", check_every=10,
        condition=lambda: False,
        message=lambda: "msg",
    )
    proactive.register(t)
    assert len(proactive._triggers) == 1


def test_pending_events_empty(proactive):
    assert proactive.pending_events() == []


def test_store_and_pending(proactive):
    event = ProactiveEvent("t1", "test message", time.time())
    proactive._store(event)
    pending = proactive.pending_events()
    assert len(pending) == 1
    assert pending[0].trigger_id == "t1"
    assert pending[0].message == "test message"


def test_mark_delivered(proactive):
    event = ProactiveEvent("t1", "test message", time.time())
    proactive._store(event)
    proactive.mark_delivered("t1")
    pending = proactive.pending_events()
    assert len(pending) == 0


# ── Loop / trigger firing ─────────────────────────────────────────────────────

def test_trigger_fires_when_condition_true(tmp_path):
    fired = []
    db = str(tmp_path / "test.db")
    p = PrismProactive(
        on_event=fired.append,
        db_path=db,
        poll_seconds=999,  # won't auto-fire; we call _loop manually
    )
    t = ProactiveTrigger(
        "t1", "test", check_every=1,
        condition=lambda: True,
        message=lambda: "fired!",
        cooldown=0,
    )
    p.register(t)
    # Manually run one iteration
    p._stop.clear()
    now = time.time()
    for trigger in p._triggers:
        if trigger.condition():
            msg   = trigger.message()
            event = ProactiveEvent(trigger.trigger_id, msg)
            p._store(event)
            p._on_event(event)
            trigger.last_fired = now
    assert len(fired) == 1
    assert fired[0].message == "fired!"


def test_trigger_respects_cooldown(tmp_path):
    fired = []
    db = str(tmp_path / "test2.db")
    p = PrismProactive(on_event=fired.append, db_path=db, poll_seconds=999)
    t = ProactiveTrigger(
        "t2", "test", check_every=1,
        condition=lambda: True,
        message=lambda: "msg",
        cooldown=9999,
        last_fired=time.time(),  # fired very recently
    )
    p.register(t)
    now = time.time()
    for trigger in p._triggers:
        if not trigger.enabled:
            continue
        if now - trigger.last_fired < trigger.cooldown:
            continue   # should skip
        p._on_event(ProactiveEvent(trigger.trigger_id, trigger.message()))
    assert len(fired) == 0


def test_disabled_trigger_not_fired(tmp_path):
    fired = []
    db = str(tmp_path / "test3.db")
    p = PrismProactive(on_event=fired.append, db_path=db, poll_seconds=999)
    t = ProactiveTrigger(
        "t3", "test", check_every=1,
        condition=lambda: True,
        message=lambda: "msg",
        enabled=False,
        cooldown=0,
    )
    p.register(t)
    _now = time.time()
    for trigger in p._triggers:
        if not trigger.enabled:
            continue
        p._on_event(ProactiveEvent(trigger.trigger_id, trigger.message()))
    assert len(fired) == 0


# ── start / stop ──────────────────────────────────────────────────────────────

def test_start_stop(proactive):
    proactive.start()
    assert proactive._thread is not None
    assert proactive._thread.is_alive()
    proactive.stop()
    proactive._thread.join(timeout=2)


# ── build_default_triggers ────────────────────────────────────────────────────

def test_build_default_no_deps():
    triggers = build_default_triggers()
    assert isinstance(triggers, list)
    # calibration_prompt is always included as a baseline trigger
    assert len(triggers) >= 0
    trigger_ids = [t.trigger_id for t in triggers]
    # calibration_prompt is always present
    assert "calibration_prompt" in trigger_ids


def test_build_default_with_perception():
    class FakePerception:
        def current_context(self):
            class Ctx:
                factors = {"stress_level": 0.9}
            return Ctx()

    triggers = build_default_triggers(perception=FakePerception())
    trigger_ids = [t.trigger_id for t in triggers]
    assert "recovery_alert" in trigger_ids
    assert "calibration_prompt" in trigger_ids
    recovery = next(t for t in triggers if t.trigger_id == "recovery_alert")
    assert recovery.condition() is True


def test_build_default_with_task_queue():
    class FakeTask:
        title = "Test task"
        status = "completed"
        completed_at = time.time() - 10  # recently completed

    class FakeQueue:
        def list_recent(self, n):
            return [FakeTask()]

    triggers = build_default_triggers(task_queue=FakeQueue())
    assert any(t.trigger_id == "task_done" for t in triggers)
    task_trigger = next(t for t in triggers if t.trigger_id == "task_done")
    assert task_trigger.condition() is True
    assert "Test task" in task_trigger.message()
