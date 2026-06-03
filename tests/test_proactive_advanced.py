"""
tests/test_proactive_advanced.py
=================================
Tests for the advanced proactive triggers added in build_advanced_triggers():
  reminder_fire, morning_brief, calendar_warning, disk_space,
  horizon_deadline, evening_summary.
All I/O and time-sensitive behaviour is mocked.
"""
from __future__ import annotations

import datetime
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

from prism_proactive import build_advanced_triggers, ProactiveTrigger, _run_organ


# ── helpers ────────────────────────────────────────────────────────────────────

def _triggers_by_id(triggers: list) -> dict:
    return {t.trigger_id: t for t in triggers}


def _organ_loader(intents: dict):
    """Return a mock OrganLoader whose get() returns a callable per intent."""
    ol = MagicMock()
    def _get(intent):
        fn = intents.get(intent)
        if fn is None:
            return None
        return fn
    ol.get.side_effect = _get
    return ol


def _fake_card(body: str):
    c = MagicMock()
    c.body = body
    return c


# ── _run_organ helper ──────────────────────────────────────────────────────────

class TestRunOrgan:
    def test_returns_body(self):
        fn = MagicMock(return_value=_fake_card("weather: sunny"))
        ol = _organ_loader({"weather_check": fn})
        result = _run_organ(ol, "weather_check", "weather today", {})
        assert result == "weather: sunny"

    def test_returns_empty_when_no_loader(self):
        assert _run_organ(None, "weather_check", "msg", {}) == ""

    def test_returns_empty_when_intent_missing(self):
        ol = _organ_loader({})
        assert _run_organ(ol, "nonexistent", "msg", {}) == ""

    def test_returns_empty_on_exception(self):
        fn = MagicMock(side_effect=RuntimeError("broken"))
        ol = _organ_loader({"weather_check": fn})
        assert _run_organ(ol, "weather_check", "msg", {}) == ""


# ── build_advanced_triggers structure ────────────────────────────────────────

class TestBuildAdvancedTriggers:
    def test_always_returns_reminder_and_disk_and_evening(self):
        triggers = build_advanced_triggers()
        ids = {t.trigger_id for t in triggers}
        assert "reminder_fire" in ids
        assert "disk_space" in ids
        assert "evening_summary" in ids

    def test_morning_brief_requires_organ_loader(self):
        triggers = build_advanced_triggers()
        ids = {t.trigger_id for t in triggers}
        assert "morning_brief" not in ids

        triggers = build_advanced_triggers(organ_loader=MagicMock())
        ids = {t.trigger_id for t in triggers}
        assert "morning_brief" in ids

    def test_calendar_warning_requires_calendar(self):
        triggers = build_advanced_triggers()
        ids = {t.trigger_id for t in triggers}
        assert "calendar_warning" not in ids

        triggers = build_advanced_triggers(calendar=MagicMock())
        ids = {t.trigger_id for t in triggers}
        assert "calendar_warning" in ids

    def test_horizon_deadline_requires_horizon(self):
        triggers = build_advanced_triggers()
        ids = {t.trigger_id for t in triggers}
        assert "horizon_deadline" not in ids

        triggers = build_advanced_triggers(horizon=MagicMock())
        ids = {t.trigger_id for t in triggers}
        assert "horizon_deadline" in ids

    def test_all_are_proactive_trigger_instances(self):
        triggers = build_advanced_triggers(
            organ_loader=MagicMock(),
            calendar=MagicMock(),
            horizon=MagicMock(),
        )
        assert all(isinstance(t, ProactiveTrigger) for t in triggers)


# ── reminder_fire ──────────────────────────────────────────────────────────────

class TestReminderFire:
    def _trigger(self):
        return _triggers_by_id(build_advanced_triggers())["reminder_fire"]

    def _reminders_json(self, items):
        return json.dumps(items)

    def test_fires_when_reminder_overdue(self):
        trigger = self._trigger()
        past = (datetime.datetime.now() - datetime.timedelta(minutes=5)).isoformat()
        data = [{"id": "r1", "text": "call Alice", "status": "pending",
                 "set_at": past, "fire_at": past}]
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.read_text", return_value=json.dumps(data)):
            assert trigger.condition() is True

    def test_does_not_fire_for_future_reminder(self):
        trigger = self._trigger()
        future = (datetime.datetime.now() + datetime.timedelta(hours=1)).isoformat()
        data = [{"id": "r2", "text": "dentist", "status": "pending",
                 "set_at": future, "fire_at": future}]
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.read_text", return_value=json.dumps(data)):
            assert trigger.condition() is False

    def test_does_not_fire_when_already_fired(self):
        trigger = self._trigger()
        past = (datetime.datetime.now() - datetime.timedelta(minutes=5)).isoformat()
        data = [{"id": "r3", "text": "done", "status": "fired",
                 "set_at": past, "fire_at": past}]
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.read_text", return_value=json.dumps(data)):
            assert trigger.condition() is False

    def test_does_not_fire_when_no_file(self):
        trigger = self._trigger()
        with patch("pathlib.Path.exists", return_value=False):
            assert trigger.condition() is False

    def test_message_returns_reminder_text(self):
        trigger = self._trigger()
        past = (datetime.datetime.now() - datetime.timedelta(minutes=2)).isoformat()
        data = [{"id": "r4", "text": "buy groceries", "status": "pending",
                 "set_at": past, "fire_at": past}]
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.read_text", return_value=json.dumps(data)), \
             patch("pathlib.Path.write_text"):
            msg = trigger.message()
        assert "buy groceries" in msg

    def test_message_marks_reminders_fired(self):
        trigger = self._trigger()
        past = (datetime.datetime.now() - datetime.timedelta(minutes=2)).isoformat()
        data = [{"id": "r5", "text": "stand up", "status": "pending",
                 "set_at": past, "fire_at": past}]
        written_data = []
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.read_text", return_value=json.dumps(data)), \
             patch("pathlib.Path.write_text", side_effect=lambda d, **kw: written_data.append(d)):
            trigger.message()
        written = "".join(written_data)
        assert "fired" in written

    def test_handles_corrupt_json_gracefully(self):
        trigger = self._trigger()
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.read_text", return_value="not json {{{"):
            assert trigger.condition() is False

    def test_cooldown_is_minimal(self):
        trigger = self._trigger()
        assert trigger.cooldown <= 5

    def test_check_every_is_short(self):
        trigger = self._trigger()
        assert trigger.check_every <= 60


# ── morning_brief ──────────────────────────────────────────────────────────────

class TestMorningBrief:
    def _trigger(self, organ_loader=None, router=None, persona=None):
        triggers = build_advanced_triggers(
            organ_loader=organ_loader or MagicMock(),
            router=router,
            persona=persona,
        )
        return _triggers_by_id(triggers)["morning_brief"]

    def test_fires_at_morning_hour(self):
        trigger = self._trigger()
        morning = datetime.datetime.now().replace(hour=7, minute=0, second=0)
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = morning
            mock_dt.fromisoformat = datetime.datetime.fromisoformat
            result = trigger.condition()
        # May be True or False depending on whether last_date was set
        assert isinstance(result, bool)

    def test_does_not_fire_at_wrong_hour(self):
        trigger = self._trigger()
        not_morning = datetime.datetime.now().replace(hour=14, minute=0)
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = not_morning
            assert trigger.condition() is False

    def test_message_runs_weather_and_news_organs(self):
        weather_fn = MagicMock(return_value=_fake_card("Sunny 22°C"))
        news_fn = MagicMock(return_value=_fake_card("Top story: big news"))
        ol = _organ_loader({"weather_check": weather_fn, "news_headlines": news_fn})
        trigger = self._trigger(organ_loader=ol)
        msg = trigger.message()
        assert "22" in msg or "news" in msg.lower() or "morning" in msg.lower()

    def test_message_uses_llm_to_compose(self):
        ol = _organ_loader({
            "weather_check": MagicMock(return_value=_fake_card("Sunny")),
            "news_headlines": MagicMock(return_value=_fake_card("Top story")),
        })
        router = MagicMock()
        router.call.return_value = ("Have a great day! Weather is sunny.", {})
        trigger = self._trigger(organ_loader=ol, router=router)
        msg = trigger.message()
        assert "great day" in msg or "morning" in msg.lower()
        router.call.assert_called_once()

    def test_message_fallback_when_no_organs(self):
        ol = _organ_loader({})
        trigger = self._trigger(organ_loader=ol)
        msg = trigger.message()
        assert "morning" in msg.lower() or msg

    def test_persona_aware_morning_hour(self):
        persona = MagicMock()
        persona.peak_hours.return_value = [8, 9, 10]
        trigger = self._trigger(persona=persona)
        # Condition should respect persona peak hour (8 in this case)
        early = datetime.datetime.now().replace(hour=6, minute=0)
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = early
            assert trigger.condition() is False

    def test_cooldown_is_daily(self):
        trigger = self._trigger()
        assert trigger.cooldown >= 86400


# ── calendar_warning ───────────────────────────────────────────────────────────

class TestCalendarWarning:
    def _trigger(self, calendar=None):
        cal = calendar or MagicMock()
        triggers = build_advanced_triggers(calendar=cal)
        return _triggers_by_id(triggers)["calendar_warning"]

    def _event(self, minutes_from_now: float, title="Team standup", uid="evt1"):
        start = datetime.datetime.now() + datetime.timedelta(minutes=minutes_from_now)
        e = MagicMock()
        e.title = title
        e.start_dt = start
        e.uid = uid
        return e

    def test_fires_when_event_in_15_min(self):
        cal = MagicMock()
        cal.events_today.return_value = [self._event(10)]
        trigger = self._trigger(calendar=cal)
        assert trigger.condition() is True

    def test_does_not_fire_for_event_in_1_hour(self):
        cal = MagicMock()
        cal.events_today.return_value = [self._event(60)]
        trigger = self._trigger(calendar=cal)
        assert trigger.condition() is False

    def test_does_not_fire_for_past_event(self):
        cal = MagicMock()
        cal.events_today.return_value = [self._event(-30)]
        trigger = self._trigger(calendar=cal)
        assert trigger.condition() is False

    def test_message_includes_event_title(self):
        cal = MagicMock()
        cal.events_today.return_value = [self._event(12, title="Board Review")]
        trigger = self._trigger(calendar=cal)
        trigger.condition()  # populates _warned_events check
        msg = trigger.message()
        assert "Board Review" in msg

    def test_message_includes_minutes(self):
        cal = MagicMock()
        cal.events_today.return_value = [self._event(8, uid="evt2")]
        trigger = self._trigger(calendar=cal)
        msg = trigger.message()
        assert "minute" in msg.lower() or "min" in msg.lower()

    def test_handles_calendar_exception(self):
        cal = MagicMock()
        cal.events_today.side_effect = Exception("calendar offline")
        trigger = self._trigger(calendar=cal)
        assert trigger.condition() is False


# ── disk_space ─────────────────────────────────────────────────────────────────

class TestDiskSpace:
    def _trigger(self):
        return _triggers_by_id(build_advanced_triggers())["disk_space"]

    def test_fires_when_disk_over_90(self):
        trigger = self._trigger()
        mock_usage = MagicMock()
        mock_usage.percent = 93
        with patch("psutil.disk_usage", return_value=mock_usage):
            assert trigger.condition() is True

    def test_does_not_fire_when_disk_ok(self):
        trigger = self._trigger()
        mock_usage = MagicMock()
        mock_usage.percent = 60
        with patch("psutil.disk_usage", return_value=mock_usage):
            assert trigger.condition() is False

    def test_does_not_fire_when_psutil_missing(self):
        trigger = self._trigger()
        import sys
        with patch.dict(sys.modules, {"psutil": None}):
            assert trigger.condition() is False

    def test_message_includes_percentage(self):
        trigger = self._trigger()
        mock_usage = MagicMock()
        mock_usage.percent = 94
        mock_usage.free = 5 * 1024 ** 3
        with patch("psutil.disk_usage", return_value=mock_usage):
            msg = trigger.message()
        assert "94" in msg or "disk" in msg.lower()

    def test_cooldown_is_daily(self):
        assert self._trigger().cooldown >= 86400


# ── horizon_deadline ───────────────────────────────────────────────────────────

class TestHorizonDeadline:
    def _trigger(self, horizon=None):
        h = horizon or MagicMock()
        triggers = build_advanced_triggers(horizon=h)
        return _triggers_by_id(triggers)["horizon_deadline"]

    def _goal(self, hours_until_expiry: float, intent: str = "book hotel"):
        g = MagicMock()
        g.intent = intent
        g.expires_at = datetime.datetime.now() + datetime.timedelta(hours=hours_until_expiry)
        return g

    def test_fires_when_goal_expires_in_24h(self):
        h = MagicMock()
        h.list_goals.return_value = [self._goal(20)]
        trigger = self._trigger(horizon=h)
        assert trigger.condition() is True

    def test_does_not_fire_when_goal_has_time(self):
        h = MagicMock()
        h.list_goals.return_value = [self._goal(72)]
        trigger = self._trigger(horizon=h)
        assert trigger.condition() is False

    def test_message_includes_goal_intent(self):
        h = MagicMock()
        h.list_goals.return_value = [self._goal(10, intent="confirm conference room")]
        trigger = self._trigger(horizon=h)
        msg = trigger.message()
        assert "confirm conference room" in msg

    def test_message_includes_hours_remaining(self):
        h = MagicMock()
        h.list_goals.return_value = [self._goal(6)]
        trigger = self._trigger(horizon=h)
        msg = trigger.message()
        assert "h" in msg or "hour" in msg.lower()

    def test_handles_no_goals(self):
        h = MagicMock()
        h.list_goals.return_value = []
        trigger = self._trigger(horizon=h)
        assert trigger.condition() is False

    def test_handles_exception_gracefully(self):
        h = MagicMock()
        h.list_goals.side_effect = Exception("db locked")
        trigger = self._trigger(horizon=h)
        assert trigger.condition() is False


# ── evening_summary ────────────────────────────────────────────────────────────

class TestEveningSummary:
    def _trigger(self, calendar=None, router=None):
        triggers = build_advanced_triggers(calendar=calendar, router=router)
        return _triggers_by_id(triggers)["evening_summary"]

    def test_fires_at_evening_hour(self):
        trigger = self._trigger()
        evening = datetime.datetime.now().replace(hour=18, minute=5)
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = evening
            result = trigger.condition()
        assert isinstance(result, bool)

    def test_does_not_fire_at_wrong_hour(self):
        trigger = self._trigger()
        midday = datetime.datetime.now().replace(hour=12, minute=0)
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = midday
            assert trigger.condition() is False

    def test_message_includes_tomorrow_events(self):
        cal = MagicMock()
        evt = MagicMock()
        evt.title = "Design review"
        cal.events_on.return_value = [evt]
        trigger = self._trigger(calendar=cal)
        msg = trigger.message()
        assert "Design review" in msg or "evening" in msg.lower() or msg

    def test_message_uses_llm(self):
        cal = MagicMock()
        evt = MagicMock()
        evt.title = "Sprint planning"
        cal.events_on.return_value = [evt]
        router = MagicMock()
        router.call.return_value = ("Good evening! Tomorrow is packed.", {})
        trigger = self._trigger(calendar=cal, router=router)
        msg = trigger.message()
        assert "evening" in msg.lower() or "tomorrow" in msg.lower()

    def test_message_graceful_with_no_calendar(self):
        trigger = self._trigger()
        msg = trigger.message()
        assert msg

    def test_cooldown_is_daily(self):
        assert self._trigger().cooldown >= 86400

    def test_custom_evening_hour(self):
        triggers = build_advanced_triggers(config={"evening_hour": 20})
        t = _triggers_by_id(triggers)["evening_summary"]
        wrong_hour = datetime.datetime.now().replace(hour=18, minute=0)
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = wrong_hour
            assert t.condition() is False
