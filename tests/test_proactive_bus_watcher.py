"""Tests for prism_proactive_bus_watcher.ProactiveBusWatcher"""
from __future__ import annotations

from unittest.mock import MagicMock

from prism_proactive_bus_watcher import (
    _COOLDOWNS,
    WATCHED_SIGNALS,
    ProactiveBusWatcher,
)


def _make_watcher():
    proactive = MagicMock()
    bus       = MagicMock()
    return ProactiveBusWatcher(proactive=proactive, organ_bus=bus), proactive, bus


# ── register() ───────────────────────────────────────────────────────────────

def test_register_calls_bus_register():
    watcher, _, bus = _make_watcher()
    watcher.register()
    bus.register.assert_called_once()
    args, kwargs = bus.register.call_args
    assert set(kwargs.get("signal_types", [])) == WATCHED_SIGNALS or \
           set(bus.register.call_args[1].get("signal_types", [])) == WATCHED_SIGNALS


def test_register_uses_watcher_as_organ_name():
    watcher, _, bus = _make_watcher()
    watcher.register()
    kwargs = bus.register.call_args[1]
    assert kwargs.get("organ_name") == "proactive_watcher"


# ── _handle() ────────────────────────────────────────────────────────────────

def test_handle_health_alert_schedules_notification():
    watcher, proactive, _ = _make_watcher()
    # Clear cooldown state
    _COOLDOWNS.clear()
    watcher._handle({
        "signal_type": "health_alert",
        "source":      "health_summary",
        "message":     "HRV dropped 25% overnight",
    })
    proactive.schedule_in.assert_called_once()
    _, kwargs = proactive.schedule_in.call_args
    assert kwargs.get("seconds") == 5


def test_handle_finance_alert_schedules_notification():
    watcher, proactive, _ = _make_watcher()
    _COOLDOWNS.clear()
    watcher._handle({
        "signal_type": "finance_alert",
        "source":      "finance_summary",
        "message":     "Budget 90% used for dining",
    })
    proactive.schedule_in.assert_called_once()


def test_handle_goal_triggered():
    watcher, proactive, _ = _make_watcher()
    _COOLDOWNS.clear()
    watcher._handle({
        "signal_type": "goal_triggered",
        "source":      "horizon",
        "message":     "Flight price dropped below $300",
    })
    proactive.schedule_in.assert_called_once()


def test_handle_cooldown_suppresses_repeat():
    watcher, proactive, _ = _make_watcher()
    _COOLDOWNS.clear()
    watcher._handle({"signal_type": "health_alert", "source": "x", "message": "first"})
    watcher._handle({"signal_type": "health_alert", "source": "x", "message": "second"})
    # Only the first should schedule
    assert proactive.schedule_in.call_count == 1


def test_handle_different_signal_types_not_suppressed():
    watcher, proactive, _ = _make_watcher()
    _COOLDOWNS.clear()
    watcher._handle({"signal_type": "health_alert",   "source": "x", "message": "a"})
    watcher._handle({"signal_type": "finance_alert",  "source": "x", "message": "b"})
    assert proactive.schedule_in.call_count == 2


def test_handle_proactive_error_does_not_raise():
    watcher, proactive, _ = _make_watcher()
    _COOLDOWNS.clear()
    proactive.schedule_in.side_effect = RuntimeError("push unavailable")
    watcher._handle({"signal_type": "task_completed", "source": "x", "message": "done"})
    # Must not propagate the error


# ── _format_notification() ────────────────────────────────────────────────────

def test_format_health_alert():
    watcher, _, _ = _make_watcher()
    n = watcher._format_notification("health_alert", "health_summary", "HRV low", {})
    assert "health" in n.lower() or "hrv" in n.lower()


def test_format_finance_alert():
    watcher, _, _ = _make_watcher()
    n = watcher._format_notification("finance_alert", "finance_summary", "Budget exceeded", {})
    assert "finance" in n.lower() or "budget" in n.lower()


def test_format_unknown_signal_type():
    watcher, _, _ = _make_watcher()
    n = watcher._format_notification("unknown_type", "source", "some message", {})
    assert "some message" in n


def test_format_truncates_long_message():
    watcher, _, _ = _make_watcher()
    long_msg = "x" * 500
    n = watcher._format_notification("health_alert", "src", long_msg, {})
    assert len(n) < 500


# ── WATCHED_SIGNALS ───────────────────────────────────────────────────────────

def test_watched_signals_contains_expected():
    expected = {"health_alert", "finance_alert", "goal_triggered", "task_completed"}
    assert expected.issubset(WATCHED_SIGNALS)
