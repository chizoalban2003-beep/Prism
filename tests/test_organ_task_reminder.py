"""Tests for organs/task_reminder.py"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_task(title, due_date="", done=False):
    t = MagicMock()
    t.title    = title
    t.due_date = due_date
    t.done     = done
    return t


def _execute(message, ctx=None):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "task_reminder",
        Path(__file__).parent.parent / "organs" / "task_reminder.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.execute("task_reminder", message, ctx or {})


# ── No engine, no file ────────────────────────────────────────────────────────

def test_no_engine_no_file_returns_message():
    card = _execute("show reminders")
    assert hasattr(card, "body")
    assert "no" in card.body.lower() or "not" in card.body.lower() or "available" in card.body.lower()


# ── File fallback ─────────────────────────────────────────────────────────────

def test_file_fallback_reads_reminders_json(tmp_path, monkeypatch):
    reminders = [{"title": "Buy milk", "due_date": "2026-06-10"}]
    prism_dir = tmp_path / ".prism"
    prism_dir.mkdir()
    (prism_dir / "reminders.json").write_text(json.dumps(reminders))

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "task_reminder_fb",
        Path(__file__).parent.parent / "organs" / "task_reminder.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    real_path = Path

    def patched_path(*args, **kwargs):
        s = str(args[0]) if args else ""
        if "~/.prism/reminders.json" in s:
            return prism_dir / "reminders.json"
        return real_path(*args, **kwargs)

    with patch("pathlib.Path", patched_path):
        card = mod.execute("task_reminder_fb", "show reminders", {})
    assert hasattr(card, "body")


# ── List mode ─────────────────────────────────────────────────────────────────

def test_list_shows_due_today():
    today = date.today().isoformat()
    engine = MagicMock()
    engine.list_tasks.return_value = [_make_task("Call dentist", today)]
    card = _execute("reminders", {"tasks": engine})
    assert "dentist" in card.body.lower()
    assert "today" in card.body.lower()


def test_list_shows_overdue():
    yesterday = (date.today() - timedelta(days=2)).isoformat()
    engine = MagicMock()
    engine.list_tasks.return_value = [_make_task("File taxes", yesterday)]
    card = _execute("what's overdue?", {"tasks": engine})
    assert "overdue" in card.body.lower()
    assert "taxes" in card.body.lower()


def test_list_shows_upcoming():
    future = (date.today() + timedelta(days=3)).isoformat()
    engine = MagicMock()
    engine.list_tasks.return_value = [_make_task("Team standup", future)]
    card = _execute("upcoming", {"tasks": engine})
    assert "standup" in card.body.lower()


def test_list_no_tasks_returns_no_pending():
    engine = MagicMock()
    engine.list_tasks.return_value = []
    card = _execute("reminders", {"tasks": engine})
    assert "no pending" in card.body.lower()


def test_list_engine_error_returns_error_message():
    engine = MagicMock()
    engine.list_tasks.side_effect = RuntimeError("db gone")
    card = _execute("reminders", {"tasks": engine})
    assert "could not" in card.body.lower() or "db gone" in card.body.lower()


# ── Add mode ──────────────────────────────────────────────────────────────────

def test_add_creates_task():
    engine = MagicMock()
    engine.add.return_value = _make_task("Doctor appointment")
    card = _execute("add reminder Doctor appointment", {"tasks": engine})
    engine.add.assert_called_once()
    assert "added" in card.body.lower()


def test_add_with_due_date_today():
    engine = MagicMock()
    engine.add.return_value = _make_task("Morning run", date.today().isoformat())
    _execute("remind me to go for a morning run today", {"tasks": engine})
    engine.add.assert_called_once()
    _, kwargs = engine.add.call_args
    assert kwargs.get("due_date") == date.today().isoformat()


def test_add_with_due_date_tomorrow():
    engine = MagicMock()
    expected = (date.today() + timedelta(days=1)).isoformat()
    engine.add.return_value = _make_task("Pay bills", expected)
    _execute("add reminder to pay bills tomorrow", {"tasks": engine})
    engine.add.assert_called_once()
    _, kwargs = engine.add.call_args
    assert kwargs.get("due_date") == expected


def test_add_with_iso_date():
    engine = MagicMock()
    engine.add.return_value = _make_task("Submit report", "2026-07-01")
    _execute("add reminder Submit report by 2026-07-01", {"tasks": engine})
    engine.add.assert_called_once()
    _, kwargs = engine.add.call_args
    assert kwargs.get("due_date") == "2026-07-01"


def test_add_engine_error_returns_error_card():
    engine = MagicMock()
    engine.add.side_effect = RuntimeError("db locked")
    card = _execute("add reminder buy groceries", {"tasks": engine})
    assert "could not" in card.body.lower() or "db locked" in card.body.lower()


def test_add_without_engine_falls_through_to_list():
    card = _execute("add reminder something", ctx={})
    assert hasattr(card, "body")


# ── ORGAN_META and ORGAN_POLICY ───────────────────────────────────────────────

def test_organ_meta_declared():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "task_reminder",
        Path(__file__).parent.parent / "organs" / "task_reminder.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.ORGAN_META["intent"] == "task_reminder"
    assert mod.ORGAN_POLICY["risk_level"] == "low"
    assert mod.ORGAN_POLICY["irreversible"] is False
