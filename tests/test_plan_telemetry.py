"""Tests for prism_plan_telemetry — per-step DailyPlan execution log."""
from __future__ import annotations

import sqlite3

import pytest

from prism_plan_telemetry import (
    STATUS_ABANDONED,
    STATUS_DONE,
    STATUS_IN_PROGRESS,
    STATUS_PENDING,
    STATUS_SKIPPED,
    PlanTelemetry,
)


@pytest.fixture
def telemetry(tmp_path):
    return PlanTelemetry(db_path=str(tmp_path / "pt.db"))


def _plan(*titles):
    from sports_pro import DailyPlan, DailyTask
    return DailyPlan(
        primary_focus = "Fitness building",
        activation    = 0.6,
        fulcrum       = 0.5,
        tasks         = [DailyTask("07:00", 30, "warmup", t, "") for t in titles],
        rationale     = "test plan",
    )


def test_record_plan_creates_rows(telemetry):
    pid = telemetry.record_plan(_plan("run", "lift"), request="plan my day")
    p = telemetry.get_plan(pid)
    assert p is not None
    assert p["request"] == "plan my day"
    assert p["primary_focus"] == "Fitness building"
    assert len(p["steps"]) == 2
    assert all(s["status"] == STATUS_PENDING for s in p["steps"])


def test_mark_step_terminal_stamps_completed_at(telemetry):
    pid = telemetry.record_plan(_plan("run"))
    assert telemetry.mark_step(pid, 0, STATUS_DONE) is True
    p = telemetry.get_plan(pid)
    step = p["steps"][0]
    assert step["status"] == STATUS_DONE
    assert step["started_at"] is not None
    assert step["completed_at"] is not None


def test_mark_step_in_progress_only_stamps_started(telemetry):
    pid = telemetry.record_plan(_plan("run"))
    telemetry.mark_step(pid, 0, STATUS_IN_PROGRESS)
    step = telemetry.get_plan(pid)["steps"][0]
    assert step["started_at"] is not None
    assert step["completed_at"] is None


def test_mark_step_invalid_status_raises(telemetry):
    pid = telemetry.record_plan(_plan("run"))
    with pytest.raises(ValueError):
        telemetry.mark_step(pid, 0, "bogus")


def test_mark_unknown_step_returns_false(telemetry):
    pid = telemetry.record_plan(_plan("run"))
    assert telemetry.mark_step(pid, 99, STATUS_DONE) is False


def test_completion_stats_counts(telemetry):
    pid = telemetry.record_plan(_plan("a", "b", "c", "d"))
    telemetry.mark_step(pid, 0, STATUS_DONE)
    telemetry.mark_step(pid, 1, STATUS_DONE)
    telemetry.mark_step(pid, 2, STATUS_ABANDONED)
    stats = telemetry.completion_stats(pid)
    assert stats["total"] == 4
    assert stats[STATUS_DONE] == 2
    assert stats[STATUS_ABANDONED] == 1
    assert stats[STATUS_PENDING] == 1
    assert stats["completion_rate"] == 0.5


def test_supersede_chains_old_to_new(telemetry):
    old_id = telemetry.record_plan(_plan("run"))
    new_id = telemetry.record_plan(_plan("walk"))
    assert telemetry.supersede(old_id, new_id) is True
    assert telemetry.get_plan(old_id)["superseded_by"] == new_id
    assert telemetry.supersede(old_id, new_id) is False  # already superseded


def test_latest_plan_skips_superseded(telemetry):
    old_id = telemetry.record_plan(_plan("a"))
    new_id = telemetry.record_plan(_plan("b"))
    telemetry.supersede(old_id, new_id)
    assert telemetry.latest_plan()["plan_id"] == new_id


def test_latest_plan_none_when_empty(telemetry):
    assert telemetry.latest_plan() is None


def test_telemetry_summary_mentions_done_and_stalled(telemetry):
    pid = telemetry.record_plan(_plan("Morning run", "Evening lift", "Stretching"))
    telemetry.mark_step(pid, 0, STATUS_DONE)
    telemetry.mark_step(pid, 1, STATUS_ABANDONED)
    telemetry.mark_step(pid, 2, STATUS_SKIPPED)
    summary = telemetry.telemetry_summary(pid)
    assert "Morning run" in summary
    assert "Evening lift" in summary
    assert "Stretching" in summary
    assert "Completed" in summary
    assert "Stalled" in summary
    assert "Skipped" in summary


def test_outcome_record_id_stored(telemetry):
    pid = telemetry.record_plan(_plan("run"))
    telemetry.mark_step(pid, 0, STATUS_DONE, outcome_record_id="abc12345")
    step = telemetry.get_plan(pid)["steps"][0]
    assert step["outcome_record_id"] == "abc12345"


def test_persistence_across_instances(tmp_path):
    db = str(tmp_path / "shared.db")
    pid = PlanTelemetry(db_path=db).record_plan(_plan("run"))
    fresh = PlanTelemetry(db_path=db)
    assert fresh.get_plan(pid) is not None


def test_schema_has_expected_columns(telemetry):
    with sqlite3.connect(telemetry._db) as con:
        plans_cols = {r[1] for r in con.execute("PRAGMA table_info(plans)")}
        steps_cols = {r[1] for r in con.execute("PRAGMA table_info(plan_steps)")}
    assert {"plan_id", "request", "primary_focus", "rationale", "created_at",
            "superseded_by"} <= plans_cols
    assert {"plan_id", "step_index", "title", "time_slot", "duration_min",
            "category", "notes", "status", "outcome_record_id",
            "started_at", "completed_at"} <= steps_cols
