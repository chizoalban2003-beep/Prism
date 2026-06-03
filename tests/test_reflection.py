"""Tests for prism_reflection.PrismReflection"""
from __future__ import annotations

from unittest.mock import MagicMock

from prism_reflection import PrismReflection, ReflectionReport


def _reflection(**kwargs) -> PrismReflection:
    return PrismReflection(**kwargs)


# ── No dependencies ───────────────────────────────────────────────────────────

def test_run_no_deps_returns_report():
    r = _reflection()
    report = r.run()
    assert isinstance(report, ReflectionReport)


def test_run_no_llm_returns_heuristic_summary():
    tracker = MagicMock()
    tracker.stats.return_value = {
        "total": 10, "done": 8, "abandoned": 2,
        "user_corrected": 0, "completion_rate": 0.8,
        "avg_steps": 3.5, "avg_policy_flags": 0.2,
    }
    tracker.recent.return_value = []
    r = _reflection(outcome_tracker=tracker)
    report = r.run()
    assert report.summary != ""
    assert "10" in report.summary or "0.8" in report.summary or "8" in report.summary


def test_run_no_llm_includes_stale_goals():
    import time
    horizon = MagicMock()
    stale_goal = MagicMock()
    stale_goal.goal_id    = "abc123"
    stale_goal.intent     = "Old goal that never completed"
    stale_goal.created_at = time.time() - 20 * 86400
    horizon.list_goals.return_value = [stale_goal]
    r = _reflection(horizon=horizon)
    report = r.run()
    assert any(g["goal_id"] == "abc123" for g in report.unresolved_goals)


def test_run_no_llm_fresh_goal_not_stale():
    import time
    horizon = MagicMock()
    fresh_goal = MagicMock()
    fresh_goal.goal_id    = "fresh1"
    fresh_goal.intent     = "New goal just added"
    fresh_goal.created_at = time.time() - 2 * 86400
    horizon.list_goals.return_value = [fresh_goal]
    r = _reflection(horizon=horizon)
    report = r.run()
    assert not any(g["goal_id"] == "fresh1" for g in report.unresolved_goals)


# ── Heuristic patterns ────────────────────────────────────────────────────────

def test_heuristic_patterns_low_completion():
    tracker = MagicMock()
    tracker.stats.return_value = {
        "total": 10, "done": 4, "abandoned": 6, "user_corrected": 0,
        "completion_rate": 0.4, "avg_steps": 5, "avg_policy_flags": 0,
    }
    tracker.recent.return_value = []
    r = _reflection(outcome_tracker=tracker)
    report = r.run()
    assert any("completion" in p.lower() for p in report.patterns)


def test_heuristic_patterns_recurring_topic():
    tracker = MagicMock()
    tracker.stats.return_value = {
        "total": 4, "done": 4, "abandoned": 0, "user_corrected": 0,
        "completion_rate": 1.0, "avg_steps": 2, "avg_policy_flags": 0,
    }
    goals = []
    for _ in range(3):
        rec = MagicMock()
        rec.outcome = "done"
        rec.goal = "check health metrics daily"
        goals.append(rec)
    tracker.recent.return_value = goals
    r = _reflection(outcome_tracker=tracker)
    report = r.run()
    assert any("health" in p.lower() or "metrics" in p.lower() for p in report.patterns)


# ── With LLM ─────────────────────────────────────────────────────────────────

def test_run_with_llm_parses_response():
    import json
    tracker = MagicMock()
    tracker.stats.return_value = {
        "total": 5, "done": 5, "abandoned": 0, "user_corrected": 0,
        "completion_rate": 1.0, "avg_steps": 2, "avg_policy_flags": 0,
    }
    tracker.recent.return_value = []

    router = MagicMock()
    router.call.return_value = (json.dumps({
        "summary": "Good week — 5 chains all completed.",
        "patterns": ["User often asks about weather"],
        "belief_proposals": [],
        "flag_goal_ids": [],
    }), {})

    r = _reflection(outcome_tracker=tracker, llm_router=router)
    report = r.run()
    assert report.summary == "Good week — 5 chains all completed."
    assert "User often asks about weather" in report.patterns


def test_run_llm_failure_falls_back_gracefully():
    tracker = MagicMock()
    tracker.stats.return_value = {
        "total": 0, "done": 0, "abandoned": 0, "user_corrected": 0,
        "completion_rate": 0.0, "avg_steps": 0, "avg_policy_flags": 0,
    }
    tracker.recent.return_value = []

    router = MagicMock()
    router.call.side_effect = RuntimeError("LLM unavailable")

    r = _reflection(outcome_tracker=tracker, llm_router=router)
    report = r.run()
    assert report.error != ""
    assert isinstance(report, ReflectionReport)


# ── apply proposals ───────────────────────────────────────────────────────────

def test_apply_proposals_updates_belief():
    import json
    soul = MagicMock()
    existing = MagicMock()
    existing.confidence = 0.7
    soul.get_belief.return_value = existing
    soul.list_beliefs.return_value = []
    soul.list_lenses.return_value = []

    tracker = MagicMock()
    tracker.stats.return_value = {
        "total": 3, "done": 3, "abandoned": 0, "user_corrected": 0,
        "completion_rate": 1.0, "avg_steps": 2, "avg_policy_flags": 0,
    }
    tracker.recent.return_value = []

    router = MagicMock()
    router.call.return_value = (json.dumps({
        "summary": "OK",
        "patterns": [],
        "belief_proposals": [{"node_id": "b1", "new_confidence": 0.8, "rationale": "good"}],
        "flag_goal_ids": [],
    }), {})

    r = _reflection(outcome_tracker=tracker, soul=soul, llm_router=router, auto_apply=True)
    report = r.run()
    assert report.applied is True
    soul.update_belief.assert_called_once()


def test_apply_proposals_rejects_large_delta():
    import json
    soul = MagicMock()
    existing = MagicMock()
    existing.confidence = 0.5
    soul.get_belief.return_value = existing
    soul.list_beliefs.return_value = []
    soul.list_lenses.return_value = []

    tracker = MagicMock()
    tracker.stats.return_value = {
        "total": 1, "done": 1, "abandoned": 0, "user_corrected": 0,
        "completion_rate": 1.0, "avg_steps": 1, "avg_policy_flags": 0,
    }
    tracker.recent.return_value = []

    router = MagicMock()
    router.call.return_value = (json.dumps({
        "summary": "OK",
        "patterns": [],
        "belief_proposals": [{"node_id": "b1", "new_confidence": 0.9, "rationale": "big jump"}],
        "flag_goal_ids": [],
    }), {})

    r = _reflection(outcome_tracker=tracker, soul=soul, llm_router=router, auto_apply=True)
    r.run()
    # Delta of 0.4 exceeds 0.15 limit — should NOT be applied
    soul.update_belief.assert_not_called()


# ── summarise_for_chat() ──────────────────────────────────────────────────────

def test_summarise_for_chat_returns_string():
    r = _reflection()
    s = r.summarise_for_chat()
    assert isinstance(s, str)
    assert "reflection" in s.lower()
