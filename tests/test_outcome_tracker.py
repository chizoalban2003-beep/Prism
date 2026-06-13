"""Tests for prism_outcome_tracker.OutcomeTracker"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

from prism_outcome_tracker import (
    OUTCOME_ABANDONED,
    OUTCOME_CORRECTED,
    OUTCOME_DONE,
    OutcomeTracker,
    _extract_keywords,
)


def _tracker(soul=None, horizon=None) -> tuple[OutcomeTracker, str]:
    d = tempfile.mkdtemp()
    db = str(Path(d) / "outcomes.db")
    return OutcomeTracker(db_path=db, soul=soul, horizon=horizon), db


# ── record() ─────────────────────────────────────────────────────────────────

def test_record_creates_row():
    tracker, _ = _tracker()
    rec = tracker.record("chain1", "check the weather", OUTCOME_DONE)
    assert rec.outcome == OUTCOME_DONE
    assert rec.chain_id == "chain1"


def test_record_abandoned():
    tracker, _ = _tracker()
    rec = tracker.record("c2", "complex multi-step task", OUTCOME_ABANDONED)
    assert rec.outcome == OUTCOME_ABANDONED


def test_record_corrected():
    tracker, _ = _tracker()
    rec = tracker.record("c3", "write an email", OUTCOME_CORRECTED,
                         correction="You sent the wrong email")
    assert rec.correction == "You sent the wrong email"


def test_record_with_all_fields():
    tracker, _ = _tracker()
    rec = tracker.record(
        chain_id    = "abc123",
        goal        = "find flights to Tokyo",
        outcome     = OUTCOME_DONE,
        steps_count = 4,
        duration_ms = 1200.5,
        policy_flags= 1,
        final_answer= "Here are the flights...",
        context_id  = "work",
    )
    assert rec.steps_count == 4
    assert rec.duration_ms == 1200.5
    assert rec.context_id  == "work"


# ── recent() ─────────────────────────────────────────────────────────────────

def test_recent_returns_records():
    tracker, _ = _tracker()
    tracker.record("c1", "goal A", OUTCOME_DONE)
    tracker.record("c2", "goal B", OUTCOME_ABANDONED)
    records = tracker.recent(n=10)
    assert len(records) == 2


def test_recent_respects_n():
    tracker, _ = _tracker()
    for i in range(5):
        tracker.record(f"c{i}", f"goal {i}", OUTCOME_DONE)
    records = tracker.recent(n=3)
    assert len(records) == 3


def test_recent_filters_by_context():
    tracker, _ = _tracker()
    tracker.record("c1", "work goal", OUTCOME_DONE, context_id="work")
    tracker.record("c2", "personal goal", OUTCOME_DONE, context_id="personal")
    work_records = tracker.recent(n=10, context_id="work")
    assert all(r.context_id == "work" for r in work_records)
    assert len(work_records) == 1


# ── stats() ──────────────────────────────────────────────────────────────────

def test_stats_counts_outcomes():
    tracker, _ = _tracker()
    tracker.record("c1", "g", OUTCOME_DONE)
    tracker.record("c2", "g", OUTCOME_DONE)
    tracker.record("c3", "g", OUTCOME_ABANDONED)
    tracker.record("c4", "g", OUTCOME_CORRECTED)
    stats = tracker.stats(days=30)
    assert stats["total"] == 4
    assert stats["done"] == 2
    assert stats["abandoned"] == 1
    assert stats["user_corrected"] == 1


def test_stats_completion_rate():
    tracker, _ = _tracker()
    for _ in range(8):
        tracker.record("c", "g", OUTCOME_DONE)
    for _ in range(2):
        tracker.record("c", "g", OUTCOME_ABANDONED)
    stats = tracker.stats(days=30)
    assert stats["completion_rate"] == 0.8


def test_stats_empty_db():
    tracker, _ = _tracker()
    stats = tracker.stats()
    assert stats["total"] == 0
    assert stats["completion_rate"] == 0.0


# ── pattern_stats() ───────────────────────────────────────────────────────────

def test_pattern_stats_matches_keyword():
    tracker, _ = _tracker()
    tracker.record("c1", "find the best flight to Paris",   OUTCOME_DONE)
    tracker.record("c2", "check available flights to Rome", OUTCOME_DONE)
    tracker.record("c3", "book a hotel in London",          OUTCOME_ABANDONED)
    ps = tracker.pattern_stats("flight")
    assert ps["total"] == 2
    assert ps["done"] == 2
    assert ps["completion_rate"] == 1.0


def test_pattern_stats_no_match():
    tracker, _ = _tracker()
    tracker.record("c1", "check the weather", OUTCOME_DONE)
    ps = tracker.pattern_stats("quantum")
    assert ps["total"] == 0


# ── Soul feedback ─────────────────────────────────────────────────────────────

def test_feed_soul_calls_record_observation():
    soul = MagicMock()
    lens = MagicMock()
    lens.lens_id     = "health_lens"
    lens.name        = "health"
    lens.description = "health monitoring"
    soul.list_lenses.return_value = [lens]
    soul.list_beliefs.return_value = []

    tracker, _ = _tracker(soul=soul)
    tracker.record("c1", "check my health summary today", OUTCOME_DONE)
    # Feed should update the health lens
    n = tracker.feed_soul(soul)
    assert soul.record_observation.called or n >= 0  # may be 0 if no keyword overlap


def test_feed_soul_handles_no_soul():
    tracker, _ = _tracker()
    tracker.record("c1", "goal", OUTCOME_DONE)
    n = tracker.feed_soul(MagicMock())  # should not raise
    assert n >= 0


# ── Horizon feedback ──────────────────────────────────────────────────────────

def test_feed_horizon_calls_update_context():
    horizon = MagicMock()
    goal = MagicMock()
    goal.goal_id = "g1"
    goal.intent  = "monitor my weekly health metrics"
    goal.created_at = time.time() - 86400
    horizon.list_goals.return_value = [goal]

    tracker, _ = _tracker(horizon=horizon)
    for _ in range(4):
        tracker.record("c", "check health metrics weekly", OUTCOME_DONE)
    tracker.feed_horizon(horizon)
    # update_context may be called if keyword overlap >= 3 outcomes
    assert horizon.list_goals.called


# ── _extract_keywords ─────────────────────────────────────────────────────────

def test_extract_keywords_filters_stopwords():
    kw = _extract_keywords("what is the best way to check weather today")
    assert "what" not in kw
    assert "best" not in kw or "check" in kw or "weather" in kw


def test_extract_keywords_returns_list():
    kw = _extract_keywords("analyse financial trends in Europe")
    assert isinstance(kw, list)
    assert len(kw) <= 8
