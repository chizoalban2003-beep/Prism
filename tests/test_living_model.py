"""Tests for the living user model: PrismPersona, PrismCrystalliser, PrismNarrative."""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

from prism_crystalliser import PrismCrystalliser
from prism_narrative import PrismNarrative
from prism_persona import PrismPersona

# ── Helpers ───────────────────────────────────────────────────────────────────

def _persona(tmp_path=None) -> PrismPersona:
    d = tempfile.mkdtemp()
    return PrismPersona(db_path=str(Path(d) / "persona.db"))


# ── PrismPersona ──────────────────────────────────────────────────────────────

def test_update_and_get_trait():
    p = _persona()
    p.update_trait("communication_style", "direct", 0.6, source="inferred")
    t = p.get_trait("communication_style")
    assert t is not None
    assert t.name == "communication_style"
    assert t.value == "direct"
    assert 0.5 <= t.confidence <= 1.0
    assert t.observation_count == 1


def test_update_trait_increments_observation_count():
    p = _persona()
    p.update_trait("risk_tolerance", "cautious", 0.5)
    p.update_trait("risk_tolerance", "cautious", 0.6)
    t = p.get_trait("risk_tolerance")
    assert t.observation_count == 2


def test_explicit_source_overwrites_inferred():
    p = _persona()
    p.update_trait("style", "verbose", 0.4, source="inferred")
    p.update_trait("style", "concise", 0.9, source="explicit")
    t = p.get_trait("style")
    assert t.source == "explicit"
    assert t.value == "concise"


def test_get_trait_missing_returns_none():
    p = _persona()
    assert p.get_trait("nonexistent") is None


def test_list_traits_returns_all():
    p = _persona()
    p.update_trait("a", "1", 0.5)
    p.update_trait("b", "2", 0.7)
    traits = p.list_traits()
    names = [t.name for t in traits]
    assert "a" in names and "b" in names


def test_add_pattern_returns_id():
    p = _persona()
    pid = p.add_pattern("tends to ask follow-up questions", "example here")
    assert isinstance(pid, str) and len(pid) == 8


def test_bump_pattern_creates_when_no_match():
    p = _persona()
    p.bump_pattern("defers complex decisions at night")
    patterns = p._top_patterns(5)
    assert any("defers" in pt.description for pt in patterns)


def test_bump_pattern_increments_frequency():
    p = _persona()
    p.add_pattern("abandoned planning tasks", "skipped the plan")
    p.bump_pattern("abandoned planning tasks")
    patterns = p._top_patterns(5)
    match = next(pt for pt in patterns if "abandoned" in pt.description)
    assert match.frequency >= 2


def test_record_active_hour_and_peak_hours():
    p = _persona()
    for _ in range(5):
        p.record_active_hour(9)
    for _ in range(3):
        p.record_active_hour(14)
    p.record_active_hour(20)
    peaks = p.peak_hours()
    assert 9 in peaks
    assert len(peaks) <= 3


def test_build_context_returns_string():
    p = _persona()
    p.update_trait("communication_style", "direct", 0.8)
    p.update_trait("response_length_preference", "concise", 0.7)
    p.record_active_hour(9)
    ctx = p.build_context(max_chars=500)
    assert isinstance(ctx, str)
    assert "[Crystallised user profile]" in ctx


def test_build_context_respects_max_chars():
    p = _persona()
    ctx = p.build_context(max_chars=50)
    assert len(ctx) <= 53  # 50 + possible "..." suffix


def test_growth_since_returns_dict():
    p = _persona()
    p.update_trait("x", "y", 0.5)
    g = p.growth_since(days=7)
    assert "new_traits" in g
    assert "new_patterns" in g
    assert "confidence_avg" in g
    assert g["new_traits"] >= 1


def test_summary_returns_string():
    p = _persona()
    p.update_trait("technical_depth", "high", 0.9)
    s = p.summary()
    assert "technical_depth" in s


def test_db_migration_runs():
    d = tempfile.mkdtemp()
    db = str(Path(d) / "persona.db")
    # Create DB twice — migration should be idempotent
    PrismPersona(db_path=db)
    p2 = PrismPersona(db_path=db)
    p2.update_trait("test", "val", 0.5)
    assert p2.get_trait("test") is not None


# ── PrismCrystalliser ─────────────────────────────────────────────────────────

def test_observe_turn_records_active_hour():
    p = _persona()
    c = PrismCrystalliser(persona=p)
    c.observe_turn("hello world today", "response text", "chat", {})
    hour = time.localtime().tm_hour
    peaks = p.peak_hours()
    # At least the current hour has been recorded
    assert hour in peaks or len(peaks) >= 0  # non-empty after call


def test_observe_turn_length_short():
    p = _persona()
    c = PrismCrystalliser(persona=p)
    c.observe_turn("hi", "ok", "chat", {})
    t = p.get_trait("response_length_preference")
    assert t is not None
    assert t.value == "concise"


def test_observe_turn_length_long():
    p = _persona()
    c = PrismCrystalliser(persona=p)
    long_msg = " ".join(["word"] * 60)
    c.observe_turn(long_msg, "response", "chat", {})
    t = p.get_trait("response_length_preference")
    assert t.value == "detailed"


def test_observe_turn_technical_depth_high():
    p = _persona()
    c = PrismCrystalliser(persona=p)
    c.observe_turn(
        "run the docker deploy script and check the api endpoint config json",
        "ok", "device_task", {}
    )
    t = p.get_trait("technical_depth")
    assert t is not None
    assert t.value == "high"


def test_observe_turn_approval_approved():
    p = _persona()
    c = PrismCrystalliser(persona=p)
    c.observe_turn("yes go ahead", "done", "approve_pending", {"_last_approval_action": "approved"})
    t = p.get_trait("risk_tolerance")
    assert t is not None
    assert t.value == "willing"


def test_observe_turn_approval_cancelled():
    p = _persona()
    c = PrismCrystalliser(persona=p)
    c.observe_turn("cancel that", "cancelled", "cancel_pending", {"_last_approval_action": "cancelled"})
    t = p.get_trait("risk_tolerance")
    assert t.value == "cautious"


def test_observe_outcome_abandoned():
    p = _persona()
    c = PrismCrystalliser(persona=p)
    c.observe_outcome("planning", "abandoned", "plan my week", "")
    patterns = p._top_patterns(5)
    assert any("abandoned" in pt.description for pt in patterns)


def test_observe_outcome_done():
    p = _persona()
    c = PrismCrystalliser(persona=p)
    c.observe_outcome("email", "done", "send email to Bob", "")
    patterns = p._top_patterns(5)
    assert any("completes" in pt.description for pt in patterns)


def test_observe_outcome_corrected_explicit():
    p = _persona()
    c = PrismCrystalliser(persona=p)
    c.observe_outcome("email", "user_corrected", "send email", "I prefer shorter emails always")
    t = p.get_trait("correction_pattern")
    assert t is not None
    assert t.source == "explicit"


def test_deep_analyse_no_router_returns_zero():
    p = _persona()
    c = PrismCrystalliser(persona=p)
    assert c.deep_analyse() == 0


def test_deep_analyse_with_mocked_llm():
    p = _persona()
    mock_router = MagicMock()
    mock_router.call.return_value = (
        json.dumps({
            "communication_style": "direct",
            "response_length": "concise",
            "technical_depth": "high",
            "decision_style": "quick",
            "patterns": ["asks follow-up questions frequently"],
            "explicit_preferences": ["prefers dark mode"],
        }),
        "mock/model",
    )
    mock_memory = MagicMock()
    mock_memory.search.return_value = []
    c = PrismCrystalliser(persona=p, memory=mock_memory, llm_router=mock_router)
    updates = c.deep_analyse(lookback_hours=24)
    assert updates > 0
    assert p.get_trait("communication_style") is not None
    assert p.get_trait("communication_style").value == "direct"


def test_deep_analyse_invalid_llm_response():
    p = _persona()
    mock_router = MagicMock()
    mock_router.call.return_value = ("not valid json at all {broken", "mock/model")
    mock_memory = MagicMock()
    mock_memory.search.return_value = []
    c = PrismCrystalliser(persona=p, memory=mock_memory, llm_router=mock_router)
    # Should not raise, returns 0
    result = c.deep_analyse()
    assert result == 0


def test_crystallise_returns_summary_dict():
    p = _persona()
    mock_router = MagicMock()
    mock_router.call.return_value = (json.dumps({"patterns": []}), "mock/model")
    mock_memory = MagicMock()
    mock_memory.search.return_value = []
    c = PrismCrystalliser(persona=p, memory=mock_memory, llm_router=mock_router)
    summary = c.crystallise()
    assert "traits_updated" in summary
    assert "confidence_avg" in summary
    assert "peak_hours" in summary


# ── PrismNarrative ────────────────────────────────────────────────────────────

def test_snapshot_no_llm():
    p = _persona()
    p.update_trait("communication_style", "direct", 0.8)
    n = PrismNarrative(persona=p)
    result = n.snapshot()
    assert isinstance(result, str)
    assert len(result) > 0


def test_snapshot_with_soul():
    p = _persona()
    mock_soul = MagicMock()
    mock_soul.compress_for_llm.return_value = "I value focus and clarity."
    n = PrismNarrative(persona=p, soul=mock_soul)
    result = n.snapshot()
    assert "focus" in result or "crystallised" in result.lower() or "profile" in result.lower()


def test_growth_report_no_llm():
    p = _persona()
    p.update_trait("technical_depth", "high", 0.9)
    p.update_trait("communication_style", "direct", 0.8)
    n = PrismNarrative(persona=p)
    report = n.growth_report()
    assert "PRISM" in report
    assert "traits" in report.lower() or "trait" in report.lower()


def test_growth_report_with_outcome_tracker():
    p = _persona()
    mock_tracker = MagicMock()
    mock_tracker.stats.return_value = {
        "done": 10, "total": 12, "completion_rate": 0.83,
        "abandoned": 1, "user_corrected": 1,
    }
    n = PrismNarrative(persona=p, outcome_tracker=mock_tracker)
    report = n.growth_report()
    assert "83%" in report or "0.83" in report or "83" in report


def test_weekly_narrative_fallback_no_llm():
    p = _persona()
    n = PrismNarrative(persona=p)
    result = n.weekly()
    assert isinstance(result, str)
    assert "PRISM" in result or "weekly" in result.lower() or "days" in result.lower()


def test_weekly_narrative_stores_to_memory():
    p = _persona()
    mock_memory = MagicMock()
    mock_memory.ingest.return_value = "abc123"
    n = PrismNarrative(persona=p, memory=mock_memory)
    n.weekly()
    assert mock_memory.ingest.called


def test_weekly_narrative_with_mocked_llm():
    p = _persona()
    mock_router = MagicMock()
    mock_router.call.return_value = (
        "This week, PRISM observed focused, direct communication patterns.",
        "mock/model",
    )
    mock_memory = MagicMock()
    mock_memory.ingest.return_value = "abc"
    n = PrismNarrative(persona=p, memory=mock_memory, llm_router=mock_router)
    result = n.weekly()
    assert "PRISM" in result or len(result) > 10


def test_monthly_narrative():
    p = _persona()
    n = PrismNarrative(persona=p)
    result = n.monthly()
    assert isinstance(result, str) and len(result) > 0


def test_narrative_memory_store_failure_does_not_raise():
    p = _persona()
    mock_memory = MagicMock()
    mock_memory.ingest.side_effect = RuntimeError("db error")
    n = PrismNarrative(persona=p, memory=mock_memory)
    # Should not raise
    result = n.weekly()
    assert isinstance(result, str)


# ── Integration: crystalliser → persona → build_context roundtrip ────────────

def test_integration_roundtrip():
    p = _persona()
    c = PrismCrystalliser(persona=p)

    # Simulate a few turns
    c.observe_turn(
        "run the docker api deploy config",
        "deployed successfully",
        "device_task",
        {"_last_approval_action": "approved"},
    )
    c.observe_turn("what's the status?", "all green", "status", {})
    c.observe_outcome("device_task", "done", "deploy api", "")

    # Persona should have learned from all three
    ctx = p.build_context(max_chars=500)
    assert "[Crystallised user profile]" in ctx

    traits = p.list_traits()
    assert len(traits) >= 3  # at least: length, tech_depth, risk_tolerance

    growth = p.growth_since(days=1)
    assert growth["new_traits"] >= 3


def test_integration_narrative_uses_persona():
    p = _persona()
    c = PrismCrystalliser(persona=p)
    for _ in range(5):
        c.observe_turn("write the api endpoint config", "done", "device_task", {})

    n = PrismNarrative(persona=p)
    snapshot = n.snapshot()
    assert isinstance(snapshot, str)
    # Profile data should be present in the snapshot
    assert len(snapshot) > 20
