"""Tests for prism_horizon.py — cross-session long-horizon goal planner."""

from __future__ import annotations

import time

import pytest

from prism_horizon import HorizonGoal, HorizonGoalStatus, HorizonPlanner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _planner(tmp_path) -> HorizonPlanner:
    return HorizonPlanner(db_path=str(tmp_path / "horizon.db"))


# ---------------------------------------------------------------------------
# HorizonGoal serialisation
# ---------------------------------------------------------------------------

class TestHorizonGoalSerialisation:
    def test_roundtrip(self):
        g = HorizonGoal(
            goal_id="abc1",
            intent="Book a flight when price < $300",
            trigger_condition="price below 300",
            completion_condition="flight booked",
            accumulated_context={"last_price": 450},
            completed_steps=["searched flights"],
        )
        row = g.to_row()
        g2  = HorizonGoal.from_row(row)
        assert g2.goal_id              == "abc1"
        assert g2.intent               == g.intent
        assert g2.accumulated_context  == {"last_price": 450}
        assert g2.completed_steps      == ["searched flights"]
        assert g2.status               == HorizonGoalStatus.WATCHING

    def test_default_status_is_watching(self):
        g = HorizonGoal(goal_id="x", intent="T", trigger_condition="c")
        assert g.status == HorizonGoalStatus.WATCHING

    def test_is_expired_false_when_no_deadline(self):
        g = HorizonGoal(goal_id="x", intent="T", trigger_condition="c")
        assert g.is_expired() is False

    def test_is_expired_true_past_deadline(self):
        g = HorizonGoal(
            goal_id="x", intent="T", trigger_condition="c",
            expires_at=time.time() - 1,
        )
        assert g.is_expired() is True


# ---------------------------------------------------------------------------
# HorizonPlanner — basic CRUD
# ---------------------------------------------------------------------------

class TestHorizonPlannerCRUD:
    def test_add_returns_id(self, tmp_path):
        p = _planner(tmp_path)
        gid = p.add("Monitor price", "price < 300")
        assert isinstance(gid, str) and len(gid) > 0

    def test_get_returns_goal(self, tmp_path):
        p = _planner(tmp_path)
        gid = p.add("Monitor price", "price < 300", completion_condition="booked")
        g   = p.get(gid)
        assert g.intent              == "Monitor price"
        assert g.trigger_condition   == "price < 300"
        assert g.completion_condition == "booked"

    def test_get_unknown_returns_none(self, tmp_path):
        p = _planner(tmp_path)
        assert p.get("nope") is None

    def test_list_goals_empty(self, tmp_path):
        p = _planner(tmp_path)
        assert p.list_goals() == []

    def test_list_goals_all(self, tmp_path):
        p = _planner(tmp_path)
        p.add("G1", "c1")
        p.add("G2", "c2")
        assert len(p.list_goals()) == 2

    def test_list_goals_by_status(self, tmp_path):
        p = _planner(tmp_path)
        gid = p.add("G", "c")
        p.complete(gid)
        assert len(p.list_goals(HorizonGoalStatus.COMPLETED)) == 1
        assert len(p.list_goals(HorizonGoalStatus.WATCHING))  == 0

    def test_complete_marks_goal(self, tmp_path):
        p   = _planner(tmp_path)
        gid = p.add("Task", "cond")
        assert p.complete(gid) is True
        g   = p.get(gid)
        assert g.status       == HorizonGoalStatus.COMPLETED
        assert g.completed_at is not None

    def test_complete_idempotent(self, tmp_path):
        p   = _planner(tmp_path)
        gid = p.add("Task", "cond")
        p.complete(gid)
        assert p.complete(gid) is False

    def test_abandon_marks_goal(self, tmp_path):
        p   = _planner(tmp_path)
        gid = p.add("Task", "cond")
        assert p.abandon(gid, reason="user cancelled") is True
        g   = p.get(gid)
        assert g.status == HorizonGoalStatus.ABANDONED
        assert g.notes  == "user cancelled"

    def test_update_context_merges(self, tmp_path):
        p   = _planner(tmp_path)
        gid = p.add("Watch", "cond")
        p.update_context(gid, last_price=450, date="2026-06-02")
        g   = p.get(gid)
        assert g.accumulated_context["last_price"] == 450
        assert g.accumulated_context["date"]       == "2026-06-02"

    def test_update_context_unknown_raises(self, tmp_path):
        p = _planner(tmp_path)
        with pytest.raises(KeyError):
            p.update_context("nope", x=1)

    def test_record_step(self, tmp_path):
        p   = _planner(tmp_path)
        gid = p.add("Task", "cond")
        p.record_step(gid, "searched flights")
        p.record_step(gid, "selected seat 14A")
        g   = p.get(gid)
        assert "searched flights" in g.completed_steps
        assert "selected seat 14A" in g.completed_steps

    def test_status_summary(self, tmp_path):
        p = _planner(tmp_path)
        p.add("G1", "c1")
        p.add("G2", "c2")
        s = p.status()
        assert s["total"]               == 2
        assert s["counts"]["watching"]  == 2
        assert len(s["goals"])          == 2


# ---------------------------------------------------------------------------
# Probe-based trigger evaluation
# ---------------------------------------------------------------------------

class TestHorizonPlannerProbe:
    def test_probe_triggers_when_condition_met(self, tmp_path):
        p   = _planner(tmp_path)
        gid = p.add("Book flight when price < 300", "price < 300")
        p.register_probe(gid, lambda ctx: ctx.get("last_price", 999) < 300)
        p.update_context(gid, last_price=250)

        triggered = p.on_session_start()
        assert gid in triggered
        assert p.get(gid).status == HorizonGoalStatus.TRIGGERED

    def test_probe_not_triggered_when_condition_not_met(self, tmp_path):
        p   = _planner(tmp_path)
        gid = p.add("Book flight when price < 300", "price < 300")
        p.register_probe(gid, lambda ctx: ctx.get("last_price", 999) < 300)
        p.update_context(gid, last_price=450)

        triggered = p.on_session_start()
        assert gid not in triggered
        assert p.get(gid).status == HorizonGoalStatus.WATCHING

    def test_probe_at_add_time(self, tmp_path):
        p   = _planner(tmp_path)
        gid = p.add("Always trigger", "cond", probe=lambda ctx: True)
        assert gid in p.on_session_start()

    def test_probe_exception_does_not_crash(self, tmp_path):
        p   = _planner(tmp_path)
        gid = p.add("Bad probe", "cond")
        p.register_probe(gid, lambda ctx: 1 / 0)   # ZeroDivisionError
        triggered = p.on_session_start()            # must not raise
        assert gid not in triggered

    def test_register_probe_unknown_raises(self, tmp_path):
        p = _planner(tmp_path)
        with pytest.raises(KeyError):
            p.register_probe("nope", lambda ctx: True)

    def test_check_now_returns_bool(self, tmp_path):
        p   = _planner(tmp_path)
        gid = p.add("Test", "cond")
        p.register_probe(gid, lambda ctx: True)
        assert p.check_now(gid) is True

    def test_check_now_does_not_change_status(self, tmp_path):
        p   = _planner(tmp_path)
        gid = p.add("Test", "cond")
        p.register_probe(gid, lambda ctx: True)
        p.check_now(gid)
        assert p.get(gid).status == HorizonGoalStatus.WATCHING

    def test_check_now_unknown_raises(self, tmp_path):
        p = _planner(tmp_path)
        with pytest.raises(KeyError):
            p.check_now("nope")


# ---------------------------------------------------------------------------
# SQLite persistence across planner instances
# ---------------------------------------------------------------------------

class TestHorizonPlannerPersistence:
    def test_goals_survive_reload(self, tmp_path):
        p1  = HorizonPlanner(db_path=str(tmp_path / "h.db"))
        gid = p1.add("Persist me", "cond", completion_condition="done")
        p1.update_context(gid, price=350)
        del p1

        p2 = HorizonPlanner(db_path=str(tmp_path / "h.db"))
        g  = p2.get(gid)
        assert g is not None
        assert g.intent                       == "Persist me"
        assert g.accumulated_context["price"] == 350

    def test_completed_steps_survive_reload(self, tmp_path):
        p1  = HorizonPlanner(db_path=str(tmp_path / "h.db"))
        gid = p1.add("Multi-step", "cond")
        p1.record_step(gid, "step one done")
        del p1

        p2 = HorizonPlanner(db_path=str(tmp_path / "h.db"))
        assert "step one done" in p2.get(gid).completed_steps

    def test_terminal_status_survives_reload(self, tmp_path):
        p1  = HorizonPlanner(db_path=str(tmp_path / "h.db"))
        gid = p1.add("Complete me", "cond")
        p1.complete(gid)
        del p1

        p2 = HorizonPlanner(db_path=str(tmp_path / "h.db"))
        assert p2.get(gid).status == HorizonGoalStatus.COMPLETED


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

class TestHorizonPlannerSessionLifecycle:
    def test_session_count_increments(self, tmp_path):
        p   = _planner(tmp_path)
        gid = p.add("Watch", "never true")
        p.register_probe(gid, lambda ctx: False)
        p.on_session_start()
        assert p.get(gid).session_count == 1
        p.on_session_start()
        assert p.get(gid).session_count == 2

    def test_on_session_end_pauses_triggered_goal(self, tmp_path):
        p   = _planner(tmp_path)
        gid = p.add("Trigger me", "cond")
        p.register_probe(gid, lambda ctx: True)
        p.on_session_start()
        assert p.get(gid).status == HorizonGoalStatus.TRIGGERED

        p.on_session_end()
        assert p.get(gid).status == HorizonGoalStatus.PAUSED

    def test_paused_goal_resumed_next_session(self, tmp_path):
        p   = _planner(tmp_path)
        gid = p.add("Resume me", "cond")
        p.register_probe(gid, lambda ctx: True)
        p.on_session_start()
        p.on_session_end()
        assert p.get(gid).status == HorizonGoalStatus.PAUSED

        activated = p.on_session_start()
        assert gid in activated

    def test_completed_goals_skipped(self, tmp_path):
        p   = _planner(tmp_path)
        gid = p.add("Done", "cond")
        p.complete(gid)
        assert gid not in p.on_session_start()

    def test_abandoned_goals_skipped(self, tmp_path):
        p   = _planner(tmp_path)
        gid = p.add("Gone", "cond")
        p.abandon(gid)
        assert gid not in p.on_session_start()

    def test_expired_goal_abandoned_on_session_start(self, tmp_path):
        p   = _planner(tmp_path)
        # Insert a goal with an already-past expiry directly
        g = HorizonGoal(
            goal_id="exp1",
            intent="Expired",
            trigger_condition="cond",
            expires_at=time.time() - 1.0,   # already in the past
        )
        p._upsert(g)
        p.on_session_start()
        assert p.get("exp1").status == HorizonGoalStatus.ABANDONED


# ---------------------------------------------------------------------------
# Execution prompt construction
# ---------------------------------------------------------------------------

class TestExecutionPrompt:
    def test_includes_intent(self, tmp_path):
        p = _planner(tmp_path)
        g = HorizonGoal(
            goal_id="t1",
            intent="Book a flight to Lisbon",
            trigger_condition="price < 300",
        )
        prompt = p._build_execution_prompt(g, resume=False)
        assert "Book a flight to Lisbon" in prompt

    def test_includes_accumulated_context(self, tmp_path):
        p = _planner(tmp_path)
        g = HorizonGoal(
            goal_id="t2",
            intent="Book flight",
            trigger_condition="cond",
            accumulated_context={"last_price": 290},
        )
        prompt = p._build_execution_prompt(g, resume=False)
        assert "last_price" in prompt
        assert "290" in prompt

    def test_resume_includes_done_steps(self, tmp_path):
        p = _planner(tmp_path)
        g = HorizonGoal(
            goal_id="t3",
            intent="Book flight",
            trigger_condition="cond",
            completed_steps=["searched for flights", "selected option A"],
        )
        prompt = p._build_execution_prompt(g, resume=True)
        assert "searched for flights" in prompt
        assert "skip completed steps" in prompt.lower()

    def test_no_resume_omits_steps_block(self, tmp_path):
        p = _planner(tmp_path)
        g = HorizonGoal(
            goal_id="t4",
            intent="Fresh goal",
            trigger_condition="cond",
            completed_steps=["old step"],
        )
        prompt = p._build_execution_prompt(g, resume=False)
        assert "skip completed" not in prompt.lower()

    def test_includes_completion_condition(self, tmp_path):
        p = _planner(tmp_path)
        g = HorizonGoal(
            goal_id="t5",
            intent="Book flight",
            trigger_condition="cond",
            completion_condition="flight is booked",
        )
        prompt = p._build_execution_prompt(g, resume=False)
        assert "flight is booked" in prompt


# ---------------------------------------------------------------------------
# LLM condition evaluation (mock LLM)
# ---------------------------------------------------------------------------

class TestLLMEvaluation:
    def _mock_llm(self, answer: str):
        class MockRouter:
            def call(self, prompt, **kwargs):
                return answer, "mock/model"
        return MockRouter()

    def test_llm_yes_triggers(self, tmp_path):
        p = HorizonPlanner(
            llm_router=self._mock_llm("yes"),
            db_path=str(tmp_path / "h.db"),
        )
        gid = p.add("Check something", "condition")
        triggered = p.on_session_start()
        assert gid in triggered

    def test_llm_no_does_not_trigger(self, tmp_path):
        p = HorizonPlanner(
            llm_router=self._mock_llm("no"),
            db_path=str(tmp_path / "h.db"),
        )
        gid = p.add("Check something", "condition")
        triggered = p.on_session_start()
        assert gid not in triggered

    def test_llm_error_does_not_crash(self, tmp_path):
        class BrokenRouter:
            def call(self, prompt, **kwargs):
                raise RuntimeError("LLM offline")

        p = HorizonPlanner(
            llm_router=BrokenRouter(),
            db_path=str(tmp_path / "h.db"),
        )
        gid = p.add("Will fail gracefully", "cond")
        triggered = p.on_session_start()   # must not raise
        assert gid not in triggered

    def test_probe_takes_priority_over_llm(self, tmp_path):
        # LLM says 'yes' but probe says False — probe should win
        p = HorizonPlanner(
            llm_router=self._mock_llm("yes"),
            db_path=str(tmp_path / "h.db"),
        )
        gid = p.add("Probe beats LLM", "cond")
        p.register_probe(gid, lambda ctx: False)
        triggered = p.on_session_start()
        assert gid not in triggered


# ---------------------------------------------------------------------------
# as_proactive_trigger integration
# ---------------------------------------------------------------------------

class TestAsProactiveTrigger:
    def test_returns_proactive_trigger(self, tmp_path):
        from prism_proactive import ProactiveTrigger
        p   = _planner(tmp_path)
        gid = p.add("Watch for price drop", "price < 300")
        pt  = p.as_proactive_trigger(gid)
        assert isinstance(pt, ProactiveTrigger)
        assert gid in pt.trigger_id

    def test_unknown_goal_raises(self, tmp_path):
        p = _planner(tmp_path)
        with pytest.raises(KeyError):
            p.as_proactive_trigger("nope")

    def test_condition_callable_returns_bool(self, tmp_path):
        p   = _planner(tmp_path)
        gid = p.add("Watch", "cond")
        p.register_probe(gid, lambda ctx: True)
        pt  = p.as_proactive_trigger(gid)
        assert pt.condition() is True   # triggers because probe returns True

    def test_condition_false_for_completed_goal(self, tmp_path):
        p   = _planner(tmp_path)
        gid = p.add("Done", "cond")
        p.register_probe(gid, lambda ctx: True)
        p.complete(gid)
        pt  = p.as_proactive_trigger(gid)
        assert pt.condition() is False  # completed goal should not re-trigger
