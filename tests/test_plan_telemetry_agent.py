"""Agent-level wiring tests for plan telemetry (M12d).

Exercises only the methods we touched, with the rest of PrismAgent stubbed,
so we don't pay the full init cost.
"""
from __future__ import annotations

import types
from unittest.mock import MagicMock

import pytest

from prism_plan_telemetry import STATUS_DONE, PlanTelemetry


def _make_plan(*titles):
    from sports_pro import DailyPlan, DailyTask
    return DailyPlan(
        primary_focus="Recovery",
        activation=0.4,
        fulcrum=0.5,
        tasks=[DailyTask("07:00", 30, "rest", t, "") for t in titles],
        rationale="test",
    )


@pytest.fixture
def stub_agent(tmp_path):
    """Bind PrismAgent.replan as an unbound method onto a stub object."""
    from prism_agent import PrismAgent

    agent = types.SimpleNamespace()
    agent._plan_telemetry  = PlanTelemetry(db_path=str(tmp_path / "pt.db"))
    agent._kde             = MagicMock()
    agent._last_plan_request = "plan my day"
    agent._last_plan_id    = None
    agent._last_plan       = None
    agent.replan = types.MethodType(PrismAgent.replan, agent)
    return agent


def test_replan_includes_telemetry_summary_in_kde_prompt(stub_agent):
    plan = _make_plan("Morning run", "Evening lift")
    prior_id = stub_agent._plan_telemetry.record_plan(plan, "plan my day")
    stub_agent._plan_telemetry.mark_step(prior_id, 0, STATUS_DONE)
    stub_agent._last_plan_id = prior_id

    replan_plan = _make_plan("Stretch")
    stub_agent._kde.ask.return_value = types.SimpleNamespace(output=replan_plan)

    stub_agent.replan(instructions="add stretching")

    sent_message = stub_agent._kde.ask.call_args[0][0]
    assert "Previous plan status" in sent_message
    assert "Morning run" in sent_message
    assert "add stretching" in sent_message


def test_replan_supersedes_prior_plan(stub_agent):
    prior_id = stub_agent._plan_telemetry.record_plan(_make_plan("a"), "plan my day")
    stub_agent._last_plan_id = prior_id
    new_plan = _make_plan("b")
    stub_agent._kde.ask.return_value = types.SimpleNamespace(output=new_plan)

    stub_agent.replan()

    assert stub_agent._last_plan_id != prior_id
    old = stub_agent._plan_telemetry.get_plan(prior_id)
    assert old["superseded_by"] == stub_agent._last_plan_id


def test_replan_without_prior_plan_skips_summary(stub_agent):
    stub_agent._last_plan_id = None
    stub_agent._kde.ask.return_value = types.SimpleNamespace(output=_make_plan("x"))
    stub_agent.replan(instructions="fresh start")
    sent_message = stub_agent._kde.ask.call_args[0][0]
    assert "Previous plan status" not in sent_message


def test_replan_no_kde_returns_text_card():
    from prism_agent import PrismAgent

    agent = types.SimpleNamespace(
        _plan_telemetry=None, _kde=None,
        _last_plan_request="plan my day", _last_plan_id=None, _last_plan=None,
    )
    agent.replan = types.MethodType(PrismAgent.replan, agent)
    card = agent.replan()
    assert "KDE" in card.body or "isn't available" in card.body
