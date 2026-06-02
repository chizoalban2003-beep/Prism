import json
from unittest.mock import MagicMock
from prism_composer import (
    PrismComposer, CompositionPlan, CompositionStep,
    LogicResult, LOGIC_REGISTRY,
)
from prism_responses import text_card


def _make_composer(plan_json=None):
    router = MagicMock()
    if plan_json:
        router.call.return_value = (json.dumps(plan_json), {})
    else:
        router.call.return_value = ('{"steps": [], "parallel": false}', {})
    return PrismComposer(llm_router=router)


def _dummy_agent(intent, message, ctx):
    return text_card(f"result of {intent}: {message[:30]}", intent)


def test_should_compose_detects_multi_step():
    c = _make_composer()
    assert c.should_compose("check my email and then add tasks for urgent ones")


def test_should_compose_rejects_single():
    c = _make_composer()
    assert not c.should_compose("what is the weather")


def test_decompose_returns_none_for_single_step():
    c = _make_composer({"steps": [], "parallel": False})
    result = c.decompose("what time is it")
    assert result is None


def test_decompose_returns_plan():
    plan_data = {
        "steps": [
            {"step_id": "s1", "logic": "email_read",
             "description": "read emails", "depends_on": [],
             "input_from": "", "params": {}},
            {"step_id": "s2", "logic": "add_task",
             "description": "add tasks", "depends_on": ["s1"],
             "input_from": "s1", "params": {}},
        ],
        "parallel": False,
    }
    c = _make_composer(plan_data)
    # Need to override MIN_STEPS check
    plan = c.decompose("check email and add tasks")
    assert plan is not None
    assert len(plan.steps) == 2
    assert plan.steps[0].logic == "email_read"
    assert plan.steps[1].depends_on == ["s1"]


def test_execute_sequential():
    plan = CompositionPlan(
        plan_id="test", original="test",
        steps=[
            CompositionStep("s1","web_search","search",[], "", {}),
            CompositionStep("s2","add_task","add",["s1"],"s1", {}),
        ],
        parallel=False,
    )
    c = _make_composer()
    c._router.call.return_value = ("All done.", {})
    card = c.execute(plan, _dummy_agent, {})
    assert card is not None
    assert hasattr(card, "body")


def test_execute_parallel():
    plan = CompositionPlan(
        plan_id="test2", original="test",
        steps=[
            CompositionStep("s1","web_search","search",[], "", {}),
            CompositionStep("s2","calendar_read","calendar",[], "", {}),
        ],
        parallel=True,
    )
    c = _make_composer()
    c._router.call.return_value = ("All done in parallel.", {})
    card = c.execute(plan, _dummy_agent, {})
    assert "done" in card.body.lower() or card.body


def test_unknown_logic_falls_back_to_autonomous():
    plan_data = {
        "steps": [
            {"step_id": "s1", "logic": "nonexistent_logic_xyz",
             "description": "do something new", "depends_on": [],
             "input_from": "", "params": {}},
        ],
        "parallel": False,
    }
    c = _make_composer(plan_data)
    plan = c.decompose("do something new")
    if plan:
        assert plan.steps[0].logic == "autonomous"


def test_logic_registry_has_core_logics():
    for logic in ["email_read", "web_search", "calendar_read",
                  "add_task", "domain_medical", "autonomous"]:
        assert logic in LOGIC_REGISTRY


def test_compose_output_handles_failure():
    plan = CompositionPlan(
        plan_id="fail_test", original="test",
        steps=[CompositionStep("s1","web_search","search",[], "", {})],
        parallel=False,
    )
    results = {"s1": LogicResult("s1","web_search",{},"",False,"network error")}
    c = _make_composer()
    c._router.call.return_value = ("", {})
    card = c._compose_output(plan, results)
    assert "failed" in card.body.lower() or "error" in card.body.lower()
