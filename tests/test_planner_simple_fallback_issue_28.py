"""
tests/test_planner_simple_fallback_issue_28.py
==============================================
Graceful planner degradation (issue #28-82): when the big structured
extraction prompt fails (weak local model times out or emits
unparseable JSON), the planner tries a tiny plain-text single-shot
plan before surfacing the "Planner LLM unavailable" card.
"""
from __future__ import annotations

from unittest.mock import patch

from prism_planner import PrismPlanner

SIMPLE_RAW = """GOAL: Get the day under control
1. Write down the three things that actually matter today.
2. Do the hardest one first, before email.
3) Block 30 minutes for a walk after lunch.
some chatter the model added that is not a step
4. Review what got done at 5pm.
"""


def _planner() -> PrismPlanner:
    return PrismPlanner(ollama_host="http://127.0.0.1:1", request_timeout=1)


class TestSimplePlanParsing:
    def test_parses_goal_and_numbered_steps(self):
        p = _planner()
        with patch.object(p, "_call_ollama", return_value=SIMPLE_RAW):
            plan = p._simple_plan("plan my day")
        assert plan is not None
        assert plan.recommended.name == "Get the day under control"
        actions = [s.action for s in plan.recommended.steps]
        assert len(actions) == 4
        assert actions[0].startswith("Write down")
        assert "chatter" not in " ".join(actions)

    def test_empty_llm_response_returns_none(self):
        p = _planner()
        with patch.object(p, "_call_ollama", return_value=""):
            assert p._simple_plan("plan my day") is None

    def test_no_numbered_lines_returns_none(self):
        p = _planner()
        with patch.object(p, "_call_ollama",
                          return_value="I cannot make plans, sorry."):
            assert p._simple_plan("plan my day") is None


class TestPlanDegradationChain:
    def test_extraction_failure_falls_to_simple_plan(self):
        p = _planner()
        with patch.object(p, "_extract_task_profile", return_value=None), \
             patch.object(p, "_call_ollama", return_value=SIMPLE_RAW):
            plan = p.plan("plan my day")
        assert "simple mode" in plan.context_summary
        assert plan.recommended.steps, "simple plan should carry steps"

    def test_total_failure_still_reaches_fallback_card(self):
        p = _planner()
        with patch.object(p, "_extract_task_profile", return_value=None), \
             patch.object(p, "_call_ollama", return_value=""):
            plan = p.plan("plan my day")
        assert "Planner LLM unavailable" in plan.context_summary

    def test_healthy_extraction_never_calls_simple_plan(self):
        p = _planner()
        profile = {
            "domain": "general", "entity": "user", "timeline": "1 day",
            "context_summary": "ok",
            "strategies": [{
                "name": "Do it", "position": 0.5, "payoff": 1.0,
                "cost": 0.1, "risk": 10, "probability": 0.9,
            }],
            "factors": [],
        }
        with patch.object(p, "_extract_task_profile", return_value=profile), \
             patch.object(p, "_generate_action_plan") as gen, \
             patch.object(p, "_simple_plan") as simple:
            gen.return_value = None
            try:
                p.plan("plan my day")
            except Exception:
                pass  # downstream mocks are incomplete; we only assert routing
            simple.assert_not_called()
