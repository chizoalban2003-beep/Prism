from __future__ import annotations

import json
import urllib.error
import urllib.request
from unittest.mock import patch

import pytest

from prism_planner import (
    ActionStep,
    PlanOfAction,
    PrismPlanner,
    StrategyPlan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_TASK_PROFILE = {
    "domain": "fitness",
    "entity": "individual",
    "timeline": "6 months",
    "strategies": [
        {"name": "Walk only",         "position": 0.1, "payoff": 30,  "cost": 2,  "risk": 5,  "probability": 0.90},
        {"name": "Structured plan",   "position": 0.4, "payoff": 80,  "cost": 10, "risk": 20, "probability": 0.75},
        {"name": "High mileage push", "position": 0.7, "payoff": 120, "cost": 20, "risk": 50, "probability": 0.55},
        {"name": "Elite programme",   "position": 0.9, "payoff": 160, "cost": 40, "risk": 80, "probability": 0.35},
    ],
    "factors": [
        {"id": "fitness_level", "label": "Fitness level", "value": 0.2, "weight": 2.0, "direction": 1},
        {"id": "time_per_week", "label": "Time per week", "value": 0.4, "weight": 2.0, "direction": 1},
    ],
    "context_summary": "Low fitness, moderate time — a structured moderate plan is optimal.",
}

_MINIMAL_ACTION_PLAN = {
    "steps": [
        {"order": 1, "action": "Start easy runs", "timeline": "week 1-2",
         "resource": "running shoes", "outcome": "complete 3km without stopping"},
    ],
    "resources": ["running shoes", "coach app"],
    "expected_outcome": "Finish a marathon in under 5 hours.",
    "risks": ["injury", "overtraining"],
    "why_recommended": "Matches current low fitness with progressive overload.",
}


class _MockResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_setup_creates_instance():
    p = PrismPlanner.setup()
    assert isinstance(p, PrismPlanner)


def test_setup_passes_kwargs():
    p = PrismPlanner.setup(ollama_model="llama3", claude_api_key="key123", prefer_claude=True)
    assert p.ollama_model == "llama3"
    assert p.claude_api_key == "key123"
    assert p.prefer_claude is True


def test_prefer_claude_false_when_no_key():
    p = PrismPlanner(prefer_claude=True, claude_api_key=None)
    assert p.prefer_claude is False


def test_prefer_claude_true_when_key_provided():
    p = PrismPlanner(prefer_claude=True, claude_api_key="abc")
    assert p.prefer_claude is True


# ---------------------------------------------------------------------------
# _parse_json
# ---------------------------------------------------------------------------

def test_parse_json_plain():
    p = PrismPlanner()
    result = p._parse_json('{"a": 1}')
    assert result == {"a": 1}


def test_parse_json_fenced_json():
    p = PrismPlanner()
    result = p._parse_json('```json\n{"x": 2}\n```')
    assert result == {"x": 2}


def test_parse_json_fenced_no_lang():
    p = PrismPlanner()
    result = p._parse_json('```\n{"y": 3}\n```')
    assert result == {"y": 3}


def test_parse_json_embedded():
    p = PrismPlanner()
    result = p._parse_json('Some preamble {"z": 4} trailing text')
    assert result == {"z": 4}


def test_parse_json_empty_returns_none():
    assert PrismPlanner()._parse_json("") is None


def test_parse_json_invalid_returns_none():
    assert PrismPlanner()._parse_json("not json at all") is None


# ---------------------------------------------------------------------------
# _build_beam / _rank_strategies (pure engine — no LLM)
# ---------------------------------------------------------------------------

def test_build_beam_adds_all_planks():
    p = PrismPlanner()
    beam = p._build_beam(_MINIMAL_TASK_PROFILE, {})
    assert len(beam.planks) == len(_MINIMAL_TASK_PROFILE["strategies"])


def test_rank_strategies_sums_to_one():
    p = PrismPlanner()
    ranked = p._rank_strategies(_MINIMAL_TASK_PROFILE, {})
    total = sum(act for act, _ in ranked)
    assert abs(total - 1.0) < 1e-9


def test_rank_strategies_sorted_descending():
    p = PrismPlanner()
    ranked = p._rank_strategies(_MINIMAL_TASK_PROFILE, {})
    activations = [act for act, _ in ranked]
    assert activations == sorted(activations, reverse=True)


def test_rank_strategies_context_shifts_fulcrum():
    p = PrismPlanner()
    low  = p._rank_strategies(_MINIMAL_TASK_PROFILE, {"fitness_level": 0.0, "time_per_week": 0.0})
    high = p._rank_strategies(_MINIMAL_TASK_PROFILE, {"fitness_level": 1.0, "time_per_week": 1.0})
    low_beam  = PrismPlanner()._build_beam(_MINIMAL_TASK_PROFILE, {"fitness_level": 0.0, "time_per_week": 0.0})
    high_beam = PrismPlanner()._build_beam(_MINIMAL_TASK_PROFILE, {"fitness_level": 1.0, "time_per_week": 1.0})
    # Higher context values push fulcrum toward aggressive end
    assert high_beam.fulcrum.position() > low_beam.fulcrum.position()


def test_rank_strategies_returns_all_strategies():
    p = PrismPlanner()
    ranked = p._rank_strategies(_MINIMAL_TASK_PROFILE, {})
    assert len(ranked) == len(_MINIMAL_TASK_PROFILE["strategies"])


# ---------------------------------------------------------------------------
# _fallback_plan
# ---------------------------------------------------------------------------

def test_fallback_plan_returns_plan_of_action():
    p = PrismPlanner()
    result = p._fallback_plan("build a rocket")
    assert isinstance(result, PlanOfAction)
    assert result.task == "build a rocket"
    assert result.domain == "unknown"
    assert result.fulcrum_position == 0.5


def test_fallback_plan_recommended_flags_llm_unavailable():
    p = PrismPlanner()
    result = p._fallback_plan("task")
    assert "LLM unavailable" in result.recommended.risks[0]


def test_fallback_plan_has_single_stub_strategy():
    p = PrismPlanner()
    result = p._fallback_plan("task")
    assert len(result.all_strategies) == 1
    assert result.recommended is result.all_strategies[0]


# ---------------------------------------------------------------------------
# PlanOfAction helpers
# ---------------------------------------------------------------------------

def _make_strategy(name="A", activation=0.6, why="because") -> StrategyPlan:
    return StrategyPlan(
        name=name, position=0.5, activation=activation, expected_value=50,
        risk_score=20, steps=[], timeline="1 month",
        resources=[], expected_outcome="done", risks=[], why_recommended=why,
    )


def test_plan_of_action_top():
    strategies = [_make_strategy(name=str(i), activation=1/(i+1)) for i in range(6)]
    plan = PlanOfAction(
        task="t", domain="d", entity="e", timeline="30d", fulcrum_position=0.5,
        recommended=strategies[0], all_strategies=strategies, context_summary="ok",
    )
    assert plan.top(3) == strategies[:3]
    assert plan.top(1) == [strategies[0]]


def test_plan_of_action_top_default():
    strategies = [_make_strategy(name=str(i)) for i in range(5)]
    plan = PlanOfAction(
        task="t", domain="d", entity="e", timeline="30d", fulcrum_position=0.5,
        recommended=strategies[0], all_strategies=strategies, context_summary="ok",
    )
    assert len(plan.top()) == 3


def test_to_chat_response_contains_task():
    strategies = [_make_strategy(name="Plan A", activation=0.7, why="fits budget")]
    plan = PlanOfAction(
        task="run a marathon", domain="fitness", entity="individual",
        timeline="6 months", fulcrum_position=0.4,
        recommended=strategies[0], all_strategies=strategies,
        context_summary="Low fitness but motivated.",
    )
    text = plan.to_chat_response()
    assert "run a marathon" in text
    assert "Plan A" in text
    assert "70%" in text


def test_to_chat_response_marks_optimal():
    strategies = [_make_strategy(name="S1", why="w"), _make_strategy(name="S2", why="w")]
    plan = PlanOfAction(
        task="t", domain="d", entity="e", timeline="", fulcrum_position=0.5,
        recommended=strategies[0], all_strategies=strategies, context_summary="ctx",
    )
    text = plan.to_chat_response()
    assert "★ Optimal" in text
    assert "Alt 1" in text


# ---------------------------------------------------------------------------
# Full plan() with mocked LLM
# ---------------------------------------------------------------------------

def _make_planner_with_mock_llm(task_profile_json: str, action_plan_json: str) -> PrismPlanner:
    """Return a PrismPlanner whose _call_llm alternates between two responses."""
    planner = PrismPlanner()
    call_count = [0]

    def fake_call_llm(prompt: str) -> str:
        call_count[0] += 1
        if call_count[0] == 1:
            return task_profile_json
        return action_plan_json

    planner._call_llm = fake_call_llm  # type: ignore[method-assign]
    return planner


def test_plan_returns_plan_of_action():
    planner = _make_planner_with_mock_llm(
        json.dumps(_MINIMAL_TASK_PROFILE),
        json.dumps(_MINIMAL_ACTION_PLAN),
    )
    result = planner.plan("run a marathon")
    assert isinstance(result, PlanOfAction)


def test_plan_task_preserved():
    planner = _make_planner_with_mock_llm(
        json.dumps(_MINIMAL_TASK_PROFILE),
        json.dumps(_MINIMAL_ACTION_PLAN),
    )
    result = planner.plan("run a marathon")
    assert result.task == "run a marathon"


def test_plan_domain_from_profile():
    planner = _make_planner_with_mock_llm(
        json.dumps(_MINIMAL_TASK_PROFILE),
        json.dumps(_MINIMAL_ACTION_PLAN),
    )
    result = planner.plan("run a marathon")
    assert result.domain == "fitness"


def test_plan_recommended_has_steps():
    planner = _make_planner_with_mock_llm(
        json.dumps(_MINIMAL_TASK_PROFILE),
        json.dumps(_MINIMAL_ACTION_PLAN),
    )
    result = planner.plan("run a marathon")
    assert len(result.recommended.steps) == 1
    assert isinstance(result.recommended.steps[0], ActionStep)


def test_plan_all_strategies_count():
    planner = _make_planner_with_mock_llm(
        json.dumps(_MINIMAL_TASK_PROFILE),
        json.dumps(_MINIMAL_ACTION_PLAN),
    )
    result = planner.plan("run a marathon", n_plans=2)
    assert len(result.all_strategies) == len(_MINIMAL_TASK_PROFILE["strategies"])


def test_plan_strategies_beyond_n_plans_have_no_steps():
    planner = _make_planner_with_mock_llm(
        json.dumps(_MINIMAL_TASK_PROFILE),
        json.dumps(_MINIMAL_ACTION_PLAN),
    )
    result = planner.plan("run a marathon", n_plans=1)
    for s in result.all_strategies[1:]:
        assert s.steps == []


def test_plan_fulcrum_position_in_unit_interval():
    planner = _make_planner_with_mock_llm(
        json.dumps(_MINIMAL_TASK_PROFILE),
        json.dumps(_MINIMAL_ACTION_PLAN),
    )
    result = planner.plan("run a marathon")
    assert 0.0 <= result.fulcrum_position <= 1.0


def test_plan_uses_identity_profile():
    planner = PrismPlanner()
    call_count = [0]

    def fake_call_llm(prompt: str) -> str:
        call_count[0] += 1
        if call_count[0] == 1:
            return json.dumps(_MINIMAL_TASK_PROFILE)
        return json.dumps(_MINIMAL_ACTION_PLAN)

    planner._call_llm = fake_call_llm  # type: ignore[method-assign]
    identity = {"domains": {"fitness": {"value": 0.8}}}
    result = planner.plan("run a marathon", identity_profile=identity)
    assert isinstance(result, PlanOfAction)


def test_plan_fallback_when_llm_returns_empty():
    planner = PrismPlanner()
    planner._call_llm = lambda _: ""  # type: ignore[method-assign]
    result = planner.plan("some task")
    assert isinstance(result, PlanOfAction)
    assert result.domain == "unknown"


def test_plan_fallback_when_llm_returns_invalid_json():
    planner = PrismPlanner()
    planner._call_llm = lambda _: "not json"  # type: ignore[method-assign]
    result = planner.plan("some task")
    assert isinstance(result, PlanOfAction)
    assert result.domain == "unknown"


# ---------------------------------------------------------------------------
# _call_ollama
# ---------------------------------------------------------------------------

def test_call_ollama_returns_response_field():
    planner = PrismPlanner()
    payload = json.dumps({"response": "hello"}).encode()
    with patch.object(urllib.request, "urlopen", return_value=_MockResponse(payload)):
        result = planner._call_ollama("prompt")
    assert result == "hello"


def test_call_ollama_returns_empty_on_network_error():
    planner = PrismPlanner()
    with patch.object(urllib.request, "urlopen", side_effect=urllib.error.URLError("offline")):
        result = planner._call_ollama("prompt")
    assert result == ""


# ---------------------------------------------------------------------------
# _call_claude
# ---------------------------------------------------------------------------

def test_call_claude_sends_api_key_header():
    planner = PrismPlanner(claude_api_key="my-secret-key", prefer_claude=True)
    captured = {}

    def fake_urlopen(request, timeout=0):
        captured["headers"] = dict(request.header_items())
        body = {"content": [{"type": "text", "text": "response text"}]}
        return _MockResponse(json.dumps(body).encode())

    with patch.object(urllib.request, "urlopen", side_effect=fake_urlopen):
        result = planner._call_claude("prompt")

    assert result == "response text"
    lowered = {k.lower(): v for k, v in captured["headers"].items()}
    assert lowered["x-api-key"] == "my-secret-key"


def test_call_claude_falls_back_to_ollama_on_error():
    planner = PrismPlanner(claude_api_key="key", prefer_claude=True)
    ollama_payload = json.dumps({"response": "ollama fallback"}).encode()

    call_count = [0]

    def fake_urlopen(request, timeout=0):
        call_count[0] += 1
        if "anthropic" in request.full_url:
            raise urllib.error.URLError("claude offline")
        return _MockResponse(ollama_payload)

    with patch.object(urllib.request, "urlopen", side_effect=fake_urlopen):
        result = planner._call_claude("prompt")

    assert result == "ollama fallback"


def test_call_llm_routes_to_claude_when_prefer_claude():
    planner = PrismPlanner(claude_api_key="key", prefer_claude=True)
    called_with = []

    def fake_claude(prompt):
        called_with.append("claude")
        return "{}"

    planner._call_claude = fake_claude  # type: ignore[method-assign]
    planner._call_llm("any prompt")
    assert called_with == ["claude"]


def test_call_llm_routes_to_ollama_when_no_key():
    planner = PrismPlanner(prefer_claude=True, claude_api_key=None)
    called_with = []

    def fake_ollama(prompt):
        called_with.append("ollama")
        return "{}"

    planner._call_ollama = fake_ollama  # type: ignore[method-assign]
    planner._call_llm("any prompt")
    assert called_with == ["ollama"]
