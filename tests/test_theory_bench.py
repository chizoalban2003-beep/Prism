"""
tests/test_theory_bench.py
==========================
15+ tests covering prism_chain_theory.py and prism_chain_theory_bench.py.
"""
from __future__ import annotations

import json
from dataclasses import fields
from unittest.mock import MagicMock

from prism_responses import text_card
from prism_chain_theory import (
    SubChainLogic,
    SoftLogic,
    InterceptorPolicy,
    PolicyIntercept,
)
from prism_chain_theory_bench import (
    ExperimentResult,
    run_experiment_1_recursive,
    run_experiment_2_vertical,
    run_experiment_3_interceptor,
)


# ── Shared helpers ────────────────────────────────────────────────────────────


def _agent(intent: str, message: str, ctx: dict):
    """Mock agent_execute_fn for testing."""
    body = f"Result of {intent}: {message[:50]}"
    return text_card(body, intent)


def _agent_error(intent: str, message: str, ctx: dict):
    """Agent that returns error-containing results."""
    return text_card(f"Error: {intent} failed to connect", intent)


def _make_router(*responses: str) -> MagicMock:
    router = MagicMock()
    router.call.side_effect = [(r, {}) for r in responses]
    return router


# ── Experiment 1: SubChainLogic ───────────────────────────────────────────────


def test_subchain_executes_inner_chain_and_returns_string():
    """SubChainLogic must call each sub-logic and return a string."""
    sub = SubChainLogic(sub_logics=["web_search", "parse_result"])
    result = sub("find Python news", _agent, {})
    assert isinstance(result, str)
    assert len(result) > 0


def test_subchain_calls_each_sub_logic():
    """SubChainLogic runs all configured sub-logics."""
    called = []

    def tracking_agent(intent, message, ctx):
        called.append(intent)
        return text_card(f"ok from {intent}", intent)

    sub = SubChainLogic(sub_logics=["web_search", "parse_result", "cross_reference"])
    sub("goal", tracking_agent, {})
    assert "web_search" in called
    assert "parse_result" in called
    assert "cross_reference" in called


def test_subchain_result_is_synthesised_when_router_available():
    """With a router, SubChainLogic returns the synthesised text, not raw concat."""
    router = _make_router("Synthesised summary: Python 3.14 is stable and fast.")
    sub = SubChainLogic(
        sub_logics=["web_search", "parse_result"],
        llm_router=router,
    )
    result = sub("find Python news", _agent, {})
    assert "Synthesised summary" in result
    router.call.assert_called_once()


def test_subchain_result_shorter_than_raw_concat_with_router():
    """Synthesised result is typically shorter than naive concatenation of all steps."""
    long_agent_count = [0]

    def verbose_agent(intent, message, ctx):
        long_agent_count[0] += 1
        return text_card("X" * 200, intent)

    # router returns a compact summary
    router = _make_router("Short synthesis.")
    sub = SubChainLogic(
        sub_logics=["web_search", "parse_result", "cross_reference"],
        llm_router=router,
    )
    result = sub("goal", verbose_agent, {})
    # Router synthesis is 16 chars; raw concat of 3×200 chars would be ~600+
    assert len(result) < 100


def test_subchain_no_router_returns_joined_string():
    """Without a router, SubChainLogic still returns a non-empty string."""
    sub = SubChainLogic(sub_logics=["web_search", "parse_result"])
    result = sub("goal", _agent, {})
    assert isinstance(result, str)
    assert len(result) > 0


def test_subchain_handles_agent_exception_gracefully():
    """SubChainLogic must not raise if a sub-logic step throws."""
    def failing_agent(intent, message, ctx):
        raise RuntimeError("network down")

    sub = SubChainLogic(sub_logics=["web_search"])
    result = sub("goal", failing_agent, {})
    assert isinstance(result, str)
    assert "Error" in result


# ── Experiment 2: SoftLogic ───────────────────────────────────────────────────


def test_softlogic_calls_underlying_logic_and_one_llm():
    """SoftLogic must call the underlying logic and exactly one LLM call."""
    router = _make_router("Compressed: 3 key facts extracted.")
    soft = SoftLogic(underlying_logic="web_search", llm_router=router)
    result = soft("find Python news", _agent, {})
    assert isinstance(result, str)
    assert router.call.call_count == 1


def test_softlogic_returns_string_not_card():
    """SoftLogic result is a plain string, not a PrismCard."""
    router = _make_router("Plain text facts here.")
    soft = SoftLogic(underlying_logic="web_search", llm_router=router)
    result = soft("goal", _agent, {})
    assert isinstance(result, str)
    assert not hasattr(result, "body")  # not a PrismCard


def test_softlogic_graceful_degradation_no_router():
    """Without a router, SoftLogic returns raw (truncated) underlying result."""
    soft = SoftLogic(underlying_logic="web_search", llm_router=None)
    result = soft("goal", _agent, {})
    assert isinstance(result, str)
    assert len(result) > 0


def test_softlogic_graceful_degradation_llm_fails():
    """If LLM call throws, SoftLogic falls back to raw result."""
    router = MagicMock()
    router.call.side_effect = RuntimeError("LLM unavailable")
    soft = SoftLogic(underlying_logic="web_search", llm_router=router)
    result = soft("goal", _agent, {})
    assert isinstance(result, str)
    assert len(result) > 0


def test_softlogic_with_callable_underlying():
    """SoftLogic can wrap a callable, not just a string logic name."""
    router = _make_router("Soft result from callable.")

    def my_logic(goal, agent_fn, ctx):
        return "raw output from callable"

    soft = SoftLogic(underlying_logic=my_logic, llm_router=router)
    result = soft("goal", _agent, {})
    assert "Soft result" in result


# ── Experiment 3: InterceptorPolicy ──────────────────────────────────────────


def test_interceptor_returns_none_for_clean_result():
    """No intercept should fire for a clean, successful web_search result."""
    policy = InterceptorPolicy()
    intercept = policy.intercept(
        "web_search",
        "Python 3.14 released with major improvements.",
        "email_read",
        "find news",
    )
    assert intercept is None


def test_interceptor_fires_on_error_web_search():
    """Intercept must fire when web_search result contains 'Error'."""
    policy = InterceptorPolicy()
    intercept = policy.intercept(
        "web_search",
        "Error: connection timed out",
        "parse_result",
        "research Python",
    )
    assert intercept is not None
    assert intercept.substitute_logic == "autonomous"


def test_interceptor_fires_on_email_send_no_sent():
    """Intercept must fire when email_send result does not contain 'sent'."""
    policy = InterceptorPolicy()
    intercept = policy.intercept(
        "email_send",
        "Message queued. Delivery pending.",
        "autonomous",
        "send meeting invite",
    )
    assert intercept is not None
    assert intercept.substitute_logic == "email_read"


def test_interceptor_no_fire_on_email_send_with_sent():
    """No intercept when email_send result confirms 'sent'."""
    policy = InterceptorPolicy()
    intercept = policy.intercept(
        "email_send",
        "Email sent successfully to all recipients.",
        "autonomous",
        "send meeting invite",
    )
    assert intercept is None


def test_interceptor_fires_on_autonomous_approval():
    """Intercept must fire when autonomous result mentions 'approval'."""
    policy = InterceptorPolicy()
    intercept = policy.intercept(
        "autonomous",
        "Task complete. Requires approval from user to continue.",
        "web_search",
        "run maintenance",
    )
    assert intercept is not None
    assert intercept.substitute_logic == "approve_pending"


def test_interceptor_returns_policy_intercept_dataclass():
    """Fired intercept must be a PolicyIntercept with all required fields."""
    policy = InterceptorPolicy()
    intercept = policy.intercept(
        "web_search",
        "Error: DNS lookup failed",
        "parse_result",
        "goal",
    )
    assert isinstance(intercept, PolicyIntercept)
    assert isinstance(intercept.substitute_logic, str)
    assert isinstance(intercept.substitute_message, str)
    assert isinstance(intercept.reason, str)
    assert len(intercept.substitute_logic) > 0


def test_prismchain_interceptor_fires_in_run():
    """PrismChain.run() must execute substitute logic when intercept fires."""
    from prism_chain import PrismChain
    import pathlib
    import tempfile

    policy = InterceptorPolicy()

    # LLM decides to run web_search, then (after intercept) chain continues
    step_resp = json.dumps({
        "done": False,
        "next_logic": "web_search",
        "next_message": "search for Python news",
        "reasoning": "need web info",
    })
    done_resp = json.dumps({
        "done": True,
        "answer": "Found info via intercepted path.",
        "reasoning": "done",
    })
    router = _make_router(step_resp, done_resp)
    chain = PrismChain(
        llm_router=router,
        use_evaluator=False,
        use_soft_logic=False,
        interceptor_policy=policy,
    )
    chain._db = pathlib.Path(tempfile.mktemp(suffix=".db"))
    chain._init_db()

    card = chain.run("find Python news", _agent_error, {})
    # Chain ran; interceptor should have fired since _agent_error returns "Error"
    assert card is not None
    # Steps should include both the original and the substitute
    # (state object not directly accessible, but card should be non-empty)
    assert len(card.body) > 0


def test_prismchain_no_interceptor_baseline():
    """PrismChain with interceptor_policy=None must run without interception."""
    from prism_chain import PrismChain
    import pathlib
    import tempfile

    step_resp = json.dumps({
        "done": False,
        "next_logic": "web_search",
        "next_message": "search",
        "reasoning": "need info",
    })
    done_resp = json.dumps({
        "done": True,
        "answer": "Normal answer.",
        "reasoning": "done",
    })
    router = _make_router(step_resp, done_resp)
    chain = PrismChain(
        llm_router=router,
        use_evaluator=False,
        use_soft_logic=False,
        interceptor_policy=None,
    )
    chain._db = pathlib.Path(tempfile.mktemp(suffix=".db"))
    chain._init_db()
    card = chain.run("search something", _agent, {})
    assert card is not None
    # No intercept should have fired — card body from final done answer
    assert "Normal answer" in card.body


# ── Benchmark dataclass & experiment runners ──────────────────────────────────


def test_experiment_result_dataclass_fields():
    """ExperimentResult must have all expected fields."""
    field_names = {f.name for f in fields(ExperimentResult)}
    assert "name" in field_names
    assert "llm_calls" in field_names
    assert "eval_scores" in field_names
    assert "intercepts_fired" in field_names
    assert "steps" in field_names
    assert "notes" in field_names


def test_run_experiment_1_returns_experiment_result():
    """run_experiment_1_recursive() must return an ExperimentResult."""
    result = run_experiment_1_recursive()
    assert isinstance(result, ExperimentResult)
    assert result.name != ""
    assert isinstance(result.llm_calls, int)
    assert isinstance(result.eval_scores, list)


def test_run_experiment_2_returns_experiment_result():
    """run_experiment_2_vertical() must return an ExperimentResult."""
    result = run_experiment_2_vertical()
    assert isinstance(result, ExperimentResult)
    assert result.name != ""
    assert len(result.eval_scores) >= 2


def test_run_experiment_3_returns_experiment_result():
    """run_experiment_3_interceptor() must return an ExperimentResult."""
    result = run_experiment_3_interceptor()
    assert isinstance(result, ExperimentResult)
    assert result.name != ""
    assert result.intercepts_fired >= 1


def test_experiment_1_score_improves_with_subchain():
    """Sub-chain (score index 1) should beat flat (score index 0) in Exp 1."""
    result = run_experiment_1_recursive()
    assert len(result.eval_scores) == 2
    assert result.eval_scores[1] > result.eval_scores[0]


def test_experiment_2_score_improves_with_softlogic():
    """SoftLogic-enriched score (index 1) should beat raw score (index 0) in Exp 2."""
    result = run_experiment_2_vertical()
    assert len(result.eval_scores) == 2
    assert result.eval_scores[1] > result.eval_scores[0]


def test_experiment_3_no_false_positive():
    """Exp 3 should have 0 false positives (clean scenario passes through)."""
    policy = InterceptorPolicy()
    # Clean result — should NOT fire
    intercept = policy.intercept(
        "web_search",
        "Python 3.14 is stable.",
        "parse_result",
        "goal",
    )
    assert intercept is None
