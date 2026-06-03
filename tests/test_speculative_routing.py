"""Tests for LLMRouter speculative=True routing."""
from __future__ import annotations

from unittest.mock import MagicMock

from prism_llm_router import LLMOption, LLMRouter


def _router_with_mock(responses: list[tuple[str, str]]) -> LLMRouter:
    import time
    router = LLMRouter()
    it = iter(responses)
    router._call_option = MagicMock(side_effect=lambda opt, *a, **kw: next(it)[0])
    opt1 = LLMOption(provider="ollama", model="qwen", endpoint="", available=True, capability=1)
    opt2 = LLMOption(provider="claude", model="claude-sonnet", endpoint="", available=True, capability=3)
    router._options    = [opt2, opt1]
    router._discovered = True
    router._last_scan  = time.time()  # keep cache valid so discover() won't re-probe
    return router


_LONG_CONFIDENT = (
    "The capital of France is Paris, a major European city situated on the Seine river "
    "in the north-central part of the country. It has been the capital since the late "
    "tenth century and is home to landmarks such as the Eiffel Tower and the Louvre."
)


def test_speculative_returns_fast_when_confident():
    router = _router_with_mock([
        (_LONG_CONFIDENT, "ollama/qwen"),
        ("Paris is the capital.", "claude/sonnet"),  # should not be called
    ])
    resp, model = router.call("What is the capital of France?", min_capability=2, speculative=True)
    assert "Paris" in resp
    assert router._call_option.call_count == 1  # only fast model called


def test_speculative_escalates_on_uncertainty():
    router = _router_with_mock([
        ("I don't know the answer to this question.", "ollama/qwen"),
        ("The answer is 42.", "claude/sonnet"),
    ])
    resp, model = router.call("Complex question?", min_capability=2, speculative=True)
    assert router._call_option.call_count == 2  # escalated
    assert resp == "The answer is 42."


def test_speculative_escalates_on_short_response():
    router = _router_with_mock([
        ("Short.", "ollama/qwen"),
        (_LONG_CONFIDENT, "claude/sonnet"),
    ])
    resp, _ = router.call("?", min_capability=2, speculative=True)
    assert router._call_option.call_count == 2


def test_speculative_false_bypasses_fast_model():
    router = _router_with_mock([
        (_LONG_CONFIDENT, "claude/sonnet"),
    ])
    resp, _ = router.call("Q", min_capability=2, speculative=False)
    assert router._call_option.call_count == 1


def test_speculative_ignored_when_min_capability_1():
    router = _router_with_mock([
        (_LONG_CONFIDENT, "ollama/qwen"),
    ])
    resp, _ = router.call("Q", min_capability=1, speculative=True)
    assert router._call_option.call_count == 1
