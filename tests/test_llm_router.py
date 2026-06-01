from __future__ import annotations

from prism_llm_router import LLMRouter, LLMOption


def test_discover_non_empty():
    router = LLMRouter()
    options = router.discover()
    assert len(options) > 0


def test_stdlib_available():
    router = LLMRouter()
    options = router.discover()
    providers = [o.provider for o in options]
    assert "stdlib" in providers


def test_best_returns_none_or_option():
    router = LLMRouter()
    result = router.best()
    assert result is None or isinstance(result, LLMOption)


def test_status_has_best_key():
    router = LLMRouter()
    summary = router.status_summary()
    assert "best" in summary


def test_call_tuple():
    router = LLMRouter()
    result = router.call("ping")
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], str)
    assert isinstance(result[1], str)
