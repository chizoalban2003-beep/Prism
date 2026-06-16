from __future__ import annotations

import time

from prism_llm_router import LLMOption, LLMRouter


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
    # Force the stdlib fallback so this test only checks return shape and
    # never hits a real LLM. Without _last_scan set to "now", discover() would
    # re-scan on every call (cache window 60 s, sentinel is 0.0) and hit
    # whatever Ollama happens to be running on the host.
    router = LLMRouter()
    router._options = []
    router._discovered = True
    router._last_scan = time.time()
    result = router.call("ping")
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], str)
    assert isinstance(result[1], str)
