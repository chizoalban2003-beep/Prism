"""Smoke tests for the `offline_llm` fixture itself."""
from __future__ import annotations

import pytest


def test_offline_llm_blocks_router_calls(offline_llm):
    from prism_llm_router import LLMRouter
    r = LLMRouter()
    text, model = r.call("hello", min_capability=1)
    assert text == ""
    assert model == "stdlib/stdlib"
    assert offline_llm.calls and offline_llm.calls[0][0] == "hello"


def test_offline_llm_queue_and_default_reply(offline_llm):
    from prism_llm_router import LLMRouter
    offline_llm.set_reply("default")
    offline_llm.queue_reply("first")
    r = LLMRouter()
    assert r.call("a")[0] == "first"
    assert r.call("b")[0] == "default"


def test_offline_llm_blocks_urlopen_in_router_module(offline_llm):
    import prism_llm_router
    with pytest.raises(RuntimeError, match="offline_llm"):
        prism_llm_router.urllib.request.urlopen("http://example.com")
