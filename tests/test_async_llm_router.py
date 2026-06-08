"""Tests for async LLM router methods — fully additive, no sync code changed."""
from __future__ import annotations

import asyncio
from unittest.mock import patch

from prism_llm_router import LLMOption, LLMRouter


def test_async_call_fallback_when_httpx_absent():
    """async_call() must fall back to asyncio.to_thread(call()) when httpx is absent."""
    router = LLMRouter()
    with patch.object(router, "call", return_value=("mocked response", "mock/model")) as mock_call, \
         patch("prism_llm_router._HTTPX_AVAILABLE", False):
        result = asyncio.run(router.async_call("hello"))
    assert result == ("mocked response", "mock/model")
    mock_call.assert_called_once()


def test_async_call_no_providers():
    """async_call() returns ('', 'none') when no providers are available."""
    router = LLMRouter()
    with patch.object(router, "best", return_value=None):
        result = asyncio.run(router.async_call("hello"))
    assert result == ("", "none")


def test_async_call_uses_httpx_when_available():
    """async_call() dispatches through _async_call_option when httpx is present."""
    router = LLMRouter()
    opt = LLMOption("ollama", "mistral", "http://localhost:11434", True, 10.0, 2)

    async def mock_option(*args, **kwargs):
        return "token response"

    with patch.object(router, "best", return_value=opt), \
         patch("prism_llm_router._HTTPX_AVAILABLE", True):
        router._async_call_option = mock_option
        result = asyncio.run(router.async_call("hello", min_capability=1))
    assert result == ("token response", "ollama/mistral")


def test_async_call_stream_fallback_no_httpx():
    """async_call_stream() falls back to yielding full async_call() result when httpx absent."""
    router = LLMRouter()

    async def _collect():
        tokens = []
        with patch.object(router, "async_call", return_value=("full response", "mock/model")), \
             patch("prism_llm_router._HTTPX_AVAILABLE", False):
            async for token in router.async_call_stream("hello"):
                tokens.append(token)
        return tokens

    tokens = asyncio.run(_collect())
    assert tokens == ["full response"]


def test_async_call_stream_no_providers():
    """async_call_stream() yields nothing when no providers match."""
    router = LLMRouter()

    async def _collect():
        tokens = []
        with patch.object(router, "best", return_value=None), \
             patch("prism_llm_router._HTTPX_AVAILABLE", True):
            async for token in router.async_call_stream("hello"):
                tokens.append(token)
        return tokens

    tokens = asyncio.run(_collect())
    assert tokens == []


def test_sync_call_unchanged():
    """Verify sync call() signature and return type are unaffected."""
    router = LLMRouter()
    with patch.object(router, "_call_option", return_value="sync response"), \
         patch.object(router, "discover", return_value=[
             LLMOption("ollama", "mistral", "http://localhost:11434", True, 5.0, 2)
         ]):
        result = router.call("hello", min_capability=2)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == "sync response"
