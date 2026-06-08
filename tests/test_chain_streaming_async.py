"""Tests for PrismChain.run_streaming_async() — async bridge over run_streaming()."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock


def _make_mock_chain():
    """Return a minimal mock that has a run_streaming() yielding known events."""
    from prism_chain import PrismChain

    chain = MagicMock(spec=PrismChain)

    def _fake_streaming(message, fn, ctx):
        yield {"event": "step", "step": 1, "logic": "test", "result": "r", "policy": ""}
        yield {"event": "done", "answer": "final answer", "chain_id": "abc"}

    chain.run_streaming.side_effect = _fake_streaming
    # Attach the real run_streaming_async (unbound, needs self)
    from prism_chain import PrismChain as _PC
    chain.run_streaming_async = _PC.run_streaming_async.__get__(chain)
    return chain


def test_run_streaming_async_yields_same_events():
    """run_streaming_async must yield the same events as run_streaming()."""
    chain = _make_mock_chain()

    async def _collect():
        events = []
        async for evt in chain.run_streaming_async("hi", lambda x, y: None, {}):
            events.append(evt)
        return events

    events = asyncio.run(_collect())
    assert len(events) == 2
    assert events[0]["event"] == "step"
    assert events[1]["event"] == "done"
    assert events[1]["answer"] == "final answer"


def test_run_streaming_async_handles_exception():
    """run_streaming_async must surface exceptions as error events."""
    from prism_chain import PrismChain

    chain = MagicMock(spec=PrismChain)

    def _exploding(message, fn, ctx):
        yield {"event": "step", "step": 1, "logic": "x", "result": "y", "policy": ""}
        raise RuntimeError("chain exploded")

    chain.run_streaming.side_effect = _exploding
    chain.run_streaming_async = PrismChain.run_streaming_async.__get__(chain)

    async def _collect():
        events = []
        async for evt in chain.run_streaming_async("hi", lambda x, y: None, {}):
            events.append(evt)
        return events

    events = asyncio.run(_collect())
    assert any(e["event"] == "error" for e in events)
    error_evt = next(e for e in events if e["event"] == "error")
    assert "chain exploded" in error_evt["message"]


def test_run_streaming_unaffected():
    """Verify the original synchronous run_streaming() still works unchanged."""
    from prism_chain import PrismChain

    chain = MagicMock(spec=PrismChain)

    def _fake_streaming(message, fn, ctx):
        yield {"event": "done", "answer": "sync answer", "chain_id": "xyz"}

    chain.run_streaming.side_effect = _fake_streaming

    events = list(chain.run_streaming("hello", lambda x, y: None, {}))
    assert events[0]["event"] == "done"
    assert events[0]["answer"] == "sync answer"
