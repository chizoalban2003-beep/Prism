"""Tests for PrismChain.run_streaming() and step_callback"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from prism_chain import ChainStep, PrismChain


def _make_router(responses: list[str]) -> MagicMock:
    router = MagicMock()
    router.call.side_effect = [(r, {}) for r in responses]
    return router


def _noop_execute(intent: str, message: str, ctx: dict):
    from prism_responses import text_card
    return text_card(f"result for {intent}", intent)


# ── step_callback ─────────────────────────────────────────────────────────────

def test_step_callback_called_on_each_step():
    step1 = json.dumps({
        "done": False,
        "next_logic": "web_search",
        "next_message": "search query",
        "reasoning": "need info",
    })
    done_resp = json.dumps({
        "done": True,
        "answer": "Final answer here",
        "reasoning": "done",
    })
    # Provide enough responses for decision + evaluator calls
    router = MagicMock()
    router.call.side_effect = [(step1, {})] + [(done_resp, {})] * 20
    chain = PrismChain(llm_router=router)

    steps_received = []
    chain.run(
        "find something",
        _noop_execute,
        {},
        step_callback=lambda s: steps_received.append(s),
    )
    assert len(steps_received) >= 1
    assert all(isinstance(s, ChainStep) for s in steps_received)


def test_step_callback_not_called_when_none():
    done_resp = json.dumps({
        "done": True,
        "answer": "Immediate answer",
        "reasoning": "done",
    })
    router = _make_router([done_resp])
    chain = PrismChain(llm_router=router)
    # Should not raise even when callback=None (default)
    card = chain.run("quick question", _noop_execute, {})
    assert card is not None


def test_step_callback_exception_does_not_abort_chain():
    step1 = json.dumps({
        "done": False,
        "next_logic": "web_search",
        "next_message": "query",
        "reasoning": "reason",
    })
    done_resp = json.dumps({
        "done": True,
        "answer": "Final answer",
        "reasoning": "done",
    })
    router = MagicMock()
    router.call.side_effect = [(step1, {})] + [(done_resp, {})] * 20
    chain = PrismChain(llm_router=router)

    def bad_callback(step):
        raise RuntimeError("callback failed")

    card = chain.run("find something", _noop_execute, {}, step_callback=bad_callback)
    assert card is not None  # chain completed despite callback error


# ── run_streaming() ───────────────────────────────────────────────────────────

def test_run_streaming_yields_done_event():
    done_resp = json.dumps({
        "done": True,
        "answer": "Streaming final answer",
        "reasoning": "done",
    })
    router = _make_router([done_resp])
    chain = PrismChain(llm_router=router)

    events = list(chain.run_streaming("quick question", _noop_execute, {}))
    done_events = [e for e in events if e.get("event") == "done"]
    assert len(done_events) == 1
    assert done_events[0]["answer"] != ""


def test_run_streaming_yields_step_events():
    step1 = json.dumps({
        "done": False,
        "next_logic": "web_search",
        "next_message": "search query",
        "reasoning": "need info",
    })
    done_resp = json.dumps({
        "done": True,
        "answer": "Final answer",
        "reasoning": "done",
    })
    router = MagicMock()
    router.call.side_effect = [(step1, {})] + [(done_resp, {})] * 20
    chain = PrismChain(llm_router=router)

    events = list(chain.run_streaming("complex task", _noop_execute, {}))
    step_events = [e for e in events if e.get("event") == "step"]
    assert len(step_events) >= 1
    for e in step_events:
        assert "logic" in e
        assert "step" in e


def test_run_streaming_no_router_yields_error():
    chain = PrismChain(llm_router=None)
    events = list(chain.run_streaming("something", _noop_execute, {}))
    # chain.run returns None when no router — streaming should yield error or done
    assert any(e.get("event") in ("error", "done") for e in events)


def test_run_streaming_events_are_json_serialisable():
    done_resp = json.dumps({
        "done": True,
        "answer": "Serialisable answer",
        "reasoning": "done",
    })
    router = _make_router([done_resp])
    chain = PrismChain(llm_router=router)
    events = list(chain.run_streaming("test", _noop_execute, {}))
    for e in events:
        json.dumps(e, default=str)  # must not raise


# ── outcome_tracker integration ───────────────────────────────────────────────

def test_chain_records_outcome_on_done():
    done_resp = json.dumps({
        "done": True,
        "answer": "Final answer",
        "reasoning": "done",
    })
    router = _make_router([done_resp])
    tracker = MagicMock()
    chain = PrismChain(llm_router=router, outcome_tracker=tracker)
    chain.run("test goal", _noop_execute, {})
    tracker.record.assert_called_once()
    _, kwargs = tracker.record.call_args
    assert kwargs["outcome"] == "done"


def test_chain_records_outcome_on_abandoned():
    # Force chain to run out of steps by making LLM never return done=True
    never_done = json.dumps({
        "done": False,
        "next_logic": "web_search",
        "next_message": "keep searching",
        "reasoning": "not done",
    })
    router = MagicMock()
    router.call.return_value = (never_done, {})
    tracker = MagicMock()
    chain = PrismChain(llm_router=router, outcome_tracker=tracker)
    chain.run("neverending task", _noop_execute, {})
    tracker.record.assert_called_once()
    _, kwargs = tracker.record.call_args
    assert kwargs["outcome"] == "abandoned"


def test_chain_context_id_passed_to_outcome_tracker():
    done_resp = json.dumps({"done": True, "answer": "ok", "reasoning": ""})
    router = _make_router([done_resp])
    tracker = MagicMock()
    chain = PrismChain(llm_router=router, outcome_tracker=tracker, context_id="work")
    chain.run("task", _noop_execute, {})
    _, kwargs = tracker.record.call_args
    assert kwargs["context_id"] == "work"
