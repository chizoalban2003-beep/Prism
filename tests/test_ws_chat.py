"""
test_ws_chat.py
===============
Tests for the /ws/chat WebSocket endpoint.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prism_asgi import app
from prism_state import _set_state

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_streaming_agent(events=None):
    """Return a mock agent whose _chain streams the given events."""
    from prism_chain import PrismChain

    if events is None:
        events = [
            {"event": "step", "step": 1, "logic": "test", "result": "r", "policy": ""},
            {"event": "done", "answer": "ws answer", "chain_id": "abc",
             "card_type": "text", "card_data": {}, "card_title": ""},
        ]

    agent = MagicMock()
    agent._execute = MagicMock(return_value="ok")

    chain = MagicMock(spec=PrismChain)

    captured_events = list(events)

    def _fake_streaming(message, fn, ctx):
        yield from captured_events

    chain.run_streaming.side_effect = _fake_streaming
    chain.run_streaming_async = PrismChain.run_streaming_async.__get__(chain)
    agent._chain = chain

    return agent


@pytest.fixture()
def client():
    agent = _make_streaming_agent()
    _set_state(agent=agent)
    return TestClient(app)


@pytest.fixture()
def no_agent_client():
    _set_state(agent=None)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------

class TestWsConnection:

    def test_connects_successfully(self, client):
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"message": "hello"})
            events = []
            while True:
                evt = ws.receive_json()
                events.append(evt)
                if evt.get("event") in ("done", "error"):
                    break
        assert any(e["event"] == "done" for e in events)

    def test_agent_not_ready_closes_with_error(self, no_agent_client):
        with no_agent_client.websocket_connect("/ws/chat") as ws:
            evt = ws.receive_json()
        assert evt["event"] == "error"
        assert "agent not ready" in evt["message"]

    def test_empty_message_returns_error_and_stays_open(self, client):
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"message": ""})
            evt = ws.receive_json()
            assert evt["event"] == "error"
            assert "'message' required" in evt["message"]
            # Connection is still open — can send a valid message
            ws.send_json({"message": "hello"})
            events = []
            while True:
                e = ws.receive_json()
                events.append(e)
                if e.get("event") in ("done", "error"):
                    break
        assert any(e["event"] == "done" for e in events)

    def test_q_alias_accepted(self, client):
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"q": "hello via q"})
            events = []
            while True:
                evt = ws.receive_json()
                events.append(evt)
                if evt.get("event") in ("done", "error"):
                    break
        assert any(e["event"] == "done" for e in events)


# ---------------------------------------------------------------------------
# Event streaming
# ---------------------------------------------------------------------------

class TestWsStreaming:

    def test_streams_step_events_before_done(self, client):
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"message": "hi"})
            events = []
            while True:
                evt = ws.receive_json()
                events.append(evt)
                if evt.get("event") in ("done", "error"):
                    break
        types = [e["event"] for e in events]
        assert "step" in types
        assert types[-1] == "done"

    def test_done_event_contains_answer(self, client):
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"message": "what is 2+2?"})
            events = []
            while True:
                evt = ws.receive_json()
                events.append(evt)
                if evt.get("event") in ("done", "error"):
                    break
        done = next(e for e in events if e["event"] == "done")
        assert done["answer"] == "ws answer"

    def test_step_event_shape(self, client):
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"message": "test"})
            events = []
            while True:
                evt = ws.receive_json()
                events.append(evt)
                if evt.get("event") in ("done", "error"):
                    break
        step = next(e for e in events if e["event"] == "step")
        assert "step" in step
        assert "logic" in step
        assert "result" in step

    def test_error_event_on_chain_failure(self):
        from prism_chain import PrismChain

        agent = MagicMock()
        agent._execute = MagicMock()
        chain = MagicMock(spec=PrismChain)

        def _explode(msg, fn, ctx):
            raise RuntimeError("chain boom")

        chain.run_streaming.side_effect = _explode
        chain.run_streaming_async = PrismChain.run_streaming_async.__get__(chain)
        agent._chain = chain

        _set_state(agent=agent)
        c = TestClient(app)
        with c.websocket_connect("/ws/chat") as ws:
            ws.send_json({"message": "crash please"})
            evt = ws.receive_json()
        assert evt["event"] == "error"
        assert "chain boom" in evt["message"]


# ---------------------------------------------------------------------------
# Multi-turn
# ---------------------------------------------------------------------------

class TestWsMultiTurn:

    def test_multiple_messages_on_same_connection(self):
        call_count = {"n": 0}

        from prism_chain import PrismChain

        agent = MagicMock()
        agent._execute = MagicMock()
        chain = MagicMock(spec=PrismChain)

        def _fake(msg, fn, ctx):
            call_count["n"] += 1
            yield {"event": "done", "answer": f"answer {call_count['n']}",
                   "chain_id": "x", "card_type": "text", "card_data": {}, "card_title": ""}

        chain.run_streaming.side_effect = _fake
        chain.run_streaming_async = PrismChain.run_streaming_async.__get__(chain)
        agent._chain = chain

        _set_state(agent=agent)
        c = TestClient(app)

        with c.websocket_connect("/ws/chat") as ws:
            ws.send_json({"message": "turn 1"})
            e1 = ws.receive_json()
            assert e1["answer"] == "answer 1"

            ws.send_json({"message": "turn 2"})
            e2 = ws.receive_json()
            assert e2["answer"] == "answer 2"

        assert call_count["n"] == 2

    def test_error_turn_does_not_close_connection(self):
        from prism_chain import PrismChain

        call_count = {"n": 0}
        agent = MagicMock()
        agent._execute = MagicMock()
        chain = MagicMock(spec=PrismChain)

        def _fake(msg, fn, ctx):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("first turn fails")
            yield {"event": "done", "answer": "recovered",
                   "chain_id": "", "card_type": "text", "card_data": {}, "card_title": ""}

        chain.run_streaming.side_effect = _fake
        chain.run_streaming_async = PrismChain.run_streaming_async.__get__(chain)
        agent._chain = chain

        _set_state(agent=agent)
        c = TestClient(app)

        with c.websocket_connect("/ws/chat") as ws:
            ws.send_json({"message": "fail"})
            e1 = ws.receive_json()
            assert e1["event"] == "error"

            ws.send_json({"message": "recover"})
            e2 = ws.receive_json()
            assert e2["event"] == "done"
            assert e2["answer"] == "recovered"


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------

class TestWsSessionPersistence:

    def test_session_id_persists_messages(self, tmp_path):
        import tempfile

        from prism_session_manager import reset_session_manager

        with tempfile.TemporaryDirectory() as d:
            sm = reset_session_manager(db_path=f"{d}/ws_test.db")
            sess = sm.create_session("ws-test-session")

            agent = _make_streaming_agent()
            _set_state(agent=agent)
            c = TestClient(app)

            with c.websocket_connect("/ws/chat") as ws:
                ws.send_json({"message": "remember this", "session_id": sess.session_id})
                while True:
                    evt = ws.receive_json()
                    if evt.get("event") in ("done", "error"):
                        break

            history = sm.get_history(sess.session_id)
            assert len(history) == 2
            assert history[0].role == "user"
            assert history[0].content == "remember this"
            assert history[1].role == "assistant"
            assert history[1].content == "ws answer"

    def test_no_session_id_still_works(self, client):
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"message": "no session needed"})
            while True:
                evt = ws.receive_json()
                if evt.get("event") in ("done", "error"):
                    break
        assert evt["event"] == "done"


# ---------------------------------------------------------------------------
# Cascade fast-path: simple requests route through agent.chat (not the chain)
# so they work even with no LLM backend (cf. FrugalGPT cascade).
# ---------------------------------------------------------------------------

class TestWsFastPath:

    def _fast_path_agent(self):
        from prism_chain import PrismChain
        from prism_responses import text_card

        agent = MagicMock()
        agent._execute = MagicMock()
        agent.chat = MagicMock(return_value=text_card("8.05 km", "Convert"))
        chain = MagicMock(spec=PrismChain)
        chain.should_chain.return_value = False  # simple → fast-path
        # If the chain were (wrongly) used, this would raise.
        chain.run_streaming.side_effect = AssertionError("chain must not run")
        chain.run_streaming_async = PrismChain.run_streaming_async.__get__(chain)
        agent._chain = chain
        return agent

    def test_simple_request_uses_chat_not_chain(self):
        agent = self._fast_path_agent()
        _set_state(agent=agent)
        c = TestClient(app)
        with c.websocket_connect("/ws/chat") as ws:
            ws.send_json({"message": "convert 5 miles to km"})
            events = []
            while True:
                evt = ws.receive_json()
                events.append(evt)
                if evt.get("event") in ("done", "error"):
                    break
        assert [e["event"] for e in events][-1] == "done"
        done = next(e for e in events if e["event"] == "done")
        assert done["answer"] == "8.05 km"
        agent.chat.assert_called_once()
