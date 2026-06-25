"""Chat empty-input guard for issue #28 bug 46.

Live test:

* ``POST /chat {"message":""}`` returned the policy_inspect organ — a
  full dump of every loaded organ's risk policy. That's because the
  empty string fell through to the tier dispatcher, which the LLM
  classifier (or first regex) treated as a generic "show me what you
  can do" query.
* ``POST /chat {"message":"   "}`` (whitespace-only) created a real
  task with body ``Added: `` — an empty task in the user's task list.

Both are user-hostile and waste budget on LLM calls. Add a guard at
the top of ``PrismAgent.chat`` that returns a help card instead.
"""
from __future__ import annotations

from pathlib import Path


class TestEmptyInputReturnsHelpCard:
    """Without spinning a full agent, we pin the guard textually.

    Real-agent integration tests live in tests/test_chat_endpoint.py;
    here we just want a fast unit-shaped pin so future refactors that
    move the guard down past the dispatch step trip the test.
    """

    def _agent_source(self) -> str:
        return (Path(__file__).resolve().parent.parent / "prism_agent.py").read_text()

    def test_chat_guards_empty_before_dispatch(self):
        src = self._agent_source()
        chat_idx = src.find("def chat(self, message: str")
        assert chat_idx > 0, "PrismAgent.chat method must exist"
        dispatch_idx = src.find("self._tier_dispatcher().dispatch", chat_idx)
        assert dispatch_idx > 0, "dispatch call must exist below chat()"

        guard_idx = src.find('if not (message or "").strip():', chat_idx)
        assert guard_idx > 0, (
            "PrismAgent.chat must guard empty/whitespace input — without "
            "it '' routes to policy_inspect and '   ' creates an empty task "
            "(issue #28-46)."
        )
        assert guard_idx < dispatch_idx, (
            "empty-input guard must run BEFORE tier_dispatcher().dispatch"
        )

    def test_guard_returns_helpful_card(self):
        src = self._agent_source()
        chat_idx = src.find("def chat(self, message: str")
        guard_idx = src.find('if not (message or "").strip():', chat_idx)
        next_dispatch = src.find("dispatch(", guard_idx)
        between = src[guard_idx:next_dispatch]
        # The guard body must produce a card and return — not just `pass`
        # or fall through.
        assert "return text_card(" in between, (
            "guard must build a text_card and return — falling through "
            "re-introduces the bug."
        )


class TestEmptyInputViaRealAgent:
    """End-to-end: build a PrismAgent (with stub deps) and confirm an
    empty / whitespace message returns the help card without invoking
    the tier dispatcher.

    We stub the dispatcher so a regression that bypasses the guard
    raises immediately instead of routing somewhere unexpected.
    """

    def _make_agent(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        monkeypatch.setenv("PRISM_HOME", str(tmp_path))
        # PrismAgent's constructor pulls in a lot — let it run, then
        # replace the dispatcher with one that raises if invoked.
        import prism_agent as _mod
        agent = _mod.PrismAgent.__new__(_mod.PrismAgent)
        # Minimal attribute scaffolding for chat() to run its guard.
        agent._instructions = MagicMock()
        agent._instructions.parse_from_chat.return_value = None
        agent._instructions.parse_fact.return_value = None
        agent._instructions.parse_forget.return_value = None
        agent._memory = None
        agent._chat_history = []
        agent._tts = None
        agent._perception = None
        agent._chain = MagicMock()
        agent._calendar = None
        agent._email = None
        agent._context_manager = None
        agent._persona = None
        agent._crystalliser = None

        # If chat() reaches the dispatcher, that's the bug — fail loudly.
        def _boom(*_a, **_kw):
            raise AssertionError("empty-input guard let the message through")
        # Use a plain object with a callable attribute so Python doesn't
        # bind `self` (a class method would; the dispatcher in production
        # is also a callable attribute on an instance).
        _disp = type("D", (), {})()
        _disp.dispatch = _boom
        agent._tier_dispatcher = lambda: _disp
        agent._should_suppress_logging = lambda _m: False
        agent._memory_graph = None
        agent._suppress_logging = False
        return agent

    def test_empty_message_short_circuits(self, tmp_path, monkeypatch):
        agent = self._make_agent(tmp_path, monkeypatch)
        card = agent.chat("")
        # If we got here, the guard worked.
        assert "What would you like help with" in (card.body or "")
        assert card.title == "PRISM"

    def test_whitespace_message_short_circuits(self, tmp_path, monkeypatch):
        agent = self._make_agent(tmp_path, monkeypatch)
        card = agent.chat("    \t  \n ")
        assert "What would you like help with" in (card.body or "")

    def test_real_message_still_dispatches(self, tmp_path, monkeypatch):
        # Sanity: the guard must not eat real input.
        agent = self._make_agent(tmp_path, monkeypatch)
        # Swap the boom-dispatcher for one that records the call.
        from prism_responses import text_card
        recorded: dict = {}

        def _ok(msg, ctx, initial_card=None):
            recorded["msg"] = msg
            return text_card("ok", "stub")
        _disp = type("D", (), {})()
        _disp.dispatch = _ok
        agent._tier_dispatcher = lambda: _disp

        card = agent.chat("what time is it")
        assert recorded.get("msg") == "what time is it", (
            "non-empty input must still reach the dispatcher"
        )
        assert card.body == "ok"
