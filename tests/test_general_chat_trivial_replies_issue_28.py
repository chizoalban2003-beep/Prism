"""general_chat trivial-token short-circuit + system prompt for #28 bug 53.

Live probes uncovered the assistant turning trivial acks into essays:

  user: "thanks"   →  1532-char "Hypothesis: the user's single word
                       'thanks' is a conclusionary utterance, likely
                       indicating confirmation of a prior interaction's
                       success or a polite closing signal…"
  user: "ok"       →  332-char "Acknowledged. PRISM is online in
                       **factual-audit** mode. I am ready. Please
                       state your claim…"
  user: "bye"      →  "Goodbye. If you have any further requests
                       requiring factual audit or analysis, I am
                       available."

Root cause: the general_chat handler called the LLM with no system
prompt. Some models (DeepSeek in particular) then improvised a
"factual-audit" persona and over-analysed every casual phrase.

Two-part fix:

1. ``_trivial_chat_reply`` returns a fixed friendly reply for a small
   set of bare acks/greetings — no LLM call needed.
2. The general_chat LLM call now passes a system prompt that steers
   the model to one-or-two-sentence conversational replies.
"""
from __future__ import annotations

import inspect

import prism_agent


class TestTrivialChatReplyHelper:

    def test_thanks_returns_youre_welcome(self):
        assert prism_agent._trivial_chat_reply("thanks") == "You're welcome."

    def test_thank_you_returns_youre_welcome(self):
        assert prism_agent._trivial_chat_reply("thank you") == "You're welcome."

    def test_ok_returns_got_it(self):
        assert prism_agent._trivial_chat_reply("ok") == "Got it."

    def test_bye_returns_goodbye(self):
        assert prism_agent._trivial_chat_reply("bye") == "Goodbye."

    def test_hi_returns_greeting(self):
        out = prism_agent._trivial_chat_reply("hi")
        assert out is not None and "help" in out.lower()

    def test_punctuation_stripped(self):
        assert prism_agent._trivial_chat_reply("Thanks!") == "You're welcome."
        assert prism_agent._trivial_chat_reply("ok?") == "Got it."
        assert prism_agent._trivial_chat_reply("HELLO.") == "Hi! How can I help?"

    def test_whitespace_stripped(self):
        assert prism_agent._trivial_chat_reply("  thanks  ") == "You're welcome."

    def test_non_trivial_returns_none(self):
        # Real questions must NOT short-circuit.
        assert prism_agent._trivial_chat_reply(
            "explain hash maps") is None
        assert prism_agent._trivial_chat_reply(
            "thanks for explaining hash maps") is None
        assert prism_agent._trivial_chat_reply(
            "what time is it") is None

    def test_empty_returns_none(self):
        # Empty input is handled elsewhere (#28-46); this helper
        # should report "not a trivial reply" so the caller can defer.
        assert prism_agent._trivial_chat_reply("") is None
        assert prism_agent._trivial_chat_reply("   ") is None


class TestGeneralChatHandlerWiredToHelper:
    """``chat`` must invoke ``_trivial_chat_reply`` for general_chat
    intents and skip the LLM when it returns a string. Test by reading
    the source — wiring is one line, so a source-level assertion is
    appropriate."""

    def test_execute_source_calls_trivial_chat_reply(self):
        src = inspect.getsource(prism_agent.PrismAgent._execute)
        assert "_trivial_chat_reply" in src


class TestSystemPromptSteersChatLLM:
    """For non-trivial chat, the LLM call must pass a system prompt
    that anchors the model to a conversational PRISM persona."""

    def test_execute_source_passes_system_prompt(self):
        src = inspect.getsource(prism_agent.PrismAgent._execute)
        # The router.call invocation for general_chat must include a
        # `system=` kwarg — without it, the model improvises personas.
        assert "system=system" in src or 'system="' in src
        # The prompt itself must mention PRISM and conversational tone.
        assert "PRISM" in src
        assert "conversational" in src or "concise" in src
