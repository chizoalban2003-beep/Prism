"""
tests/test_chat_ux_fixes_issue_28.py
====================================
Three UX/routing fixes found while testing the live daemon (#28-129):
1. small-talk ("what is up") no longer routes to wikipedia_lookup
2. "i want to send an email" no longer hijacked by the universal_plan planner
3. kinetic proactive notifications read as plain suggestions, not raw
   telemetry ([Kinetic/...] Z=-2.8, ΔA=0.19)
"""
from __future__ import annotations

import types

from prism_intents import INTENTS
from prism_routing import route_intent


def _route(m):
    return route_intent(m, INTENTS, lambda _m: "general_chat")


class TestSmallTalkNotWikipedia:
    def test_greetings_and_smalltalk(self):
        for m in ("what is up", "what is new", "what is going on",
                  "what is wrong", "what is the matter"):
            assert _route(m) == "general_chat", m

    def test_real_lookups_still_wikipedia(self):
        assert _route("what is a black hole") == "wikipedia_lookup"
        assert _route("tell me about the Roman empire") == "wikipedia_lookup"
        assert _route("what is photosynthesis") == "wikipedia_lookup"


class TestConcreteActionNotPlanner:
    def test_email_intent_wins_over_planner(self):
        assert _route("i want to send an email") == "email_send"
        assert _route(
            "i want to send an email saying hello to a@b.com") == "email_send"
        assert _route("i'd like to send an email to bob") == "email_send"

    def test_real_goals_still_planner(self):
        for m in ("i want to lose weight", "i want to learn spanish",
                  "help me reach a goal", "what is the best way to save money"):
            assert _route(m) == "universal_plan", m


class TestKineticMessagesAreHuman:
    def _window(self, lever, crisis=False):
        from prism_kinetic_engine import ActionWindow
        sig = types.SimpleNamespace(domain="energy", signal_type="on_power",
                                    z_score=-2.8)
        return ActionWindow(window_id="w", lever_id=lever, source_signal=sig,
                            v_potential=0.5, v_current=0.0, c_friction=0.3,
                            delta_a=0.19, is_crisis=crisis)

    def test_no_telemetry_jargon_in_user_message(self):
        for lever in ("defer_decision", "intervene_now", "proactive_assist"):
            msg = self._window(lever).to_proactive_message()
            # none of the internal maths leaks to the user
            for junk in ("[Kinetic", "Z=", "ΔA", "threshold", "z_score"):
                assert junk not in msg, (lever, msg)
            assert len(msg) > 20  # a real sentence

    def test_defer_decision_reads_naturally(self):
        msg = self._window("defer_decision").to_proactive_message()
        assert "energy" in msg.lower() and "decision" in msg.lower()

    def test_unknown_lever_has_safe_default(self):
        msg = self._window("some_new_lever").to_proactive_message()
        assert "Kinetic" not in msg and len(msg) > 10

    def test_debug_line_keeps_telemetry(self):
        # the maths is preserved for logs, just not shown to the user
        dbg = self._window("defer_decision").debug_line()
        assert "Z=" in dbg and "defer_decision" in dbg
