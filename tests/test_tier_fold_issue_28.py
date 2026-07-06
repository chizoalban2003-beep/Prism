"""
tests/test_tier_fold_issue_28.py
================================
RFC step 5 (#28-111): chain/composer triggers try the policied tool
loop first; the legacy tier runs unchanged whenever the loop declines
(offline / disabled), and [tool_loop].fold_tiers=false restores the
legacy order outright. Orchestrator and expert chain keep precedence.
"""
from __future__ import annotations

from prism_chat_tiers import TierDispatcher
from prism_responses import text_card


class _Recorder:
    def __init__(self, name, card=None, wants=True):
        self.name = name
        self.card = card
        self.wants = wants
        self.ran = False

    # chain / composer / orchestrator surface
    def should_chain(self, m):
        return self.wants

    def should_compose(self, m):
        return self.wants

    def should_orchestrate(self, m):
        return self.wants

    def run(self, message, execute, ctx):
        self.ran = True
        return self.card

    def orchestrate(self, message, execute, ctx):
        self.ran = True
        return self.card

    def decompose(self, message):
        self.ran = True
        return None


def _dispatcher(*, loop_card="loop answer", fold=True,
                chain_wants=True, compose_wants=True, orch=None):
    chain = _Recorder("chain", card=text_card("chain answer", "Chain"),
                      wants=chain_wants)
    composer = _Recorder("composer", wants=compose_wants)
    loop_calls = []

    def tool_loop(message, context, multistep=False):
        loop_calls.append({"message": message, "multistep": multistep})
        return text_card(loop_card, "PRISM") if loop_card else None

    d = TierDispatcher(
        orchestrator=orch,
        chain_expert=_Recorder("expert", wants=False),
        chain=chain,
        composer=composer,
        execute=lambda intent, m, c: text_card("single intent", "Fallback"),
        route=lambda m: "general_chat",
        tool_loop=tool_loop,
        fold_tiers=fold,
    )
    return d, chain, composer, loop_calls


MULTISTEP_MSG = "check the weather and then add a note about what to wear"


class TestFold:
    def test_loop_takes_the_chain_trigger(self):
        d, chain, _, loop_calls = _dispatcher()
        card = d.dispatch(MULTISTEP_MSG, {})
        assert card.body == "loop answer"
        assert chain.ran is False
        assert loop_calls and loop_calls[0]["multistep"] is True

    def test_loop_declines_then_legacy_chain_runs(self):
        d, chain, _, loop_calls = _dispatcher(loop_card=None)
        card = d.dispatch(MULTISTEP_MSG, {})
        assert card.body == "chain answer"
        assert chain.ran is True
        assert len(loop_calls) == 1  # tried once, declined, no retry

    def test_fold_disabled_restores_legacy_order(self):
        d, chain, _, loop_calls = _dispatcher(fold=False)
        card = d.dispatch(MULTISTEP_MSG, {})
        assert card.body == "chain answer"
        assert loop_calls == []

    def test_orchestrator_keeps_precedence(self):
        orch = _Recorder("orch", card=text_card("orchestrated", "Orch"))
        d, _, _, loop_calls = _dispatcher(orch=orch)
        card = d.dispatch(MULTISTEP_MSG, {})
        assert card.body == "orchestrated"
        assert loop_calls == []

    def test_no_trigger_no_loop(self):
        d, _, _, loop_calls = _dispatcher(chain_wants=False,
                                          compose_wants=False)
        card = d.dispatch("short message here please thanks kindly", {})
        assert card.body == "single intent"
        assert loop_calls == []
