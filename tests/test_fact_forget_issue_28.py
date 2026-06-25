"""Symmetric forget my X for issue #28 bug 27.

Live test: ``forget my birthday`` returned ``My birthday is March 15``
— PRISM happily recalled the value the user wanted gone, because
there was no parse hook for "forget my X" and the message fell
through to memory_recall.

Fix: a new ``PrismInstructions.parse_forget`` classmethod parallel
to ``parse_fact``, plus a branch in ``PrismAgent.chat()`` that
deletes the matching tagged entries before the message ever
reaches intent routing.
"""
from __future__ import annotations

from prism_instructions import PrismInstructions


class TestParseForget:
    def test_basic(self):
        assert PrismInstructions.parse_forget("forget my birthday") == "birthday"

    def test_with_please(self):
        assert PrismInstructions.parse_forget("please forget my partner") == "partner"

    def test_with_about(self):
        assert PrismInstructions.parse_forget("forget about my favourite colour") == "favourite colour"

    def test_with_that(self):
        assert PrismInstructions.parse_forget("forget that my car is blue") is None
        # "forget that my X" without "is/are" should not match — but our
        # regex requires no trailing value, so:
        assert PrismInstructions.parse_forget("forget that my partner") == "partner"

    def test_trailing_punctuation(self):
        assert PrismInstructions.parse_forget("forget my birthday.") == "birthday"
        assert PrismInstructions.parse_forget("forget my birthday!") == "birthday"
        assert PrismInstructions.parse_forget("forget my birthday?") == "birthday"


class TestParseForgetNoFalsePositives:
    def test_forget_about_it_is_not_a_forget(self):
        assert PrismInstructions.parse_forget("forget about it") is None

    def test_i_forgot_is_not_a_forget(self):
        assert PrismInstructions.parse_forget("I forgot my keys") is None

    def test_dont_forget_is_not_a_forget(self):
        # "don't forget" is part of remember-flow, not delete-flow.
        assert PrismInstructions.parse_forget("don't forget my birthday") is None

    def test_empty(self):
        assert PrismInstructions.parse_forget("") is None
        assert PrismInstructions.parse_forget(None) is None


class TestAgentForgetFlow:
    """End-to-end through PrismAgent.chat with a real memory store."""

    def _build_agent(self, tmp_path):
        from prism_agent import PrismAgent
        from prism_memory import PrismMemory

        db = tmp_path / "mem.db"
        mem = PrismMemory(db_path=str(db))
        agent = PrismAgent(kde_agent=None, ksa_agent=None)
        agent._memory = mem
        return agent, mem

    def test_forget_deletes_stored_fact(self, tmp_path):
        agent, mem = self._build_agent(tmp_path)
        # Store a fact.
        mem.ingest("My birthday is March 15.", source="fact",
                   title="my birthday", tags=["fact", "birthday"])
        # Confirm it's recallable.
        results = mem.search("birthday", top_n=5)
        assert any("March 15" in r.entry.content for r in results)
        # Forget.
        card = agent.chat("forget my birthday")
        assert card.title == "Fact forgotten"
        assert "Forgotten" in card.body or "won't" in card.body.lower()
        # Confirm it's gone.
        results = mem.search("birthday", top_n=5)
        assert not any("March 15" in r.entry.content for r in results)

    def test_forget_nothing_stored_explains(self, tmp_path):
        agent, _ = self._build_agent(tmp_path)
        card = agent.chat("forget my birthday")
        assert card.title == "Fact forgotten"
        assert "don't have" in card.body.lower() or "nothing" in card.body.lower()
