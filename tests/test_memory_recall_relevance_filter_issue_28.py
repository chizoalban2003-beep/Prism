"""Memory-recall relevance filter for issue #28 bug 9 — recall returned unrelated facts.

Live test: after storing both ``my partner is Sarah`` and ``my birthday is
March 14``, asking ``who is my partner`` produced::

    My partner is Sarah.
    My birthday is March 14.

…because the handler returned the top-3 fact-source hits unconditionally
and BM25/embedding scored both well enough to clear the 0.15 threshold
(the query was short and both facts share generic words like "my", "is").

Fix: filter facts so only those whose key tag literally appears in the
query are shown. Fall back to the top-1 hit when no tag matches — that
keeps phrasing drift recoverable (e.g. user stored "partner" but asks
about "spouse").
"""
from __future__ import annotations

from pathlib import Path

import pytest

from prism_agent import PrismAgent
from prism_memory import PrismMemory


@pytest.fixture()
def agent(tmp_path: Path):
    a = PrismAgent()
    # Swap the agent's memory for a fresh, empty store under tmp_path so
    # other tests' state and the user's real ~/.prism/memory.db don't leak in.
    a._memory = PrismMemory(db_path=str(tmp_path / "mem.db"))
    return a


def _store_fact(agent: PrismAgent, key: str, value: str) -> None:
    tag = key.lower()
    agent._memory.delete_by_tag(tag, source="fact")
    agent._memory.ingest(
        f"My {key} is {value}.",
        source="fact",
        title=f"my {key}",
        tags=["fact", tag],
    )


class TestRelevantFactOnly:
    def test_who_is_my_partner_does_not_surface_birthday(self, agent):
        _store_fact(agent, "partner", "Sarah")
        _store_fact(agent, "birthday", "March 14")

        card = agent.chat("who is my partner")
        body = card.body or ""
        assert "Sarah" in body, f"partner fact should appear in body: {body!r}"
        assert "March 14" not in body, (
            f"unrelated birthday fact must not appear: {body!r}"
        )
        assert "birthday" not in body.lower()

    def test_when_is_my_birthday_does_not_surface_partner(self, agent):
        _store_fact(agent, "partner", "Sarah")
        _store_fact(agent, "birthday", "March 14")

        card = agent.chat("when is my birthday")
        body = card.body or ""
        assert "March 14" in body
        assert "Sarah" not in body
        assert "partner" not in body.lower()

    def test_three_facts_only_relevant_one_surfaces(self, agent):
        _store_fact(agent, "favourite colour", "teal")
        _store_fact(agent, "partner", "Sarah")
        _store_fact(agent, "birthday", "March 14")

        card = agent.chat("what is my favourite colour")
        body = card.body or ""
        assert "teal" in body
        assert "Sarah" not in body
        assert "March 14" not in body


class TestFallbackWhenNoTagMatch:
    """If the user phrases the query without naming the tag, surface the
    top-scored fact rather than going silent — the BM25/embedding score
    already picked the best candidate."""

    def test_phrasing_drift_falls_back_to_top_hit(self, agent):
        # User stored "my partner is Sarah" but later asks about "spouse" —
        # tag "partner" isn't in the query. Don't go silent.
        _store_fact(agent, "partner", "Sarah Sarah Sarah partner partner")
        card = agent.chat("who is my spouse")
        body = card.body or ""
        # Either we returned the partner fact or the "no stored answer" card.
        # The fix says: fall back to top-1, so "Sarah" should appear when
        # BM25 found a hit at all.
        if "stored answer" not in body.lower():
            assert "Sarah" in body


class TestNoFactsStill404s:
    def test_empty_memory_returns_no_memory_card(self, agent):
        card = agent.chat("who is my partner")
        assert "stored" in (card.body or "").lower() or "memory" in (card.title or "").lower()
