"""Fact-upsert fix for issue #28 bug 7 — duplicate facts surfaced together on recall.

Live test: the user said "remember that my favourite colour is teal".
Then asked "what is my favourite colour?". PRISM replied with all three
historical answers concatenated::

    My favourite colour is teal.
    My favourite colour is blue.
    My favourite colour is blue.

Root cause: the fact-store path called ``memory.ingest()`` every time
without removing prior entries that carried the same key tag, so every
"my X is Y" assertion was *additive*. Recall surfaces the top-3 hits
joined by newlines, which (correctly, given the data) returned all
three as if they were equally true.

Fix: ``PrismMemory.delete_by_tag()`` removes prior entries that carry
the same key tag, and the agent's fact-store branch calls it before
ingesting. These tests pin the upsert.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from prism_memory import PrismMemory


@pytest.fixture()
def memory(tmp_path: Path) -> PrismMemory:
    return PrismMemory(db_path=str(tmp_path / "mem.db"))


class TestDeleteByTag:
    def test_removes_entries_matching_tag(self, memory):
        memory.ingest("My favourite colour is blue.",
                      source="fact", title="my colour",
                      tags=["fact", "favourite colour"])
        memory.ingest("My favourite colour is red.",
                      source="fact", title="my colour",
                      tags=["fact", "favourite colour"])
        removed = memory.delete_by_tag("favourite colour", source="fact")
        assert removed == 2
        # Search confirms they're gone.
        hits = memory.search("favourite colour", top_n=5)
        assert not [h for h in hits if "favourite colour" in (h.entry.tags or [])]

    def test_leaves_unrelated_entries_alone(self, memory):
        memory.ingest("My favourite colour is blue.",
                      source="fact", title="my colour",
                      tags=["fact", "favourite colour"])
        memory.ingest("My favourite food is sushi.",
                      source="fact", title="my food",
                      tags=["fact", "favourite food"])
        memory.delete_by_tag("favourite colour", source="fact")
        # Food fact survives.
        hits = memory.search("favourite food sushi", top_n=5)
        assert any("favourite food" in (h.entry.tags or []) for h in hits)

    def test_source_filter_protects_other_sources(self, memory):
        # Conversation entry carrying the same tag-string by accident must
        # not get cleared when we only meant to upsert facts.
        memory.ingest("freeform mention of favourite colour topic discussion",
                      source="conversation", title="user: ...",
                      tags=["role:user", "favourite colour"])
        memory.ingest("My favourite colour is blue.",
                      source="fact", title="my colour",
                      tags=["fact", "favourite colour"])
        removed = memory.delete_by_tag("favourite colour", source="fact")
        assert removed == 1  # only the fact, not the conversation
        # Conversation entry still present.
        hits = memory.search("favourite colour topic discussion", top_n=5)
        assert any(h.entry.source == "conversation" for h in hits)

    def test_no_matches_returns_zero(self, memory):
        assert memory.delete_by_tag("nonexistent tag", source="fact") == 0


class TestUpsertSemantics:
    """The headline regression — re-stating a fact replaces prior values."""

    def test_two_assertions_same_key_only_keeps_latest(self, memory):
        # Simulate what the agent fact-store branch does.
        def _store(key: str, value: str) -> None:
            tag = key.lower()
            memory.delete_by_tag(tag, source="fact")
            memory.ingest(
                f"My {key} is {value}.",
                source="fact", title=f"my {key}",
                tags=["fact", tag],
            )

        _store("favourite colour", "blue")
        _store("favourite colour", "red")
        _store("favourite colour", "teal")

        hits = memory.search("favourite colour", top_n=5)
        fact_hits = [h for h in hits if h.entry.source == "fact"]
        assert len(fact_hits) == 1, (
            f"expected exactly one fact entry, got {len(fact_hits)}: "
            f"{[h.entry.content for h in fact_hits]}"
        )
        assert "teal" in fact_hits[0].entry.content
        assert "blue" not in fact_hits[0].entry.content
        assert "red"  not in fact_hits[0].entry.content
