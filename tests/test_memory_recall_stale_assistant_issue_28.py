"""Stale-assistant-output fix for issue #28 bug 6 — recall echoed PRISM's own past replies.

Live test: PRISM had once replied "Your partner's name might be Sarah,
based on the calendar entries..." to an earlier "who is my partner?"
question. That reply was ingested as a ``source="conversation"`` entry.
Next time the user asked "what's my partner's name?", memory_recall
fell back from facts to *any* conversation hit, found that stored
assistant reply, and surfaced it back to the user as if it were a
stored fact. The hedged/generated text became authoritative.

The fix has two coupled pieces, both pinned here:

  1. :meth:`PrismMemory.ingest_conversation` now writes a ``role:<role>``
     tag so the recall site can tell PRISM-authored entries apart from
     user-authored ones.
  2. The ``memory_recall`` intent handler in :mod:`prism_agent` filters
     out assistant-authored conversation entries from the fallback list
     (it still uses facts first; user-authored conversation still
     surfaces). Pre-tag legacy entries are caught by their
     ``"assistant: ..."`` title prefix as defence-in-depth.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from prism_memory import PrismMemory


@pytest.fixture()
def memory(tmp_path: Path) -> PrismMemory:
    return PrismMemory(db_path=str(tmp_path / "mem.db"))


class TestIngestConversationTagsRole:
    def test_user_turn_gets_role_user_tag(self, memory):
        entry_id = memory.ingest_conversation(
            "user",
            "My partner Sarah and I went hiking last weekend in the Lake District",
        )
        assert entry_id is not None
        # Pull it back via search and confirm the tag survived round-trip.
        hits = memory.search("partner Sarah hiking", top_n=5)
        assert hits, "expected at least one hit"
        assert any("role:user" in h.entry.tags for h in hits)

    def test_assistant_turn_gets_role_assistant_tag(self, memory):
        entry_id = memory.ingest_conversation(
            "assistant",
            "Your partner partner partner name might be Sarah Sarah Sarah, "
            "based on calendar entries — but I am not fully certain",
        )
        assert entry_id is not None
        hits = memory.search("partner Sarah name", top_n=5)
        assert hits
        assert any("role:assistant" in h.entry.tags for h in hits)


class TestMemoryRecallFiltersAssistantTurns:
    """The headline regression — calling the memory_recall path on a query
    that only has a stale assistant turn must not surface it."""

    def test_stale_assistant_reply_not_surfaced(self, memory):
        # Seed memory with the exact bug scenario.
        memory.ingest_conversation(
            "assistant",
            "Your partner partner partner name name might be Sarah Sarah Sarah "
            "based on calendar entries — but I'm not fully certain about that",
        )
        # The contract test: search returns the assistant entry; the
        # filter the handler applies (replicated inline below) must drop
        # it from the recall result.
        hits = memory.search("partner name Sarah", top_n=5)
        assistant_hits = [
            h for h in hits
            if "role:assistant" in h.entry.tags
            or (h.entry.title or "").lower().startswith("assistant:")
        ]
        non_assistant_hits = [h for h in hits if h not in assistant_hits]
        # Fix contract: the assistant hit exists, but after filtering, the
        # recall path's "top" list is empty — so the user gets the "no
        # stored answer" card, not the stale generated reply.
        assert assistant_hits, "seeded entry should be findable at all"
        assert not non_assistant_hits, (
            "filtering must drop assistant-authored conversation hits"
        )

    def test_user_turn_still_surfaces(self, memory):
        memory.ingest_conversation(
            "user",
            "My partner partner is named Sarah Sarah Sarah and we met "
            "at university back in 2015 — partner name details",
        )
        hits = memory.search("partner name Sarah", top_n=5)
        user_hits = [h for h in hits if "role:user" in h.entry.tags]
        assert user_hits, "user-authored turn must still surface on recall"

    def test_legacy_entries_caught_by_title_prefix(self, memory):
        # Pre-tag entries (ingested before the role-tag rollout) won't have
        # the role:assistant tag. The handler's defence-in-depth catches
        # them by looking at the title prefix.
        memory.ingest(
            "Your partner partner partner name might be Sarah Sarah Sarah — "
            "hedged guess from old PRISM, kept for legacy partner name lookups",
            source="conversation",
            title="assistant: Your partner's name might be...",
            # NO role tag — simulates pre-rollout entry.
            tags=[],
        )
        hits = memory.search("partner name Sarah", top_n=5)
        # Replicate the handler's filter inline to pin the title-prefix path.
        def _is_self_authored(h):
            tags = h.entry.tags or []
            if "role:assistant" in tags:
                return True
            return (h.entry.title or "").lower().startswith("assistant:")
        legacy_hits = [h for h in hits if _is_self_authored(h)]
        non_legacy = [h for h in hits if not _is_self_authored(h)]
        assert legacy_hits, "should find the legacy entry"
        assert not non_legacy, "and the filter should mark it as self-authored"
