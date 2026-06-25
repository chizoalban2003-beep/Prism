"""Memory search hex/alphanumeric token fix for issue #28 bug 44.

Live test: ingesting ``"PRISM e2e probe marker deadbeef-9d3a"`` then
searching for ``"deadbeef"`` returned ``count: 0``. Plain English
queries (``"restaurant"``, ``"Wolseley"``) hit fine.

Two compounding causes:

1. BM25 tokenised with bare ``str.split()``. ``"deadbeef-9d3a"`` is
   one whitespace-delimited token, so ``d_terms.count("deadbeef") == 0``.

2. ``search()`` used cosine OR BM25 — never both. When an embedding
   was available, BM25 was skipped entirely, so even tokens that
   embeddings can't meaningfully represent (hashes, UUIDs, error
   codes, build IDs, file paths) were dropped below the 0.15 surface
   threshold.

Fix:
* ``_tokenise()`` splits on ``\\w+`` so hyphen/period/slash all break.
* ``search()`` takes ``max(cosine, BM25)`` so exact-token recall
  always has a path to surface results.
"""
from __future__ import annotations

import prism_memory as _mod


class TestTokenise:
    def test_hyphenated_hex_splits(self):
        # The reported bug: tokeniser must break on "-".
        assert _mod.PrismMemory._tokenise("deadbeef-9d3a") == ["deadbeef", "9d3a"]

    def test_dot_separated_splits(self):
        # File paths / version strings — common in error reports.
        assert _mod.PrismMemory._tokenise("config.toml") == ["config", "toml"]

    def test_slash_path_splits(self):
        assert _mod.PrismMemory._tokenise("/usr/local/bin/prism") == [
            "usr", "local", "bin", "prism",
        ]

    def test_uuid_form_splits(self):
        toks = _mod.PrismMemory._tokenise("8b5694fa-a1b2-c3d4-e5f6-708090a0b0c0")
        assert "8b5694fa" in toks
        assert "708090a0b0c0" in toks

    def test_plain_english_still_lowercased(self):
        assert _mod.PrismMemory._tokenise("My Favourite Restaurant") == [
            "my", "favourite", "restaurant",
        ]


class TestBM25:
    def test_bm25_finds_hex_substring(self):
        # The reported bug surface — BM25 alone must return >0 for the
        # hex prefix even when the stored form is hyphenated.
        score = _mod.PrismMemory._bm25(
            "deadbeef",
            "PRISM e2e probe marker deadbeef-9d3a",
        )
        assert score > 0.0

    def test_bm25_uuid_lookup(self):
        score = _mod.PrismMemory._bm25(
            "8b5694fa",
            "Horizon goal 8b5694fa-pending-execution",
        )
        assert score > 0.0

    def test_bm25_zero_when_no_overlap(self):
        score = _mod.PrismMemory._bm25(
            "unrelated", "PRISM e2e probe marker deadbeef-9d3a",
        )
        assert score == 0.0


class TestHybridSearch:
    """End-to-end: ingest + search round-trips alphanumeric tokens."""

    def test_hex_token_round_trip(self, tmp_path):
        # Drop into a temp db so we don't touch the live one.
        store = _mod.PrismMemory(db_path=str(tmp_path / "mem.db"))
        store.ingest(
            content="PRISM e2e probe marker deadbeef-9d3a",
            source="probe",
            tags=["e2e"],
        )
        results = store.search("deadbeef", top_n=5)
        # The reported bug: this used to be []. Must surface the entry.
        assert results, "search('deadbeef') must find ingested 'deadbeef-9d3a'"
        assert results[0].entry.content.startswith("PRISM e2e probe marker")

    def test_uuid_token_round_trip(self, tmp_path):
        store = _mod.PrismMemory(db_path=str(tmp_path / "mem.db"))
        store.ingest(
            content="Task 8b5694fa completed at 2026-06-25T17:00:00Z",
            source="task_log",
        )
        results = store.search("8b5694fa", top_n=5)
        assert results

    def test_plain_english_unchanged(self, tmp_path):
        store = _mod.PrismMemory(db_path=str(tmp_path / "mem.db"))
        store.ingest(
            content="My favourite restaurant is The Wolseley in London",
            source="note",
        )
        results = store.search("Wolseley", top_n=5)
        assert results
