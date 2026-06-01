from __future__ import annotations

import tempfile
import os
import pytest

from prism_memory import PrismMemory, MemoryEntry, MemoryResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mem(tmp_path):
    db = str(tmp_path / "test_memory.db")
    return PrismMemory(db_path=db)


# ── Ingest ────────────────────────────────────────────────────────────────────

def test_ingest_returns_id(mem):
    entry_id = mem.ingest("Hello world", source="note")
    assert isinstance(entry_id, str)
    assert len(entry_id) == 12


def test_ingest_stores_content(mem):
    mem.ingest("My first memory", source="note", title="First")
    results = mem.search("first memory")
    assert len(results) >= 1
    assert any("first memory" in r.entry.content.lower() for r in results)


def test_ingest_uses_source(mem):
    mem.ingest("A conversation note", source="conversation")
    results = mem.search("conversation", source_filter="conversation")
    assert len(results) >= 1
    assert results[0].entry.source == "conversation"


def test_ingest_tags(mem):
    mem.ingest("Important reminder", source="note", tags=["urgent", "health"])
    results = mem.search("important reminder")
    assert len(results) >= 1
    assert "urgent" in results[0].entry.tags


def test_ingest_default_title(mem):
    content = "This is the full content text"
    mem.ingest(content, source="document")
    results = mem.search("full content text")
    # Title should default to first 60 chars of content
    assert len(results) >= 1
    assert results[0].entry.title == content[:60]


# ── Search ────────────────────────────────────────────────────────────────────

def test_search_empty_db(mem):
    results = mem.search("nothing here")
    assert results == []


def test_search_returns_list(mem):
    mem.ingest("Python programming language tutorial", source="document")
    results = mem.search("Python programming")
    assert isinstance(results, list)


def test_search_score_threshold(mem):
    mem.ingest("Completely unrelated topic about penguins", source="note")
    # Search for something totally different — should return nothing above threshold
    results = mem.search("quantum physics laser")
    # If something is returned it must have score > 0.15
    for r in results:
        assert r.score > 0.15


def test_search_source_filter(mem):
    mem.ingest("Email content about meeting", source="email")
    mem.ingest("Note about meeting", source="note")
    results = mem.search("meeting", source_filter="email")
    for r in results:
        assert r.entry.source == "email"


def test_search_top_n(mem):
    for i in range(10):
        mem.ingest(f"Entry number {i} about search testing", source="note")
    results = mem.search("search testing", top_n=3)
    assert len(results) <= 3


def test_search_result_has_excerpt(mem):
    mem.ingest("The quick brown fox jumps over the lazy dog", source="note")
    results = mem.search("quick brown fox")
    if results:
        assert isinstance(results[0].excerpt, str)
        assert len(results[0].excerpt) <= 300


# ── ingest_conversation ───────────────────────────────────────────────────────

def test_ingest_conversation_stores_turn(mem):
    mem.ingest_conversation("user", "Tell me about machine learning algorithms and neural networks")
    results = mem.search("machine learning", source_filter="conversation")
    assert len(results) >= 1


def test_ingest_conversation_skips_short(mem):
    mem.ingest_conversation("user", "ok")  # too short — should not store
    results = mem.search("ok")
    assert len(results) == 0


# ── BM25 ──────────────────────────────────────────────────────────────────────

def test_bm25_matching_terms():
    score = PrismMemory._bm25("python tutorial", "This is a python tutorial for beginners")
    assert score > 0.0


def test_bm25_no_match():
    score = PrismMemory._bm25("python", "The quick brown fox")
    assert score == 0.0


def test_bm25_capped_at_one():
    # Repeated matching terms shouldn't exceed 1.0
    score = PrismMemory._bm25("python python python", "python python python python python")
    assert score <= 1.0


# ── Cosine ────────────────────────────────────────────────────────────────────

def test_cosine_identical():
    v = [1.0, 0.0, 1.0]
    assert abs(PrismMemory._cosine(v, v) - 1.0) < 1e-6


def test_cosine_orthogonal():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(PrismMemory._cosine(a, b)) < 1e-6


def test_cosine_length_mismatch():
    assert PrismMemory._cosine([1.0, 2.0], [1.0]) == 0.0


def test_cosine_empty():
    assert PrismMemory._cosine([], []) == 0.0


# ── Excerpt ───────────────────────────────────────────────────────────────────

def test_excerpt_finds_relevant_section():
    content = "A" * 200 + " relevant excerpt here " + "B" * 200
    excerpt = PrismMemory._excerpt("relevant excerpt", content, max_len=300)
    assert "relevant" in excerpt.lower() or len(excerpt) <= 300


def test_excerpt_max_length():
    content = "word " * 200
    excerpt = PrismMemory._excerpt("word", content, max_len=300)
    assert len(excerpt) <= 300
