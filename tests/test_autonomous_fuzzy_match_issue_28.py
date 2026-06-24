"""Autonomous tool fuzzy-match fix for issue #28 bug 22.

Live test: ``calculate the SHA256 hash of hello world`` returned a
**FIFA World Cup 2026 prediction** — the autonomous engine's fuzzy
cache match treated a single shared word ("world") as sufficient to
reuse a cached `world_cup_predictor` tool.

Fix: require at least 2 distinctive task keywords to appear in the
tool's name/description, and exclude generic stopwords ("world",
"hello", "calculate", ...) that carry no topical meaning. Pick the
tool with the highest match count, ties broken by insertion order.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from prism_autonomous import AcquiredTool, PrismAutonomous


def _engine() -> PrismAutonomous:
    """A PrismAutonomous instance with a fresh, empty cache directory."""
    tmp = tempfile.mkdtemp()
    with patch.object(PrismAutonomous, "TOOL_DIR", Path(tmp)):
        eng = PrismAutonomous(llm_router=None)
    eng.TOOL_DIR = Path(tmp)
    return eng


def _tool(name: str, description: str) -> AcquiredTool:
    return AcquiredTool(
        tool_id      = name[:10].ljust(10, "x"),
        name         = name,
        description  = description,
        code         = "def execute(t,p): return 'x'",
        requirements = [],
        entry_fn     = "execute",
    )


class TestNoFalsePositives:
    """The cluster of false positives observed live."""

    def test_sha256_hello_world_does_not_match_world_cup(self):
        eng = _engine()
        wc = _tool(
            "world_cup_predictor",
            "Predicts World Cup winner probabilities using recent team performance",
        )
        eng._tools[wc.tool_id] = wc
        assert eng._find_cached_tool("calculate the SHA256 hash of hello world") is None

    def test_make_video_game_does_not_steal_video_summarizer(self):
        eng = _engine()
        vs = _tool("video_summarizer", "Summarize video transcripts")
        eng._tools[vs.tool_id] = vs
        # Only "video" overlaps — one shared keyword shouldn't be enough.
        assert eng._find_cached_tool("make a video game for me") is None

    def test_single_generic_word_overlap_rejected(self):
        eng = _engine()
        t = _tool("foo_bar", "this tool generates random results today")
        eng._tools[t.tool_id] = t
        # All overlapping words are in stopword set.
        assert eng._find_cached_tool("generate random results today") is None


class TestRealReuseStillWorks:
    """Tools should still be reused when there's actual topical overlap."""

    def test_two_strong_keyword_overlap_matches(self):
        eng = _engine()
        t = _tool(
            "sha256_calculator",
            "Computes SHA256 hash digest of arbitrary text input",
        )
        eng._tools[t.tool_id] = t
        # "sha256" + "digest" overlap.
        match = eng._find_cached_tool("compute the SHA256 digest of bytes")
        assert match is not None
        assert match.name == "sha256_calculator"

    def test_exact_task_hash_match_still_works(self):
        eng = _engine()
        # The exact-hash code path doesn't go through the fuzzy guard.
        task = "this is the exact same task"
        import hashlib
        tid = hashlib.sha256(task.encode()).hexdigest()[:10]
        t = _tool("anything", "description has no overlap with task")
        t.tool_id = tid
        eng._tools[tid] = t
        match = eng._find_cached_tool(task)
        assert match is not None
        assert match.tool_id == tid


class TestHighestScoreWins:
    def test_picks_tool_with_more_keyword_matches(self):
        eng = _engine()
        weak = _tool("kibble", "translate spanish words and phrases")
        strong = _tool(
            "spanish_translator",
            "translate english phrases into spanish with grammar adjustment",
        )
        eng._tools["weak______"] = weak
        eng._tools["strong____"] = strong
        # "translate", "spanish", "phrases" — strong wins (matches 3 vs weak's 2).
        match = eng._find_cached_tool("translate these spanish phrases and grammar")
        assert match is strong
