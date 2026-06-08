"""Tests for prism_speculative — 12 tests."""
from __future__ import annotations

from unittest.mock import MagicMock

import prism_speculative as _spec_mod
from prism_silicon_policy import ExecutionBudget
from prism_speculative import SpeculativeDecodingPipeline, get_pipeline


def _budget(capability_ceil: int = 3, max_tokens: int = 1500, speculative: bool = False) -> ExecutionBudget:
    return ExecutionBudget(
        capability_ceil=capability_ceil,
        max_tokens=max_tokens,
        speculative=speculative,
    )


def _make_router(draft_response: str = "", verify_response: str = "") -> MagicMock:
    """Return a mock router whose call() returns draft then verify responses."""
    router = MagicMock()
    router.call.side_effect = [
        (draft_response, "draft_model/small"),
        (verify_response, "verify_model/large"),
    ]
    return router


class TestSpeculativeDecodingPipeline:
    def setup_method(self):
        # Reset the module-level singleton so tests are isolated
        _spec_mod._pipeline = None

    def test_draft_always_called(self):
        # The pipeline always invokes router at capability=1 for draft
        router = _make_router(draft_response="x " * 50, verify_response="x " * 50)
        pipeline = SpeculativeDecodingPipeline(router)
        budget = _budget(capability_ceil=3, max_tokens=1500)
        pipeline.call("test prompt", budget=budget)
        first_call = router.call.call_args_list[0]
        assert first_call[1].get("min_capability", first_call[0][1] if len(first_call[0]) > 1 else None) == 1 or \
               first_call.kwargs.get("min_capability") == 1

    def test_bypass_on_capability_1_budget(self):
        # capability_ceil=1 → bypass, only draft call
        router = _make_router(draft_response="x " * 50, verify_response="should not be called")
        pipeline = SpeculativeDecodingPipeline(router)
        result = pipeline.call("test prompt", budget=_budget(capability_ceil=1))
        assert result.verified is False
        assert result.verify_model == ""
        assert router.call.call_count == 1

    def test_bypass_on_short_draft(self):
        # draft under 30 words → bypass verification
        short_draft = "short response only five words"
        router = _make_router(draft_response=short_draft, verify_response="irrelevant")
        pipeline = SpeculativeDecodingPipeline(router)
        result = pipeline.call("test", budget=_budget(capability_ceil=3))
        assert result.verified is False
        assert router.call.call_count == 1

    def test_verify_called_on_healthy_budget(self):
        # capability=3, non-speculative, long draft → verify stage runs
        long_draft = "word " * 40  # 40 words >= _MIN_VERIFY_TOKENS
        verify_resp = "word " * 40
        router = _make_router(draft_response=long_draft, verify_response=verify_resp)
        pipeline = SpeculativeDecodingPipeline(router)
        result = pipeline.call("test prompt", budget=_budget(capability_ceil=3, speculative=False))
        assert result.verified is True
        assert router.call.call_count == 2

    def test_corrected_flag_when_different(self):
        # target returns different text → corrected=True
        long_draft = "word " * 40
        different = "different " * 40
        router = _make_router(draft_response=long_draft, verify_response=different)
        pipeline = SpeculativeDecodingPipeline(router)
        result = pipeline.call("test", budget=_budget(capability_ceil=3, speculative=False))
        assert result.corrected is True
        assert result.response == different.strip() or result.response == different

    def test_corrected_false_when_same(self):
        # target returns exact same text → corrected=False
        text = "word " * 40
        router = _make_router(draft_response=text, verify_response=text)
        pipeline = SpeculativeDecodingPipeline(router)
        result = pipeline.call("test", budget=_budget(capability_ceil=3, speculative=False))
        assert result.corrected is False
        assert result.response == text

    def test_stats_increment(self):
        router = MagicMock()
        # First call: long draft → verify
        router.call.side_effect = [
            ("word " * 40, "draft/m"),
            ("word " * 40, "verify/m"),
            # Second call: short draft → bypass
            ("short", "draft/m"),
        ]
        pipeline = SpeculativeDecodingPipeline(router)
        pipeline.call("q1", budget=_budget(capability_ceil=3, speculative=False))
        pipeline.call("q2", budget=_budget(capability_ceil=1))
        assert pipeline.stats["drafts"] == 2
        assert pipeline.stats["verifications"] == 1
        assert pipeline.stats["bypasses"] == 1

    def test_correction_rate_zero_no_corrections(self):
        text = "word " * 40
        router = _make_router(draft_response=text, verify_response=text)
        pipeline = SpeculativeDecodingPipeline(router)
        pipeline.call("test", budget=_budget(capability_ceil=3, speculative=False))
        assert pipeline.correction_rate == 0.0

    def test_bypass_rate(self):
        # All calls bypass → bypass_rate = 1.0
        router = MagicMock()
        router.call.return_value = ("short", "m")
        pipeline = SpeculativeDecodingPipeline(router)
        for _ in range(3):
            pipeline.call("test", budget=_budget(capability_ceil=1))
        assert pipeline.bypass_rate == 1.0

    def test_speculative_result_fields(self):
        router = _make_router(draft_response="short", verify_response="n/a")
        pipeline = SpeculativeDecodingPipeline(router)
        result = pipeline.call("test", budget=_budget(capability_ceil=1))
        assert hasattr(result, "response")
        assert hasattr(result, "verified")
        assert hasattr(result, "draft_model")
        assert hasattr(result, "verify_model")
        assert hasattr(result, "latency_ms")
        assert hasattr(result, "corrected")
        assert isinstance(result.latency_ms, float)

    def test_get_pipeline_singleton_with_router(self):
        _spec_mod._pipeline = None
        router = MagicMock()
        p1 = get_pipeline(router=router)
        p2 = get_pipeline(router=router)
        assert p1 is p2
        assert p1 is not None

    def test_get_pipeline_returns_none_without_router(self):
        _spec_mod._pipeline = None
        result = get_pipeline(router=None)
        assert result is None
