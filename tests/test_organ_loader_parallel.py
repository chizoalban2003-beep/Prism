"""Tests for OrganLoader.execute_parallel()."""
from __future__ import annotations

import tempfile
import textwrap
from pathlib import Path

from prism_organ_loader import OrganLoader

_LOW_RISK_ORGAN = textwrap.dedent("""
    ORGAN_META   = {"intent": "fast_organ", "description": "fast", "version": "1.0"}
    ORGAN_POLICY = {"risk_level": "low", "requires_approval": False, "irreversible": False}
    def execute(intent, message, ctx):
        from prism_responses import text_card
        return text_card("fast result", intent)
""").strip()

_IRREV_ORGAN = textwrap.dedent("""
    ORGAN_META   = {"intent": "irrev_organ", "description": "irrev", "version": "1.0"}
    ORGAN_POLICY = {"risk_level": "high", "requires_approval": True, "irreversible": True}
    def execute(intent, message, ctx):
        from prism_responses import text_card
        return text_card("should not run", intent)
""").strip()

_SLOW_ORGAN = textwrap.dedent("""
    ORGAN_META   = {"intent": "slow_organ", "description": "slow", "version": "1.0"}
    ORGAN_POLICY = {"risk_level": "low", "requires_approval": False, "irreversible": False}
    def execute(intent, message, ctx):
        import time
        time.sleep(0.05)
        from prism_responses import text_card
        return text_card("slow result", intent)
""").strip()


def _loader_with(organs: dict[str, str]) -> tuple[OrganLoader, Path]:
    d = tempfile.mkdtemp()
    for name, code in organs.items():
        Path(d, f"{name}.py").write_text(code)
    return OrganLoader(bundled_dir=Path(d), user_dir=Path(d) / "user"), Path(d)


def test_execute_parallel_runs_safe_organs():
    loader, _ = _loader_with({"fast_organ": _LOW_RISK_ORGAN})
    results = loader.execute_parallel(["fast_organ"], "test", {})
    assert "fast_organ" in results
    assert results["fast_organ"] is not None


def test_execute_parallel_skips_irreversible():
    loader, _ = _loader_with({"irrev_organ": _IRREV_ORGAN})
    results = loader.execute_parallel(["irrev_organ"], "test", {})
    assert results == {}


def test_execute_parallel_runs_multiple_concurrently():
    loader, _ = _loader_with({"fast_organ": _LOW_RISK_ORGAN, "slow_organ": _SLOW_ORGAN})
    results = loader.execute_parallel(["fast_organ", "slow_organ"], "test", {})
    assert "fast_organ" in results
    assert "slow_organ" in results


def test_execute_parallel_handles_missing_intent():
    loader, _ = _loader_with({})
    results = loader.execute_parallel(["nonexistent"], "test", {})
    assert results == {}


def test_execute_parallel_mixed_safe_unsafe():
    loader, _ = _loader_with({"fast_organ": _LOW_RISK_ORGAN, "irrev_organ": _IRREV_ORGAN})
    results = loader.execute_parallel(["fast_organ", "irrev_organ"], "test", {})
    assert "fast_organ" in results
    assert "irrev_organ" not in results
