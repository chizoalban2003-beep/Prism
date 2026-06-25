"""LLM router fast-phase fix for issue #28 bug 40.

Live test: ``/llm/status`` reported ``best: ollama/tinyllama:latest``
despite the openai_compat (DeepSeek) endpoint having a 60x lower
measured latency (1530ms vs 91106ms). The CRYSTAL-phase "fast" hint
locked routing to the first ollama option regardless of how slow it
actually was, violating the "fast" semantic.

Fix: in the "fast" branch, only return the local ollama option if its
measured ``latency_ms`` is below ``_FAST_LOCAL_LATENCY_CAP_MS``
(10s). Otherwise fall through to the ranked selection, which orders
by measured latency.
"""
from __future__ import annotations

import time

from prism_llm_router import (
    _FAST_LOCAL_LATENCY_CAP_MS,
    LLMOption,
    LLMRouter,
)


def _router_with_options(opts: list[LLMOption]) -> LLMRouter:
    """Build a router with a pre-seeded option list (skipping discovery)."""
    r = LLMRouter()
    # Match discover()'s sort key.
    r._options = sorted(
        opts,
        key=lambda o: (-o.capability, o.latency_ms if o.available else 9999),
    )
    r._discovered = True
    r._last_scan = time.time()  # keep cache fresh so discover() returns _options
    return r


class TestFastPhaseLatencyGate:
    def test_slow_local_falls_through_to_cloud(self):
        # The reported bug: tinyllama 91s, deepseek 1.5s, "fast" picked
        # tinyllama purely because it was local.
        opts = [
            LLMOption(
                provider="openai_compat", model="gpt-4", endpoint="",
                available=True, latency_ms=1530.0, capability=1,
            ),
            LLMOption(
                provider="ollama", model="tinyllama:latest", endpoint="",
                available=True, latency_ms=91106.0, capability=1,
            ),
        ]
        r = _router_with_options(opts)
        best = r.best(phase_hint="fast")
        assert best is not None
        # Must NOT pick the slow local — must fall through to the
        # actually-fast cloud option.
        assert best.provider != "ollama" or best.latency_ms <= _FAST_LOCAL_LATENCY_CAP_MS

    def test_fast_local_still_preferred(self):
        # Healthy local should still win in "fast" mode.
        opts = [
            LLMOption(
                provider="openai_compat", model="gpt-4", endpoint="",
                available=True, latency_ms=1530.0, capability=1,
            ),
            LLMOption(
                provider="ollama", model="phi:latest", endpoint="",
                available=True, latency_ms=400.0, capability=1,
            ),
        ]
        r = _router_with_options(opts)
        best = r.best(phase_hint="fast")
        assert best is not None
        assert best.provider == "ollama"

    def test_no_local_falls_through(self):
        opts = [
            LLMOption(
                provider="openai_compat", model="gpt-4", endpoint="",
                available=True, latency_ms=1530.0, capability=1,
            ),
        ]
        r = _router_with_options(opts)
        best = r.best(phase_hint="fast")
        assert best is not None
        assert best.provider == "openai_compat"

    def test_cap_boundary_exactly_at_cap_is_accepted(self):
        opts = [
            LLMOption(
                provider="openai_compat", model="gpt-4", endpoint="",
                available=True, latency_ms=500.0, capability=1,
            ),
            LLMOption(
                provider="ollama", model="phi:latest", endpoint="",
                available=True, latency_ms=float(_FAST_LOCAL_LATENCY_CAP_MS),
                capability=1,
            ),
        ]
        r = _router_with_options(opts)
        best = r.best(phase_hint="fast")
        assert best.provider == "ollama"

    def test_cap_boundary_just_above_falls_through(self):
        opts = [
            LLMOption(
                provider="openai_compat", model="gpt-4", endpoint="",
                available=True, latency_ms=500.0, capability=1,
            ),
            LLMOption(
                provider="ollama", model="phi:latest", endpoint="",
                available=True, latency_ms=_FAST_LOCAL_LATENCY_CAP_MS + 1,
                capability=1,
            ),
        ]
        r = _router_with_options(opts)
        best = r.best(phase_hint="fast")
        assert best.provider == "openai_compat"


class TestStandardSelectionUnchanged:
    """The non-"fast" path must keep ranking by (capability, latency)."""

    def test_standard_picks_highest_capability(self):
        opts = [
            LLMOption(
                provider="ollama", model="tinyllama:latest", endpoint="",
                available=True, latency_ms=200.0, capability=1,
            ),
            LLMOption(
                provider="openai_compat", model="deepseek-r1", endpoint="",
                available=True, latency_ms=2000.0, capability=3,
            ),
        ]
        r = _router_with_options(opts)
        best = r.best(min_capability=2)
        assert best.model == "deepseek-r1"

    def test_no_phase_hint_picks_by_rank_then_latency(self):
        opts = [
            LLMOption(
                provider="openai_compat", model="gpt-4", endpoint="",
                available=True, latency_ms=1500.0, capability=1,
            ),
            LLMOption(
                provider="ollama", model="tinyllama:latest", endpoint="",
                available=True, latency_ms=91000.0, capability=1,
            ),
        ]
        r = _router_with_options(opts)
        best = r.best()
        # Same capability, lower latency wins.
        assert best.provider == "openai_compat"
