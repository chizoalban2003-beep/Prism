"""LLMRouter budget gating for issue #28 bug 54.

Live probe: PRISM has spent $17.4381 of a $5.00 daily ceiling (348% over)
and continues to dispatch to DeepSeek on every chat turn. The budget card
displays the overage but no gate is enforced — ``PrismBudget`` is
constructed in ``PrismAgent.__init__`` but never consulted by the LLM
router.

Per the user directive that PRISM should be a bridge governed by
"permissions, instructions, notifications, budgets and policies",
budgets must be load-bearing — not decorative. This test pins:

1. ``LLMRouter`` accepts a ``budget_policy`` parameter (or attribute).
2. When the policy says ``not allowed`` for a paid provider, that
   provider is skipped — the router falls through to the next option.
3. Free providers (ollama / stdlib / local) are still tried because
   ``BudgetPolicy.check`` returns ``allowed=True`` for them by default.
4. With no ``budget_policy`` wired (legacy callers), no gating happens.
"""
from __future__ import annotations

from dataclasses import dataclass
from unittest import mock

from prism_llm_router import LLMOption, LLMRouter


@dataclass
class _Decision:
    allowed: bool
    reason: str = ""


class _StubBudget:
    """Minimal BudgetPolicy substitute that records every check()."""
    def __init__(self, allow: bool = True, free_providers=("ollama", "stdlib", "local")):
        self.allow = allow
        self.free_providers = set(free_providers)
        self.calls: list[str] = []

    def check(self, provider: str = "") -> _Decision:
        self.calls.append(provider)
        if provider.lower() in self.free_providers:
            return _Decision(allowed=True, reason="local — free")
        return _Decision(allowed=self.allow,
                         reason="daily budget reached" if not self.allow else "ok")


class TestRouterAcceptsBudgetPolicy:

    def test_constructor_accepts_budget_policy(self):
        b = _StubBudget()
        r = LLMRouter(budget_policy=b)
        assert r._budget_policy is b

    def test_default_is_none(self):
        r = LLMRouter()
        assert r._budget_policy is None


class TestRouterSkipsBlockedProviders:

    def _wire(self, *opts):
        r = LLMRouter()
        r._options = list(opts)
        r._discovered = True
        return r

    def test_paid_provider_blocked_when_over_budget(self):
        deepseek = LLMOption(provider="openai_compat", model="gpt-4",
                             endpoint="https://example.invalid",
                             available=True, capability=3, latency_ms=1500)
        ollama = LLMOption(provider="ollama", model="tinyllama",
                           endpoint="http://localhost:11434",
                           available=True, capability=1, latency_ms=400)
        r = self._wire(deepseek, ollama)
        r._budget_policy = _StubBudget(allow=False)
        with mock.patch.object(r, "_call_option",
                               return_value="local reply") as call:
            text, model = r.call("hello", min_capability=1)
        assert text == "local reply"
        assert "ollama" in model
        # Only the local option was actually invoked.
        called_providers = [c.args[0].provider for c in call.call_args_list]
        assert "openai_compat" not in called_providers
        assert called_providers == ["ollama"]

    def test_free_provider_not_blocked(self):
        ollama = LLMOption(provider="ollama", model="tinyllama",
                           endpoint="http://localhost:11434",
                           available=True, capability=1, latency_ms=400)
        r = self._wire(ollama)
        r._budget_policy = _StubBudget(allow=False)
        with mock.patch.object(r, "_call_option",
                               return_value="ok") as call:
            text, _ = r.call("hi", min_capability=1)
        assert text == "ok"
        assert call.call_count == 1

    def test_no_budget_policy_no_gating(self):
        deepseek = LLMOption(provider="openai_compat", model="gpt-4",
                             endpoint="https://example.invalid",
                             available=True, capability=3, latency_ms=1500)
        r = self._wire(deepseek)
        assert r._budget_policy is None
        with mock.patch.object(r, "_call_option",
                               return_value="paid reply") as call:
            text, _ = r.call("hi", min_capability=1)
        assert text == "paid reply"
        assert call.call_count == 1

    def test_budget_check_error_does_not_crash(self):
        class _BoomBudget:
            def check(self, provider=""):
                raise RuntimeError("ledger offline")
        deepseek = LLMOption(provider="openai_compat", model="gpt-4",
                             endpoint="https://example.invalid",
                             available=True, capability=3, latency_ms=1500)
        r = self._wire(deepseek)
        r._budget_policy = _BoomBudget()
        # Errors are swallowed; the call still goes through.
        with mock.patch.object(r, "_call_option",
                               return_value="paid reply"):
            text, _ = r.call("hi", min_capability=1)
        assert text == "paid reply"


class TestAgentWiresBudgetIntoRouter:
    """End-to-end wiring: PrismAgent should attach its budget to the
    router so the gate is live in production, not just in tests."""

    def test_agent_source_wires_budget_to_router(self):
        import inspect

        import prism_agent
        src = inspect.getsource(prism_agent.PrismAgent.__init__)
        # The wiring is one line; assert it survives refactors.
        assert "_router._budget_policy" in src or "budget_policy=self._budget" in src
