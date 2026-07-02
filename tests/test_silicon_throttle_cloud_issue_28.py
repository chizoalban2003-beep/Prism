"""
tests/test_silicon_throttle_cloud_issue_28.py
=============================================
The silicon pressure throttle must clamp max_tokens for LOCAL calls
only (issue #28-90). Cloud calls don't burn this machine's silicon,
and clamping them truncates structured output mid-JSON — observed
live: the 700-token "high pressure" cap cut the planner's extraction
JSON from DeepSeek, failing the whole plan while the provider was
perfectly healthy.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import prism_llm_router
from prism_llm_router import LLMOption, LLMRouter


class _StubPolicy:
    def current_budget(self, delta_b=0.0, phase_name="STABLE"):
        return SimpleNamespace(
            capability_ceil=2, max_tokens=700, speculative=False,
            throttle_reason="high pressure (test)",
        )


class _StubPolicyModule:
    @staticmethod
    def get_policy():
        return _StubPolicy()


def _router_with(option: LLMOption) -> LLMRouter:
    r = LLMRouter(config={})
    r._options = [option]
    r._discovered = True
    return r


def _captured_max_tokens(option: LLMOption, requested: int) -> int:
    r = _router_with(option)
    seen: dict = {}

    def fake_call_option(opt, prompt, max_tokens, *a, **kw):
        seen["max_tokens"] = max_tokens
        return "response text"

    with patch.object(prism_llm_router, "_silicon_policy_mod",
                      _StubPolicyModule), \
         patch.object(r, "_call_option", side_effect=fake_call_option), \
         patch.object(r, "discover", return_value=[option]):
        text, model = r.call("prompt", min_capability=1,
                             max_tokens=requested)
    assert text == "response text"
    return seen["max_tokens"]


class TestSiliconThrottleScope:
    def test_cloud_call_ignores_silicon_clamp(self):
        opt = LLMOption("openai_compat", "deepseek-chat",
                        "https://api.deepseek.com", True, 100.0, 2)
        assert _captured_max_tokens(opt, requested=1500) == 1500

    def test_claude_call_ignores_silicon_clamp(self):
        opt = LLMOption("claude", "claude-opus-4-8",
                        "https://api.anthropic.com", True, 100.0, 3)
        assert _captured_max_tokens(opt, requested=1500) == 1500

    def test_local_call_still_clamped(self):
        opt = LLMOption("ollama", "tinyllama", "http://localhost:11434",
                        True, 100.0, 2)
        assert _captured_max_tokens(opt, requested=1500) == 700


class TestCompatPingLabel:
    def test_ping_labels_option_with_configured_model(self):
        r = LLMRouter(config={"openai_model": "deepseek-chat"})
        with patch("urllib.request.urlopen", side_effect=OSError("down")):
            opt = r._ping_openai_compat("https://api.deepseek.com", "k")
        assert opt.model == "deepseek-chat"

    def test_deepseek_models_ranked_capable(self):
        from prism_llm_router import _rank
        chat = LLMOption("openai_compat", "deepseek-chat", "", True)
        reasoner = LLMOption("openai_compat", "deepseek-reasoner", "", True)
        assert _rank(chat) >= 2
        assert _rank(reasoner) == 3


class TestFastHintRespectsPreference:
    def test_explicit_cloud_preference_beats_fast_local_shortcut(self):
        cloud = LLMOption("openai_compat", "deepseek-chat", "h", True, 100.0, 2)
        local = LLMOption("ollama", "tinyllama", "h", True, 100.0, 1)
        r = LLMRouter(config={}, preferred="openai_compat")
        with patch.object(r, "discover", return_value=[local, cloud]):
            picked = r.best(min_capability=1, phase_hint="fast")
        assert picked is not None and picked.provider == "openai_compat"

    def test_no_preference_keeps_fast_local_shortcut(self):
        cloud = LLMOption("openai_compat", "deepseek-chat", "h", True, 100.0, 2)
        local = LLMOption("ollama", "tinyllama", "h", True, 100.0, 1)
        r = LLMRouter(config={}, preferred="")
        with patch.object(r, "discover", return_value=[local, cloud]):
            picked = r.best(min_capability=1, phase_hint="fast")
        assert picked is not None and picked.provider == "ollama"
