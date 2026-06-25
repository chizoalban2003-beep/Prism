"""LLMRouter request_timeout for issue #28 bug 55.

Live probe: after #28-54 shipped budget gating, DeepSeek calls are
correctly blocked when over budget. The router then falls through to
ollama tinyllama — and hangs for 120 s before failing with "timed out",
because the sync provider calls were hard-coded to ``timeout=120``.

The planner had this same anti-pattern and got fixed in #28-52 (30 s
default, configurable). The router needs the same treatment so that
fallbacks don't freeze chat for two minutes on a misconfigured local
model.

Probe (post-fix expectation): an over-budget chat falls through to
ollama, which fails fast (≤30 s) instead of locking the UI for 120 s.
"""
from __future__ import annotations

from unittest import mock

from prism_llm_router import LLMRouter, LLMOption


class TestDefaultTimeoutIsChatFriendly:

    def test_default_is_30s_not_120s(self):
        r = LLMRouter()
        assert r.request_timeout == 30.0

    def test_explicit_override_persists(self):
        r = LLMRouter(request_timeout=10.0)
        assert r.request_timeout == 10.0


class TestTimeoutFlowsToOllama:

    def test_call_ollama_uses_request_timeout(self):
        r = LLMRouter(request_timeout=7.5)
        opt = LLMOption(provider="ollama", model="tinyllama",
                        endpoint="http://localhost:11434",
                        available=True, capability=1)
        with mock.patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value.read.return_value = b'{"response":"ok"}'
            r._call_ollama(opt, "hi", max_tokens=100, system="",
                           json_mode=False, history=None)
        kwargs = urlopen.call_args.kwargs
        args = urlopen.call_args.args
        assert kwargs.get("timeout") == 7.5 or \
               (len(args) >= 2 and args[1] == 7.5)


class TestTimeoutFlowsToClaude:

    def test_call_claude_uses_request_timeout(self):
        r = LLMRouter(config={"claude_api_key": "fake"}, request_timeout=12.0)
        opt = LLMOption(provider="claude", model="claude-sonnet-4",
                        endpoint="https://api.anthropic.com",
                        available=True, capability=3)
        with mock.patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value.read.return_value = \
                b'{"content":[{"text":"ok"}]}'
            r._call_claude(opt, "hi", max_tokens=100, system="",
                           json_mode=False, history=None)
        kwargs = urlopen.call_args.kwargs
        args = urlopen.call_args.args
        assert kwargs.get("timeout") == 12.0 or \
               (len(args) >= 2 and args[1] == 12.0)


class TestTimeoutFlowsToOpenAI:

    def test_call_openai_uses_request_timeout(self):
        r = LLMRouter(config={"openai_api_key": "fake"}, request_timeout=8.0)
        opt = LLMOption(provider="openai_compat", model="gpt-4o-mini",
                        endpoint="https://api.openai.com",
                        available=True, capability=2)
        with mock.patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value.read.return_value = \
                b'{"choices":[{"message":{"content":"ok"}}]}'
            r._call_openai(opt, "hi", max_tokens=100, system="",
                           json_mode=False, history=None)
        kwargs = urlopen.call_args.kwargs
        args = urlopen.call_args.args
        assert kwargs.get("timeout") == 8.0 or \
               (len(args) >= 2 and args[1] == 8.0)


class TestNoLegacy120Hangs:
    """Defence-in-depth: no sync provider call should silently fall back
    to a 120-second urlopen. Catch the regression at the source-grep
    level so a future refactor that drops `self.request_timeout` is
    flagged immediately."""

    def test_no_hard_coded_120s_in_sync_provider_calls(self):
        import inspect
        import prism_llm_router
        for name in ("_call_claude", "_call_ollama", "_call_openai"):
            src = inspect.getsource(getattr(prism_llm_router.LLMRouter, name))
            assert "timeout=120" not in src, \
                f"{name} still hard-codes timeout=120 — use self.request_timeout"
            assert "self.request_timeout" in src, \
                f"{name} no longer references self.request_timeout"
