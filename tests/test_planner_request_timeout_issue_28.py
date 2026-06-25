"""PrismPlanner request_timeout for issue #28 bug 52.

Live test: typing "good morning" — which the routing table deliberately
sends to ``universal_plan`` so the user gets a daily briefing — froze
the chat for 120 seconds before reporting "Planner LLM unavailable —
model 'tinyllama' timed out after 120s".

The routing decision is *correct*: "good morning" → daily plan is a
product choice (see ``test_list_tasks_today_issue_28::test_good_morning``
and ``test_translate_intent_routing_issue_28::test_plain_morning``).
The bug is the planner's hard-coded 120s urlopen timeout — far too
long for an interactive chat path. tinyllama can't structure a plan
that fast no matter how long it has, so all the 120 s buys is a worse
UX.

Fix: make the timeout configurable on the planner, default 30 s.
"""
from __future__ import annotations

import socket
from unittest import mock

from prism_planner import PrismPlanner


class TestDefaultTimeoutIsChatFriendly:

    def test_default_is_30s_not_120s(self):
        p = PrismPlanner()
        assert p.request_timeout == 30.0

    def test_explicit_override_persists(self):
        p = PrismPlanner(request_timeout=10.0)
        assert p.request_timeout == 10.0


class TestTimeoutFlowsToOllamaCall:

    def test_ollama_call_uses_request_timeout(self):
        p = PrismPlanner(request_timeout=7.5)
        with mock.patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = b'{"response":"ok"}'
            p._call_ollama("hi")
        # Inspect kwargs of the urlopen call.
        assert urlopen.call_args.kwargs.get("timeout") == 7.5 or \
               (len(urlopen.call_args.args) >= 2 and urlopen.call_args.args[1] == 7.5)

    def test_timeout_message_reflects_configured_value(self):
        p = PrismPlanner(request_timeout=8.0)
        with mock.patch("urllib.request.urlopen", side_effect=socket.timeout("timed out")):
            result = p._call_ollama("hi")
        assert result == ""
        # Error string must say 8s, not the old hard-coded 120s.
        assert "8s" in p._last_ollama_error
        assert "120s" not in p._last_ollama_error


class TestTimeoutFlowsToClaudeCall:

    def test_claude_call_uses_request_timeout(self):
        p = PrismPlanner(claude_api_key="fake", prefer_claude=True,
                         request_timeout=12.0)
        with mock.patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = \
                b'{"content":[{"text":"ok"}]}'
            p._call_claude("hi")
        assert urlopen.call_args.kwargs.get("timeout") == 12.0 or \
               (len(urlopen.call_args.args) >= 2 and urlopen.call_args.args[1] == 12.0)
