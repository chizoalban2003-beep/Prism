from __future__ import annotations

import json
import urllib.error
import urllib.request
from unittest.mock import patch

from prism_collaborator import PrismCollaborator, ResearchResult


class _MockResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_research_parses_fenced_json_from_ollama():
    collaborator = PrismCollaborator()
    payload = {
        "response": '```json\n{"findings":{"transport_cost":1.8},"confidence":0.9}\n```'
    }

    with patch.object(urllib.request, "urlopen", return_value=_MockResponse(json.dumps(payload).encode())):
        result = collaborator.research("uber surge price", ["transport_cost"], prefer_local=True)

    assert result.source == "ollama"
    assert result.findings["transport_cost"] == 1.8
    assert result.to_factor_updates() == {"transport_cost": 1.0}


def test_research_falls_back_to_heuristics_when_ollama_fails():
    collaborator = PrismCollaborator()

    with patch.object(urllib.request, "urlopen", side_effect=urllib.error.URLError("offline")):
        result = collaborator.research(
            "Pizza Palace official website online ordering",
            ["website_url", "has_online_ordering", "phone_number"],
            prefer_local=True,
        )

    assert result.source == "local_heuristic"
    assert result.findings["website_url"].startswith("https://www.")
    assert result.findings["has_online_ordering"] is True
    assert result.findings["phone_number"].startswith("+44 20 ")


def test_research_uses_web_search_when_enabled_after_ollama_failure():
    collaborator = PrismCollaborator(use_web_search=True)
    expected = ResearchResult(
        query="topic",
        findings={"result_count": 3},
        raw_response="html",
        source="web_search",
        confidence=0.35,
    )

    with patch.object(collaborator, "_call_ollama_raw", side_effect=OSError("offline")):
        with patch.object(collaborator, "_call_web_search", return_value=expected):
            result = collaborator.research("topic", prefer_local=True)

    assert result is expected


def test_call_claude_raw_sends_api_key_and_extracts_text():
    collaborator = PrismCollaborator(claude_api_key="secret-key")
    captured = {}

    def _fake_urlopen(request, timeout=0):
        captured["headers"] = dict(request.header_items())
        body = {
            "content": [
                {"type": "text", "text": '{"findings":{"eta_min":12},"confidence":0.8}'}
            ]
        }
        return _MockResponse(json.dumps(body).encode())

    with patch.object(urllib.request, "urlopen", side_effect=_fake_urlopen):
        raw = collaborator._call_claude_raw("prompt")

    assert json.loads(raw)["findings"]["eta_min"] == 12
    assert captured["headers"]["X-api-key"] == "secret-key"


def test_synthesise_tool_requires_sandbox_test_when_testing():
    collaborator = PrismCollaborator(claude_api_key="key")
    generated = """
class DemoExecutor:
    def execute(self, **kwargs):
        return {"success": True}
"""

    with patch.object(collaborator, "_call_claude_raw", return_value=generated):
        ok, message = collaborator.synthesise_tool(
            {
                "task_name": "demo",
                "description": "demo",
                "inputs": {"item_name": "str"},
                "expected_output": {"order_id": "str"},
            }
        )

    assert ok is False
    assert "Sandbox test failed" in message
