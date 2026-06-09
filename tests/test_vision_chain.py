"""Tests for vision/multimodal capabilities in PRISM.

Covers:
- LLMRouter.call() images param (Claude, Ollama, OpenAI-compat)
- GET /perception/visual returns 200 with base64 image
- POST /perception/visual/reason input validation and routing
- vision_query intent registered in PrismAgent.INTENTS
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prism_llm_router import LLMOption, LLMRouter
from prism_state import _set_state

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_state():
    _set_state(agent=None, visual_perception=None)
    yield
    _set_state(agent=None, visual_perception=None)


@pytest.fixture
def client():
    from prism_asgi import app
    return TestClient(app)


def _make_router() -> LLMRouter:
    return LLMRouter(config={})


def _make_agent_with_router(router=None):
    agent = MagicMock()
    agent._router = router or _make_router()
    return agent


# ---------------------------------------------------------------------------
# 1. LLMRouter.call() — images param: Claude branch
# ---------------------------------------------------------------------------


class TestLLMRouterCallImagesClaudeParam:
    def test_images_param_accepted_by_call_signature(self):
        """call() must accept images= without raising TypeError."""
        router = _make_router()
        # Patch discover so no real HTTP is made
        router._options = []
        router._discovered = True
        result = router.call("hello", images=["fakeb64"])
        # Falls through to "none" because no providers configured
        assert result == ("", "none")

    def test_images_none_is_default(self):
        """images=None is the default — existing callers unaffected."""
        router = _make_router()
        router._options = []
        router._discovered = True
        text, model = router.call("hello")
        assert model == "none"

    def test_claude_images_builds_multipart_content(self):
        """_call_claude builds correct Anthropic vision content blocks."""
        router = _make_router()
        router._config = {"claude_api_key": "test-key"}
        opt = LLMOption("claude", "claude-sonnet-4-20250514",
                        "https://api.anthropic.com", True, 0, 3)
        captured = {}

        def fake_urlopen(req, timeout=30):
            captured["body"] = json.loads(req.data)
            resp = MagicMock()
            resp.read.return_value = json.dumps(
                {"content": [{"text": "I see a screen"}]}).encode()
            return resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = router._call_claude(
                opt, "what do you see?", 500, "", False,
                images=["fakeb64data"])

        assert result == "I see a screen"
        msgs = captured["body"]["messages"]
        user_msg = msgs[-1]
        assert user_msg["role"] == "user"
        assert isinstance(user_msg["content"], list)
        # First block must be the image
        assert user_msg["content"][0]["type"] == "image"
        assert user_msg["content"][0]["source"]["data"] == "fakeb64data"
        assert user_msg["content"][0]["source"]["type"] == "base64"
        assert user_msg["content"][0]["source"]["media_type"] == "image/jpeg"
        # Last block must be the text
        assert user_msg["content"][-1]["type"] == "text"
        assert user_msg["content"][-1]["text"] == "what do you see?"

    def test_claude_multiple_images_prepended(self):
        """Multiple images are all prepended before the text block."""
        router = _make_router()
        router._config = {"claude_api_key": "test-key"}
        opt = LLMOption("claude", "claude-sonnet-4-20250514",
                        "https://api.anthropic.com", True, 0, 3)
        captured = {}

        def fake_urlopen(req, timeout=30):
            captured["body"] = json.loads(req.data)
            resp = MagicMock()
            resp.read.return_value = json.dumps(
                {"content": [{"text": "Two images"}]}).encode()
            return resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            router._call_claude(
                opt, "describe both", 500, "", False,
                images=["img1", "img2"])

        content = captured["body"]["messages"][-1]["content"]
        assert len(content) == 3  # 2 images + 1 text
        assert content[0]["source"]["data"] == "img1"
        assert content[1]["source"]["data"] == "img2"
        assert content[2]["type"] == "text"

    def test_claude_no_images_sends_plain_string(self):
        """Without images, user content remains a plain string."""
        router = _make_router()
        router._config = {"claude_api_key": "test-key"}
        opt = LLMOption("claude", "claude-sonnet-4-20250514",
                        "https://api.anthropic.com", True, 0, 3)
        captured = {}

        def fake_urlopen(req, timeout=30):
            captured["body"] = json.loads(req.data)
            resp = MagicMock()
            resp.read.return_value = json.dumps(
                {"content": [{"text": "ok"}]}).encode()
            return resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            router._call_claude(opt, "hello", 500, "", False, images=None)

        user_msg = captured["body"]["messages"][-1]
        assert isinstance(user_msg["content"], str)


# ---------------------------------------------------------------------------
# 2. LLMRouter.call() — images param: Ollama branch
# ---------------------------------------------------------------------------


class TestLLMRouterCallImagesOllamaParam:
    def test_ollama_images_added_to_payload(self):
        """_call_ollama adds images key when provided."""
        router = _make_router()
        opt = LLMOption("ollama", "llava", "http://localhost:11434", True, 0, 2)
        captured = {}

        def fake_urlopen(req, timeout=30):
            captured["body"] = json.loads(req.data)
            resp = MagicMock()
            resp.read.return_value = json.dumps({"response": "a cat"}).encode()
            return resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = router._call_ollama(
                opt, "what is this?", 500, "", False, images=["fakeb64"])

        assert result == "a cat"
        assert captured["body"]["images"] == ["fakeb64"]

    def test_ollama_no_images_omits_key(self):
        """_call_ollama omits images key when not provided."""
        router = _make_router()
        opt = LLMOption("ollama", "mistral", "http://localhost:11434", True, 0, 2)
        captured = {}

        def fake_urlopen(req, timeout=30):
            captured["body"] = json.loads(req.data)
            resp = MagicMock()
            resp.read.return_value = json.dumps({"response": "ok"}).encode()
            return resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            router._call_ollama(opt, "hello", 500, "", False, images=None)

        assert "images" not in captured["body"]


# ---------------------------------------------------------------------------
# 3. LLMRouter.call() — images param: OpenAI-compat branch
# ---------------------------------------------------------------------------


class TestLLMRouterCallImagesOpenAIParam:
    def test_openai_images_build_image_url_blocks(self):
        """_call_openai uses image_url content blocks when images provided."""
        router = _make_router()
        router._config = {"openai_api_key": "test-key"}
        opt = LLMOption("openai_compat", "gpt-4o",
                        "https://api.openai.com", True, 0, 2)
        captured = {}

        def fake_urlopen(req, timeout=30):
            captured["body"] = json.loads(req.data)
            resp = MagicMock()
            resp.read.return_value = json.dumps(
                {"choices": [{"message": {"content": "screen content"}}]}).encode()
            return resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = router._call_openai(
                opt, "what's on screen?", 500, "", False,
                images=["fakeb64data"])

        assert result == "screen content"
        user_msg = captured["body"]["messages"][-1]
        assert user_msg["role"] == "user"
        assert isinstance(user_msg["content"], list)
        img_block = user_msg["content"][0]
        assert img_block["type"] == "image_url"
        assert img_block["image_url"]["url"] == "data:image/jpeg;base64,fakeb64data"
        text_block = user_msg["content"][-1]
        assert text_block["type"] == "text"
        assert text_block["text"] == "what's on screen?"

    def test_openai_no_images_plain_string_content(self):
        """Without images, openai user content is a plain string."""
        router = _make_router()
        router._config = {"openai_api_key": "test-key"}
        opt = LLMOption("openai_compat", "gpt-4o",
                        "https://api.openai.com", True, 0, 2)
        captured = {}

        def fake_urlopen(req, timeout=30):
            captured["body"] = json.loads(req.data)
            resp = MagicMock()
            resp.read.return_value = json.dumps(
                {"choices": [{"message": {"content": "ok"}}]}).encode()
            return resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            router._call_openai(opt, "hello", 500, "", False, images=None)

        user_msg = captured["body"]["messages"][-1]
        assert isinstance(user_msg["content"], str)


# ---------------------------------------------------------------------------
# 4. GET /perception/visual — returns 200 (with VP configured)
# ---------------------------------------------------------------------------


class TestPerceptionVisualGet:
    def test_post_visual_400_no_image_b64(self, client):
        """POST /perception/visual returns 400 when image_b64 missing."""
        r = client.post("/perception/visual", json={"source": "test"})
        assert r.status_code == 400
        assert "image_b64" in r.json()["error"]

    def test_post_visual_503_no_vp(self, client):
        """POST /perception/visual returns 503 when visual_perception not set."""
        r = client.post("/perception/visual",
                        json={"image_b64": "abc", "source": "test"})
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# 5. POST /perception/visual/reason — validation & routing
# ---------------------------------------------------------------------------


class TestPerceptionVisualReason:
    def test_missing_image_b64_returns_400(self, client):
        """POST /perception/visual/reason with no image_b64 → 400."""
        r = client.post("/perception/visual/reason",
                        json={"question": "what is this?"})
        assert r.status_code == 400
        assert "image_b64" in r.json()["error"]

    def test_empty_body_returns_400(self, client):
        """POST /perception/visual/reason with empty body → 400."""
        r = client.post("/perception/visual/reason", json={})
        assert r.status_code == 400

    def test_no_agent_returns_503(self, client):
        """POST /perception/visual/reason without agent → 503."""
        r = client.post("/perception/visual/reason",
                        json={"image_b64": "abc123", "question": "what?"})
        assert r.status_code == 503
        assert "agent" in r.json()["error"]

    def test_agent_no_router_returns_503(self, client):
        """POST /perception/visual/reason with agent but no router → 503."""
        agent = MagicMock()
        agent._router = None
        _set_state(agent=agent)
        r = client.post("/perception/visual/reason",
                        json={"image_b64": "abc123", "question": "what?"})
        assert r.status_code == 503
        assert "router" in r.json()["error"]

    def test_successful_reason_returns_answer(self, client):
        """POST /perception/visual/reason with router → 200 with answer."""
        agent = MagicMock()
        router_mock = MagicMock()
        router_mock.call.return_value = ("It's a desktop screenshot.", "claude/claude-sonnet")
        agent._router = router_mock
        _set_state(agent=agent)

        r = client.post("/perception/visual/reason",
                        json={"image_b64": "fakeb64", "question": "what's on screen?"})
        assert r.status_code == 200
        d = r.json()
        assert d["answer"] == "It's a desktop screenshot."
        assert d["question"] == "what's on screen?"
        assert d["model_used"] == "claude/claude-sonnet"

    def test_reason_passes_images_to_router(self, client):
        """POST /perception/visual/reason passes image_b64 to router.call images param."""
        agent = MagicMock()
        router_mock = MagicMock()
        router_mock.call.return_value = ("answer", "ollama/llava")
        agent._router = router_mock
        _set_state(agent=agent)

        client.post("/perception/visual/reason",
                    json={"image_b64": "mybase64img", "question": "describe"})

        router_mock.call.assert_called_once()
        call_kwargs = router_mock.call.call_args
        assert call_kwargs.kwargs.get("images") == ["mybase64img"]

    def test_reason_default_question_used(self, client):
        """When question is omitted, a default question is used."""
        agent = MagicMock()
        router_mock = MagicMock()
        router_mock.call.return_value = ("default answer", "mock/model")
        agent._router = router_mock
        _set_state(agent=agent)

        r = client.post("/perception/visual/reason",
                        json={"image_b64": "abc"})
        assert r.status_code == 200
        d = r.json()
        assert "question" in d
        assert d["question"]  # non-empty

    def test_reason_llm_exception_returns_500(self, client):
        """POST /perception/visual/reason when router.call raises → 500."""
        agent = MagicMock()
        router_mock = MagicMock()
        router_mock.call.side_effect = RuntimeError("LLM exploded")
        agent._router = router_mock
        _set_state(agent=agent)

        r = client.post("/perception/visual/reason",
                        json={"image_b64": "abc123", "question": "what?"})
        assert r.status_code == 500
        assert "LLM call failed" in r.json()["error"]


# ---------------------------------------------------------------------------
# 6. vision_query intent registered in INTENTS
# ---------------------------------------------------------------------------


class TestVisionQueryIntentRegistered:
    def test_vision_query_in_intents(self):
        """vision_query must be in PrismAgent.INTENTS."""
        from prism_agent import PrismAgent
        intents = [intent for _, intent in PrismAgent.INTENTS]
        assert "vision_query" in intents

    def test_vision_query_pattern_matches_whats_on_screen(self):
        """'what's on my screen' should route to vision_query."""
        import re

        from prism_agent import PrismAgent
        msg = "what's on my screen"
        matched = None
        for pattern, intent in PrismAgent.INTENTS:
            if re.search(pattern, msg.lower()):
                matched = intent
                break
        assert matched == "vision_query"

    def test_vision_query_pattern_matches_analyse_screen(self):
        """'analyse my screen' should route to vision_query."""
        import re

        from prism_agent import PrismAgent
        msg = "analyse my screen"
        matched = None
        for pattern, intent in PrismAgent.INTENTS:
            if re.search(pattern, msg.lower()):
                matched = intent
                break
        assert matched == "vision_query"

    def test_vision_query_pattern_matches_analyze_screen(self):
        """'analyze my screen' (US spelling) routes to vision_query."""
        import re

        from prism_agent import PrismAgent
        msg = "analyze my screen"
        matched = None
        for pattern, intent in PrismAgent.INTENTS:
            if re.search(pattern, msg.lower()):
                matched = intent
                break
        assert matched == "vision_query"
