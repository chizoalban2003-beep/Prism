"""
tests/test_llm_provider_selection_issue_28.py
=============================================
LLM provider selection end-to-end (issue #28-87):

- The planner follows the provider picked in /settings/llm via the
  agent's LLMRouter instead of its own boot-time Claude/Ollama wiring.
- The Claude save flow persists the chosen model (claude_model) and a
  masked key ("••••1234") never overwrites the stored API key.
- No source file references the retired claude-sonnet-4-20250514
  (retired 2026-06-15 — requests to it 404).
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import prism_state
from prism_planner import PrismPlanner
from prism_routes_infra import router as infra_router

REPO = Path(__file__).resolve().parent.parent


class TestPlannerFollowsRouter:
    def test_router_is_preferred_when_present(self):
        router = MagicMock()
        router.call.return_value = ("ROUTED", "claude/claude-opus-4-8")
        p = PrismPlanner(llm_router=router)
        assert p._call_llm("plan something") == "ROUTED"
        router.call.assert_called_once()

    def test_router_failure_falls_back_to_direct_path(self):
        router = MagicMock()
        router.call.side_effect = RuntimeError("router down")
        p = PrismPlanner(llm_router=router)
        with patch.object(p, "_call_ollama", return_value="OLLAMA") as direct:
            assert p._call_llm("plan something") == "OLLAMA"
            direct.assert_called_once()

    def test_router_empty_response_falls_back(self):
        router = MagicMock()
        router.call.return_value = ("", "none")
        p = PrismPlanner(llm_router=router)
        with patch.object(p, "_call_ollama", return_value="OLLAMA"):
            assert p._call_llm("plan something") == "OLLAMA"

    def test_no_router_uses_legacy_paths(self):
        p = PrismPlanner(claude_api_key="sk-ant-test")
        with patch.object(p, "_call_claude", return_value="CLAUDE") as c:
            assert p._call_llm("plan") == "CLAUDE"
            c.assert_called_once()

    def test_claude_model_configurable_and_current(self):
        p = PrismPlanner()
        assert p.claude_model == "claude-opus-4-8"


class TestSettingsSaveFlow:
    @pytest.fixture
    def client(self):
        app = FastAPI()
        app.include_router(infra_router)
        prism_state._state.clear()
        yield TestClient(app)
        prism_state._state.clear()

    def test_claude_save_persists_model(self, client):
        captured = {}
        with patch("prism_settings_llm.write_llm_config",
                   side_effect=lambda u: captured.update(u)):
            r = client.post("/settings/llm", json={
                "provider": "claude", "key": "sk-ant-new",
                "model": "claude-sonnet-5",
            })
        assert r.status_code == 200
        assert captured["claude_model"] == "claude-sonnet-5"
        assert captured["claude_api_key"] == "sk-ant-new"

    def test_claude_model_defaults_to_opus(self, client):
        captured = {}
        with patch("prism_settings_llm.write_llm_config",
                   side_effect=lambda u: captured.update(u)):
            client.post("/settings/llm",
                        json={"provider": "claude", "key": "sk-ant-new"})
        assert captured["claude_model"] == "claude-opus-4-8"

    def test_masked_key_keeps_stored_key(self, client):
        captured = {}
        with patch("prism_settings_llm.write_llm_config",
                   side_effect=lambda u: captured.update(u)), \
             patch("prism_settings_llm.read_llm_config",
                   return_value={"claude_api_key": "sk-ant-REAL"}):
            client.post("/settings/llm", json={
                "provider": "claude", "key": "••••••••1234",
                "model": "claude-haiku-4-5",
            })
        assert captured["claude_api_key"] == "sk-ant-REAL"
        assert captured["claude_model"] == "claude-haiku-4-5"


class TestNoRetiredModelIDs:
    def test_retired_sonnet_4_id_gone_from_source(self):
        # claude-sonnet-4-20250514 retired 2026-06-15; any request 404s.
        out = subprocess.run(
            ["grep", "-rln", "--include=*.py", "--include=*.toml",
             "claude-sonnet-4-20250514", str(REPO)],
            capture_output=True, text=True,
        )
        offenders = [
            line for line in out.stdout.splitlines()
            # The router comment documenting the retirement is allowed;
            # this test file mentions the ID by necessity.
            if "test_llm_provider_selection" not in line
            and "prism_llm_router.py" not in line
        ]
        assert not offenders, f"retired model ID still referenced: {offenders}"

    def test_settings_page_offers_current_models(self):
        from prism_settings_llm import get_llm_settings_html
        html = get_llm_settings_html()
        assert "claude-opus-4-8" in html
        assert "claude_model" in html
