"""
tests/test_config_groundwork_issue_28.py
========================================
Groundwork config fixes (issue #28-91), found when a key rotation left
the system half-configured and "a lot of things stopped working":

1. /settings/llm must write to the config file that actually WINS —
   ~/.prism/prism_config.toml overlays the repo file, so saving a key
   to the repo file while an overlay exists silently loses the save.
2. A cloud (https) OpenAI-compatible endpoint with no key is
   unavailable without a network probe; keyless stays valid for
   local/http hosts (LM Studio, llama.cpp).
3. PRISM_HERMETIC_CONFIG makes load_toml_config skip both untracked
   config files so tests never inherit a developer's live LLM setup.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import prism_settings_llm
from prism_agent_bootstrap import DEFAULT_CONFIG, load_toml_config
from prism_llm_router import LLMRouter


class TestSettingsWriteTarget:
    def test_prefers_user_overlay_when_present(self, tmp_path, monkeypatch):
        overlay = tmp_path / ".prism" / "prism_config.toml"
        overlay.parent.mkdir()
        overlay.write_text('[llm]\nopenai_api_key = ""\n')
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        assert prism_settings_llm._config_path() == overlay

        prism_settings_llm.write_llm_config({"openai_api_key": "sk-new"})
        assert 'openai_api_key = "sk-new"' in overlay.read_text()
        assert prism_settings_llm.read_llm_config()["openai_api_key"] == "sk-new"

    def test_falls_back_to_repo_file_without_overlay(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        assert prism_settings_llm._config_path() == prism_settings_llm._REPO_CONFIG


class TestKeylessCloudEndpoint:
    def test_https_host_without_key_skips_probe(self):
        r = LLMRouter(config={"openai_model": "deepseek-chat"})
        with patch("urllib.request.urlopen") as net:
            opt = r._ping_openai_compat("https://api.deepseek.com", "")
        net.assert_not_called()
        assert opt.available is False
        assert "no API key" in opt.notes

    def test_local_http_host_without_key_still_probed(self):
        r = LLMRouter(config={"openai_model": "local-model"})
        with patch("urllib.request.urlopen") as net:
            r._ping_openai_compat("http://localhost:1234", "")
        net.assert_called_once()


class TestHermeticConfig:
    def test_hermetic_skips_repo_and_user_files(self, tmp_path, monkeypatch):
        poisoned = tmp_path / "prism_config.toml"
        poisoned.write_text('[llm]\npreferred = "openai_compat"\n')
        monkeypatch.setenv("PRISM_HERMETIC_CONFIG", "1")
        cfg = load_toml_config(poisoned)
        assert cfg == DEFAULT_CONFIG
        assert cfg.get("llm", {}).get("preferred", "") != "openai_compat"

    def test_non_hermetic_still_reads_files(self, tmp_path, monkeypatch):
        repo = tmp_path / "prism_config.toml"
        repo.write_text('[llm]\npreferred = "ollama/x"\n')
        monkeypatch.delenv("PRISM_HERMETIC_CONFIG", raising=False)
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        cfg = load_toml_config(repo)
        assert cfg["llm"]["preferred"] == "ollama/x"

    def test_conftest_sets_hermetic_for_the_suite(self):
        import os
        assert os.environ.get("PRISM_HERMETIC_CONFIG") == "1"


class TestRouterTimeoutConfig:
    def test_agent_wires_llm_request_timeout(self):
        """[llm].request_timeout must reach LLMRouter — the chat-path
        counterpart of [agent].planner_timeout for slow hardware."""
        import inspect

        import prism_agent as pa
        assert "request_timeout" in inspect.getsource(pa)
        r = LLMRouter(config={}, request_timeout=120.0)
        assert r.request_timeout == 120.0
