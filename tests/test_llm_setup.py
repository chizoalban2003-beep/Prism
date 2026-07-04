"""
Tests for prism_setup_llm, prism_settings_llm, and the agent's new
LLM config path (full [llm] dict → LLMRouter constructor).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_toml(llm: dict, path: Path) -> None:
    lines = ["[llm]"]
    for k, v in llm.items():
        lines.append(f"{k} = {json.dumps(v)}")
    path.write_text("\n".join(lines) + "\n")


# ── prism_settings_llm ─────────────────────────────────────────────────────────

class TestSettingsLLMReadWrite:

    def test_read_llm_config_returns_dict(self, tmp_path, monkeypatch):
        cfg = tmp_path / "prism_config.toml"
        _make_toml({"claude_api_key": "sk-test", "preferred": "claude"}, cfg)
        import prism_settings_llm as m
        monkeypatch.setattr(m, "_CONFIG_PATH", cfg)
        result = m.read_llm_config()
        assert result["claude_api_key"] == "sk-test"
        assert result["preferred"] == "claude"

    def test_read_llm_config_missing_file(self, tmp_path, monkeypatch):
        import prism_settings_llm as m
        monkeypatch.setattr(m, "_CONFIG_PATH", tmp_path / "nofile.toml")
        assert m.read_llm_config() == {}

    def test_write_llm_config_creates_section(self, tmp_path, monkeypatch):
        cfg = tmp_path / "prism_config.toml"
        cfg.write_text("[user]\nname = \"Alice\"\n")
        import prism_settings_llm as m
        monkeypatch.setattr(m, "_CONFIG_PATH", cfg)
        m.write_llm_config({"claude_api_key": "sk-ant-abc", "preferred": "claude"})
        text = cfg.read_text()
        assert "claude_api_key" in text
        assert "sk-ant-abc" in text
        assert "[user]" in text  # other sections preserved

    def test_write_llm_config_updates_existing(self, tmp_path, monkeypatch):
        cfg = tmp_path / "prism_config.toml"
        _make_toml({"claude_api_key": "old-key", "preferred": ""}, cfg)
        import prism_settings_llm as m
        monkeypatch.setattr(m, "_CONFIG_PATH", cfg)
        m.write_llm_config({"claude_api_key": "new-key", "preferred": "claude"})
        result = m.read_llm_config()
        assert result["claude_api_key"] == "new-key"
        assert result["preferred"] == "claude"
        # old key gone
        assert "old-key" not in cfg.read_text()

    def test_write_llm_config_merges_keys(self, tmp_path, monkeypatch):
        cfg = tmp_path / "prism_config.toml"
        _make_toml({"ollama_host": "http://localhost:11434", "claude_api_key": ""}, cfg)
        import prism_settings_llm as m
        monkeypatch.setattr(m, "_CONFIG_PATH", cfg)
        m.write_llm_config({"claude_api_key": "sk-new"})
        result = m.read_llm_config()
        assert result["ollama_host"] == "http://localhost:11434"  # retained
        assert result["claude_api_key"] == "sk-new"              # updated

    def test_write_llm_config_fallback_list(self, tmp_path, monkeypatch):
        cfg = tmp_path / "prism_config.toml"
        cfg.write_text("")
        import prism_settings_llm as m
        monkeypatch.setattr(m, "_CONFIG_PATH", cfg)
        m.write_llm_config({"fallback": ["ollama/mistral", "claude"], "preferred": "claude"})
        result = m.read_llm_config()
        assert result["fallback"] == ["ollama/mistral", "claude"]


class TestSettingsLLMHtml:

    def test_html_returns_string(self, tmp_path, monkeypatch):
        cfg = tmp_path / "prism_config.toml"
        _make_toml({"claude_api_key": "", "ollama_host": "http://localhost:11434"}, cfg)
        import prism_settings_llm as m
        monkeypatch.setattr(m, "_CONFIG_PATH", cfg)
        html = m.get_llm_settings_html()
        assert isinstance(html, str)
        assert len(html) > 4000

    def test_html_contains_provider_cards(self, tmp_path, monkeypatch):
        cfg = tmp_path / "prism_config.toml"
        _make_toml({}, cfg)
        import prism_settings_llm as m
        monkeypatch.setattr(m, "_CONFIG_PATH", cfg)
        html = m.get_llm_settings_html()
        assert "Ollama" in html
        assert "Claude" in html
        assert "OpenAI" in html
        assert "OpenAI-compatible" in html

    def test_html_masks_existing_key(self, tmp_path, monkeypatch):
        cfg = tmp_path / "prism_config.toml"
        _make_toml({"claude_api_key": "sk-ant-supersecret123"}, cfg)
        import prism_settings_llm as m
        monkeypatch.setattr(m, "_CONFIG_PATH", cfg)
        html = m.get_llm_settings_html()
        assert "sk-ant-supersecret123" not in html
        assert "•" in html  # masked

    def test_html_settings_routes(self, tmp_path, monkeypatch):
        cfg = tmp_path / "prism_config.toml"
        _make_toml({}, cfg)
        import prism_settings_llm as m
        monkeypatch.setattr(m, "_CONFIG_PATH", cfg)
        html = m.get_llm_settings_html()
        assert "/settings/llm" in html
        assert "/settings/llm/test" in html


class TestSettingsLLMTestProvider:

    def test_test_unknown_provider(self):
        import prism_settings_llm as m
        result = m.test_provider("nonexistent_provider")
        assert result["ok"] is False
        assert "Unknown" in result["message"]

    def test_test_claude_no_key(self):
        import prism_settings_llm as m
        result = m.test_provider("claude", key="")
        assert result["ok"] is False

    def test_test_ollama_unreachable(self):
        import prism_settings_llm as m
        result = m.test_provider("ollama", host="http://127.0.0.1:19999")
        assert result["ok"] is False
        assert "models" in result  # always returns models key

    def test_test_openai_no_key(self):
        import prism_settings_llm as m
        result = m.test_provider("openai", key="", host="https://api.openai.com")
        assert result["ok"] is False

    def test_test_compat_unreachable_host(self):
        import prism_settings_llm as m
        result = m.test_provider("openai_compat", key="local",
                                 host="http://127.0.0.1:19998")
        assert result["ok"] is False


# ── prism_setup_llm ────────────────────────────────────────────────────────────

class TestSetupLLMHelpers:

    def test_read_config_missing(self, tmp_path, monkeypatch):
        import prism_setup_llm as m
        monkeypatch.setattr(m, "_CONFIG_PATH", tmp_path / "nope.toml")
        assert m._read_config() == {}

    def test_read_config_returns_full_dict(self, tmp_path, monkeypatch):
        cfg = tmp_path / "prism_config.toml"
        cfg.write_text("[llm]\nclaude_api_key = \"sk-x\"\n[user]\nname = \"Bob\"\n")
        import prism_setup_llm as m
        monkeypatch.setattr(m, "_CONFIG_PATH", cfg)
        data = m._read_config()
        assert data["llm"]["claude_api_key"] == "sk-x"
        assert data["user"]["name"] == "Bob"

    def test_write_llm_section_creates_new(self, tmp_path, monkeypatch):
        cfg = tmp_path / "prism_config.toml"
        cfg.write_text("[user]\nname = \"test\"\n")
        import prism_setup_llm as m
        monkeypatch.setattr(m, "_CONFIG_PATH", cfg)
        m._write_llm_section({"preferred": "claude", "claude_api_key": "sk-y"})
        text = cfg.read_text()
        assert "[llm]" in text
        assert "claude" in text
        assert "[user]" in text  # preserved

    def test_write_llm_section_replaces_existing(self, tmp_path, monkeypatch):
        cfg = tmp_path / "prism_config.toml"
        cfg.write_text("[llm]\nclaude_api_key = \"old\"\n\n[voice]\nenabled = true\n")
        import prism_setup_llm as m
        monkeypatch.setattr(m, "_CONFIG_PATH", cfg)
        m._write_llm_section({"claude_api_key": "new", "preferred": "claude"})
        text = cfg.read_text()
        assert "new" in text
        assert "old" not in text
        assert "[voice]" in text   # preserved

    def test_test_ollama_unreachable(self):
        import prism_setup_llm as m
        ok, msgs = m._test_ollama("http://127.0.0.1:19997")
        assert ok is False
        assert isinstance(msgs, list)

    def test_test_claude_empty_key(self):
        import prism_setup_llm as m
        ok, msg = m._test_claude("")
        assert ok is False


# ── Agent LLMRouter config path ────────────────────────────────────────────────

class TestAgentLLMRouterConfigPath:
    """Verify prism_agent.__init__ now passes full [llm] config to LLMRouter."""

    def _make_agent_config(self, tmp_path: Path, llm: dict) -> Path:
        cfg = tmp_path / "prism_config.toml"
        lines = ["[llm]"]
        for k, v in llm.items():
            lines.append(f"{k} = {json.dumps(v)}")
        cfg.write_text("\n".join(lines) + "\n")
        return cfg

    def test_agent_reads_ollama_host_from_config(self, tmp_path, monkeypatch):
        # This test IS about config loading — opt out of the suite-wide
        # hermetic mode so load_toml_config reads the file we build.
        monkeypatch.delenv("PRISM_HERMETIC_CONFIG", raising=False)
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "no-home"))
        cfg = self._make_agent_config(tmp_path, {
            "ollama_host": "http://myhost:9999",
            "ollama_model": "llama3",
            "preferred": "ollama/llama3",
        })
        with patch("prism_agent.Path") as mock_path:
            # Make __file__.parent / "prism_config.toml" return our cfg
            mock_path.return_value.parent.__truediv__ = MagicMock(return_value=cfg)
            # Patch open to use our file
            import builtins
            real_open = builtins.open
            def mock_open(path, *a, **kw):
                if "prism_config" in str(path):
                    return real_open(str(cfg), *a, **kw)
                return real_open(path, *a, **kw)
            with patch("builtins.open", side_effect=mock_open):
                from prism_agent import PrismAgent
                agent = PrismAgent()
                assert agent._router._ollama_host == "http://myhost:9999"
                assert agent._router._preferred == "ollama/llama3"

    def test_agent_router_env_overrides_config(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env-test-key")
        from prism_agent import PrismAgent
        agent = PrismAgent()
        assert agent._router._config.get("claude_api_key") == "sk-env-test-key"

    def test_agent_router_config_has_all_keys(self):
        from prism_agent import PrismAgent
        agent = PrismAgent()
        cfg = agent._router._config
        for key in ("preferred", "ollama_host", "claude_api_key", "openai_api_key"):
            assert key in cfg, f"Expected {key!r} in router config"

    def test_agent_router_fallback_list(self, tmp_path, monkeypatch):
        # Config-loading test — opt out of suite-wide hermetic mode.
        monkeypatch.delenv("PRISM_HERMETIC_CONFIG", raising=False)
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "no-home"))
        cfg = self._make_agent_config(tmp_path, {
            "preferred": "claude",
            "fallback": ["ollama/mistral", "openai"],
        })
        import builtins

        import prism_agent
        real_open = builtins.open
        def mock_open(path, *a, **kw):
            if "prism_config" in str(path):
                return real_open(str(cfg), *a, **kw)
            return real_open(path, *a, **kw)
        with patch("builtins.open", side_effect=mock_open):
            agent = prism_agent.PrismAgent()
            assert agent._router._preferred == "claude"
            assert agent._router._fallback == ["ollama/mistral", "openai"]

    def test_agent_constructor_claude_key_override(self):
        from prism_agent import PrismAgent
        agent = PrismAgent(claude_api_key="sk-ant-override")
        assert agent._router._config.get("claude_api_key") == "sk-ant-override"


# ── LLMRouter full config constructor ─────────────────────────────────────────

class TestLLMRouterFullConfig:

    def test_router_receives_all_config_keys(self):
        from prism_llm_router import LLMRouter
        cfg = {
            "claude_api_key": "sk-x",
            "openai_api_key": "sk-y",
            "openai_host":    "https://api.groq.com",
            "ollama_host":    "http://mybox:11434",
            "preferred":      "claude",
            "fallback":       ["ollama/mistral"],
        }
        r = LLMRouter(
            preferred   = cfg["preferred"],
            fallback    = cfg["fallback"],
            ollama_host = cfg["ollama_host"],
            config      = cfg,
        )
        assert r._ollama_host == "http://mybox:11434"
        assert r._preferred   == "claude"
        assert r._fallback    == ["ollama/mistral"]
        assert r._config["openai_host"] == "https://api.groq.com"

    def test_router_discovers_claude_from_config(self):
        from prism_llm_router import LLMRouter
        with patch("urllib.request.urlopen") as mock_ul:
            resp = MagicMock()
            resp.read.return_value = json.dumps({
                "model": "claude-haiku-4-5-20251001",
                "content": [{"text": "pong"}]
            }).encode()
            resp.__enter__ = lambda s: s
            resp.__exit__  = MagicMock(return_value=False)
            mock_ul.return_value = resp

            r = LLMRouter(config={"claude_api_key": "sk-fake"})
            opts = r.discover(force=True)
            claude_opts = [o for o in opts if o.provider == "claude"]
            assert len(claude_opts) == 1
            assert claude_opts[0].available is True

    def test_router_best_returns_claude_over_stdlib(self):
        from prism_llm_router import LLMOption, LLMRouter
        r = LLMRouter(config={"claude_api_key": "sk-fake"})
        with patch.object(r, "discover", return_value=[
            LLMOption("claude", "claude-opus-4-8", "https://api.anthropic.com",
                      available=True, capability=3),
            LLMOption("stdlib", "stdlib", "", available=True, capability=0),
        ]):
            best = r.best(min_capability=1)
            assert best is not None
            assert best.provider == "claude"

    def test_router_best_returns_none_when_only_stdlib(self):
        from prism_llm_router import LLMOption, LLMRouter
        r = LLMRouter()
        with patch.object(r, "discover", return_value=[
            LLMOption("stdlib", "stdlib", "", available=True, capability=0),
        ]):
            best = r.best(min_capability=1)
            assert best is None

    def test_router_openai_compat_endpoint(self):
        from prism_llm_router import LLMRouter
        with patch("urllib.request.urlopen") as mock_ul:
            resp = MagicMock()
            resp.read.return_value = json.dumps(
                {"choices": [{"message": {"content": "hi"}}]}
            ).encode()
            resp.__enter__ = lambda s: s
            resp.__exit__  = MagicMock(return_value=False)
            mock_ul.return_value = resp

            r = LLMRouter(config={
                "openai_api_key": "gsk-groq-test",
                "openai_host":    "https://api.groq.com",
            })
            # Just ensure the config is stored and accessible
            assert r._config["openai_host"] == "https://api.groq.com"
            assert r._config["openai_api_key"] == "gsk-groq-test"


# ── prism_settings_llm route smoke ────────────────────────────────────────────

class TestKDEServerLLMRoutes:
    """Light check that the LLM settings helpers are wired up."""

    def test_settings_html_importable(self):
        from prism_settings_llm import get_llm_settings_html
        html = get_llm_settings_html()
        assert "/settings/llm/test" in html

    def test_test_provider_helper_callable(self):
        from prism_settings_llm import test_provider
        result = test_provider("ollama", host="http://127.0.0.1:19996")
        assert "ok" in result
        assert "message" in result

    def test_write_config_callable(self, tmp_path, monkeypatch):
        cfg = tmp_path / "prism_config.toml"
        cfg.write_text("[llm]\npreferred = \"\"\n")
        import prism_settings_llm as m
        monkeypatch.setattr(m, "_CONFIG_PATH", cfg)
        m.write_llm_config({"preferred": "claude", "claude_api_key": "sk-test"})
        result = m.read_llm_config()
        assert result["preferred"] == "claude"
