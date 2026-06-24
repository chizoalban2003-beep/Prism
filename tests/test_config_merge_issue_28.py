"""Config-merge fix for issue #28 — user [llm]-only config wiped agent defaults.

Live test: planner produced "Planner LLM unavailable — HTTP 404 (model
'mistral' not found?)" even though ``prism_config.toml`` set
``[agent].text_model = "tinyllama"``. Root cause: ``load_toml_config``
returned the *first* file that parsed — ``~/.prism/prism_config.toml``
took precedence, and because it only contained ``[llm]``, the entire
``[agent]`` block from the repo config was discarded. Downstream,
``PrismPlanner(ollama_model="mistral")`` defaulted in.

Fix: load both files and deep-merge the user file on top of the repo
file. These tests pin that behaviour.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from prism_agent_bootstrap import _deep_merge, load_toml_config


class TestDeepMerge:
    def test_overlay_replaces_scalar(self):
        assert _deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_nested_dicts_merge_key_by_key(self):
        base = {"agent": {"text_model": "tinyllama", "ollama_host": "x"}}
        over = {"agent": {"text_model": "mistral"}}
        assert _deep_merge(base, over) == {
            "agent": {"text_model": "mistral", "ollama_host": "x"},
        }

    def test_override_only_section_preserves_base_sections(self):
        base = {"agent": {"text_model": "tinyllama"},
                "budget": {"daily_usd": 5.0}}
        over = {"llm": {"preferred": "openai"}}
        result = _deep_merge(base, over)
        assert result["agent"]["text_model"] == "tinyllama"
        assert result["budget"]["daily_usd"] == 5.0
        assert result["llm"]["preferred"] == "openai"

    def test_deep_merge_does_not_mutate_inputs(self):
        base = {"agent": {"text_model": "tinyllama"}}
        over = {"agent": {"text_model": "mistral"}}
        _deep_merge(base, over)
        assert base == {"agent": {"text_model": "tinyllama"}}
        assert over == {"agent": {"text_model": "mistral"}}


class TestLoadTomlConfigMerges:
    """The headline regression — user [llm]-only config must not erase
    the repo's [agent] block."""

    def test_user_llm_only_preserves_repo_agent(self, tmp_path):
        repo = tmp_path / "repo.toml"
        repo.write_text(
            "[agent]\n"
            'text_model = "tinyllama"\n'
            'ollama_host = "http://localhost:11434"\n'
            "\n"
            "[budget]\n"
            "daily_usd = 5.0\n"
        )
        user_dir = tmp_path / "home" / ".prism"
        user_dir.mkdir(parents=True)
        user_file = user_dir / "prism_config.toml"
        user_file.write_text(
            "[llm]\n"
            'preferred = "openai"\n'
            'openai_api_key = "sk-test"\n'
        )
        with patch("prism_agent_bootstrap.Path.home",
                   return_value=tmp_path / "home"):
            cfg = load_toml_config(repo)
        # Repo's [agent] and [budget] survive...
        assert cfg["agent"]["text_model"] == "tinyllama"
        assert cfg["budget"]["daily_usd"] == 5.0
        # ...and user's [llm] is applied on top.
        assert cfg["llm"]["preferred"] == "openai"
        assert cfg["llm"]["openai_api_key"] == "sk-test"

    def test_user_overrides_repo_when_both_set_same_key(self, tmp_path):
        repo = tmp_path / "repo.toml"
        repo.write_text('[agent]\ntext_model = "tinyllama"\n')
        user_dir = tmp_path / "home" / ".prism"
        user_dir.mkdir(parents=True)
        (user_dir / "prism_config.toml").write_text(
            '[agent]\ntext_model = "llama3.2"\n')
        with patch("prism_agent_bootstrap.Path.home",
                   return_value=tmp_path / "home"):
            cfg = load_toml_config(repo)
        assert cfg["agent"]["text_model"] == "llama3.2"

    def test_missing_user_file_returns_repo_only(self, tmp_path):
        repo = tmp_path / "repo.toml"
        repo.write_text('[agent]\ntext_model = "tinyllama"\n')
        # no ~/.prism/prism_config.toml at all
        with patch("prism_agent_bootstrap.Path.home",
                   return_value=tmp_path / "nonexistent_home"):
            cfg = load_toml_config(repo)
        assert cfg == {"agent": {"text_model": "tinyllama"}}

    def test_missing_repo_file_returns_user_only(self, tmp_path):
        user_dir = tmp_path / "home" / ".prism"
        user_dir.mkdir(parents=True)
        (user_dir / "prism_config.toml").write_text('[llm]\npreferred = "openai"\n')
        with patch("prism_agent_bootstrap.Path.home",
                   return_value=tmp_path / "home"):
            cfg = load_toml_config(Path("/nonexistent/repo.toml"))
        assert cfg == {"llm": {"preferred": "openai"}}

    def test_both_missing_returns_empty_dict(self, tmp_path):
        with patch("prism_agent_bootstrap.Path.home",
                   return_value=tmp_path / "nonexistent_home"):
            cfg = load_toml_config(Path("/nonexistent/repo.toml"))
        assert cfg == {}
