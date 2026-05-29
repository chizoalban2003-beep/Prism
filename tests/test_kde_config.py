from __future__ import annotations

from kde_config import load_config


def test_load_config_prefers_prism_config_toml(tmp_path, monkeypatch):
    config = tmp_path / "prism_config.toml"
    config.write_text(
        "\n".join([
            "[user]",
            'name = "Ada"',
            'role = "coach"',
            'sport = "Football"',
            "",
            "[agent]",
            'db_path = "/tmp/prism.db"',
            'media_dir = "/tmp/prism-media"',
            'ollama_model = "mistral"',
            'ffmpeg_path = "ffmpeg"',
        ]),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    loaded = load_config()

    assert loaded.db_path == "/tmp/prism.db"
    assert loaded.media_dir == "/tmp/prism-media"


def test_load_config_uses_explicit_path_before_default_candidates(tmp_path, monkeypatch):
    default_config = tmp_path / "prism_config.toml"
    default_config.write_text(
        "\n".join([
            "[agent]",
            'db_path = "/tmp/default.db"',
        ]),
        encoding="utf-8",
    )
    explicit_config = tmp_path / "custom.toml"
    explicit_config.write_text(
        "\n".join([
            "[agent]",
            'db_path = "/tmp/explicit.db"',
        ]),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    loaded = load_config(str(explicit_config))

    assert loaded.db_path == "/tmp/explicit.db"
