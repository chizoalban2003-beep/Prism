from __future__ import annotations

from kde_profiles import ROLE_CAPABILITIES, UserProfile, UserRole, from_toml


def test_role_capabilities_non_empty():
    assert all(ROLE_CAPABILITIES[role] for role in UserRole)


def test_athlete_has_sports_pro():
    assert "sports_pro" in ROLE_CAPABILITIES[UserRole.ATHLETE]


def test_developer_has_ksa():
    assert "ksa" in ROLE_CAPABILITIES[UserRole.DEVELOPER]


def test_from_toml_returns_profile(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        "\n".join([
            "[user]",
            'name = "Ada"',
            'role = "coach"',
            'sport = "Football"',
            'team = "City FC"',
            "",
            "[agent]",
            'db_path = "/tmp/kde.db"',
            'media_dir = "/tmp/media"',
            'ollama_model = "mistral"',
            'ollama_host = "http://localhost:11434"',
        ]),
        encoding="utf-8",
    )
    profile = from_toml(str(path))
    assert profile.name == "Ada"
    assert profile.role == UserRole.COACH
    assert profile.team == "City FC"


def test_universal_has_all():
    universal = set(ROLE_CAPABILITIES[UserRole.UNIVERSAL])
    expected = set().union(*(set(values) for role, values in ROLE_CAPABILITIES.items() if role != UserRole.UNIVERSAL))
    assert expected.issubset(universal)


def test_has_capability():
    profile = UserProfile(name="Dev", role=UserRole.DEVELOPER)
    assert profile.has("ksa") is True
