from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class UserRole(str, Enum):
    DEVELOPER = "developer"
    ATHLETE = "athlete"
    COACH = "coach"
    ANALYST = "analyst"
    PHYSIO = "physiotherapist"
    AGENT = "agent"
    UNIVERSAL = "universal"


ROLE_DESCRIPTIONS = {
    UserRole.DEVELOPER: "Task routing, file ops, code assistance — KSA mode",
    UserRole.ATHLETE: "Daily planning, session analysis, recovery, wearables",
    UserRole.COACH: "Squad management, tactical prep, opposition scouting",
    UserRole.ANALYST: "Duel networks, moment prediction, StatsBomb pipeline",
    UserRole.PHYSIO: "Player load, injury risk, return-to-play protocols",
    UserRole.AGENT: "Player reports, transfer values, contract context",
    UserRole.UNIVERSAL: "All KSA + KDE capabilities",
}

ROLE_CAPABILITIES = {
    UserRole.DEVELOPER: ["ksa", "tasks"],
    UserRole.ATHLETE: ["sports_pro", "daily_workflow", "device_hub",
                       "moment_analyzer", "prediction_engine"],
    UserRole.COACH: ["sports_pro", "daily_workflow", "duel_analyzer",
                     "moment_analyzer", "prediction_engine", "sport_tasks"],
    UserRole.ANALYST: ["duel_analyzer", "moment_analyzer", "moment_pipeline",
                       "prediction_engine", "sport_data"],
    UserRole.PHYSIO: ["sports_pro", "daily_workflow", "prediction_engine"],
    UserRole.AGENT: ["prediction_engine", "sport_tasks"],
    UserRole.UNIVERSAL: ["ksa", "sports_pro", "daily_workflow", "device_hub",
                         "duel_analyzer", "moment_analyzer", "moment_pipeline",
                         "prediction_engine", "domain_configs", "sport_tasks",
                         "sport_data", "tasks"],
}

DEFAULT_CONFIG_PATH = Path(os.environ.get("KDE_CONFIG", "~/.kde/config.toml")).expanduser()


@dataclass
class UserProfile:
    name: str
    role: UserRole
    sport: str = "Football"
    team: str = ""
    db_path: str = "~/.kde/kde.db"
    media_dir: str = "~/.kde/media"
    ollama_model: str = "mistral"
    ollama_host: str = "http://localhost:11434"
    text_model: str = "mistral"
    ffmpeg_path: str = "ffmpeg"
    poll_interval: int = 30
    auto_watch: bool = True
    capabilities: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.capabilities:
            self.capabilities = list(ROLE_CAPABILITIES.get(self.role, []))
        else:
            self.capabilities = list(self.capabilities)

    @property
    def is_developer(self) -> bool:
        return self.role in (UserRole.DEVELOPER, UserRole.UNIVERSAL)

    @property
    def is_sport(self) -> bool:
        return self.role not in (UserRole.DEVELOPER,)

    def has(self, capability: str) -> bool:
        return capability in self.capabilities


def _load_toml(path: Path) -> dict:
    try:
        import tomllib

        return tomllib.loads(path.read_text(encoding="utf-8"))
    except ImportError:
        import tomli

        return tomli.loads(path.read_text(encoding="utf-8"))


def _parse_role(raw: Optional[str]) -> UserRole:
    text = (raw or UserRole.ATHLETE.value).strip().lower()
    aliases = {
        "dev": UserRole.DEVELOPER,
        "developer": UserRole.DEVELOPER,
        "athlete": UserRole.ATHLETE,
        "coach": UserRole.COACH,
        "analyst": UserRole.ANALYST,
        "physio": UserRole.PHYSIO,
        "physiotherapist": UserRole.PHYSIO,
        "agent": UserRole.AGENT,
        "universal": UserRole.UNIVERSAL,
    }
    return aliases.get(text, UserRole.ATHLETE)


def write_toml(profile: UserProfile, path: str | None = None) -> str:
    config_path = Path(path).expanduser() if path else DEFAULT_CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join([
        "[user]",
        f'name = "{profile.name}"',
        f'role = "{profile.role.value}"',
        f'sport = "{profile.sport}"',
        f'team = "{profile.team}"',
        "",
        "[agent]",
        f'db_path = "{profile.db_path}"',
        f'media_dir = "{profile.media_dir}"',
        f'ollama_model = "{profile.ollama_model}"',
        f'ollama_host = "{profile.ollama_host}"',
        f'text_model = "{profile.text_model}"',
        f'ffmpeg_path = "{profile.ffmpeg_path}"',
        f"poll_interval = {profile.poll_interval}",
        f"auto_watch = {'true' if profile.auto_watch else 'false'}",
        "",
        "capabilities = ["
        + ", ".join(f'"{capability}"' for capability in profile.capabilities)
        + "]",
        "",
    ])
    config_path.write_text(payload, encoding="utf-8")
    return str(config_path)


def from_toml(path: str) -> UserProfile:
    data = _load_toml(Path(path).expanduser())
    user = data.get("user", {})
    agent = data.get("agent", {})
    capabilities = data.get("capabilities")
    if capabilities is None:
        capabilities = agent.get("capabilities", user.get("capabilities", []))
    profile = UserProfile(
        name=user.get("name", agent.get("name", "KDE User")),
        role=_parse_role(user.get("role", agent.get("role"))),
        sport=user.get("sport", agent.get("sport", "Football")),
        team=user.get("team", agent.get("team", "")),
        db_path=agent.get("db_path", "~/.kde/kde.db"),
        media_dir=agent.get("media_dir", "~/.kde/media"),
        ollama_model=agent.get("ollama_model", "mistral"),
        ollama_host=agent.get("ollama_host", "http://localhost:11434"),
        text_model=agent.get("text_model", agent.get("ollama_model", "mistral")),
        ffmpeg_path=agent.get("ffmpeg_path", "ffmpeg"),
        poll_interval=int(agent.get("poll_interval", 30)),
        auto_watch=bool(agent.get("auto_watch", True)),
        capabilities=list(capabilities or []),
    )
    return profile


def setup_wizard() -> UserProfile:
    print("\n=== KDE Profile Setup ===\n")
    name = input("Your name: ").strip() or "KDE User"
    roles = list(UserRole)
    for index, role in enumerate(roles, start=1):
        print(f"{index}. {role.value:<15} {ROLE_DESCRIPTIONS[role]}")
    choice = input(f"Choose role [1-{len(roles)}] (default 2): ").strip() or "2"
    try:
        role = roles[max(1, min(len(roles), int(choice))) - 1]
    except ValueError:
        role = _parse_role(choice)

    sport = "Football"
    team = ""
    if role != UserRole.DEVELOPER:
        sport = input("Sport [Football]: ").strip() or "Football"
        team = input("Team (optional): ").strip()

    profile = UserProfile(name=name, role=role, sport=sport, team=team)
    path = write_toml(profile)
    print(f"\n✓ Saved profile to {path}\n")
    return profile
