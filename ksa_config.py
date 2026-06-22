"""
ksa_config.py
=============
Kinetic State Agent — Configuration Loader

Loads agent configuration from a TOML or JSON file.
Falls back to a full set of sensible defaults when no file is provided.

Supported formats:
    .toml   — preferred (requires Python 3.11+ stdlib tomllib or the
              third-party ``tomli`` package on older Pythons)
    .json   — always available

Config file search order (first found wins):
    1. Explicit path passed to load()
    2. $KSA_CONFIG environment variable
    3. ~/.ksa/config.toml
    4. ~/.ksa/config.json
    5. ./ksa_config.toml
    6. ./ksa_config.json

Usage:
    cfg = KSAConfig.load()               # auto-discover
    cfg = KSAConfig.load("myconfig.toml")

    agent = KSAgent(
        db_path      = cfg.db_path,
        working_dir  = cfg.working_dir,
        ollama_model = cfg.ollama_model,
        ollama_host  = cfg.ollama_host,
        auto_optimise= cfg.auto_optimise,
        dry_run      = cfg.dry_run,
    )

    cfg.save("~/.ksa/config.toml")       # write current config to disk
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TOML helpers (stdlib 3.11+, fallback to tomli/tomllib)
# ---------------------------------------------------------------------------

def _load_toml(path: Path) -> dict:
    """Load a TOML file using whatever is available."""
    try:
        import tomllib  # Python 3.11+
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except ImportError:
        pass
    try:
        import tomli  # third-party back-port
        with open(path, "rb") as fh:
            return tomli.load(fh)
    except ImportError:
        pass
    raise ImportError(
        "TOML support requires Python 3.11+ or the 'tomli' package "
        "(pip install tomli)."
    )


def _dump_toml_basic(data: dict, indent: int = 0) -> str:
    """
    Minimal TOML serialiser for flat/nested dicts with scalar values.
    Not a full TOML writer — sufficient for our config schema.
    """
    lines: list[str] = []
    prefix = "  " * indent

    for key, val in data.items():
        if isinstance(val, dict):
            lines.append(f"\n{prefix}[{key}]")
            lines.append(_dump_toml_basic(val, indent + 1))
        elif isinstance(val, bool):
            lines.append(f"{prefix}{key} = {'true' if val else 'false'}")
        elif isinstance(val, str):
            escaped = val.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{prefix}{key} = "{escaped}"')
        elif isinstance(val, (int, float)):
            lines.append(f"{prefix}{key} = {val}")
        elif isinstance(val, list):
            items = ", ".join(
                f'"{v}"' if isinstance(v, str) else str(v) for v in val
            )
            lines.append(f"{prefix}{key} = [{items}]")
        elif val is None:
            pass  # TOML has no null — omit
        else:
            lines.append(f'{prefix}{key} = "{val}"')

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class TaskConfig:
    """Configuration block for a single registered task."""
    task_name:   str
    keywords:    list[str]  = field(default_factory=list)
    aliases:     list[str]  = field(default_factory=list)
    description: str        = ""
    executor:    str        = ""   # executor class name, resolved at runtime


@dataclass
class KSAConfig:
    """
    Full agent configuration.

    All paths support ``~`` expansion.  Defaults match the recommended
    production layout.
    """

    # Storage
    db_path: str = "~/.ksa/state.db"

    # Execution
    working_dir:   str  = "."
    dry_run:       bool = False
    auto_optimise: bool = True

    # Routing
    confidence_floor: float = 0.25

    # Ollama LLM resolver (optional)
    ollama_model: Optional[str] = None
    ollama_host:  str           = "http://localhost:11434"

    # Optimizer
    optimizer_step_size:             float = 0.05
    optimizer_improvement_threshold: float = 0.02
    optimizer_max_arm_length:        float = 4.0
    optimizer_min_arm_length:        float = 0.5
    optimizer_max_bias:              float = 2.0
    optimizer_min_bias:              float = -2.0

    # Registered tasks (optional pre-configuration)
    tasks: list[TaskConfig] = field(default_factory=list)

    # ── Loaders ───────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: Optional[str] = None) -> KSAConfig:
        """
        Load configuration from a file.

        If ``path`` is None, the config file search order is used.
        Returns a default KSAConfig instance if no file is found.
        """
        resolved = cls._find_config(path)
        if resolved is None:
            logger.info("No config file found; using defaults.")
            return cls()

        logger.info("Loading config from %s", resolved)
        raw = cls._read_raw(resolved)
        return cls._from_dict(raw)

    @classmethod
    def _find_config(cls, explicit: Optional[str]) -> Optional[Path]:
        candidates: list[Path] = []

        if explicit:
            candidates.append(Path(os.path.expanduser(explicit)))
        else:
            env = os.environ.get("KSA_CONFIG")
            if env:
                candidates.append(Path(os.path.expanduser(env)))

            candidates += [
                Path.home() / ".ksa" / "config.toml",
                Path.home() / ".ksa" / "config.json",
                Path("ksa_config.toml"),
                Path("ksa_config.json"),
            ]

        for p in candidates:
            if p.exists():
                return p

        return None

    @classmethod
    def _read_raw(cls, path: Path) -> dict:
        suffix = path.suffix.lower()
        if suffix == ".toml":
            return _load_toml(path)
        if suffix == ".json":
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        raise ValueError(f"Unsupported config format: {path.suffix!r}")

    @classmethod
    def _from_dict(cls, raw: dict) -> KSAConfig:
        """
        Build a KSAConfig from a raw dict.
        Unknown keys are silently ignored to allow forward compatibility.
        """
        known  = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        scalar = {k: v for k, v in raw.items() if k in known and k != "tasks"}
        cfg    = cls(**{k: v for k, v in scalar.items()})

        # Parse nested task configs
        for t in raw.get("tasks", []):
            cfg.tasks.append(
                TaskConfig(
                    task_name   = t.get("task_name", ""),
                    keywords    = t.get("keywords", []),
                    aliases     = t.get("aliases", []),
                    description = t.get("description", ""),
                    executor    = t.get("executor", ""),
                )
            )

        return cfg

    # ── Savers ────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Write the current config to ``path``.
        Supports ``.toml`` and ``.json`` based on the file extension.
        The parent directory is created if it does not exist.
        """
        dest = Path(os.path.expanduser(path))
        dest.parent.mkdir(parents=True, exist_ok=True)

        data = self._to_dict()
        suffix = dest.suffix.lower()

        if suffix == ".toml":
            content = _dump_toml_basic(data)
            dest.write_text(content, encoding="utf-8")
        elif suffix == ".json":
            dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
        else:
            raise ValueError(f"Unsupported config format: {dest.suffix!r}")

        logger.info("Config saved to %s", dest)

    def _to_dict(self) -> dict:
        return asdict(self)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def resolved_db_path(self) -> str:
        return os.path.expanduser(self.db_path)

    @property
    def resolved_working_dir(self) -> str:
        return os.path.expanduser(self.working_dir)

    def __repr__(self) -> str:
        return (
            f"KSAConfig("
            f"db={self.db_path!r}, "
            f"dry_run={self.dry_run}, "
            f"auto_optimise={self.auto_optimise}, "
            f"tasks={len(self.tasks)})"
        )
