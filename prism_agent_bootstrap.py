"""
prism_agent_bootstrap.py
========================
Pure config-loading helpers extracted from ``PrismAgent.__init__``.

Two functions:

* :func:`load_toml_config` — reads ``prism_config.toml`` from the package
  directory. Returns ``{}`` when no ``tomllib``/``tomli`` parser is
  available, when the file is missing, or when the parse fails. Never
  raises.
* :func:`build_llm_config` — merges the ``[llm]`` section over the
  default key set, honours an explicit ``claude_api_key`` argument, and
  falls back to ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` environment
  variables for any keys still empty.

Both are pure: no module-level state, no ``self`` parameter, no logging
side-effects. They make the agent's bootstrap path easy to read and
keep config-shape decisions out of the constructor.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional


def load_toml_config(path: Path) -> dict:
    """Load a TOML config file. Returns ``{}`` on any failure."""
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            return {}
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except Exception:
        return {}


def build_llm_config(
    toml_config: Mapping[str, Any],
    *,
    claude_api_key: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
) -> dict:
    """Build the dict passed to :class:`LLMRouter`.

    Layers (last write wins):
    1. Default keys with empty values.
    2. ``[llm]`` section from the TOML config.
    3. Explicit ``claude_api_key`` argument (constructor override).
    4. Env vars ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY``, but ONLY for
       keys that are still empty after steps 1-3.
    """
    if env is None:
        env = os.environ
    cfg: dict[str, Any] = {
        "preferred":      "",
        "fallback":       [],
        "ollama_host":    "http://localhost:11434",
        "claude_api_key": "",
        "openai_api_key": "",
    }
    cfg.update(toml_config.get("llm", {}) or {})
    if claude_api_key:
        cfg["claude_api_key"] = claude_api_key
    if not cfg.get("claude_api_key"):
        cfg["claude_api_key"] = env.get("ANTHROPIC_API_KEY", "")
    if not cfg.get("openai_api_key"):
        cfg["openai_api_key"] = env.get("OPENAI_API_KEY", "")
    return cfg
