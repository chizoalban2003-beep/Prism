"""
prism_agent_bootstrap.py
========================
Pure config-loading helpers extracted from ``PrismAgent.__init__``.

Two functions:

* :func:`load_toml_config` — reads ``prism_config.toml``. Tries
  ``~/.prism/prism_config.toml`` first (the documented user-config
  location) and falls back to the path passed in. Returns ``{}`` when
  no ``tomllib``/``tomli`` parser is available, when the file is
  missing, or when the parse fails. Never raises.
* :func:`build_llm_config` — merges the ``[llm]`` section over the
  default key set, honours an explicit ``claude_api_key`` argument, and
  falls back to ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` environment
  variables for any keys still empty.

Both are pure: no module-level state, no ``self`` parameter, no logging
side-effects. They make the agent's bootstrap path easy to read and
keep config-shape decisions out of the constructor.
"""
from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively overlay ``override`` onto ``base`` without mutating either.

    Nested dicts are merged key-by-key; everything else is replaced. Used
    so a user config that only sets ``[llm]`` doesn't wipe out the repo
    config's ``[agent]`` / ``[budget]`` / etc. sections.
    """
    out: dict = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_toml_config(path: Path) -> dict:
    """Load a TOML config file. Returns ``{}`` on any failure.

    Loads the repo-local ``path`` as the base, then deep-merges the user's
    ``~/.prism/prism_config.toml`` on top of it so a user file that only
    sets ``[llm]`` keeps the repo's ``[agent]`` / ``[budget]`` / etc.
    sections intact. Either file may be missing; both missing returns ``{}``.
    """
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            return {}

    def _safe_load(p: Path) -> dict:
        try:
            with open(p, "rb") as fh:
                return tomllib.load(fh) or {}
        except Exception:
            return {}

    repo_cfg = _safe_load(path)
    user_cfg = _safe_load(Path.home() / ".prism" / "prism_config.toml")
    if not repo_cfg and not user_cfg:
        return {}
    return _deep_merge(repo_cfg, user_cfg)


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


def safe_init(
    label: str,
    builder: Callable[[], Any],
    *,
    logger: logging.Logger,
) -> Optional[Any]:
    """Call ``builder()`` and return its result; on any exception log
    ``"<label> not available: <exc>"`` at WARNING level and return ``None``.

    Used to collapse the try/import/construct/warn/None pattern that
    repeats throughout ``PrismAgent.__init__``. Pass the agent's own
    logger so warnings retain their original origin in log streams.
    """
    try:
        return builder()
    except Exception as exc:
        logger.warning("%s not available: %s", label, exc)
        return None


def safe_init_class(
    label: str,
    module_path: str,
    attr: str,
    *args: Any,
    logger: logging.Logger,
    info_on_success: Optional[str] = None,
    **kwargs: Any,
) -> Optional[Any]:
    """Import ``module_path``, look up ``attr`` on it, call with the given
    arguments, and return the result. Logs a WARNING and returns ``None``
    on any failure (import error, missing attribute, constructor exception).

    When ``info_on_success`` is supplied and the call succeeds, logs that
    string at INFO level — handy for the "X ready" pattern. Builders that
    need to compute the success message from the constructed instance
    should use :func:`safe_init` with a closure instead.
    """
    try:
        mod = importlib.import_module(module_path)
        target = getattr(mod, attr)
        result = target(*args, **kwargs)
    except Exception as exc:
        logger.warning("%s not available: %s", label, exc)
        return None
    if info_on_success:
        logger.info(info_on_success)
    return result
