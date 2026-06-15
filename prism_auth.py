"""
prism_auth.py
=============
Bearer-token auth for the local HTTP server.

The ASGI server is bound to 127.0.0.1, but loopback is not by itself a
trust boundary: any process running as the user (and any webpage the
user visits, modulo CORS) can talk to it. The bearer token raises the
bar to "anything able to read ~/.prism/auth_token", which on a Unix
system means the user's own processes only.

Resolution order:
    1. PRISM_AUTH_DISABLE=1 (testing escape hatch) -> None, auth off
    2. PRISM_AUTH_TOKEN env var
    3. ~/.prism/auth_token (created by ensure_token() at daemon startup)
    4. None -> auth off, middleware passes traffic through

Clients (tray, PWA, CLI) should read the token from the file and send
`Authorization: Bearer <token>` on every request.
"""
from __future__ import annotations

import logging
import os
import secrets
from pathlib import Path

logger = logging.getLogger(__name__)

TOKEN_FILE = Path.home() / ".prism" / "auth_token"

_DISABLE_VALUES = frozenset({"1", "true", "yes", "on"})


def _auth_disabled() -> bool:
    return os.environ.get("PRISM_AUTH_DISABLE", "").lower() in _DISABLE_VALUES


def _file_token() -> str | None:
    try:
        if TOKEN_FILE.exists():
            value = TOKEN_FILE.read_text(encoding="utf-8").strip()
            return value or None
    except OSError as exc:
        logger.warning("auth: failed to read %s: %s", TOKEN_FILE, exc)
    return None


def get_token() -> str | None:
    """Return the active auth token, or None if auth is disabled."""
    if _auth_disabled():
        return None
    env = os.environ.get("PRISM_AUTH_TOKEN")
    if env:
        return env
    return _file_token()


def ensure_token() -> str:
    """Idempotently materialise an auth token and return it.

    Generates ~/.prism/auth_token (chmod 600) on first call. Called by
    prism_daemon at startup so the production server is always
    authenticated. Honours PRISM_AUTH_TOKEN if set.
    """
    env = os.environ.get("PRISM_AUTH_TOKEN")
    if env:
        return env
    existing = _file_token()
    if existing:
        return existing
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    TOKEN_FILE.write_text(token, encoding="utf-8")
    try:
        os.chmod(TOKEN_FILE, 0o600)
    except OSError as exc:
        logger.warning("auth: chmod 0600 failed on %s: %s", TOKEN_FILE, exc)
    return token
