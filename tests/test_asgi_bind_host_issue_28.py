"""
tests/test_asgi_bind_host_issue_28.py
=====================================
Bind-host policy (issue #28-88): loopback-only by default, with a
single sanctioned exception for containers.

Before: serve() hard-asserted 127.0.0.1, so inside Docker the app
listened on container-loopback and the published port connected to
nothing — while the in-container healthcheck (curl localhost) passed,
masking the breakage.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

import prism_asgi


def _serve(host, monkeypatch, allow=None):
    if allow is None:
        monkeypatch.delenv("PRISM_BIND_ALL_INTERFACES", raising=False)
    else:
        monkeypatch.setenv("PRISM_BIND_ALL_INTERFACES", allow)
    with patch("uvicorn.run") as run:
        prism_asgi.serve(host=host, port=9)
        return run


class TestBindPolicy:
    def test_loopback_allowed_by_default(self, monkeypatch):
        run = _serve("127.0.0.1", monkeypatch)
        run.assert_called_once()

    def test_all_interfaces_rejected_by_default(self, monkeypatch):
        with pytest.raises(AssertionError):
            _serve("0.0.0.0", monkeypatch)

    def test_all_interfaces_allowed_with_container_optin(self, monkeypatch):
        run = _serve("0.0.0.0", monkeypatch, allow="1")
        run.assert_called_once()

    def test_specific_external_ip_always_rejected(self, monkeypatch):
        # The opt-in sanctions 0.0.0.0-in-container only — never a
        # specific external interface.
        with pytest.raises(AssertionError):
            _serve("192.168.1.10", monkeypatch, allow="1")

    def test_dockerfile_sets_the_optin(self):
        from pathlib import Path
        dockerfile = (Path(prism_asgi.__file__).parent / "Dockerfile").read_text()
        assert "PRISM_BIND_ALL_INTERFACES=1" in dockerfile
        assert "PRISM_HOST=0.0.0.0" in dockerfile
