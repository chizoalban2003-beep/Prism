"""
tests/test_ui_shell_auth_exempt_issue_28.py
===========================================
First-run auth UX for issue #28: the static UI shell must be served
without a bearer token, because a browser cannot render the token
prompt (or install the PWA, which needs sw.js + manifest.json) if the
shell itself answers 401. Data endpoints stay gated; the ?token=
exchange sets the prism_auth cookie for the rest of the session.

Before this fix, following the QUICKSTART ("open http://127.0.0.1:8742
and paste the auth token into the prompt") dead-ended at a raw
{"error":"unauthorized"} — there was no prompt, and no HTML at all.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

TOKEN = "test-token-issue-28"

SHELL_PATHS = ["/", "/app", "/index.html", "/mobile",
               "/manifest.json", "/sw.js", "/icon.svg"]


@pytest.fixture
def authed_app(monkeypatch):
    # conftest globally sets PRISM_AUTH_DISABLE=1; re-enable auth here.
    monkeypatch.delenv("PRISM_AUTH_DISABLE", raising=False)
    monkeypatch.setenv("PRISM_AUTH_TOKEN", TOKEN)
    from prism_asgi import app
    return app


@pytest.fixture
def client(authed_app):
    return TestClient(authed_app)


class TestShellExempt:
    @pytest.mark.parametrize("path", SHELL_PATHS)
    def test_shell_served_without_token(self, client, path):
        r = client.get(path)
        assert r.status_code == 200, f"{path} returned {r.status_code}"

    def test_chat_page_contains_token_prompt(self, client):
        r = client.get("/")
        assert "tokenBar" in r.text
        assert "auth_token" in r.text  # tells the user where to find it

    def test_mobile_page_contains_token_gate(self, client):
        r = client.get("/mobile")
        assert "showTokenGate" in r.text


class TestDataStillGated:
    @pytest.mark.parametrize("path", ["/status", "/organs", "/instructions",
                                      "/identity/dashboard", "/metrics"])
    def test_data_endpoint_401_without_token(self, client, path):
        r = client.get(path)
        assert r.status_code == 401, f"{path} returned {r.status_code}"

    def test_wrong_token_rejected(self, client):
        r = client.get("/organs", params={"token": "wrong"})
        assert r.status_code == 401

    def test_bearer_header_accepted(self, client):
        r = client.get("/organs", headers={"Authorization": f"Bearer {TOKEN}"})
        assert r.status_code == 200


class TestTokenCookieExchange:
    def test_query_token_sets_cookie_for_session(self, client):
        r = client.get("/organs", params={"token": TOKEN})
        assert r.status_code == 200
        assert client.cookies.get("prism_auth") == TOKEN
        # Subsequent request: no header, no query param — cookie carries it.
        r2 = client.get("/organs")
        assert r2.status_code == 200

    def test_cookie_requests_slide_expiry(self, client):
        """Every cookie-authenticated response re-issues the cookie so an
        active browser session never expires mid-use (issue #28-84)."""
        client.get("/organs", params={"token": TOKEN})
        r = client.get("/organs")
        set_cookie = r.headers.get("set-cookie", "")
        assert "prism_auth=" in set_cookie
        assert "Max-Age=86400" in set_cookie

    def test_bearer_requests_do_not_set_cookie(self, client):
        r = client.get("/organs", headers={"Authorization": f"Bearer {TOKEN}"})
        assert "prism_auth" not in r.headers.get("set-cookie", "")
