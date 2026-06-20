"""Regression tests for the review-recommendation refactors."""
from __future__ import annotations

import types

import pytest
from fastapi.testclient import TestClient

from prism_state import _set_state


@pytest.fixture()
def client():
    from prism_asgi import app
    agent = types.SimpleNamespace()
    agent.chat = lambda msg, ctx=None: types.SimpleNamespace(
        body="ok", to_json=lambda: {"type": "text", "title": "Chat",
                                    "body": "ok", "data": {}, "actions": []})
    _set_state(agent=agent)
    return TestClient(app, raise_server_exceptions=False)


# ── /chat returns 400 on malformed JSON ─────────────────────────────────────────

class TestChatMalformedJSON:
    def test_malformed_json_400(self, client):
        r = client.post("/chat", content=b"{not valid json",
                        headers={"Content-Type": "application/json"})
        assert r.status_code == 400

    def test_non_object_body_400(self, client):
        r = client.post("/chat", json=[1, 2, 3])
        assert r.status_code == 400

    def test_valid_json_still_ok(self, client):
        r = client.post("/chat", json={"message": "hi"})
        assert r.status_code == 200


# ── ensure_mobile_secret: persistent, random, env-overridable ───────────────────

class TestMobileSecret:
    def test_generates_and_persists(self, tmp_path, monkeypatch):
        import prism_auth
        monkeypatch.delenv("PRISM_MOBILE_SECRET", raising=False)
        monkeypatch.setattr(prism_auth, "MOBILE_SECRET_FILE", tmp_path / "ms")
        s1 = prism_auth.ensure_mobile_secret()
        s2 = prism_auth.ensure_mobile_secret()
        assert s1 and s1 == s2                      # persisted, idempotent
        assert s1 != "prism-default-secret"         # not the weak default
        assert len(s1) >= 20

    def test_env_override(self, tmp_path, monkeypatch):
        import prism_auth
        monkeypatch.setenv("PRISM_MOBILE_SECRET", "override-secret")
        monkeypatch.setattr(prism_auth, "MOBILE_SECRET_FILE", tmp_path / "ms")
        assert prism_auth.ensure_mobile_secret() == "override-secret"


# ── CrystallizationEngine naming clash resolved with back-compat alias ──────────

class TestNamingAlias:
    def test_new_name_exists(self):
        import digital_identity
        assert hasattr(digital_identity, "DigitalIdentityEngine")

    def test_old_name_is_alias(self):
        import digital_identity
        assert digital_identity.CrystallisationEngine is digital_identity.DigitalIdentityEngine

    def test_phase_engine_is_distinct(self):
        import digital_identity
        import prism_phase
        assert prism_phase.CrystallizationEngine is not digital_identity.DigitalIdentityEngine


# ── INTENTS extracted to its own module, agent still wired ──────────────────────

class TestIntentsExtraction:
    def test_intents_module(self):
        import prism_intents
        assert isinstance(prism_intents.INTENTS, list)
        assert prism_intents.INTENTS and isinstance(prism_intents.INTENTS[0], tuple)

    def test_agent_uses_extracted_table(self):
        import prism_agent
        import prism_intents
        assert prism_agent.PrismAgent.INTENTS is prism_intents.INTENTS
