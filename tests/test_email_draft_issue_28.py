"""
tests/test_email_draft_issue_28.py
==================================
Credential-free email drafts (Limit 1): when SMTP is unconfigured, email_send
composes and saves an .eml draft instead of hitting a setup wall; when
configured it sends as before, and a runtime send failure falls back to a
draft rather than losing the message.

Uses conftest's hermetic HOME so ~/.prism/drafts is throwaway.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load():
    spec = importlib.util.spec_from_file_location(
        "email_send", "organs/email_send.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E = _load()


class _Router:
    def __init__(self, to="alice@example.com", subject="Lunch",
                 body="Hi Alice, lunch Friday?"):
        self._p = (to, subject, body)

    def call(self, *a, **k):
        import json
        to, subject, body = self._p
        return json.dumps({"to": to, "subject": subject, "body": body}), None


class _Email:
    def __init__(self, configured=True, send_ok=True):
        self.configured = configured
        self._ok = send_ok
        self.sent = []

    def send(self, to, subject, body):
        self.sent.append((to, subject, body))
        return self._ok


class TestDraftWhenUnconfigured:
    def test_writes_eml_and_reports(self):
        card = E.execute("email_send", "email alice about lunch",
                         {"router": _Router(), "email": None})
        assert card.card_data.get("drafted") is True
        p = Path(card.card_data["path"])
        assert p.exists() and p.suffix == ".eml"
        text = p.read_text()
        assert "To: alice@example.com" in text
        assert "Subject: Lunch" in text
        assert "unsent" in text  # honest marker
        assert "lunch" in text.lower()

    def test_unconfigured_email_object_also_drafts(self):
        card = E.execute("email_send", "email bob@x.com hi",
                         {"router": _Router(to="bob@x.com"),
                          "email": _Email(configured=False)})
        assert card.card_data.get("drafted") is True


class TestSendWhenConfigured:
    def test_configured_sends_not_drafts(self):
        mail = _Email(configured=True, send_ok=True)
        card = E.execute("email_send", "email alice hi",
                         {"router": _Router(), "email": mail})
        assert mail.sent and mail.sent[0][0] == "alice@example.com"
        assert card.card_data.get("drafted") is not True
        assert "Sent" in card.title or "Sent" in card.body

    def test_send_failure_falls_back_to_draft(self):
        mail = _Email(configured=True, send_ok=False)
        card = E.execute("email_send", "email alice hi",
                         {"router": _Router(), "email": mail})
        assert "draft" in card.body.lower()


class TestNoRouter:
    def test_no_router_is_honest(self):
        card = E.execute("email_send", "email alice", {"router": None})
        assert "parse" in card.body.lower()
