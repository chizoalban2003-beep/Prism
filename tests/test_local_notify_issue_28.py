"""
tests/test_local_notify_issue_28.py
===================================
Credential-free local notification (Limit 1): prism_local_notify always logs
to the inbox and fires a popup only when a notifier + display exist; the
notify_desktop organ wraps it; phone_call degrades to a local reminder when
Twilio is unconfigured instead of a hard wall.

Uses conftest's hermetic HOME so ~/.prism/notifications.jsonl is throwaway.
"""
from __future__ import annotations

import importlib.util

import prism_local_notify as ln
from prism_intents import INTENTS
from prism_routing import route_intent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


N = _load("notify_desktop", "organs/notify_desktop.py")
P = _load("phone_call", "organs/phone_call.py")


def _route(m):
    return route_intent(m, INTENTS, lambda _m: "")


class TestDeliver:
    def test_always_logs_to_inbox(self, monkeypatch):
        # force "no popup" regardless of host tooling
        monkeypatch.setattr(ln, "_fire_popup", lambda *a, **k: None)
        rep = ln.deliver("T", "hello world", source="test")
        assert rep["logged"] is True
        assert rep["popup"] is None
        recent = ln.recent(5)
        assert recent and recent[-1]["body"] == "hello world"

    def test_popup_reported_when_available(self, monkeypatch):
        monkeypatch.setattr(ln, "_fire_popup", lambda *a, **k: "notify-send")
        rep = ln.deliver("T", "ping", source="test")
        assert rep["popup"] == "notify-send"

    def test_no_display_means_no_popup(self, monkeypatch):
        monkeypatch.setattr(ln, "_has_display", lambda: False)
        monkeypatch.setattr(ln.shutil, "which", lambda t: "/usr/bin/" + t)
        # even with notifiers "installed", no display → no popup attempt
        assert ln._fire_popup("t", "b", "normal") is None


class TestNotifyOrgan:
    def test_parse_body(self):
        assert N._parse("notify me that the build finished")[1] == "the build finished"
        assert N._parse("ping me about lunch")[1] == "lunch"

    def test_execute_logs(self, monkeypatch):
        monkeypatch.setattr(ln, "_fire_popup", lambda *a, **k: None)
        card = N.execute("notify_desktop", "alert me: deploy done", {})
        assert card.card_data["logged"] is True
        assert "deploy done" in card.body

    def test_empty_body_asks(self):
        card = N.execute("notify_desktop", "notify me", {})
        assert "what" in card.body.lower()


class TestPhoneDegrade:
    def test_no_twilio_degrades_to_local_reminder(self, monkeypatch):
        monkeypatch.setattr(ln, "_fire_popup", lambda *a, **k: None)
        card = P.execute("phone_call", "call mum say happy birthday",
                         {"twilio_config": {}})
        assert "isn't configured" in card.body or "reminder" in card.body.lower()
        # the message survived into the local inbox
        assert any("happy birthday" in r["body"] for r in ln.recent(5))


class TestRouting:
    def test_notify_vs_reminder(self):
        assert _route("notify me that build finished") == "notify_desktop"
        assert _route("alert me: server down") == "notify_desktop"
        assert _route("send me a notification") == "notify_desktop"
        assert _route("remind me to stretch in 2 hours") == "reminder_set"
        assert _route("remind me to call mum") == "reminder_set"
