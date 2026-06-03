"""
tests/test_pa_organs.py
=======================
Tests for the PA gap organs: email_send, calendar_write, phone_call.
All external I/O (SMTP, Google Calendar, Twilio) is mocked.
"""
from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

# ── helpers ───────────────────────────────────────────────────────────────────

def _load(organ_name: str):
    spec = importlib.util.spec_from_file_location(
        organ_name,
        f"organs/{organ_name}.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _router(json_out: str = '{"to":"bob@x.com","subject":"Hi","body":"Hello Bob"}'):
    r = MagicMock()
    r.call.return_value = (json_out, "model")
    return r


# ── email_send ────────────────────────────────────────────────────────────────

class TestEmailSend:
    organ = _load("email_send")

    def _email(self, configured=True, send_ok=True):
        e = MagicMock()
        e.configured = configured
        e.send.return_value = send_ok
        return e

    def test_not_configured(self):
        card = self.organ.execute("email_send", "send email", {"email": self._email(configured=False)})
        assert "not configured" in card.body.lower()

    def test_no_email_in_ctx(self):
        card = self.organ.execute("email_send", "send email", {})
        assert "not configured" in card.body.lower()

    def test_no_router(self):
        card = self.organ.execute("email_send", "send email", {"email": self._email()})
        assert "no llm router" in card.body.lower()

    def test_sends_successfully(self):
        ctx = {"email": self._email(), "router": _router()}
        card = self.organ.execute("email_send", "send email to bob@x.com saying hello", ctx)
        assert "sent" in card.body.lower()
        assert "bob@x.com" in card.body

    def test_send_failure(self):
        ctx = {"email": self._email(send_ok=False), "router": _router()}
        card = self.organ.execute("email_send", "send email to bob@x.com", ctx)
        assert "failed" in card.body.lower()

    def test_bad_json_from_router(self):
        ctx = {"email": self._email(), "router": _router("not json {{{")}
        card = self.organ.execute("email_send", "send email to bob", ctx)
        assert "could not parse" in card.body.lower()

    def test_missing_to_field(self):
        ctx = {"email": self._email(), "router": _router('{"subject":"Hi","body":"Hello"}')}
        card = self.organ.execute("email_send", "send email", ctx)
        assert "no recipient" in card.body.lower()

    def test_organ_meta(self):
        assert self.organ.ORGAN_META["intent"] == "email_send"
        assert self.organ.ORGAN_POLICY["requires_approval"] is True
        assert self.organ.ORGAN_POLICY["irreversible"] is True


# ── calendar_write ────────────────────────────────────────────────────────────

class TestCalendarWrite:
    organ = _load("calendar_write")

    def _cal(self, configured=True):
        c = MagicMock()
        c.configured = configured
        return c

    def test_not_configured(self):
        card = self.organ.execute("calendar_write", "schedule meeting", {"calendar": self._cal(False)})
        assert "not configured" in card.body.lower()

    def test_no_calendar_in_ctx(self):
        card = self.organ.execute("calendar_write", "schedule meeting", {})
        assert "not configured" in card.body.lower()

    def test_free_slot_found(self):
        from datetime import datetime
        cal = self._cal()
        cal.find_free_slot.return_value = datetime(2026, 6, 10, 14, 0)
        card = self.organ.execute("calendar_write", "when am i free", {"calendar": cal})
        assert "free slot" in card.body.lower()
        assert "14:00" in card.body

    def test_free_slot_none(self):
        cal = self._cal()
        cal.find_free_slot.return_value = None
        card = self.organ.execute("calendar_write", "find a free slot", {"calendar": cal})
        assert "no free slots" in card.body.lower()

    def test_create_event_success(self):
        cal = self._cal()
        cal.parse_event_from_text.return_value = {
            "start_iso": "2026-06-10T14:00:00",
            "title": "Team Meeting",
            "duration_mins": 60,
            "location": "",
            "attendees": [],
        }
        fake_event = MagicMock()
        fake_event.__str__ = lambda self: "Team Meeting @ 2026-06-10 14:00"
        cal.create_event.return_value = fake_event
        ctx = {"calendar": cal, "router": MagicMock()}
        card = self.organ.execute("calendar_write", "schedule Team Meeting on June 10 at 2pm", ctx)
        assert "created" in card.body.lower()

    def test_create_event_parse_fails(self):
        cal = self._cal()
        cal.parse_event_from_text.return_value = {}
        ctx = {"calendar": cal, "router": MagicMock()}
        card = self.organ.execute("calendar_write", "book something vague", ctx)
        assert "could not parse" in card.body.lower()

    def test_create_event_api_fails(self):
        cal = self._cal()
        cal.parse_event_from_text.return_value = {
            "start_iso": "2026-06-10T14:00:00", "title": "X",
            "duration_mins": 30, "location": "", "attendees": [],
        }
        cal.create_event.return_value = None
        ctx = {"calendar": cal, "router": MagicMock()}
        card = self.organ.execute("calendar_write", "schedule X on June 10", ctx)
        assert "creation failed" in card.body.lower()

    def test_organ_meta(self):
        assert self.organ.ORGAN_META["intent"] == "calendar_write"
        assert self.organ.ORGAN_POLICY["requires_approval"] is True
        assert self.organ.ORGAN_POLICY["irreversible"] is False


# ── phone_call ────────────────────────────────────────────────────────────────

class TestPhoneCall:
    organ = _load("phone_call")

    _CFG = {
        "account_sid": "ACtest",
        "auth_token": "token123",
        "from_number": "+15005550006",
    }

    def test_no_credentials(self):
        card = self.organ.execute("phone_call", "call +447700900000", {})
        assert "twilio credentials" in card.body.lower()

    def test_twilio_not_installed(self):
        ctx = {"twilio_config": self._CFG}
        with patch.dict(sys.modules, {"twilio": None, "twilio.rest": None}):
            # Remove cached import if any
            for k in list(sys.modules):
                if k.startswith("twilio"):
                    sys.modules[k] = None  # type: ignore[assignment]
            card = self.organ.execute("phone_call", "call +447700900000", ctx)
        assert "not installed" in card.body.lower() or "twilio" in card.body.lower()

    def test_no_phone_number(self):
        ctx = {"twilio_config": self._CFG}
        mock_client = MagicMock()
        with patch("organs.phone_call.Client", return_value=mock_client, create=True):
            # Inject the module-level import bypass
            twilio_mod = MagicMock()
            twilio_mod.rest.Client = MagicMock(return_value=mock_client)
            with patch.dict(sys.modules, {"twilio": twilio_mod, "twilio.rest": twilio_mod.rest}):
                card = self.organ.execute("phone_call", "call my friend", ctx)
        assert "no phone number" in card.body.lower()

    def test_voice_call_success(self):
        ctx = {"twilio_config": self._CFG}
        mock_call = MagicMock()
        mock_call.sid = "CA123"
        mock_client = MagicMock()
        mock_client.calls.create.return_value = mock_call
        twilio_mod = MagicMock()
        twilio_mod.rest.Client = MagicMock(return_value=mock_client)
        with patch.dict(sys.modules, {"twilio": twilio_mod, "twilio.rest": twilio_mod.rest}):
            card = self.organ.execute(
                "phone_call", "call +447700900000 and say Hello there", ctx
            )
        assert "initiated" in card.body.lower() or "ca123" in card.body.lower()

    def test_sms_success(self):
        ctx = {"twilio_config": self._CFG}
        mock_msg = MagicMock()
        mock_msg.sid = "SM456"
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg
        twilio_mod = MagicMock()
        twilio_mod.rest.Client = MagicMock(return_value=mock_client)
        with patch.dict(sys.modules, {"twilio": twilio_mod, "twilio.rest": twilio_mod.rest}):
            card = self.organ.execute(
                "phone_call", "text +447700900000 say meeting at 3pm", ctx
            )
        assert "sms sent" in card.body.lower() or "sm456" in card.body.lower()

    def test_call_failure(self):
        ctx = {"twilio_config": self._CFG}
        mock_client = MagicMock()
        mock_client.calls.create.side_effect = Exception("Twilio API error")
        twilio_mod = MagicMock()
        twilio_mod.rest.Client = MagicMock(return_value=mock_client)
        with patch.dict(sys.modules, {"twilio": twilio_mod, "twilio.rest": twilio_mod.rest}):
            card = self.organ.execute("phone_call", "call +447700900000", ctx)
        assert "failed" in card.body.lower()

    def test_organ_meta(self):
        assert self.organ.ORGAN_META["intent"] == "phone_call"
        assert self.organ.ORGAN_POLICY["requires_approval"] is True
        assert self.organ.ORGAN_POLICY["irreversible"] is True
        assert self.organ.ORGAN_POLICY["max_per_session"] == 3
