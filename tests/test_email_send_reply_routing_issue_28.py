"""email_send routing fix for issue #28 bug 64.

Live probes::

  user: "reply to that email"      → email_read (opens inbox!)
  user: "reply to the last email"  → email_read

Root cause: the broad email catch-all at prism_intents line ~388
maps SEND-shaped alternatives (``send.*email``, ``draft.*email``,
``reply.*email``) to ``email_read`` — the comment says "avoid
duplication" but conflating actions breaks the product.

Fix: split the broad rule. Send-shaped phrases (send/draft/reply)
become a separate email_send rule; the remaining read-shaped
alternatives stay in email_read.
"""
from __future__ import annotations

from prism_intents import INTENTS
from prism_routing import route_intent


def _route(message: str) -> str:
    return route_intent(message, INTENTS, lambda _m: None)


class TestReplyVariants:

    def test_reply_to_that_email(self):
        assert _route("reply to that email") == "email_send"

    def test_reply_to_the_last_email(self):
        assert _route("reply to the last email") == "email_send"

    def test_reply_to_this_email(self):
        assert _route("reply to this email") == "email_send"

    def test_reply_to_email_from_bob(self):
        assert _route("reply to email from bob") == "email_send"


class TestDraftVariants:

    def test_draft_an_email(self):
        assert _route("draft an email") == "email_send"

    def test_draft_a_reply(self):
        assert _route("draft a reply") == "email_send"

    def test_draft_an_email_to_bob(self):
        assert _route("draft an email to bob") == "email_send"


class TestSendVariants:

    def test_send_an_email(self):
        assert _route("send an email") == "email_send"

    def test_send_mail(self):
        assert _route("send mail") == "email_send"

    def test_send_email_to_bob(self):
        assert _route("send email to bob") == "email_send"


class TestEmailReadStillWorks:

    def test_check_my_email(self):
        assert _route("check my email") == "email_read"

    def test_show_my_inbox(self):
        assert _route("show my inbox") == "email_read"

    def test_any_unread_emails(self):
        assert _route("any unread emails") == "email_read"

    def test_email_summary(self):
        assert _route("email summary") == "email_read"

    def test_any_new_mail(self):
        assert _route("any new mail") == "email_read"
