"""send_push regex fix for issue #28 bug 14 — modifier words and "an" broke routing.

Live test: ``send me a test notification`` was being routed to add_task
("Task added: Test Notification") instead of send_push. The pattern
required ``notification`` to follow the article ``a`` literally — no room
for an adjective. ``send me an alert about X`` similarly fell through
because the regex only accepted ``a``, not ``an``.

Fix: allow ``an?`` and an optional one-word modifier between the article
and the noun. The existing direct verbs (``notify me``, ``ping me``,
``alert me``) are unchanged.
"""
from __future__ import annotations

from prism_intents import INTENTS
from prism_routing import route_intent


def _route(message: str) -> str:
    return route_intent(message, INTENTS, lambda _m: None)


class TestModifierCoverage:
    def test_send_me_a_test_notification(self):
        assert _route("send me a test notification") == "send_push"

    def test_send_me_a_push_notification(self):
        assert _route("send me a push notification") == "send_push"

    def test_push_me_an_urgent_alert(self):
        assert _route("push me an urgent alert") == "send_push"

    def test_send_me_an_alert(self):
        assert _route("send me an alert") == "send_push"

    def test_send_me_a_reminder(self):
        assert _route("send me a reminder") == "send_push"


class TestNoRegression:
    def test_send_me_a_notification(self):
        assert _route("send me a notification") == "send_push"

    def test_push_a_notification(self):
        assert _route("push a notification") == "send_push"

    def test_notify_me_of_progress(self):
        assert _route("notify me of progress") == "send_push"

    def test_ping_me_when_done(self):
        # "ping me" matches send_push directly, but "when done" makes it a
        # horizon trigger — horizon_add wins because it precedes send_push
        # and matches "(?:tell|alert|notify|remind|ping) me when".
        # Confirm: it does not silently fall through to general_chat.
        assert _route("ping me when done") in {"horizon_add", "send_push"}


class TestNoOverreach:
    """The new modifier slot shouldn't grab arbitrary nouns."""

    def test_send_me_a_long_email_does_not_match_push(self):
        # No "notification/alert/reminder" noun → don't match send_push.
        assert _route("send me a long email") != "send_push"

    def test_push_a_button_does_not_match(self):
        # "push" without a notification noun.
        assert _route("push a button") != "send_push"
