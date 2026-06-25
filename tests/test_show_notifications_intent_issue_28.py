"""show_notifications intent for issue #28 bug 56.

Live probe: "my notifications" returned a "Current context" card from
general_chat instead of listing pending proactive events. The proactive
loop *records* notifications into ``~/.prism/proactive.db`` and
``PrismProactive.pending_events()`` returns them — there was just no
read intent wiring the chat to that store.

Two-part fix:

1. ``prism_intents.INTENTS`` learns a ``show_notifications`` regex that
   matches "my notifications", "show notifications",
   "any alerts", bare "notifications", etc. Placed *before* ``send_push``
   so read intents don't get caught by "ping me / alert me / notify me".
2. ``handle_pa_intent`` learns a ``show_notifications`` branch that
   calls ``agent._proactive.pending_events(n=10)`` and formats them as a
   card with relative timestamps.

Send-intent regressions (e.g. "send me a test notification" →
``send_push``) must still pass — this fix is additive.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from prism_intents import INTENTS
from prism_pa_intents import handle_pa_intent
from prism_routing import route_intent


# ---------------------------------------------------------------------
# 1. Routing — phrases the user actually typed must land on the new
#    intent, while send_push phrases keep their existing behaviour.
# ---------------------------------------------------------------------

class TestRoutingShowNotifications:

    def _it(self, msg: str) -> str:
        return route_intent(msg, INTENTS, lambda _m: None)

    def test_my_notifications(self):
        assert self._it("my notifications") == "show_notifications"

    def test_show_notifications(self):
        assert self._it("show notifications") == "show_notifications"

    def test_list_notifications(self):
        assert self._it("list my notifications") == "show_notifications"

    def test_any_alerts(self):
        assert self._it("any alerts") == "show_notifications"

    def test_bare_notifications(self):
        assert self._it("notifications") == "show_notifications"

    def test_bare_notifications_question(self):
        assert self._it("notifications?") == "show_notifications"

    def test_what_are_my_notifications(self):
        assert self._it("what are my notifications") == "show_notifications"


class TestRoutingSendPushNotRegressed:
    """Adding show_notifications must not break send_push routing."""

    def _it(self, msg: str) -> str:
        return route_intent(msg, INTENTS, lambda _m: None)

    def test_send_me_a_test_notification(self):
        assert self._it("send me a test notification") == "send_push"

    def test_push_me_an_urgent_alert(self):
        assert self._it("push me an urgent alert") == "send_push"

    def test_notify_me_of_progress(self):
        assert self._it("notify me of progress") == "send_push"

    def test_ping_me_in_an_hour(self):
        assert self._it("ping me in an hour") == "send_push"


# ---------------------------------------------------------------------
# 2. Handler — given a populated proactive store, the card lists events.
# ---------------------------------------------------------------------

@dataclass
class _StubEvent:
    trigger_id: str
    message:    str
    timestamp:  float


class _StubProactive:
    def __init__(self, events):
        self._events = events
        self.calls = 0

    def pending_events(self, n: int = 5):
        self.calls += 1
        return list(self._events[:n])


class _Agent:
    def __init__(self, proactive=None):
        self._proactive = proactive


class TestHandlerListsPendingEvents:

    def test_lists_events_with_relative_timestamps(self):
        import time
        now = time.time()
        events = [
            _StubEvent("budget", "Daily budget 80% spent", now - 30),
            _StubEvent("calendar", "Meeting in 15 min", now - 600),
            _StubEvent("recovery", "HRV low — rest today", now - 7200),
        ]
        agent = _Agent(proactive=_StubProactive(events))
        card = handle_pa_intent(agent, "show_notifications",
                                "my notifications", {})
        assert card is not None
        assert "Notifications" in card.title
        assert "Daily budget" in card.body
        assert "Meeting in 15 min" in card.body
        # Relative time format ("Xs/m/h ago") must appear.
        assert "ago" in card.body

    def test_empty_returns_friendly_card(self):
        agent = _Agent(proactive=_StubProactive([]))
        card = handle_pa_intent(agent, "show_notifications",
                                "any alerts", {})
        assert card is not None
        assert card.title == "Notifications"
        assert "No new notifications" in card.body

    def test_no_proactive_returns_friendly_card(self):
        agent = _Agent(proactive=None)
        card = handle_pa_intent(agent, "show_notifications",
                                "my notifications", {})
        assert card is not None
        assert "aren't running" in card.body or "not running" in card.body.lower()

    def test_proactive_error_doesnt_crash(self):
        class _Boom:
            def pending_events(self, n=5):
                raise RuntimeError("db locked")
        agent = _Agent(proactive=_Boom())
        card = handle_pa_intent(agent, "show_notifications",
                                "my notifications", {})
        assert card is not None
        assert "Couldn't" in card.body or "couldn't" in card.body
