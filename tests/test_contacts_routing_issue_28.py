"""contacts routing fix for issue #28 bug 63.

Live probes::

  user: "show contact John"       → general_chat (no match)
  user: "what's John's number"    → general_chat
  user: "what is John's email"    → wikipedia_lookup (steals "what is ...")

The existing contacts regex requires the literal noun "contact / person
/ colleague / client / friend" after the verb. Natural phrasings that
just use a person's name don't hit it.

Fix:

1. Widen contacts to claim "show contact <Name>", "find John",
   "what's X's (number|phone|email|address)".
2. Hoist above wikipedia_lookup so "what is John's email" isn't stolen.
"""
from __future__ import annotations

from prism_intents import INTENTS
from prism_routing import route_intent


def _route(message: str) -> str:
    return route_intent(message, INTENTS, lambda _m: None)


class TestNameContactLookup:

    def test_show_contact_john(self):
        assert _route("show contact John") == "contacts"

    def test_find_contact_john(self):
        assert _route("find contact John") == "contacts"

    def test_contact_info_for_john(self):
        assert _route("contact info for John") == "contacts"


class TestWhatsPersonsX:

    def test_whats_johns_number(self):
        assert _route("what's John's number") == "contacts"

    def test_what_is_johns_email(self):
        assert _route("what is John's email") == "contacts"

    def test_whats_johns_phone(self):
        assert _route("what's John's phone") == "contacts"

    def test_whats_johns_address(self):
        assert _route("what's John's address") == "contacts"


class TestNoRegression:

    def test_existing_find_contact(self):
        assert _route("find my contact") == "contacts"

    def test_wikipedia_lookup_still_works(self):
        assert _route("look up python on wikipedia") == "wikipedia_lookup"

    def test_what_is_python(self):
        # No possessive — wikipedia should still claim general "what is X".
        assert _route("what is python") == "wikipedia_lookup"
