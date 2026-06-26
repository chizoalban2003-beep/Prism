"""system_lock routing fix for issue #28 bug 67.

Live probes::

  user: "lock screen"        → smart_home (returns "0 devices on")
  user: "lock my screen"     → smart_home
  user: "lock the screen"    → smart_home
  user: "lock my computer"   → smart_home

The smart_home regex at prism_intents.py line ~384 contains a bare
``\\b(?:un)?lock\\b`` which claims any "lock"/"unlock" phrase for IoT
smart locks. But locking the local screen is a fundamental hardware-
bridge action PRISM should handle directly — not delegate to a smart
home that probably isn't even set up.

Fix: add a ``system_lock`` intent hoisted above smart_home, scoped to
"lock <screen|computer|session|workstation|desktop>".
"""
from __future__ import annotations

from prism_intents import INTENTS
from prism_routing import route_intent


def _route(message: str) -> str:
    return route_intent(message, INTENTS, lambda _m: None)


class TestLockScreenVariants:

    def test_lock_screen(self):
        assert _route("lock screen") == "system_lock"

    def test_lock_my_screen(self):
        assert _route("lock my screen") == "system_lock"

    def test_lock_the_screen(self):
        assert _route("lock the screen") == "system_lock"

    def test_lock_my_computer(self):
        assert _route("lock my computer") == "system_lock"

    def test_lock_workstation(self):
        assert _route("lock my workstation") == "system_lock"

    def test_lock_desktop(self):
        assert _route("lock my desktop") == "system_lock"


class TestSmartHomeStillWorks:

    def test_unlock_front_door(self):
        # Real smart-home unlock action — must stay on smart_home.
        assert _route("unlock the front door") == "smart_home"

    def test_smart_home_phrase(self):
        assert _route("smart home") == "smart_home"

    def test_home_assistant_phrase(self):
        assert _route("home assistant") == "smart_home"


class TestNoOverclaim:

    def test_lock_a_file_not_system_lock(self):
        # "lock this file" should not route to system_lock.
        assert _route("lock this file") != "system_lock"
