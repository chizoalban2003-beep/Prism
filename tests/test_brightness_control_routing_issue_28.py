"""brightness_control routing for issue #28 bug 72.

Live probes (every brightness phrase falls through to general_chat
which then hangs on the missing LLM)::

  user: "brightness up"            → TIMEOUT (general_chat → LLM)
  user: "dim screen"               → TIMEOUT
  user: "set brightness to 50"     → TIMEOUT
  user: "increase brightness"      → TIMEOUT

PRISM has zero local-display bridge. This is a fundamental hardware
action; the dim/bright verbs are unambiguous.

Fix: dedicated ``brightness_control`` intent + organ. Hoisted above
spotify_control / organ_proposal. Scoped tightly to avoid collisions
with smart_home_control's "dim the lights".
"""
from __future__ import annotations

from prism_intents import INTENTS
from prism_routing import route_intent


def _route(message: str) -> str:
    return route_intent(message, INTENTS, lambda _m: None)


class TestBrightnessUpDown:

    def test_brightness_up(self):
        assert _route("brightness up") == "brightness_control"

    def test_brightness_down(self):
        assert _route("brightness down") == "brightness_control"

    def test_increase_brightness(self):
        assert _route("increase brightness") == "brightness_control"

    def test_decrease_brightness(self):
        assert _route("decrease brightness") == "brightness_control"

    def test_screen_brightness_up(self):
        assert _route("screen brightness up") == "brightness_control"

    def test_brighter_screen(self):
        assert _route("make the screen brighter") == "brightness_control"

    def test_dimmer_screen(self):
        assert _route("make the screen dimmer") == "brightness_control"

    def test_dim_my_screen(self):
        assert _route("dim my screen") == "brightness_control"


class TestSetBrightness:

    def test_set_brightness_to_50(self):
        assert _route("set brightness to 50") == "brightness_control"

    def test_set_screen_brightness_to_75(self):
        assert _route("set screen brightness to 75") == "brightness_control"

    def test_brightness_to_25(self):
        assert _route("brightness to 25") == "brightness_control"


class TestNoOverclaim:

    def test_dim_the_lights_stays_smart_home(self):
        # smart_home_control claims "dim the lights" — we must not
        # poach it. Note: route may be smart_home OR smart_home_control
        # depending on which regex wins; both are correct.
        assert _route("dim the lights") in {"smart_home", "smart_home_control"}

    def test_lock_screen_still_system_lock(self):
        # Other system_* hoists must keep working.
        assert _route("lock screen") == "system_lock"

    def test_volume_up_still_volume_control(self):
        # volume_control hoist must keep working.
        assert _route("volume up") == "volume_control"
