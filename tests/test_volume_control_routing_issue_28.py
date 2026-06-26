"""volume_control routing for issue #28 bug 70.

Live probes::

  user: "volume up"            → organ_proposal ("Build new organ?")
  user: "mute"                 → organ_proposal
  user: "set volume to 50"     → organ_proposal
  user: "turn up the volume"   → organ_proposal
  user: "increase volume"      → unrelated approval card
  user: "volume down"          → veax_control (claims "decrease")

PRISM cannot adjust the local audio output volume — a fundamental
hardware-bridge action. The spotify_control organ has a `volume N`
sub-command, but only when the user explicitly references music; pure
"volume up" requests fall through.

Fix: dedicated ``volume_control`` intent + organ. Hoisted above
veax_control and organ_proposal. Scoped to genuine volume verbs so it
doesn't claim "play music" or spectrum verbs like "increase verification".
"""
from __future__ import annotations

from prism_intents import INTENTS
from prism_routing import route_intent


def _route(message: str) -> str:
    return route_intent(message, INTENTS, lambda _m: None)


class TestVolumeUpDown:

    def test_volume_up(self):
        assert _route("volume up") == "volume_control"

    def test_volume_down(self):
        assert _route("volume down") == "volume_control"

    def test_turn_up_the_volume(self):
        assert _route("turn up the volume") == "volume_control"

    def test_turn_down_the_volume(self):
        assert _route("turn down the volume") == "volume_control"

    def test_increase_volume(self):
        assert _route("increase volume") == "volume_control"

    def test_decrease_volume(self):
        assert _route("decrease volume") == "volume_control"

    def test_louder(self):
        assert _route("make it louder") == "volume_control"

    def test_quieter(self):
        assert _route("make it quieter") == "volume_control"


class TestMute:

    def test_mute(self):
        assert _route("mute") == "volume_control"

    def test_unmute(self):
        assert _route("unmute") == "volume_control"

    def test_mute_the_volume(self):
        assert _route("mute the volume") == "volume_control"

    def test_mute_audio(self):
        assert _route("mute audio") == "volume_control"


class TestSetVolume:

    def test_set_volume_to_50(self):
        assert _route("set volume to 50") == "volume_control"

    def test_volume_to_25(self):
        assert _route("volume to 25") == "volume_control"

    def test_set_volume_50_percent(self):
        assert _route("set volume to 50%") == "volume_control"


class TestNoOverclaim:

    def test_play_music_still_spotify(self):
        # spotify volume command must still go to spotify_control.
        assert _route("play music") == "spotify_control"

    def test_increase_verification_not_volume(self):
        # VEAX spectrum verb shouldn't be stolen by volume_control —
        # "increase verification" is the spectrum-tuning command.
        assert _route("increase verification") != "volume_control"

    def test_volume_of_a_sphere_not_volume_control(self):
        # "what's the volume of a sphere" is a math question, not audio.
        assert _route("what's the volume of a sphere") != "volume_control"
