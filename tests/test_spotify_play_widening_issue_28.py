"""spotify_control verb widening for issue #28 bug 66.

Live probes::

  user: "play some music"      → general_chat
  user: "play the queen song"  → general_chat
  user: "play a song"          → general_chat

The existing regex required ``play\\s+(music|spotify|song|track|playback)``
with no filler between verb and noun, so the natural "play SOME music"
or "play THE QUEEN song" missed.

Fix: allow optional filler words between the verb and the music noun.
"""
from __future__ import annotations

from prism_intents import INTENTS
from prism_routing import route_intent


def _route(message: str) -> str:
    return route_intent(message, INTENTS, lambda _m: None)


class TestPlayWithFillerWords:

    def test_play_some_music(self):
        assert _route("play some music") == "spotify_control"

    def test_play_a_song(self):
        assert _route("play a song") == "spotify_control"

    def test_play_the_queen_song(self):
        assert _route("play the queen song") == "spotify_control"

    def test_play_a_random_track(self):
        assert _route("play a random track") == "spotify_control"


class TestExistingControlsUnchanged:

    def test_play_music(self):
        assert _route("play music") == "spotify_control"

    def test_pause_music(self):
        assert _route("pause music") == "spotify_control"

    def test_next_song(self):
        assert _route("next song") == "spotify_control"

    def test_skip_track(self):
        assert _route("skip track") == "spotify_control"

    def test_whats_playing(self):
        assert _route("what's playing") == "spotify_control"


class TestNoOverclaim:

    def test_play_my_day_still_plans(self):
        # "play" without music noun must not steal universal_plan.
        assert _route("plan my day") == "universal_plan"

    def test_random_play_word_not_spotify(self):
        # "I want to play with my dog" must not be spotify_control.
        assert _route("I want to play with my dog") != "spotify_control"
