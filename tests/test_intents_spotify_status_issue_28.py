"""spotify_control intent regex extension for issue #28 bug 48.

Live test: ``what is playing on Spotify`` returned the Wikipedia
article on Spotify (the Swedish music streaming company). The
intent regex for ``spotify_control`` only matched control *verbs*
(play / pause / skip / volume), so query forms fell through to
wikipedia_lookup's ``what is X`` catch-all.

Fix: extend the spotify_control regex to also match status queries —
``what's playing on spotify``, ``what is playing right now``,
``current playing``, ``now playing``, ``spotify status``. Routing
hands the message to the spotify_control organ, which already returns
a ``status`` action when no control verb matched.
"""
from __future__ import annotations

import re

from prism_intents import INTENTS


def _route(text: str) -> str:
    """First-match regex routing (mirrors prism_routing.route_intent
    minus the LLM fallback)."""
    lowered = text.lower()
    for pattern, intent in INTENTS:
        if re.search(pattern, lowered):
            return intent
    return ""


class TestSpotifyStatusQueries:
    def test_what_is_playing_on_spotify(self):
        # The reported repro.
        assert _route("what is playing on Spotify") == "spotify_control"

    def test_whats_playing_on_spotify(self):
        assert _route("what's playing on Spotify") == "spotify_control"

    def test_whats_playing_right_now_on_spotify(self):
        assert _route("what's playing right now on Spotify") == "spotify_control"

    def test_what_song_is_playing_on_spotify(self):
        assert _route("what song is playing on Spotify") == "spotify_control"

    def test_now_playing(self):
        assert _route("now playing") == "spotify_control"

    def test_currently_playing(self):
        assert _route("currently playing") == "spotify_control"

    def test_spotify_status(self):
        assert _route("spotify status") == "spotify_control"

    def test_spotify_state(self):
        assert _route("spotify state") == "spotify_control"


class TestSpotifyControlStillWorks:
    """Sanity: the existing control-verb branches still route correctly."""

    def test_play_music(self):
        assert _route("play music") == "spotify_control"

    def test_pause_spotify(self):
        assert _route("pause spotify") == "spotify_control"

    def test_skip_song(self):
        assert _route("skip song") == "spotify_control"

    def test_volume_music(self):
        # "volume music" via the existing volume branch (verb + noun shape).
        assert _route("volume music") == "spotify_control"


class TestWikipediaNotShadowed:
    """The widened regex must not start swallowing innocuous wiki queries.

    "what is spotify" (no playback verb) is a question about the company,
    not playback. Keep that on wikipedia_lookup — otherwise we'd just
    have flipped the bug. But "what IS PLAYING on spotify" is a clear
    playback query and must hit spotify_control.
    """

    def test_what_is_spotify_remains_wiki(self):
        # No "playing"/"on now"/"status" — wikipedia_lookup is correct.
        assert _route("what is spotify") == "wikipedia_lookup"

    def test_tell_me_about_spotify_remains_wiki(self):
        assert _route("tell me about spotify") == "wikipedia_lookup"
