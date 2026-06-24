"""Synthesised-organ slug generalisation for issue #28 bug 23.

Live test: ``encrypt this message: hello`` was synthesised as
``synth_encrypt_this_message_hello`` — the slug baked the entire
prompt (including stopwords and the secret payload) into the intent
name. Any subsequent ``encrypt X`` request would synthesise a brand
new organ instead of reusing the existing one, defeating the cache.

Fix: ``PrismAgent._slugify_intent`` now drops anything after the
first colon (payload, not the action), strips URLs, removes a
conversational stopword set, and keeps only the first three
significant tokens.
"""
from __future__ import annotations

from prism_agent import PrismAgent


class _Slugger:
    """Bind only the slugify method onto a bare object so we avoid the
    full PrismAgent construction cost in these unit tests."""
    _SLUG_STOPWORDS = PrismAgent._SLUG_STOPWORDS
    _slugify_intent = PrismAgent._slugify_intent


def _slug(message: str) -> str:
    return _Slugger()._slugify_intent(message)


class TestPayloadStripping:
    def test_colon_payload_dropped(self):
        assert _slug("encrypt this message: hello") == "synth_encrypt_message"

    def test_colon_payload_dropped_with_long_secret(self):
        assert _slug("encrypt this message: my super secret value") == "synth_encrypt_message"

    def test_url_dropped(self):
        assert _slug("shorten this url https://example.com/foo") == "synth_shorten_url"

    def test_url_among_words_dropped(self):
        # The URL (path included) is stripped entirely.
        assert _slug("summarise https://example.com/page in one line") == "synth_summarise_one_line"


class TestStopwordRemoval:
    def test_conversational_scaffolding_dropped(self):
        assert _slug("what is on my calendar today") == "synth_calendar_today"

    def test_please_kindly_dropped(self):
        assert _slug("please kindly translate hello to french") == "synth_translate_hello_french"

    def test_articles_dropped(self):
        # "the", "a", "an" — pure scaffolding.
        assert _slug("send a message to the team") == "synth_send_message_team"


class TestReuseAcrossPhrasings:
    """Different phrasings of the same intent should collapse to the same slug."""

    def test_same_action_different_payload(self):
        a = _slug("encrypt this message: hello")
        b = _slug("encrypt this message: completely different secret here")
        assert a == b == "synth_encrypt_message"

    def test_url_specifics_dont_change_slug(self):
        a = _slug("shorten this url https://example.com/a")
        b = _slug("shorten this url https://anothersite.org/long/path")
        assert a == b == "synth_shorten_url"


class TestEdgeCases:
    def test_empty_message_falls_back(self):
        assert _slug("") == "synth_new_intent"

    def test_only_stopwords_keeps_originals(self):
        # If we filtered everything out, fall back to raw tokens.
        slug = _slug("the a an")
        assert slug.startswith("synth_")
        # Should still be deterministic and non-empty.
        assert slug != "synth_"

    def test_slug_length_bounded(self):
        long_msg = "encrypt asymmetrically using rotating ephemeral session keys derived from passphrase"
        slug = _slug(long_msg)
        # synth_ prefix + at most 40 char body.
        assert len(slug) <= len("synth_") + 40

    def test_unicode_punctuation_stripped(self):
        # Smart-quote curly apostrophe — must not crash the regex.
        slug = _slug("what\u2019s on my calendar today?")
        assert slug == "synth_calendar_today"
