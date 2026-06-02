from __future__ import annotations

from prism_tts import PrismTTS

# ── Setup / factory ───────────────────────────────────────────────────────────

def test_setup_returns_instance():
    tts = PrismTTS.setup()
    assert isinstance(tts, PrismTTS)


def test_disabled_by_default():
    tts = PrismTTS()
    assert not tts._enabled


# ── toggle ────────────────────────────────────────────────────────────────────

def test_toggle_enables():
    tts = PrismTTS(enabled=False)
    result = tts.toggle()
    assert result is True
    assert tts._enabled is True


def test_toggle_disables():
    tts = PrismTTS(enabled=True)
    result = tts.toggle()
    assert result is False
    assert tts._enabled is False


def test_toggle_idempotent():
    tts = PrismTTS(enabled=False)
    tts.toggle()
    tts.toggle()
    assert tts._enabled is False


# ── available ─────────────────────────────────────────────────────────────────

def test_available_returns_bool():
    tts = PrismTTS()
    assert isinstance(tts.available, bool)


# ── speak (disabled) ──────────────────────────────────────────────────────────

def test_speak_disabled_no_error():
    tts = PrismTTS(enabled=False)
    # Should not raise even though disabled
    tts.speak("Hello world")


def test_speak_empty_text_no_error():
    tts = PrismTTS(enabled=True)
    # strip_markdown of empty/whitespace returns "" → should not call engine
    tts.speak("")


# ── _strip_markdown ───────────────────────────────────────────────────────────

def test_strip_bold():
    result = PrismTTS._strip_markdown("**bold text**")
    assert "bold text" in result
    assert "**" not in result


def test_strip_italic():
    result = PrismTTS._strip_markdown("_italic text_")
    assert "italic text" in result
    assert "_" not in result


def test_strip_code():
    result = PrismTTS._strip_markdown("`code here`")
    assert "code here" in result
    assert "`" not in result


def test_strip_heading():
    result = PrismTTS._strip_markdown("## Heading text")
    assert "Heading text" in result
    assert "#" not in result


def test_strip_link():
    result = PrismTTS._strip_markdown("[click here](https://example.com)")
    assert "click here" in result
    assert "https://example.com" not in result


def test_strip_html():
    result = PrismTTS._strip_markdown("<b>bold</b>")
    assert "<b>" not in result
    assert "bold" in result


def test_strip_complex():
    md = "**PRISM** says: _hello_ world. See [docs](https://prism.ai) for `details`."
    result = PrismTTS._strip_markdown(md)
    assert "PRISM" in result
    assert "hello" in result
    assert "world" in result
    assert "**" not in result
    assert "_" not in result
    assert "`" not in result
    assert "https://prism.ai" not in result


def test_strip_plain_text_unchanged():
    text = "Hello, this is plain text with no markdown."
    result = PrismTTS._strip_markdown(text)
    assert result == text


# ── _detect_engine ────────────────────────────────────────────────────────────

def test_detect_engine_returns_string():
    tts = PrismTTS()
    assert isinstance(tts._engine, str)
