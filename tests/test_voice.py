import os
import sys
import wave
import struct
import tempfile
from unittest.mock import MagicMock, patch
from prism_voice import PrismVoice


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_wav(path: str, duration_s: float = 0.1, rate: int = 16000) -> None:
    """Write a minimal valid WAV file for transcription tests."""
    n_frames = int(rate * duration_s)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(struct.pack(f"<{n_frames}h", *([0] * n_frames)))


def _make_voice(backend="", record_lib="") -> PrismVoice:
    v = PrismVoice(enabled=True)
    v._backend    = backend
    v._record_lib = record_lib
    return v


# ── Constructor / from_config ─────────────────────────────────────────────────

def test_from_config_defaults():
    v = PrismVoice.from_config({})
    assert v._model_size == "base"
    assert v._language == "en"
    assert v._enabled is True


def test_from_config_custom():
    v = PrismVoice.from_config({"voice": {"model": "small", "language": "fr",
                                            "enabled": False, "sample_rate": 8000}})
    assert v._model_size == "small"
    assert v._language == "fr"
    assert not v._enabled
    assert v._rate == 8000


# ── available / can_record ────────────────────────────────────────────────────

def test_available_true_when_backend_set():
    v = _make_voice(backend="whisper")
    assert v.available is True


def test_available_false_when_no_backend():
    v = _make_voice()
    assert v.available is False


def test_can_record_true_when_record_lib_set():
    v = _make_voice(record_lib="sounddevice")
    assert v.can_record is True


def test_can_record_false_when_no_record_lib():
    v = _make_voice()
    assert v.can_record is False


# ── transcribe — no backend ───────────────────────────────────────────────────

def test_transcribe_returns_empty_when_no_backend():
    v = _make_voice()
    assert v.transcribe("/tmp/nonexistent.wav") == ""


def test_transcribe_returns_empty_when_disabled():
    v = _make_voice(backend="whisper")
    v._enabled = False
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        _make_wav(f.name)
        assert v.transcribe(f.name) == ""
    os.unlink(f.name)


def test_transcribe_returns_empty_for_missing_file():
    v = _make_voice(backend="speech_recognition")
    assert v.transcribe("/tmp/__nonexistent_prism_test.wav") == ""


# ── transcribe — whisper backend ──────────────────────────────────────────────

def test_transcribe_whisper_success():
    v = _make_voice(backend="whisper")
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": "hello world"}
    v._whisper = mock_model

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        _make_wav(f.name)
        result = v.transcribe(f.name)
    os.unlink(f.name)

    assert result == "hello world"


def test_transcribe_whisper_exception_returns_empty():
    v = _make_voice(backend="whisper")
    mock_model = MagicMock()
    mock_model.transcribe.side_effect = RuntimeError("model error")
    v._whisper = mock_model

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        _make_wav(f.name)
        result = v.transcribe(f.name)
    os.unlink(f.name)

    assert result == ""


# ── transcribe — faster-whisper backend ───────────────────────────────────────

def test_transcribe_faster_whisper_success():
    v = _make_voice(backend="faster_whisper")
    seg1 = MagicMock(); seg1.text = "fast "
    seg2 = MagicMock(); seg2.text = "transcription"
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([seg1, seg2], MagicMock())
    v._whisper = mock_model

    # _transcribe_faster_whisper does a runtime import — pre-install the mock
    mock_fw_mod = MagicMock()
    mock_fw_mod.WhisperModel.return_value = mock_model

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        _make_wav(f.name)
        fname = f.name

    with patch.dict(sys.modules, {"faster_whisper": mock_fw_mod}):
        result = v._transcribe_faster_whisper(__import__("pathlib").Path(fname))
    os.unlink(fname)

    assert result == "fast  transcription"


# ── transcribe — speech_recognition backend ───────────────────────────────────

def test_transcribe_sr_success():
    v = _make_voice(backend="speech_recognition")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        _make_wav(f.name)
        fname = f.name

    mock_sr = MagicMock()
    mock_sr.AudioFile.return_value.__enter__ = lambda s: s
    mock_sr.AudioFile.return_value.__exit__ = MagicMock(return_value=False)
    mock_sr.Recognizer.return_value.recognize_google.return_value = "sr result"

    with patch.dict(sys.modules, {"speech_recognition": mock_sr}):
        result = v._transcribe_sr(__import__("pathlib").Path(fname))

    os.unlink(fname)
    assert result == "sr result"


# ── listen_once ───────────────────────────────────────────────────────────────

def test_listen_once_returns_empty_when_disabled():
    v = _make_voice(backend="whisper", record_lib="sounddevice")
    v._enabled = False
    assert v.listen_once(1.0) == ""


def test_listen_once_returns_empty_when_no_record_lib():
    v = _make_voice(backend="whisper")
    assert v.listen_once(1.0) == ""


def test_listen_once_transcribes_recording():
    v = _make_voice(backend="whisper", record_lib="sounddevice")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        _make_wav(f.name)
        tmp_path = f.name

    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": "recorded speech"}
    v._whisper = mock_model

    with patch.object(v, "_record", return_value=tmp_path):
        result = v.listen_once(2.0)

    assert result == "recorded speech"
    assert not os.path.exists(tmp_path)   # temp file cleaned up


# ── background listening ──────────────────────────────────────────────────────

def test_start_background_calls_callback():
    import time
    v = _make_voice(backend="whisper", record_lib="sounddevice")
    received = []

    def fake_listen(seconds):
        return "background text"

    with patch.object(v, "listen_once", side_effect=fake_listen):
        v.start_background(callback=received.append, interval_seconds=0.05)
        time.sleep(0.2)
        v.stop_background()

    assert len(received) >= 1
    assert received[0] == "background text"


def test_start_background_does_not_duplicate_thread():
    v = _make_voice(backend="whisper", record_lib="sounddevice")

    with patch.object(v, "listen_once", return_value=""):
        v.start_background(callback=lambda t: None, interval_seconds=0.1)
        thread1 = v._bg_thread
        v.start_background(callback=lambda t: None, interval_seconds=0.1)
        thread2 = v._bg_thread
        v.stop_background()

    assert thread1 is thread2   # second call is a no-op


# ── detect backends gracefully ────────────────────────────────────────────────

def test_detect_transcription_backend_no_deps():
    """When all optional libs missing, returns empty string without raising."""
    v2 = PrismVoice(enabled=False)
    assert isinstance(v2._backend, str)


def test_detect_recording_backend_no_deps():
    v = PrismVoice(enabled=False)
    assert isinstance(v._record_lib, str)
