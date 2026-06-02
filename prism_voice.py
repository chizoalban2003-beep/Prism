from __future__ import annotations

import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class PrismVoice:
    """
    Speech-to-text input for PRISM.

    Transcription backends (tried in order, first available wins):
      1. openai-whisper  — local, fully offline, best accuracy
      2. faster-whisper  — local, lighter, ~4x faster
      3. SpeechRecognition + Google — requires internet, free tier

    Recording backends:
      1. sounddevice  — cross-platform, no extra deps on Linux
      2. PyAudio      — slightly wider platform support

    Usage:
        voice = PrismVoice.from_config(config)
        text  = voice.listen_once(seconds=5)   # record + transcribe
        text  = voice.transcribe("/tmp/clip.wav")  # file only
        voice.start_background(callback=agent.chat)  # continuous loop
    """

    SILENCE_THRESHOLD = 500   # RMS amplitude below which = silence
    SILENCE_FRAMES    = 20    # consecutive silent frames before stopping

    def __init__(
        self,
        model:      str  = "base",    # whisper model size: tiny/base/small/medium/large
        language:   str  = "en",      # ISO 639-1 language code
        enabled:    bool = True,
        sample_rate:int  = 16000,
        channels:   int  = 1,
    ):
        self._model_size  = model
        self._language    = language
        self._enabled     = enabled
        self._rate        = sample_rate
        self._channels    = channels
        self._whisper     = None       # lazy-loaded whisper model
        self._backend     = self._detect_transcription_backend()
        self._record_lib  = self._detect_recording_backend()
        self._bg_thread:  Optional[threading.Thread] = None
        self._bg_stop     = threading.Event()

    @classmethod
    def from_config(cls, config: dict) -> "PrismVoice":
        v = config.get("voice", {})
        return cls(
            model    = v.get("model", "base"),
            language = v.get("language", "en"),
            enabled  = v.get("enabled", True),
            sample_rate = v.get("sample_rate", 16000),
        )

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return bool(self._backend)

    @property
    def can_record(self) -> bool:
        return bool(self._record_lib)

    def listen_once(self, seconds: float = 5.0) -> str:
        """
        Record from microphone for `seconds` then transcribe.
        Returns the transcript, or "" if unavailable/error.
        """
        if not self._enabled:
            return ""
        if not self.can_record:
            logger.warning("PrismVoice: no recording library (install sounddevice or pyaudio)")
            return ""
        audio_path = self._record(seconds)
        if not audio_path:
            return ""
        try:
            return self.transcribe(audio_path)
        finally:
            try:
                os.unlink(audio_path)
            except Exception:
                pass

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe an existing audio file (WAV, MP3, FLAC, M4A).
        Returns transcript text, or "" on failure.
        """
        if not self._enabled or not self._backend:
            return ""
        path = Path(audio_path)
        if not path.exists():
            logger.warning("PrismVoice: file not found: %s", audio_path)
            return ""
        try:
            if self._backend == "whisper":
                return self._transcribe_whisper(path)
            if self._backend == "faster_whisper":
                return self._transcribe_faster_whisper(path)
            if self._backend == "speech_recognition":
                return self._transcribe_sr(path)
        except Exception as e:
            logger.warning("PrismVoice: transcription failed: %s", e)
        return ""

    def start_background(self, callback: Callable[[str], None],
                          interval_seconds: float = 5.0) -> None:
        """
        Start a background thread that continuously records and transcribes.
        Each non-empty transcript is passed to `callback`.
        Call stop_background() to halt.
        """
        if self._bg_thread and self._bg_thread.is_alive():
            return
        self._bg_stop.clear()
        self._bg_thread = threading.Thread(
            target=self._bg_loop,
            args=(callback, interval_seconds),
            daemon=True, name="prism-voice-bg")
        self._bg_thread.start()
        logger.info("PrismVoice: background listening started")

    def stop_background(self) -> None:
        self._bg_stop.set()

    # ── Recording ──────────────────────────────────────────────────────────

    def _record(self, seconds: float) -> Optional[str]:
        """Record from mic, save to temp WAV, return path."""
        try:
            if self._record_lib == "sounddevice":
                return self._record_sounddevice(seconds)
            if self._record_lib == "pyaudio":
                return self._record_pyaudio(seconds)
        except Exception as e:
            logger.warning("PrismVoice: recording failed: %s", e)
        return None

    def _record_sounddevice(self, seconds: float) -> str:
        import wave

        import sounddevice as sd

        logger.debug("PrismVoice: recording %.1fs via sounddevice", seconds)
        audio = sd.rec(
            int(seconds * self._rate),
            samplerate=self._rate,
            channels=self._channels,
            dtype="int16")
        sd.wait()

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(self._channels)
            wf.setsampwidth(2)   # 16-bit
            wf.setframerate(self._rate)
            wf.writeframes(audio.tobytes())
        return tmp.name

    def _record_pyaudio(self, seconds: float) -> str:
        import wave

        import pyaudio

        CHUNK = 1024
        logger.debug("PrismVoice: recording %.1fs via pyaudio", seconds)
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=self._channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=CHUNK)

        frames = []
        n_chunks = int(self._rate / CHUNK * seconds)
        for _ in range(n_chunks):
            frames.append(stream.read(CHUNK, exception_on_overflow=False))

        stream.stop_stream()
        stream.close()
        pa.terminate()

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(self._channels)
            wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self._rate)
            wf.writeframes(b"".join(frames))
        return tmp.name

    # ── Transcription ──────────────────────────────────────────────────────

    def _load_whisper(self):
        if self._whisper is None:
            import whisper
            logger.info("PrismVoice: loading whisper model '%s'…", self._model_size)
            self._whisper = whisper.load_model(self._model_size)
        return self._whisper

    def _transcribe_whisper(self, path: Path) -> str:
        model = self._load_whisper()
        opts  = {"language": self._language} if self._language else {}
        result = model.transcribe(str(path), **opts)
        return result.get("text", "").strip()

    def _transcribe_faster_whisper(self, path: Path) -> str:
        from faster_whisper import WhisperModel
        if self._whisper is None:
            logger.info("PrismVoice: loading faster-whisper model '%s'…", self._model_size)
            self._whisper = WhisperModel(self._model_size, device="cpu",
                                         compute_type="int8")
        model = self._whisper
        lang  = self._language or None
        segments, _ = model.transcribe(str(path), language=lang, beam_size=1)
        return " ".join(seg.text for seg in segments).strip()

    def _transcribe_sr(self, path: Path) -> str:
        import speech_recognition as sr
        recogniser = sr.Recognizer()
        with sr.AudioFile(str(path)) as src:
            audio = recogniser.record(src)
        return recogniser.recognize_google(audio, language=self._language)

    # ── Backend detection ──────────────────────────────────────────────────

    def _detect_transcription_backend(self) -> str:
        try:
            import whisper  # noqa
            return "whisper"
        except ImportError:
            pass
        try:
            import faster_whisper  # noqa
            return "faster_whisper"
        except ImportError:
            pass
        try:
            import speech_recognition  # noqa
            return "speech_recognition"
        except ImportError:
            pass
        logger.info("PrismVoice: no transcription backend found. "
                    "Install openai-whisper for local STT.")
        return ""

    def _detect_recording_backend(self) -> str:
        try:
            import sounddevice  # noqa
            return "sounddevice"
        except ImportError:
            pass
        try:
            import pyaudio  # noqa
            return "pyaudio"
        except ImportError:
            pass
        return ""

    # ── Background loop ────────────────────────────────────────────────────

    def _bg_loop(self, callback: Callable[[str], None],
                  interval: float) -> None:
        while not self._bg_stop.wait(0):
            text = self.listen_once(seconds=interval)
            if text:
                try:
                    callback(text)
                except Exception as e:
                    logger.debug("PrismVoice: callback error: %s", e)
            if self._bg_stop.wait(0.1):
                break
