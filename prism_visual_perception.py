"""
prism_visual_perception.py
==========================
Video/Audio Perception expansion for PRISM.

Real-time scene understanding via Ollama LLaVA; audio feature extraction;
OrganBus signal emission on detected events.

All Ollama calls are guarded with try/except — if Ollama is unavailable the
methods return stubs with confidence=0.0 rather than raising.
No heavy dependencies (no opencv, librosa, pyaudio) — pure Python + stdlib only.
"""
from __future__ import annotations

import base64
import json
import logging
import math
import struct
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Confidence threshold above which objects/actions are forwarded to OrganBus
_EMIT_CONFIDENCE_THRESHOLD = 0.4


# ---------------------------------------------------------------------------
# Domain dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SceneAnalysis:
    """Structured output of a single LLaVA scene-understanding call."""
    scene_id:         str
    objects:          list[str]    # detected objects
    actions:          list[str]    # detected actions/activities
    emotion:          str          # dominant emotion ("neutral", "focused", "stressed", …)
    confidence:       float        # 0.0 – 1.0
    raw_description:  str
    timestamp:        float
    source:           str          # "camera" | "image_file" | "video_frame"


@dataclass
class AudioFeatures:
    """Features extracted from a single audio chunk."""
    audio_id:            str
    volume_db:           float   # –60 to 0  (dBFS approximation)
    speech_rate:         float   # words-per-minute estimate (0 if no speech detected)
    dominant_frequency:  float   # Hz — crude zero-crossing estimate
    stress_indicator:    float   # 0-1 heuristic from volume + rate
    timestamp:           float


# ---------------------------------------------------------------------------
# VisualPerception
# ---------------------------------------------------------------------------

_SCENE_PROMPT = (
    "Analyse this image and return ONLY valid JSON (no extra text) with these keys: "
    '{"objects": ["list", "of", "detected", "objects"], '
    '"actions": ["list", "of", "detected", "actions", "or", "activities"], '
    '"emotion": "dominant emotion as a single word (neutral/focused/stressed/happy/sad/angry/relaxed)", '
    '"confidence": 0.0_to_1.0_float, '
    '"description": "one sentence summary"}'
)


class VisualPerception:
    """
    Scene-understanding engine using Ollama LLaVA.

    Usage::

        vp = VisualPerception(organ_bus=bus)
        scene = vp.analyse_image(image_path="/tmp/frame.jpg")
        vp.emit_scene_signals(scene)
    """

    _TIMEOUT = 60  # seconds

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model: str = "llava",
        organ_bus: Optional[Any] = None,
    ) -> None:
        self._host      = ollama_host.rstrip("/")
        self._model     = model
        self._organ_bus = organ_bus
        self._history:  list[SceneAnalysis] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse_image(
        self,
        image_path: str | None = None,
        image_bytes: bytes | None = None,
        source: str = "image_file",
    ) -> SceneAnalysis:
        """
        Analyse an image via Ollama LLaVA.

        Parameters
        ----------
        image_path :
            Path to an image file on disk.
        image_bytes :
            Raw image bytes (takes priority over image_path when supplied).
        source :
            Provenance tag stored in the returned SceneAnalysis.

        Returns
        -------
        SceneAnalysis
            If Ollama is unavailable, returns a stub with confidence=0.0.
        """
        scene_id = uuid.uuid4().hex[:12]

        # ── Load bytes ────────────────────────────────────────────────────────
        raw_bytes: bytes | None = None
        if image_bytes is not None:
            raw_bytes = image_bytes
        elif image_path is not None:
            try:
                with open(image_path, "rb") as fh:
                    raw_bytes = fh.read()
            except OSError as exc:
                logger.warning("[VisualPerception] cannot read %s: %s", image_path, exc)

        if raw_bytes is None:
            return self._stub_scene(scene_id, source, "no image data")

        b64 = base64.b64encode(raw_bytes).decode()
        return self._call_llava(b64, scene_id, source)

    def analyse_frame_base64(
        self,
        b64_data: str,
        source: str = "camera",
    ) -> SceneAnalysis:
        """Decode *b64_data* and call :meth:`analyse_image`."""
        try:
            raw = base64.b64decode(b64_data)
        except Exception as exc:
            logger.warning("[VisualPerception] base64 decode failed: %s", exc)
            return self._stub_scene(uuid.uuid4().hex[:12], source, "base64 decode error")
        return self.analyse_image(image_bytes=raw, source=source)

    def emit_scene_signals(self, scene: SceneAnalysis) -> int:
        """
        Emit OrganBus signals for each detected object and action when confidence
        is above the threshold.

        Returns the count of signals emitted.
        """
        if self._organ_bus is None:
            return 0
        if scene.confidence < _EMIT_CONFIDENCE_THRESHOLD:
            return 0

        try:
            from prism_organ_bus import NORMAL, OrganSignal
        except ImportError:
            logger.debug("[VisualPerception] prism_organ_bus not available")
            return 0

        emitted = 0

        for obj in scene.objects:
            if not obj:
                continue
            try:
                self._organ_bus.emit(OrganSignal(
                    source      = "visual_perception",
                    signal_type = "visual_object_detected",
                    payload     = {
                        "object":     obj,
                        "scene_id":   scene.scene_id,
                        "confidence": scene.confidence,
                        "emotion":    scene.emotion,
                    },
                    priority = NORMAL,
                ))
                emitted += 1
            except Exception as exc:
                logger.debug("[VisualPerception] emit object signal error: %s", exc)

        for action in scene.actions:
            if not action:
                continue
            try:
                self._organ_bus.emit(OrganSignal(
                    source      = "visual_perception",
                    signal_type = "visual_action_detected",
                    payload     = {
                        "action":     action,
                        "scene_id":   scene.scene_id,
                        "confidence": scene.confidence,
                        "emotion":    scene.emotion,
                    },
                    priority = NORMAL,
                ))
                emitted += 1
            except Exception as exc:
                logger.debug("[VisualPerception] emit action signal error: %s", exc)

        return emitted

    def recent_history(self, n: int = 20) -> list[SceneAnalysis]:
        """Return the last *n* SceneAnalysis records."""
        return list(self._history[-n:])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llava(self, b64: str, scene_id: str, source: str) -> SceneAnalysis:
        """POST to Ollama /api/generate with the image; parse response."""
        payload = json.dumps({
            "model":  self._model,
            "prompt": _SCENE_PROMPT,
            "images": [b64],
            "stream": False,
        }).encode()

        req = urllib.request.Request(
            f"{self._host}/api/generate",
            data    = payload,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self._TIMEOUT) as resp:
                outer = json.loads(resp.read().decode())
        except Exception as exc:
            logger.debug("[VisualPerception] Ollama unavailable: %s", exc)
            return self._stub_scene(scene_id, source, "ollama_unavailable")

        raw_text = outer.get("response", "")
        scene = self._parse_llava_response(raw_text, scene_id, source)
        self._history.append(scene)
        if len(self._history) > 200:
            self._history = self._history[-200:]
        return scene

    def _parse_llava_response(
        self, text: str, scene_id: str, source: str
    ) -> SceneAnalysis:
        """Extract structured data from the raw LLaVA text response."""
        data: dict[str, Any] = {}

        # Try direct JSON parse first
        try:
            data = json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        if not data:
            # Try to find first {...} block
            start = text.find("{")
            end   = text.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    data = json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass

        def _safe_list(key: str) -> list[str]:
            v = data.get(key, [])
            if isinstance(v, list):
                return [str(x) for x in v if x]
            if isinstance(v, str):
                return [v] if v else []
            return []

        def _safe_float(key: str, default: float = 0.0) -> float:
            try:
                return float(data.get(key, default))
            except (ValueError, TypeError):
                return default

        confidence = min(1.0, max(0.0, _safe_float("confidence", 0.5 if data else 0.0)))
        emotion    = str(data.get("emotion", "neutral")).lower() if data else "neutral"
        raw_desc   = str(data.get("description", text[:200]))

        return SceneAnalysis(
            scene_id        = scene_id,
            objects         = _safe_list("objects"),
            actions         = _safe_list("actions"),
            emotion         = emotion,
            confidence      = confidence,
            raw_description = raw_desc,
            timestamp       = time.time(),
            source          = source,
        )

    @staticmethod
    def _stub_scene(scene_id: str, source: str, reason: str = "") -> SceneAnalysis:
        """Return a zero-confidence stub when analysis cannot proceed."""
        return SceneAnalysis(
            scene_id        = scene_id,
            objects         = [],
            actions         = [],
            emotion         = "neutral",
            confidence      = 0.0,
            raw_description = reason,
            timestamp       = time.time(),
            source          = source,
        )


# ---------------------------------------------------------------------------
# AudioAnalyzer
# ---------------------------------------------------------------------------

# Minimum RMS value for speech detection (avoids log(0))
_RMS_SILENCE_THRESHOLD = 1e-6
# Approximate silence floor in dBFS
_DB_SILENCE = -60.0


class AudioAnalyzer:
    """
    Pure-Python heuristic audio feature extractor.

    No librosa / pyaudio / numpy required — uses only the stdlib ``struct``
    module and basic arithmetic.  Input is expected to be raw 16-bit
    little-endian PCM (mono, any sample rate).

    If the audio cannot be decoded the method returns a stub with all
    numeric fields at 0 and stress_indicator=0.0.
    """

    def extract_features(
        self,
        audio_bytes: bytes | None = None,
        audio_path: str | None = None,
        sample_rate: int = 16000,
    ) -> AudioFeatures:
        """
        Compute heuristic audio features from raw 16-bit PCM bytes.

        Parameters
        ----------
        audio_bytes :
            Raw 16-bit little-endian mono PCM samples.
        audio_path :
            Path to a file containing the same format (used if
            audio_bytes is None).
        sample_rate :
            Samples per second — used for frequency estimates.

        Returns
        -------
        AudioFeatures
            Stub (stress_indicator=0.0) if audio is unavailable.
        """
        audio_id = uuid.uuid4().hex[:12]

        raw: bytes | None = None
        if audio_bytes is not None:
            raw = audio_bytes
        elif audio_path is not None:
            try:
                with open(audio_path, "rb") as fh:
                    raw = fh.read()
            except OSError as exc:
                logger.debug("[AudioAnalyzer] cannot read %s: %s", audio_path, exc)

        if not raw or len(raw) < 2:
            return self._stub_features(audio_id)

        # ── Decode PCM ────────────────────────────────────────────────────────
        n_samples = len(raw) // 2
        try:
            samples = struct.unpack(f"<{n_samples}h", raw[: n_samples * 2])
        except struct.error:
            return self._stub_features(audio_id)

        if not samples:
            return self._stub_features(audio_id)

        # ── RMS → dBFS volume ─────────────────────────────────────────────────
        rms = math.sqrt(sum(s * s for s in samples) / len(samples)) / 32768.0
        rms = max(rms, _RMS_SILENCE_THRESHOLD)
        volume_db = max(_DB_SILENCE, 20.0 * math.log10(rms))

        # ── Zero-crossing rate → dominant frequency estimate ──────────────────
        zero_crossings = sum(
            1 for i in range(1, len(samples))
            if (samples[i - 1] >= 0) != (samples[i] >= 0)
        )
        # Each crossing = half a cycle; divide by duration to get Hz
        duration_s = len(samples) / max(sample_rate, 1)
        dominant_frequency = (zero_crossings / 2.0) / max(duration_s, 1e-9)

        # ── Speech-rate heuristic ─────────────────────────────────────────────
        # Map zero-crossing rate to a rough syllable/word rate.
        # Human speech ZCR roughly 80-300 Hz; >150 Hz implies faster speech.
        zcr_hz = dominant_frequency
        # Words-per-minute rough proxy: (ZCR - 80) / 2.0
        speech_rate = max(0.0, (zcr_hz - 80.0) / 2.0) if zcr_hz > 80 else 0.0

        # ── Stress indicator ──────────────────────────────────────────────────
        # Heuristic: normalised volume (0=silence, 1=clipping) + high speech rate
        vol_norm     = min(1.0, max(0.0, (volume_db - _DB_SILENCE) / abs(_DB_SILENCE)))
        rate_norm    = min(1.0, speech_rate / 300.0)
        stress_indicator = min(1.0, vol_norm * 0.6 + rate_norm * 0.4)

        return AudioFeatures(
            audio_id           = audio_id,
            volume_db          = round(volume_db, 2),
            speech_rate        = round(speech_rate, 2),
            dominant_frequency = round(dominant_frequency, 2),
            stress_indicator   = round(stress_indicator, 4),
            timestamp          = time.time(),
        )

    def emit_audio_signals(
        self,
        features: AudioFeatures,
        organ_bus: Optional[Any] = None,
    ) -> int:
        """
        Emit an ``audio_stress_detected`` OrganBus signal when
        ``features.stress_indicator > 0.7``.

        Returns
        -------
        int
            Number of signals emitted (0 or 1).
        """
        if organ_bus is None or features.stress_indicator <= 0.7:
            return 0

        try:
            from prism_organ_bus import HIGH, OrganSignal
        except ImportError:
            logger.debug("[AudioAnalyzer] prism_organ_bus not available")
            return 0

        try:
            organ_bus.emit(OrganSignal(
                source      = "audio_analyzer",
                signal_type = "audio_stress_detected",
                payload     = {
                    "stress_indicator":    features.stress_indicator,
                    "volume_db":           features.volume_db,
                    "speech_rate":         features.speech_rate,
                    "dominant_frequency":  features.dominant_frequency,
                    "audio_id":            features.audio_id,
                },
                priority = HIGH,
            ))
            return 1
        except Exception as exc:
            logger.debug("[AudioAnalyzer] emit_audio_signals error: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stub_features(audio_id: str) -> AudioFeatures:
        return AudioFeatures(
            audio_id           = audio_id,
            volume_db          = _DB_SILENCE,
            speech_rate        = 0.0,
            dominant_frequency = 0.0,
            stress_indicator   = 0.0,
            timestamp          = time.time(),
        )
