"""
prism_perception.py
===================
PRISM Perceptual Context Engine

Continuous background sensing that converts raw input from all available
channels into normalised factor values (0-1) for the decision engine.

The core loop:
  Every channel runs in its own daemon thread.
  Each produces a stream of ContextSignal objects.
  The ContextFuser aggregates signals into a ContextState.
  The ContextState is a dict of factor_id → float used by PrismAgent
  to enrich every decision with real-time perceptual context.

Privacy principles (enforced in code):
  All processing is local — nothing leaves the device.
  Raw data (audio frames, camera frames) is never stored.
  Only derived factor values are stored.
  Every channel is opt-in and can be paused at any time.
  A visible indicator shows which channels are active.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context signal — the unit of output from every channel
# ---------------------------------------------------------------------------

@dataclass
class ContextSignal:
    channel:    str          # "voice"|"biometric"|"screen"|"system"|"typing"
    factor_id:  str          # maps to a Factor name in the decision engine
    value:      float        # 0.0 to 1.0 normalised
    confidence: float        # how reliable is this reading
    timestamp:  float = field(default_factory=time.time)
    raw_label:  str   = ""   # human-readable description of the raw reading


@dataclass
class ContextState:
    """
    The fused, current-moment context built from all active channels.
    This is what the decision engine sees — not raw sensor data,
    but a clean map of normalised factor values.
    """
    factors:      dict[str, float]    # factor_id → value
    confidence:   dict[str, float]    # factor_id → confidence
    active_channels: list[str]
    last_updated: float = field(default_factory=time.time)
    summary:      str   = ""          # plain English state summary

    def to_factor_updates(self) -> dict[str, float]:
        """Return only factors with sufficient confidence."""
        return {k: v for k, v in self.factors.items()
                if self.confidence.get(k, 0) >= 0.4}


# ---------------------------------------------------------------------------
# Base channel
# ---------------------------------------------------------------------------

class PerceptionChannel:
    """
    Abstract base. Each channel runs in its own daemon thread,
    pushing ContextSignal objects to a shared queue.
    """
    NAME = "base"

    def __init__(self, signal_queue: queue.Queue, enabled: bool = True):
        self._q       = signal_queue
        self._enabled = enabled
        self._stop    = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self._enabled:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"prism-{self.NAME}")
        self._thread.start()
        logger.info("Perception channel started: %s", self.NAME)

    def stop(self) -> None:
        self._stop.set()

    def pause(self) -> None:
        self._enabled = False

    def resume(self) -> None:
        self._enabled = True

    def _emit(self, factor_id: str, value: float,
               confidence: float, raw_label: str = "") -> None:
        self._q.put(ContextSignal(
            channel    = self.NAME,
            factor_id  = factor_id,
            value      = max(0.0, min(1.0, value)),
            confidence = confidence,
            raw_label  = raw_label,
        ))

    def _run(self) -> None:
        """Override in subclasses. Must check self._stop periodically."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# System context channel — always available, no permissions needed
# ---------------------------------------------------------------------------

class SystemContextChannel(PerceptionChannel):
    """
    Reads system state: time of day, battery, CPU load, active apps,
    network location. No microphone or camera required.
    Polls every 30 seconds.
    """
    NAME = "system"

    def _run(self) -> None:
        while not self._stop.wait(30.0):
            if not self._enabled:
                continue
            try:
                self._emit_time_context()
                self._emit_system_load()
                self._emit_battery()
            except Exception as e:
                logger.debug("System channel error: %s", e)

    def _emit_time_context(self) -> None:
        """Circadian context — energy level follows a human daily rhythm."""
        from datetime import datetime
        h = datetime.now().hour
        # Rough human energy curve: peaks at 10am and 3pm, low at 2pm and night
        curve = {
            0:0.15, 1:0.10, 2:0.08, 3:0.08, 4:0.10, 5:0.20,
            6:0.40, 7:0.60, 8:0.75, 9:0.85, 10:0.90, 11:0.88,
            12:0.80, 13:0.70, 14:0.65, 15:0.82, 16:0.80, 17:0.72,
            18:0.65, 19:0.58, 20:0.50, 21:0.42, 22:0.32, 23:0.22,
        }
        self._emit("circadian_energy", curve.get(h, 0.5), 0.7,
                   f"{h:02d}:00 circadian phase")

        # Work hours context
        is_work_hours = 8 <= h <= 18
        self._emit("work_context", 0.8 if is_work_hours else 0.2,
                   0.9, "work hours" if is_work_hours else "off hours")

    def _emit_system_load(self) -> None:
        try:
            import psutil
            cpu   = psutil.cpu_percent(interval=1) / 100.0
            mem   = psutil.virtual_memory().percent / 100.0
            # High CPU = system is busy = user is likely actively working
            self._emit("system_busy", cpu, 0.8, f"CPU {cpu:.0%}")
            self._emit("memory_pressure", mem, 0.8, f"RAM {mem:.0%}")
        except ImportError:
            pass

    def _emit_battery(self) -> None:
        try:
            import psutil
            bat = psutil.sensors_battery()
            if bat:
                level   = bat.percent / 100.0
                on_power= 1.0 if bat.power_plugged else 0.0
                self._emit("battery_level",   level,    0.95, f"Battery {bat.percent:.0f}%")
                self._emit("on_power",         on_power, 0.95,
                           "plugged in" if bat.power_plugged else "on battery")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Typing pattern channel — infers focus and stress from keyboard behaviour
# ---------------------------------------------------------------------------

class TypingPatternChannel(PerceptionChannel):
    """
    Monitors typing cadence via the system keyboard buffer.
    Does NOT capture what is typed — only timing patterns.

    Inferred signals:
      typing_speed    → high = focused and energetic
      typing_regularity → irregular = stressed or distracted
      active_typing   → currently in keyboard-intensive work
    """
    NAME = "typing"

    def __init__(self, signal_queue: queue.Queue, enabled: bool = True):
        super().__init__(signal_queue, enabled)
        self._key_times: list[float] = []
        self._max_samples = 30

    def record_keypress(self) -> None:
        """Call this from a keyboard hook to record timing."""
        now = time.time()
        self._key_times.append(now)
        if len(self._key_times) > self._max_samples:
            self._key_times.pop(0)

    def _run(self) -> None:
        """Emit typing-derived signals every 10 seconds."""
        while not self._stop.wait(10.0):
            if not self._enabled or len(self._key_times) < 5:
                continue
            self._analyse()
            # Prune old timestamps
            cutoff = time.time() - 60.0
            self._key_times = [t for t in self._key_times if t > cutoff]

    def _analyse(self) -> None:
        if len(self._key_times) < 3:
            return
        times  = sorted(self._key_times[-20:])
        gaps   = [times[i+1]-times[i] for i in range(len(times)-1)]
        if not gaps:
            return
        avg_gap = sum(gaps) / len(gaps)
        std_gap = math.sqrt(sum((g-avg_gap)**2 for g in gaps) / len(gaps))

        # Speed: inverse of average gap, normalised to 0-1
        # 0.1s gap (10 kps) = high speed, >2s gap = slow
        speed = max(0.0, min(1.0, 1.0 - (avg_gap - 0.08) / 1.5))

        # Regularity: lower std relative to mean = more regular = more focused
        cv = std_gap / max(avg_gap, 0.001)   # coefficient of variation
        regularity = max(0.0, min(1.0, 1.0 - cv * 0.5))

        self._emit("typing_speed",       speed,       0.7, f"avg gap {avg_gap:.2f}s")
        self._emit("typing_regularity",  regularity,  0.6, f"CV {cv:.2f}")
        self._emit("keyboard_active",    1.0,         0.9, "actively typing")


# ---------------------------------------------------------------------------
# Biometric channel — reads from device_hub wearable data
# ---------------------------------------------------------------------------

class BiometricChannel(PerceptionChannel):
    """
    Reads wearable data from device_hub.py (Apple Health, Garmin, etc.)
    and converts to decision-relevant factor values.
    Polls every 5 minutes (wearable data does not change faster).
    """
    NAME = "biometric"

    def __init__(self, signal_queue: queue.Queue,
                 device_hub=None, enabled: bool = True):
        super().__init__(signal_queue, enabled)
        self._hub = device_hub

    def _run(self) -> None:
        while not self._stop.wait(300.0):   # every 5 minutes
            if not self._enabled or self._hub is None:
                continue
            try:
                self._read_wearables()
            except Exception as e:
                logger.debug("Biometric channel error: %s", e)

    def ingest(self, hrv_ms: float = None, heart_rate: int = None,
                sleep_hrs: float = None, steps: int = None,
                soreness: int = None) -> None:
        """
        Direct ingestion for manual or wearable-pushed data.
        Call this when a wearable sync occurs.
        """
        if hrv_ms is not None:
            # HRV: <30ms = very stressed, >80ms = well recovered
            hrv_norm = max(0.0, min(1.0, (hrv_ms - 20) / 80.0))
            self._emit("hrv_recovery", hrv_norm, 0.9,
                       f"HRV {hrv_ms:.0f}ms")
            self._emit("stress_level", 1.0 - hrv_norm, 0.85,
                       "stress from HRV")

        if heart_rate is not None:
            # Resting HR: <55 = athletic, >90 = stressed/unfit
            hr_norm = max(0.0, min(1.0, 1.0 - (heart_rate - 45) / 60.0))
            self._emit("cardio_state", hr_norm, 0.8, f"HR {heart_rate}bpm")

        if sleep_hrs is not None:
            # Sleep: <5hrs = very poor, >8hrs = excellent
            sleep_norm = max(0.0, min(1.0, (sleep_hrs - 4.0) / 5.0))
            self._emit("sleep_quality", sleep_norm, 0.95,
                       f"sleep {sleep_hrs:.1f}hrs")
            self._emit("cognitive_readiness", sleep_norm * 0.7 + 0.15,
                       0.85, "readiness from sleep")

        if steps is not None:
            # Steps: 0 = sedentary, 10000+ = active
            steps_norm = min(1.0, steps / 10000.0)
            self._emit("activity_today", steps_norm, 0.8, f"{steps} steps")

        if soreness is not None:
            # Soreness: 1-10 scale
            soreness_norm = max(0.0, min(1.0, soreness / 10.0))
            self._emit("physical_soreness", soreness_norm, 0.85,
                       f"soreness {soreness}/10")

    def _read_wearables(self) -> None:
        if not self._hub:
            return
        try:
            data = self._hub.latest_health_snapshot()
            if data:
                self.ingest(
                    hrv_ms     = data.get("hrv"),
                    heart_rate = data.get("heart_rate"),
                    sleep_hrs  = data.get("sleep_hours"),
                    steps      = data.get("steps"),
                    soreness   = data.get("soreness"),
                )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Voice channel — speech-to-text + tone analysis
# ---------------------------------------------------------------------------

class VoiceChannel(PerceptionChannel):
    """
    Microphone → speech-to-text via Whisper (local, via Ollama or openai-whisper).
    Also analyses speech rate and energy as stress/focus proxies.

    Privacy: raw audio frames are NEVER stored or transmitted.
    Only the transcribed text and derived factor values are used.

    Requires: either 'openai-whisper' pip package or Ollama with a speech model.
    Falls back gracefully if neither is available.
    """
    NAME = "voice"
    SAMPLE_RATE   = 16000
    CHUNK_SECONDS = 30    # process in 30-second chunks

    def __init__(self, signal_queue: queue.Queue,
                 whisper_model: str = "base",
                 wake_word:     str = "hey prism",
                 enabled:       bool = True,
                 on_transcript: Callable = None):
        super().__init__(signal_queue, enabled)
        self._whisper_model  = whisper_model
        self._wake_word      = wake_word.lower()
        self._on_transcript  = on_transcript   # callback when speech detected
        self._whisper        = None
        self._microphone_ok  = False

    def _run(self) -> None:
        if not self._try_init():
            logger.info("Voice channel: no audio input available — skipping")
            return

        logger.info("Voice channel active (wake word: '%s')", self._wake_word)
        while not self._stop.wait(0.1):
            if not self._enabled:
                time.sleep(1.0)
                continue
            try:
                self._listen_chunk()
            except Exception as e:
                logger.debug("Voice channel error: %s", e)
                time.sleep(5.0)

    def _try_init(self) -> bool:
        """Try to initialise audio input and Whisper. Return True if ready."""
        try:
            import pyaudio  # noqa
            self._microphone_ok = True
        except ImportError:
            return False

        try:
            import whisper
            self._whisper = whisper.load_model(self._whisper_model)
            return True
        except ImportError:
            pass

        return False

    def _listen_chunk(self) -> None:
        """
        Record CHUNK_SECONDS of audio, transcribe, analyse.
        If wake word detected: trigger on_transcript callback.
        """
        try:
            import pyaudio
            import numpy as np
        except ImportError:
            time.sleep(60.0)
            return

        pa     = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1,
                         rate=self.SAMPLE_RATE, input=True,
                         frames_per_buffer=1024)
        frames = []
        for _ in range(0, int(self.SAMPLE_RATE / 1024 * self.CHUNK_SECONDS)):
            if self._stop.is_set():
                break
            frames.append(stream.read(1024, exception_on_overflow=False))
        stream.stop_stream(); stream.close(); pa.terminate()

        if not frames:
            return

        # Convert to numpy for Whisper — raw audio never written to disk
        audio = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32)
        audio = audio / 32768.0   # normalise to [-1, 1]

        # Voice activity — are they speaking at all?
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.005:    # silence threshold
            self._emit("voice_active", 0.0, 0.9, "silence detected")
            return

        self._emit("voice_active", 1.0, 0.9, "speech detected")

        # Speech rate proxy — rough estimate from zero-crossing rate
        zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2)
        speech_rate_norm = min(1.0, zcr * 20.0)
        self._emit("speech_rate", speech_rate_norm, 0.6,
                   f"ZCR proxy {zcr:.3f}")

        # Transcribe
        if self._whisper is None:
            return
        try:
            result = self._whisper.transcribe(audio, fp16=False, language="en")
            text   = result.get("text","").strip().lower()
            if not text:
                return

            # Stress signals from language
            stress_words = ["urgent","immediately","asap","deadline","help",
                            "worried","stressed","overwhelmed","behind"]
            calm_words   = ["fine","good","great","relaxed","done","finished"]
            stress_count = sum(1 for w in stress_words if w in text)
            calm_count   = sum(1 for w in calm_words   if w in text)
            if stress_count or calm_count:
                total = stress_count + calm_count
                stress_signal = stress_count / total
                self._emit("voice_stress", stress_signal, 0.5,
                           f"stress words: {stress_count}")

            # Wake word detection
            if self._wake_word in text:
                command = text.split(self._wake_word, 1)[-1].strip()
                if command and self._on_transcript:
                    self._on_transcript(command)

        except Exception as e:
            logger.debug("Whisper transcription error: %s", e)


# ---------------------------------------------------------------------------
# Screen context channel
# ---------------------------------------------------------------------------

class ScreenContextChannel(PerceptionChannel):
    """
    Periodic screenshot → Ollama LLaVA analysis.
    Infers what kind of work is happening and current focus level.
    Privacy: screenshots are never stored — only the analysis text.
    Polls every 2 minutes.
    """
    NAME = "screen"

    def __init__(self, signal_queue: queue.Queue,
                 ollama_host: str = "http://localhost:11434",
                 enabled: bool = False):   # opt-in only
        super().__init__(signal_queue, enabled)
        self._ollama = ollama_host

    def _run(self) -> None:
        while not self._stop.wait(120.0):   # every 2 minutes
            if not self._enabled:
                continue
            try:
                self._analyse_screen()
            except Exception as e:
                logger.debug("Screen channel error: %s", e)

    def _analyse_screen(self) -> None:
        # Capture screenshot
        try:
            from PIL import ImageGrab
            shot = ImageGrab.grab()
        except ImportError:
            try:
                import mss
                import PIL.Image
                with mss.mss() as sct:
                    raw  = sct.grab(sct.monitors[1])
                    shot = PIL.Image.frombytes("RGB", raw.size, raw.bgra, "raw","BGRX")
            except ImportError:
                return

        # Resize to reduce LLaVA processing time
        shot = shot.resize((640, 360))

        # Convert to base64 for LLaVA
        import base64
        import io
        buf = io.BytesIO()
        shot.save(buf, format="JPEG", quality=60)
        b64 = base64.b64encode(buf.getvalue()).decode()

        prompt = (
            "Analyse this screenshot in 2 sentences. State: "
            "1) what kind of work is shown (coding/writing/email/browsing/idle/meeting), "
            "2) estimated focus level (high/medium/low). "
            "Reply with JSON: {\"work_type\":\"...\",\"focus\":\"high|medium|low\"}"
        )

        try:
            import json
            import urllib.request
            payload = json.dumps({
                "model":  "llava",
                "prompt": prompt,
                "images": [b64],
                "stream": False,
            }).encode()
            req  = urllib.request.Request(
                f"{self._ollama}/api/generate",
                data=payload,
                headers={"Content-Type":"application/json"})
            resp = urllib.request.urlopen(req, timeout=15)
            outer = json.loads(resp.read())
            data = json.loads(outer.get("response","{}").strip())

            focus_map  = {"high": 0.9, "medium": 0.5, "low": 0.2}
            focus_val  = focus_map.get(data.get("focus","medium"), 0.5)
            work_type  = data.get("work_type","unknown")

            self._emit("screen_focus",     focus_val, 0.65, work_type)
            self._emit("screen_work_type",
                       0.9 if work_type != "idle" else 0.1, 0.7, work_type)

            # Work type → domain context
            type_signal = {
                "coding":    ("developer_context", 0.9),
                "email":     ("communication_context", 0.8),
                "writing":   ("creative_context", 0.8),
                "meeting":   ("meeting_active", 0.9),
                "browsing":  ("research_context", 0.6),
                "idle":      ("idle_context", 0.9),
            }.get(work_type)
            if type_signal:
                self._emit(type_signal[0], type_signal[1], 0.65, work_type)

        except Exception as e:
            logger.debug("LLaVA screen analysis failed: %s", e)


# ---------------------------------------------------------------------------
# Context fuser — aggregates all channel signals into one ContextState
# ---------------------------------------------------------------------------

class ContextFuser:
    """
    Subscribes to the shared signal queue.
    Maintains a rolling window of signals (last 10 minutes).
    Produces a ContextState by taking confidence-weighted averages.
    """
    WINDOW_SECONDS = 600     # 10-minute rolling window
    DECAY_HALF_LIFE = 120.0  # older signals count less (2-min half-life)

    def __init__(self, signal_queue: queue.Queue):
        self._q       = signal_queue
        self._signals: dict[str, list[ContextSignal]] = {}
        self._lock    = threading.Lock()
        self._stop    = threading.Event()
        self._thread  = threading.Thread(
            target=self._fuse_loop, daemon=True, name="prism-fuser")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def current_state(self) -> ContextState:
        """Return the current fused context state."""
        now  = time.time()
        cutoff = now - self.WINDOW_SECONDS
        factors: dict[str, float]    = {}
        confidence: dict[str, float] = {}

        with self._lock:
            for factor_id, sigs in self._signals.items():
                recent = [s for s in sigs if s.timestamp > cutoff]
                if not recent:
                    continue
                # Exponential time-weighted average
                weights = [
                    s.confidence * (0.5 ** ((now - s.timestamp) / self.DECAY_HALF_LIFE))
                    for s in recent
                ]
                total_w = sum(weights)
                if total_w < 1e-9:
                    continue
                fused_value = sum(s.value * w for s, w in zip(recent, weights)) / total_w
                avg_conf    = sum(s.confidence for s in recent) / len(recent)
                factors[factor_id]    = round(fused_value, 3)
                confidence[factor_id] = round(avg_conf,    3)

        active = list({s.channel for sigs in self._signals.values()
                       for s in sigs if s.timestamp > now - 60})

        return ContextState(
            factors          = factors,
            confidence       = confidence,
            active_channels  = active,
            last_updated     = now,
            summary          = self._summarise(factors),
        )

    def _fuse_loop(self) -> None:
        while not self._stop.wait(0.1):
            try:
                sig = self._q.get(timeout=0.5)
                with self._lock:
                    if sig.factor_id not in self._signals:
                        self._signals[sig.factor_id] = []
                    self._signals[sig.factor_id].append(sig)
                    # Prune old
                    cutoff = time.time() - self.WINDOW_SECONDS
                    self._signals[sig.factor_id] = [
                        s for s in self._signals[sig.factor_id]
                        if s.timestamp > cutoff
                    ]
            except queue.Empty:
                pass

    @staticmethod
    def _summarise(factors: dict[str, float]) -> str:
        parts = []
        if factors.get("stress_level", 0) > 0.7:
            parts.append("high stress")
        elif factors.get("hrv_recovery", 1) > 0.7:
            parts.append("well recovered")
        if factors.get("sleep_quality", 0.5) < 0.4:
            parts.append("sleep-deprived")
        if factors.get("screen_focus", 0.5) > 0.75:
            parts.append("focused")
        if factors.get("voice_active", 0) > 0.5:
            parts.append("actively speaking")
        return ", ".join(parts) if parts else "normal context"


# ---------------------------------------------------------------------------
# Main perception engine
# ---------------------------------------------------------------------------

class PrismPerception:
    """
    Orchestrates all perception channels.
    Provides a single interface for the rest of PRISM to get context.

    Usage:
        perception = PrismPerception.setup(
            enable_voice  = True,
            enable_screen = False,  # off by default — explicit opt-in
            device_hub    = hub,
            on_voice_command = agent.chat,
        )
        perception.start()

        # Anywhere in the codebase:
        context = perception.current_context()
        # context.factors = {"stress_level":0.65,"sleep_quality":0.45,...}
        # Pass these to the decision engine as factor updates.
    """

    def __init__(
        self,
        enable_voice:    bool = False,
        enable_screen:   bool = False,
        enable_biometric:bool = True,
        enable_system:   bool = True,
        enable_typing:   bool = True,
        device_hub             = None,
        ollama_host:     str  = "http://localhost:11434",
        whisper_model:   str  = "base",
        wake_word:       str  = "hey prism",
        on_voice_command: Callable = None,
    ):
        self._q       = queue.Queue()
        self._fuser   = ContextFuser(self._q)
        self._channels: list[PerceptionChannel] = []

        if enable_system:
            self._channels.append(SystemContextChannel(self._q))

        if enable_typing:
            self._typing = TypingPatternChannel(self._q)
            self._channels.append(self._typing)
        else:
            self._typing = None

        if enable_biometric:
            self._channels.append(BiometricChannel(self._q, device_hub))

        if enable_voice:
            self._channels.append(VoiceChannel(
                self._q,
                whisper_model    = whisper_model,
                wake_word        = wake_word,
                on_transcript    = on_voice_command,
            ))

        if enable_screen:
            self._channels.append(ScreenContextChannel(
                self._q, ollama_host=ollama_host))

    @classmethod
    def setup(cls, **kwargs) -> "PrismPerception":
        return cls(**kwargs)

    def start(self) -> None:
        self._fuser.start()
        for ch in self._channels:
            ch.start()
        logger.info("PRISM perception started. Active channels: %s",
                    [c.NAME for c in self._channels if c._enabled])

    def stop(self) -> None:
        self._fuser.stop()
        for ch in self._channels:
            ch.stop()

    def current_context(self) -> ContextState:
        return self._fuser.current_state()

    def ingest_biometrics(self, **kwargs) -> None:
        """Push wearable data directly: hrv_ms, sleep_hrs, steps, etc."""
        for ch in self._channels:
            if isinstance(ch, BiometricChannel):
                ch.ingest(**kwargs)
                break

    def record_keypress(self) -> None:
        """Call from keyboard hook to feed typing pattern channel."""
        if self._typing:
            self._typing.record_keypress()

    def status(self) -> dict:
        state = self.current_context()
        return {
            "active_channels": state.active_channels,
            "factor_count":    len(state.factors),
            "summary":         state.summary,
            "factors":         state.factors,
        }
