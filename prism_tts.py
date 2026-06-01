from __future__ import annotations
import logging, platform, subprocess, threading
from typing import Optional

logger = logging.getLogger(__name__)

class PrismTTS:
    """
    Local text-to-speech using the best available engine:
      macOS:   system 'say' command (always available)
      Linux:   espeak-ng or festival
      Windows: pyttsx3 or PowerShell SAPI
      Any:     pyttsx3 pip package

    Usage:
        tts = PrismTTS.setup()
        tts.speak("Your plan of action is ready.")
        # Non-blocking by default — speaks in background thread
    """

    def __init__(self, voice: str = "", rate: int = 180, enabled: bool = False):
        self._voice   = voice
        self._rate    = rate
        self._enabled = enabled
        self._engine  = self._detect_engine()
        self._lock    = threading.Lock()

    @classmethod
    def setup(cls, **kwargs) -> "PrismTTS":
        return cls(**kwargs)

    def speak(self, text: str, blocking: bool = False) -> None:
        """Speak text. Non-blocking by default."""
        if not self._enabled or not self._engine:
            return
        clean = self._strip_markdown(text)
        if not clean:
            return
        if blocking:
            self._speak_sync(clean)
        else:
            t = threading.Thread(target=self._speak_sync,
                                  args=(clean,), daemon=True)
            t.start()

    def toggle(self) -> bool:
        self._enabled = not self._enabled
        return self._enabled

    @property
    def available(self) -> bool:
        return bool(self._engine)

    def _detect_engine(self) -> str:
        import shutil
        if platform.system() == "Darwin" and shutil.which("say"):
            return "say"
        if shutil.which("espeak-ng"):
            return "espeak-ng"
        if shutil.which("festival"):
            return "festival"
        try:
            import pyttsx3  # noqa
            return "pyttsx3"
        except ImportError:
            pass
        if platform.system() == "Windows":
            return "powershell"
        return ""

    def _speak_sync(self, text: str) -> None:
        with self._lock:
            try:
                if self._engine == "say":
                    args = ["say"]
                    if self._voice: args += ["-v", self._voice]
                    if self._rate:  args += ["-r", str(self._rate)]
                    args.append(text[:500])
                    subprocess.run(args, timeout=30, check=False)

                elif self._engine == "espeak-ng":
                    args = ["espeak-ng", "-s", str(self._rate)]
                    if self._voice: args += ["-v", self._voice]
                    args.append(text[:500])
                    subprocess.run(args, timeout=30, check=False)

                elif self._engine == "pyttsx3":
                    import pyttsx3
                    eng = pyttsx3.init()
                    eng.setProperty("rate", self._rate)
                    if self._voice:
                        eng.setProperty("voice", self._voice)
                    eng.say(text[:500])
                    eng.runAndWait()

                elif self._engine == "powershell":
                    safe = text[:500].replace("'","")
                    subprocess.run(
                        ["powershell","-Command",
                         f"Add-Type -AssemblyName System.speech;"
                         f"(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
                         f".Speak('{safe}')"],
                        timeout=30, check=False)
            except Exception as e:
                logger.debug("TTS error: %s", e)

    @staticmethod
    def _strip_markdown(text: str) -> str:
        import re
        text = re.sub(r'\*+|_+|`+|#{1,6}\s', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        text = re.sub(r'<[^>]+>', '', text)
        return text.strip()
