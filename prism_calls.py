from __future__ import annotations
import json, logging, os, subprocess, urllib.request, urllib.parse
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class CallResult:
    success:     bool
    call_sid:    str = ""
    status:      str = ""
    duration_s:  int = 0
    transcript:  str = ""
    error:       str = ""

class PrismCalls:
    """
    Phone call integration.

    Two methods in order of preference:
      1. Twilio — programmable calls from any number, any platform.
         Requires account at twilio.com (free trial available).
         Paid: ~$0.013/min to call UK/US numbers.

      2. macOS Continuity — uses your iPhone via the Mac.
         Free. No API key. Requires: Mac + iPhone on same WiFi,
         Handoff enabled in Settings.

    Config in prism_config.toml:
      [calls]
      provider          = "twilio"   # "twilio" | "macos" | "auto"
      twilio_account_sid = ""
      twilio_auth_token  = ""
      twilio_from_number = ""        # your Twilio number e.g. "+14155551234"
      transcribe        = true       # transcribe with Whisper after call
    """

    def __init__(self, provider="auto", account_sid="", auth_token="",
                  from_number="", transcribe=True):
        self._provider    = provider
        self._sid         = account_sid
        self._token       = auth_token
        self._from        = from_number
        self._transcribe  = transcribe

    @classmethod
    def from_config(cls, config: dict) -> "PrismCalls":
        c = config.get("calls", {})
        return cls(
            provider    = c.get("provider","auto"),
            account_sid = c.get("twilio_account_sid",""),
            auth_token  = c.get("twilio_auth_token",""),
            from_number = c.get("twilio_from_number",""),
            transcribe  = c.get("transcribe", True),
        )

    @property
    def configured(self) -> bool:
        return bool(self._sid and self._token) or self._is_macos()

    def call(self, to_number: str,
              message: str = "",
              twiml: str = None) -> CallResult:
        """
        Place an outgoing call.
        message: text to speak (converted to TwiML automatically)
        twiml:   raw TwiML XML (overrides message)
        """
        method = self._resolve_provider()
        if method == "twilio":
            return self._twilio_call(to_number, message, twiml)
        if method == "macos":
            return self._macos_call(to_number)
        return CallResult(False, error="No call provider configured. "
                          "Add Twilio credentials or use macOS Continuity.")

    def _resolve_provider(self) -> str:
        if self._provider == "twilio" and self._sid:
            return "twilio"
        if self._provider == "macos" and self._is_macos():
            return "macos"
        if self._provider == "auto":
            if self._sid and self._token: return "twilio"
            if self._is_macos():         return "macos"
        return "none"

    def _twilio_call(self, to: str, message: str,
                      twiml: str) -> CallResult:
        if not twiml:
            safe_msg = message or "Hello, this is PRISM calling."
            twiml    = f"<Response><Say>{safe_msg}</Say></Response>"
        url     = (f"https://api.twilio.com/2010-04-01/Accounts/"
                   f"{self._sid}/Calls.json")
        data    = urllib.parse.urlencode({
            "To":   to,
            "From": self._from,
            "Twiml":twiml,
        }).encode()
        import base64
        creds   = base64.b64encode(
            f"{self._sid}:{self._token}".encode()).decode()
        req     = urllib.request.Request(url, data=data,
            headers={"Authorization": f"Basic {creds}"})
        try:
            resp = urllib.request.urlopen(req, timeout=15)
            d    = json.loads(resp.read())
            return CallResult(True, call_sid=d.get("sid",""),
                               status=d.get("status",""))
        except Exception as e:
            logger.warning("Twilio call failed: %s", e)
            return CallResult(False, error=str(e)[:200])

    def _macos_call(self, to: str) -> CallResult:
        """Dial via iPhone Continuity on macOS."""
        if not self._is_macos():
            return CallResult(False, error="macOS Continuity not available")
        script = f'tell application "FaceTime" to call "{to}"'
        result = subprocess.run(["osascript","-e",script],
                                 capture_output=True, text=True)
        return CallResult(
            success = result.returncode == 0,
            status  = "dialing",
            error   = result.stderr[:200] if result.returncode != 0 else ""
        )

    @staticmethod
    def _is_macos() -> bool:
        import sys
        return sys.platform == "darwin"

    def status_summary(self) -> dict:
        return {"configured": self.configured,
                "provider":   self._resolve_provider(),
                "from_number":self._from}
