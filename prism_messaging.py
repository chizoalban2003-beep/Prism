from __future__ import annotations
import json, logging, re, subprocess, urllib.parse, urllib.request
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class Message:
    msg_id:   str
    platform: str
    sender:   str
    content:  str
    timestamp:str = ""
    chat_id:  str = ""

class PrismMessaging:
    """
    Multi-platform messaging.

    Telegram:  Easy, free, official Bot API.
               1. Create a bot at t.me/BotFather
               2. Get the token
               3. Start a chat with your bot to get your chat_id

    iMessage:  macOS only. No API key. Uses osascript.
               Works when iMessage is configured on this Mac.

    WhatsApp:  Requires Twilio WhatsApp Sandbox (free for testing)
               or Twilio WhatsApp Business (paid, requires approval).

    Config in prism_config.toml:
      [messaging]
      telegram_token    = ""      # from @BotFather
      telegram_chat_id  = ""      # your personal chat ID
      twilio_whatsapp_sid   = ""  # same as calls if configured
      twilio_whatsapp_token = ""
      twilio_whatsapp_from  = "whatsapp:+14155238886"  # Twilio sandbox number
    """

    def __init__(self, telegram_token="", telegram_chat_id="",
                  wa_sid="", wa_token="", wa_from=""):
        self._tg_token  = telegram_token
        self._tg_chat   = telegram_chat_id
        self._wa_sid    = wa_sid
        self._wa_token  = wa_token
        self._wa_from   = wa_from

    @classmethod
    def from_config(cls, config: dict) -> "PrismMessaging":
        m = config.get("messaging", {})
        return cls(
            telegram_token   = m.get("telegram_token",""),
            telegram_chat_id = m.get("telegram_chat_id",""),
            wa_sid           = m.get("twilio_whatsapp_sid",""),
            wa_token         = m.get("twilio_whatsapp_token",""),
            wa_from          = m.get("twilio_whatsapp_from",""),
        )

    @property
    def configured_platforms(self) -> list[str]:
        platforms = []
        if self._tg_token:  platforms.append("telegram")
        if self._is_macos():platforms.append("imessage")
        if self._wa_sid:    platforms.append("whatsapp")
        return platforms

    # ── Send ─────────────────────────────────────────────────────────────

    def send(self, platform: str, recipient: str,
              message: str) -> bool:
        """
        Send a message on the specified platform.
        recipient: Telegram chat_id, phone number for iMessage/WhatsApp
        """
        if platform == "telegram":
            return self._telegram_send(
                recipient or self._tg_chat, message)
        if platform == "imessage":
            return self._imessage_send(recipient, message)
        if platform == "whatsapp":
            return self._whatsapp_send(recipient, message)
        logger.warning("Unknown platform: %s", platform)
        return False

    def send_to_self(self, message: str,
                      platform: str = None) -> bool:
        """Send a notification to the user on their preferred platform."""
        target = platform or (self.configured_platforms[0]
                               if self.configured_platforms else None)
        if not target:
            return False
        if target == "telegram":
            return self._telegram_send(self._tg_chat, message)
        return False

    # ── Receive ──────────────────────────────────────────────────────────

    def get_updates(self, platform: str = "telegram",
                     n: int = 10) -> list[Message]:
        """Get recent incoming messages."""
        if platform == "telegram":
            return self._telegram_updates(n)
        return []

    # ── Telegram ─────────────────────────────────────────────────────────

    def _telegram_send(self, chat_id: str, text: str) -> bool:
        if not self._tg_token or not chat_id:
            return False
        url     = (f"https://api.telegram.org/bot{self._tg_token}"
                   f"/sendMessage")
        payload = json.dumps({"chat_id": chat_id,
                               "text": text[:4096]}).encode()
        req     = urllib.request.Request(url, data=payload,
            headers={"Content-Type":"application/json"})
        try:
            urllib.request.urlopen(req, timeout=10)
            return True
        except Exception as e:
            logger.warning("Telegram send failed: %s", e)
            return False

    def _telegram_updates(self, n: int) -> list[Message]:
        if not self._tg_token:
            return []
        url = (f"https://api.telegram.org/bot{self._tg_token}"
               f"/getUpdates?limit={n}")
        try:
            resp = urllib.request.urlopen(url, timeout=10)
            data = json.loads(resp.read())
            msgs = []
            for u in data.get("result",[]):
                msg = u.get("message",{})
                if msg:
                    msgs.append(Message(
                        msg_id  = str(u.get("update_id","")),
                        platform= "telegram",
                        sender  = str(msg.get("from",{}).get("id","")),
                        content = msg.get("text",""),
                        chat_id = str(msg.get("chat",{}).get("id","")),
                    ))
            return msgs
        except Exception as e:
            logger.debug("Telegram updates failed: %s", e)
            return []

    # ── iMessage (macOS only) ─────────────────────────────────────────────

    def _imessage_send(self, to: str, message: str) -> bool:
        if not self._is_macos():
            return False
        safe_to  = re.sub(r'[^\d\+\-\s@\.]', '', to)
        safe_msg = message.replace('"', '\\"').replace("'", "\\'")
        script   = (f'tell application "Messages" to send '
                    f'"{safe_msg}" to buddy "{safe_to}" '
                    f'of (1st service whose service type is iMessage)')
        result   = subprocess.run(["osascript","-e",script],
                                   capture_output=True, text=True)
        return result.returncode == 0

    # ── WhatsApp via Twilio ───────────────────────────────────────────────

    def _whatsapp_send(self, to: str, message: str) -> bool:
        if not self._wa_sid:
            return False
        import base64
        url   = (f"https://api.twilio.com/2010-04-01/Accounts/"
                 f"{self._wa_sid}/Messages.json")
        to_wa = f"whatsapp:{to}" if not to.startswith("whatsapp:") else to
        data  = urllib.parse.urlencode({
            "From": self._wa_from, "To": to_wa, "Body": message
        }).encode()
        creds = base64.b64encode(
            f"{self._wa_sid}:{self._wa_token}".encode()).decode()
        req   = urllib.request.Request(url, data=data,
            headers={"Authorization": f"Basic {creds}"})
        try:
            urllib.request.urlopen(req, timeout=10)
            return True
        except Exception as e:
            logger.warning("WhatsApp send failed: %s", e)
            return False

    @staticmethod
    def _is_macos() -> bool:
        import sys
        return sys.platform == "darwin"

    def status_summary(self) -> dict:
        return {"configured_platforms": self.configured_platforms}
