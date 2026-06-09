"""
prism_messaging_gateway.py
==========================
Bidirectional messaging gateway — gives PRISM a platform presence beyond
the local machine (Telegram live, WhatsApp via Twilio stub).

Usage:
    from prism_messaging_gateway import start_all_gateways, gateway_registry
    await start_all_gateways(config)
"""
from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

@dataclass
class MessageEnvelope:
    platform:  str
    sender_id: str
    chat_id:   str
    text:      str
    timestamp: float
    reply_fn:  Callable[[str], Awaitable[None]]


class PlatformGateway(ABC):
    """Abstract base for all messaging platform adapters."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    # Sub-classes may override this; default tracks running state.
    running: bool = field(default=False)  # type: ignore[assignment]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "running"):
            cls.running = False


# ---------------------------------------------------------------------------
# Shared dispatcher — POST to local PRISM /chat endpoint
# ---------------------------------------------------------------------------

async def _dispatch(envelope: MessageEnvelope) -> str:
    """POST envelope text to PRISM /chat and return the response string."""
    payload = json.dumps({
        "message":    envelope.text,
        "session_id": f"{envelope.platform}_{envelope.chat_id}",
    }).encode()
    try:
        req = urllib.request.Request(
            "http://localhost:8742/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(
            None,
            lambda: urllib.request.urlopen(req, timeout=30).read(),
        )
        return json.loads(raw).get("response", "")
    except Exception as exc:
        logger.error("Gateway dispatch error: %s", exc)
        return f"[PRISM unavailable: {exc}]"


# ---------------------------------------------------------------------------
# Telegram gateway
# ---------------------------------------------------------------------------

class TelegramGateway(PlatformGateway):
    """Live Telegram bot via python-telegram-bot v21+."""

    def __init__(self, token: str) -> None:
        self._token = token
        self._app: object = None
        self.running = False

    @property
    def name(self) -> str:
        return "telegram"

    async def start(self) -> None:
        try:
            from telegram.ext import Application, MessageHandler, filters
        except ImportError:
            logger.warning(
                "python-telegram-bot not installed — TelegramGateway disabled. "
                "Install with: pip install 'python-telegram-bot>=21.0,<22.0'"
            )
            return

        try:
            app = Application.builder().token(self._token).build()
            app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle))
            self._app = app
            await app.initialize()
            await app.start()
            await app.updater.start_polling()
            self.running = True
            logger.info("TelegramGateway started")
        except Exception as exc:
            logger.error("TelegramGateway failed to start: %s", exc)

    async def stop(self) -> None:
        if self._app is None or not self.running:
            return
        try:
            app = self._app
            await app.updater.stop()
            await app.stop()
            await app.shutdown()
            self.running = False
            logger.info("TelegramGateway stopped")
        except Exception as exc:
            logger.error("TelegramGateway stop error: %s", exc)

    async def _handle(self, update: object, context: object) -> None:
        try:
            msg = update.message  # type: ignore[attr-defined]

            async def reply(text: str) -> None:
                await msg.reply_text(text)

            envelope = MessageEnvelope(
                platform="telegram",
                sender_id=str(msg.from_user.id),
                chat_id=str(msg.chat_id),
                text=msg.text or "",
                timestamp=msg.date.timestamp(),
                reply_fn=reply,
            )
            response = await _dispatch(envelope)
            await envelope.reply_fn(response)
        except Exception as exc:
            logger.error("TelegramGateway message handler error: %s", exc)


# ---------------------------------------------------------------------------
# WhatsApp gateway (Twilio Conversations webhook stub)
# ---------------------------------------------------------------------------

class WhatsAppGateway(PlatformGateway):
    """
    WhatsApp via Twilio Conversations webhook.
    Call receive(body) from your POST /integrations/messaging/webhook/whatsapp
    handler to process an inbound Twilio webhook payload.
    """

    def __init__(self, account_sid: str, auth_token: str, from_number: str) -> None:
        self._sid       = account_sid
        self._token     = auth_token
        self._from      = from_number
        self.running    = False

    @property
    def name(self) -> str:
        return "whatsapp"

    async def start(self) -> None:
        self.running = True
        logger.info("WhatsAppGateway started (webhook mode — awaiting inbound requests)")

    async def stop(self) -> None:
        self.running = False
        logger.info("WhatsAppGateway stopped")

    def receive(self, body: dict) -> MessageEnvelope:
        """Construct a MessageEnvelope from a Twilio inbound webhook dict."""
        sender  = body.get("From", "")
        chat_id = body.get("From", sender).replace("whatsapp:", "")
        text    = body.get("Body", "")

        async def reply(text_out: str) -> None:
            await self._send_whatsapp(sender, text_out)

        import time
        return MessageEnvelope(
            platform="whatsapp",
            sender_id=sender,
            chat_id=chat_id,
            text=text,
            timestamp=time.time(),
            reply_fn=reply,
        )

    async def _send_whatsapp(self, to: str, text: str) -> None:
        """Send a WhatsApp message via Twilio Messages REST API."""
        import base64
        import urllib.parse as _up

        url = (
            f"https://api.twilio.com/2010-04-01/Accounts/{self._sid}/Messages.json"
        )
        data = _up.urlencode({
            "From": f"whatsapp:{self._from}",
            "To":   to if to.startswith("whatsapp:") else f"whatsapp:{to}",
            "Body": text,
        }).encode()
        credentials = base64.b64encode(
            f"{self._sid}:{self._token}".encode()
        ).decode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type":  "application/x-www-form-urlencoded",
            },
            method="POST",
        )
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: urllib.request.urlopen(req, timeout=15).read(),
            )
        except Exception as exc:
            logger.error("WhatsAppGateway send error: %s", exc)


# ---------------------------------------------------------------------------
# Registry + bootstrap
# ---------------------------------------------------------------------------

gateway_registry: dict[str, PlatformGateway] = {}


async def start_all_gateways(config: dict) -> None:
    """Instantiate and start all gateways whose credentials are present in config."""
    token = config.get("telegram_token", "")
    if token:
        gw = TelegramGateway(token)
        gateway_registry["telegram"] = gw
        await gw.start()

    sid   = config.get("twilio_account_sid", "")
    atoken = config.get("twilio_auth_token", "")
    from_n = config.get("twilio_whatsapp_number", "")
    if sid and atoken and from_n:
        gw2 = WhatsAppGateway(sid, atoken, from_n)
        gateway_registry["whatsapp"] = gw2
        await gw2.start()

    if not gateway_registry:
        logger.info("No messaging gateways configured — skipping startup")


async def stop_all_gateways() -> None:
    """Stop all running gateways."""
    for gw in list(gateway_registry.values()):
        try:
            await gw.stop()
        except Exception as exc:
            logger.error("Error stopping gateway %s: %s", gw.name, exc)
    gateway_registry.clear()
