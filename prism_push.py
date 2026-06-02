from __future__ import annotations
import json, logging, urllib.request, urllib.parse
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PushMessage:
    title:    str
    body:     str
    priority: str = "default"  # "min"|"low"|"default"|"high"|"urgent"
    tags:     list = None      # emoji tags e.g. ["bell","robot"]
    url:      str  = ""        # click action URL

class PrismPush:
    """
    Push notifications to any device via ntfy.sh.

    ntfy.sh is completely free, open source, no account required.
    Works on Android, iOS (via ntfy app), desktop, and browser.

    Setup (2 minutes):
      1. Install ntfy app on your phone (ntfy.sh)
      2. Choose a unique topic name e.g. "prism-yourname-2024"
      3. Subscribe to that topic in the app
      4. Add topic to prism_config.toml

    Config:
      [push]
      topic    = "prism-yourname-2024"  # your unique topic
      server   = "https://ntfy.sh"      # or self-hosted ntfy server
      priority = "default"

    Notification priorities:
      urgent → phone buzzes even in Do Not Disturb (use sparingly)
      high   → standard notification with sound
      default → normal notification
      low    → delivered silently
      min    → no notification sound or banner
    """

    PRIORITY_MAP = {
        "urgent": 5, "high": 4,
        "default": 3, "low": 2, "min": 1,
    }

    def __init__(self, topic="", server="https://ntfy.sh",
                  default_priority="default"):
        self._topic    = topic
        self._server   = server.rstrip("/")
        self._priority = default_priority

    @classmethod
    def from_config(cls, config: dict) -> "PrismPush":
        p = config.get("push", {})
        return cls(
            topic            = p.get("topic", ""),
            server           = p.get("server", "https://ntfy.sh"),
            default_priority = p.get("priority", "default"),
        )

    @property
    def configured(self) -> bool:
        return bool(self._topic)

    def send(self, title: str, body: str,
              priority: str = None,
              tags: list[str] = None,
              url: str = "") -> bool:
        """
        Send a push notification to the user's phone/device.

        Example:
            push.send("Meeting in 15 min", "Budget review at 3pm — room B4",
                       priority="high", tags=["calendar"])
        """
        if not self.configured:
            logger.debug("Push not configured — no topic set")
            return False

        headers = {
            "Title":    title,
            "Priority": str(self.PRIORITY_MAP.get(
                priority or self._priority, 3)),
        }
        if tags:
            headers["Tags"] = ",".join(tags)
        if url:
            headers["Click"] = url

        req = urllib.request.Request(
            f"{self._server}/{urllib.parse.quote(self._topic)}",
            data    = body.encode("utf-8"),
            headers = headers,
            method  = "POST",
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            logger.info("Push sent: %s", title)
            return True
        except Exception as e:
            logger.warning("Push failed: %s", e)
            return False

    def alert(self, message: str) -> bool:
        """Quick single-string alert with default priority."""
        return self.send("PRISM", message)

    def urgent(self, message: str) -> bool:
        """Urgent alert — bypasses Do Not Disturb."""
        return self.send("PRISM — Urgent", message, priority="urgent",
                          tags=["warning"])

    def task_done(self, task_title: str, status: str) -> bool:
        """Notify that a background task completed."""
        icon   = "white_check_mark" if status == "completed" else "x"
        return self.send(
            f"Task {status}: {task_title[:40]}",
            f"Your background task finished with status: {status}",
            priority = "default",
            tags     = [icon],
        )

    def status_summary(self) -> dict:
        return {
            "configured": self.configured,
            "topic":      self._topic,
            "server":     self._server,
        }
