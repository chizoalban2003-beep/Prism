"""
prism_local_notify.py
=====================
Credential-free local notification delivery.

PRISM's outbound alerting (phone/SMS/email) all needs third-party accounts.
This module is the fallback that always works with no credentials at all:

1. If a desktop notifier (``notify-send`` / ``kdialog`` / ``zenity``) is
   installed AND a display is present, fire a native popup.
2. Regardless, append the notification to ``~/.prism/notifications.jsonl`` — a
   durable local inbox the UI and proactive layer can surface. This guarantees
   delivery even on a headless box where no popup is possible.

Not an organ (so it may touch the filesystem/subprocess directly); the
``notify_desktop`` organ and the ``phone_call`` degrade path both call it.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_INBOX = Path("~/.prism/notifications.jsonl").expanduser()

# notifier → argv template builder (title, body) -> argv
_NOTIFIERS = ("notify-send", "kdialog", "zenity")


def _has_display() -> bool:
    import os
    import sys
    if sys.platform in ("darwin", "win32"):
        return True
    return bool(os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY"))


def _notifier_argv(tool: str, title: str, body: str, urgency: str) -> Optional[list[str]]:
    if tool == "notify-send":
        return ["notify-send", "-u", urgency, title, body]
    if tool == "kdialog":
        return ["kdialog", "--title", title, "--passivepopup", body, "5"]
    if tool == "zenity":
        return ["zenity", "--notification", "--text", f"{title}\n{body}"]
    return None


def _fire_popup(title: str, body: str, urgency: str) -> Optional[str]:
    """Try the first available desktop notifier. Return its name on success,
    None if none is available or the call fails."""
    if not _has_display():
        return None
    for tool in _NOTIFIERS:
        if not shutil.which(tool):
            continue
        argv = _notifier_argv(tool, title, body, urgency)
        if not argv:
            continue
        try:
            proc = subprocess.run(argv, capture_output=True, text=True,
                                  timeout=5)  # nosec B603 — argv list, shell=False
            if proc.returncode == 0:
                return tool
        except Exception:
            continue
    return None


def _log_inbox(record: dict) -> bool:
    try:
        _INBOX.parent.mkdir(parents=True, exist_ok=True)
        with _INBOX.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return True
    except Exception as exc:
        logger.warning("[local_notify] inbox write failed: %s", exc)
        return False


def deliver(title: str, body: str, urgency: str = "normal",
            source: str = "prism") -> dict:
    """Deliver a notification credential-free. Always logs to the local inbox;
    additionally fires a native popup when possible. Returns a report dict:
    {popup: <tool|None>, logged: bool, ts: float}."""
    title = (title or "PRISM").strip()
    body = (body or "").strip()
    ts = time.time()
    popup = _fire_popup(title, body, urgency)
    logged = _log_inbox({
        "ts": ts, "title": title, "body": body,
        "urgency": urgency, "source": source, "popup": popup,
    })
    return {"popup": popup, "logged": logged, "ts": ts}


def recent(limit: int = 10) -> list[dict]:
    """Return the most recent notifications from the local inbox (newest last)."""
    if not _INBOX.exists():
        return []
    try:
        lines = _INBOX.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    out = []
    for ln in lines[-limit:]:
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out
