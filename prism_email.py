from __future__ import annotations

import email as _email
import imaplib
import json
import logging
import smtplib
import time
from dataclasses import dataclass, field
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EmailMessage:
    msg_id:   str
    subject:  str
    sender:   str
    to:       list[str]
    date:     str
    body:     str
    thread_id:str = ""
    unread:   bool = True
    labels:   list[str] = field(default_factory=list)


@dataclass
class DraftReply:
    to:      str
    subject: str
    body:    str
    in_reply_to: str = ""


class PrismEmail:
    """
    Email integration via IMAP (read) and SMTP (send).
    Supports Gmail OAuth and standard IMAP/SMTP.

    Config in prism_config.toml:
      [email]
      provider    = "gmail"       # "gmail" | "imap"
      address     = "you@gmail.com"
      imap_host   = "imap.gmail.com"
      imap_port   = 993
      smtp_host   = "smtp.gmail.com"
      smtp_port   = 587
      # For Gmail: generate an App Password at myaccount.google.com/apppasswords
      password    = "your-app-password"
      max_fetch   = 20            # emails to fetch per sync
    """

    def __init__(
        self,
        address:   str = "",
        password:  str = "",
        imap_host: str = "imap.gmail.com",
        imap_port: int = 993,
        smtp_host: str = "smtp.gmail.com",
        smtp_port: int = 587,
        max_fetch: int = 20,
    ):
        self._address   = address
        self._password  = password
        self._imap_host = imap_host
        self._imap_port = imap_port
        self._smtp_host = smtp_host
        self._smtp_port = smtp_port
        self._max_fetch = max_fetch

    @classmethod
    def from_config(cls, config: dict) -> "PrismEmail":
        em = config.get("email", {})
        return cls(
            address   = em.get("address", ""),
            password  = em.get("password", ""),
            imap_host = em.get("imap_host", "imap.gmail.com"),
            imap_port = int(em.get("imap_port", 993)),
            smtp_host = em.get("smtp_host", "smtp.gmail.com"),
            smtp_port = int(em.get("smtp_port", 587)),
            max_fetch = int(em.get("max_fetch", 20)),
        )

    @property
    def configured(self) -> bool:
        return bool(self._address and self._password)

    # ── Reading ────────────────────────────────────────────────────────────

    def fetch_unread(self, folder: str = "INBOX",
                     n: int = None) -> list[EmailMessage]:
        """Fetch unread emails from the specified folder."""
        if not self.configured:
            return []
        n = n or self._max_fetch
        try:
            with imaplib.IMAP4_SSL(self._imap_host, self._imap_port) as imap:
                imap.login(self._address, self._password)
                imap.select(folder)
                _, data = imap.search(None, "UNSEEN")
                ids = data[0].split()[-n:]
                messages = []
                for uid in reversed(ids):
                    msg = self._fetch_one(imap, uid)
                    if msg:
                        messages.append(msg)
                return messages
        except Exception as e:
            logger.warning("Email fetch failed: %s", e)
            return []

    def fetch_recent(self, n: int = None) -> list[EmailMessage]:
        """Fetch most recent N emails regardless of read status."""
        if not self.configured:
            return []
        n = n or self._max_fetch
        try:
            with imaplib.IMAP4_SSL(self._imap_host, self._imap_port) as imap:
                imap.login(self._address, self._password)
                imap.select("INBOX")
                _, data = imap.search(None, "ALL")
                ids = data[0].split()[-n:]
                messages = []
                for uid in reversed(ids):
                    msg = self._fetch_one(imap, uid)
                    if msg:
                        messages.append(msg)
                return messages
        except Exception as e:
            logger.warning("Email fetch failed: %s", e)
            return []

    def _fetch_one(self, imap, uid: bytes) -> Optional[EmailMessage]:
        try:
            _, msg_data = imap.fetch(uid, "(RFC822)")
            raw = msg_data[0][1]
            msg = _email.message_from_bytes(raw)
            body = self._extract_body(msg)
            return EmailMessage(
                msg_id  = msg.get("Message-ID", uid.decode()),
                subject = msg.get("Subject", "(no subject)"),
                sender  = msg.get("From", ""),
                to      = [msg.get("To", "")],
                date    = msg.get("Date", ""),
                body    = body[:3000],
                unread  = True,
            )
        except Exception:
            return None

    def _extract_body(self, msg) -> str:
        if msg.is_multipart():
            for part in msg.walk():
                ct = part.get_content_type()
                if ct == "text/plain":
                    try:
                        return part.get_payload(decode=True).decode(
                            errors="replace")
                    except Exception:
                        pass
                if ct == "text/html":
                    try:
                        html = part.get_payload(decode=True).decode(
                            errors="replace")
                        return self._strip_html(html)
                    except Exception:
                        pass
        else:
            try:
                return msg.get_payload(decode=True).decode(errors="replace")
            except Exception:
                pass
        return ""

    @staticmethod
    def _strip_html(html: str) -> str:
        import re
        text = re.sub(r'<[^>]+>', ' ', html)
        return re.sub(r'\s+', ' ', text).strip()

    # ── Sending ─────────────────────────────────────────────────────────────

    def send(self, to: str, subject: str, body: str,
              reply_to: str = "") -> bool:
        """Send an email via SMTP."""
        if not self.configured:
            return False
        try:
            msg = MIMEMultipart()
            msg["From"]    = self._address
            msg["To"]      = to
            msg["Subject"] = subject
            if reply_to:
                msg["In-Reply-To"] = reply_to
                msg["References"]  = reply_to
            msg.attach(MIMEText(body, "plain"))
            with smtplib.SMTP(self._smtp_host, self._smtp_port) as smtp:
                smtp.starttls()
                smtp.login(self._address, self._password)
                smtp.send_message(msg)
            logger.info("Email sent to %s: %s", to, subject)
            return True
        except Exception as e:
            logger.warning("Email send failed: %s", e)
            return False

    # ── LLM-assisted drafting ───────────────────────────────────────────────

    def draft_reply(self, original: EmailMessage,
                     instruction: str, llm_router=None) -> DraftReply:
        """
        Use LLM to draft a reply to an email given a brief instruction.
        Example instruction: "decline politely, suggest next week instead"
        """
        if llm_router is None:
            body = f"Re: {original.subject}\n\n[Draft response per: {instruction}]"
        else:
            prompt = (
                f"Draft a professional email reply.\n\n"
                f"Original email from {original.sender}:\n"
                f"Subject: {original.subject}\n"
                f"Body: {original.body[:1000]}\n\n"
                f"Instruction for reply: {instruction}\n\n"
                f"Write only the reply body text, no greeting or sign-off needed."
            )
            body, _ = llm_router.call(prompt, min_capability=1, max_tokens=500)

        return DraftReply(
            to          = original.sender,
            subject     = f"Re: {original.subject}",
            body        = body,
            in_reply_to = original.msg_id,
        )

    def summarise_inbox(self, messages: list[EmailMessage],
                         llm_router=None) -> str:
        """Return a plain-English summary of unread emails."""
        if not messages:
            return "No unread emails."
        if llm_router is None:
            lines = [f"• {m.sender}: {m.subject}" for m in messages[:10]]
            return f"{len(messages)} unread emails:\n" + "\n".join(lines)
        content = "\n".join(
            f"From: {m.sender}\nSubject: {m.subject}\n{m.body[:300]}\n---"
            for m in messages[:5])
        prompt  = (f"Summarise these {len(messages)} unread emails in 3-4 sentences. "
                   f"Note any urgent items.\n\n{content}")
        summary, _ = llm_router.call(prompt, min_capability=1, max_tokens=300)
        return summary or f"{len(messages)} unread emails."

    def status_summary(self) -> dict:
        if not self.configured:
            return {"configured": False,
                    "message": "Add email config to prism_config.toml"}
        return {"configured": True, "address": self._address}
