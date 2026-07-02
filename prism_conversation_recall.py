"""
prism_conversation_recall.py
============================
Time-windowed conversation recall — "what did we talk about yesterday".

Fact recall (memory_recall) answers "what is my favourite colour" by
similarity search; it cannot answer "what did we discuss last week"
because the query shares no tokens with the stored turns. This handler
parses the timeframe from the question, pulls the user's side of the
conversation from PrismMemory's time index, and lists the topics
deterministically — no LLM call, so it works identically on tinyllama
and Claude.

Only turns longer than 50 chars are ever ingested (see
PrismMemory.ingest_conversation), so short pleasantries won't appear;
the empty-window card says so honestly.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from prism_responses import PrismCard, text_card

logger = logging.getLogger(__name__)

_MAX_TOPICS = 8
_SNIPPET_LEN = 110


def _window(message: str) -> tuple[float, float, str]:
    """Map timeframe phrasing to (start_ts, end_ts, human label)."""
    m = (message or "").lower()
    now = datetime.now()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)

    if "yesterday" in m:
        start = midnight - timedelta(days=1)
        return start.timestamp(), midnight.timestamp(), "yesterday"
    if "today" in m or "this morning" in m or "this afternoon" in m \
            or "this evening" in m or "tonight" in m:
        return midnight.timestamp(), now.timestamp() + 1, "today"
    if "last week" in m:
        this_monday = midnight - timedelta(days=midnight.weekday())
        start = this_monday - timedelta(weeks=1)
        return start.timestamp(), this_monday.timestamp(), "last week"
    if "this week" in m:
        start = midnight - timedelta(days=midnight.weekday())
        return start.timestamp(), now.timestamp() + 1, "this week"
    if "last month" in m or "past month" in m:
        start = midnight - timedelta(days=30)
        return start.timestamp(), now.timestamp() + 1, "the last month"
    # No qualifier ("what did we talk about?") — the last 7 days.
    start = midnight - timedelta(days=7)
    return start.timestamp(), now.timestamp() + 1, "the last 7 days"


def _snippet(text: str) -> str:
    text = " ".join((text or "").split())
    if len(text) > _SNIPPET_LEN:
        return text[:_SNIPPET_LEN - 1].rstrip() + "…"
    return text


def recall_card(memory: Any, message: str) -> PrismCard:
    start_ts, end_ts, label = _window(message)
    try:
        turns = memory.conversation_between(start_ts, end_ts,
                                            role="user", limit=200)
    except Exception as exc:
        logger.warning("[conversation_recall] lookup failed: %s", exc)
        return text_card(f"Conversation lookup failed: {exc}", "Memory")

    if not turns:
        return text_card(
            f"I don't have any conversation recorded from {label}. "
            "I only keep substantial messages (short one-liners aren't "
            "stored), so a quiet day can legitimately come up empty.",
            f"Nothing from {label}",
        )

    # Dedupe near-identical turns (repeated probes, retries) by content.
    seen: set[str] = set()
    lines: list[str] = []
    for t in turns:
        key = " ".join(t.content.lower().split())[:80]
        if key in seen:
            continue
        seen.add(key)
        stamp = datetime.fromtimestamp(t.timestamp).strftime("%a %H:%M")
        lines.append(f"- {stamp} — {_snippet(t.content)}")
    total = len(lines)
    shown = lines[:_MAX_TOPICS]

    body = "\n".join(shown)
    if total > _MAX_TOPICS:
        body += f"\n… and {total - _MAX_TOPICS} more."
    card = text_card(body, f"We talked about — {label}")
    card.card_data.update({
        "window":  label,
        "count":   total,
        "from_ts": start_ts,
        "to_ts":   end_ts,
    })
    return card
