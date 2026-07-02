"""
tests/test_conversation_recall_issue_28.py
==========================================
"what did we talk about yesterday" answered from the conversation
store's time index (issue #28-81). Before: routed to memory_recall,
whose similarity search shares no tokens with the stored turns, so it
always answered "No memory of that".
"""
from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta

import pytest

from prism_conversation_recall import _window, recall_card
from prism_intents import INTENTS
from prism_memory import PrismMemory
from prism_routing import route_intent


def _route(message: str) -> str:
    return route_intent(message, INTENTS, lambda m: "LLM_FALLBACK_USED")


class TestRouting:
    @pytest.mark.parametrize("msg", [
        "what did we talk about yesterday",
        "what did we discuss last week",
        "What did we chat about today?",
        "do you remember what we talked about",
        "what were we talking about",
        "summarise our conversation",
        "summarize today's conversation",
        "what was our last conversation about",
    ])
    def test_routes_to_conversation_recall(self, msg):
        assert _route(msg) == "conversation_recall", msg

    @pytest.mark.parametrize("msg,expected", [
        ("what is my favourite colour", "memory_recall"),
        ("do you remember my partner's name", "memory_recall"),
    ])
    def test_fact_recall_unaffected(self, msg, expected):
        assert _route(msg) == expected, msg


class TestWindow:
    def test_yesterday_is_previous_calendar_day(self):
        start, end, label = _window("what did we talk about yesterday")
        assert label == "yesterday"
        midnight = datetime.now().replace(hour=0, minute=0, second=0,
                                          microsecond=0)
        assert end == pytest.approx(midnight.timestamp())
        assert start == pytest.approx(
            (midnight - timedelta(days=1)).timestamp())

    def test_no_qualifier_defaults_to_seven_days(self):
        start, end, label = _window("what did we talk about")
        assert label == "the last 7 days"
        assert end - start >= 7 * 86400

    def test_today(self):
        _, _, label = _window("summarize today's conversation")
        assert label == "today"


def _memory_with_turns(tmp_path, turns):
    """turns: list of (role, content, ts_offset_days_ago)."""
    mem = PrismMemory(db_path=str(tmp_path / "memory.db"))
    for role, content, days_ago in turns:
        entry_id = mem.ingest_conversation(role, content)
        assert entry_id, f"turn too short to store: {content!r}"
        ts = time.time() - days_ago * 86400
        with sqlite3.connect(tmp_path / "memory.db") as c:
            c.execute("UPDATE memory SET timestamp=? WHERE id=?",
                      (ts, entry_id))
    return mem


LONG = " — I want the full detail on this so the turn is stored."


class TestConversationBetween:
    def test_window_and_role_filtering(self, tmp_path):
        mem = _memory_with_turns(tmp_path, [
            ("user",      "we planned the marathon training block" + LONG, 0.1),
            ("assistant", "here is the marathon plan you asked for" + LONG, 0.1),
            ("user",      "we argued about database indexes last month" + LONG, 20),
        ])
        now = time.time()
        recent = mem.conversation_between(now - 86400, now + 1, role="user")
        assert len(recent) == 1
        assert "marathon" in recent[0].content


class TestRecallCard:
    def test_lists_user_topics_for_window(self, tmp_path):
        mem = _memory_with_turns(tmp_path, [
            ("user", "let's design the horizon planner architecture" + LONG, 0.2),
            ("user", "review my portfolio allocation for the year" + LONG, 0.2),
            ("assistant", "assistant reply that must not surface" + LONG, 0.2),
        ])
        card = recall_card(mem, "what did we talk about today")
        assert "horizon planner" in card.body
        assert "portfolio" in card.body
        assert "must not surface" not in card.body
        assert card.card_data["count"] == 2

    def test_empty_window_is_honest(self, tmp_path):
        mem = PrismMemory(db_path=str(tmp_path / "memory.db"))
        card = recall_card(mem, "what did we talk about yesterday")
        assert "yesterday" in card.title.lower()
        assert "don't have any conversation" in card.body

    def test_duplicate_turns_deduped(self, tmp_path):
        mem = _memory_with_turns(tmp_path, [
            ("user", "check the weather in berlin for the trip" + LONG, 0.1),
            ("user", "check the weather in berlin for the trip" + LONG, 0.1),
        ])
        card = recall_card(mem, "what did we talk about today")
        assert card.card_data["count"] == 1
