"""Tests for prism_organ_dispatch — the extracted organ execution gate."""
from __future__ import annotations

from prism_agent import PrismAgent
from prism_organ_dispatch import dispatch_organ


def test_unknown_intent_returns_none():
    agent = PrismAgent()
    assert dispatch_organ(agent, "totally_unknown_intent_zzz", "x", {}) is None


def test_known_offline_organ_executes():
    agent = PrismAgent()
    card = dispatch_organ(agent, "unit_convert", "convert 10 km to miles", {})
    assert card is not None
    assert "mile" in card.body.lower()


def test_approval_gated_organ_returns_approval_card():
    agent = PrismAgent()
    card = dispatch_organ(agent, "email_send",
                          "send an email to alice@example.com saying hi", {})
    assert card is not None
    # email_send is requires_approval → an approval card, not execution
    ctype = getattr(getattr(card, "card_type", None), "value", None) or str(
        getattr(card, "card_type", ""))
    assert "approval" in ctype.lower() or "approval" in (card.title or "").lower()


def test_approved_flag_bypasses_gate():
    agent = PrismAgent()
    # With the approval flag set, the gate proceeds (organ may then fail for
    # lack of config, but it must NOT return an approval card).
    card = dispatch_organ(agent, "email_send", "send email",
                          {"_approved_email_send": True})
    assert card is not None
    assert "approval" not in (card.title or "").lower()
