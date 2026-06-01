"""Tests for prism_email.py — Gap Prompt 10a."""
from prism_email import DraftReply, EmailMessage, PrismEmail


def test_not_configured_empty():
    """PrismEmail() with no args should not be configured."""
    assert PrismEmail().configured is False


def test_configured_when_set():
    """PrismEmail with address + password should report configured."""
    assert PrismEmail("a@b.com", "pass").configured is True


def test_strip_html():
    """_strip_html should remove tags and return plain text."""
    assert PrismEmail._strip_html("<b>bold</b>") == "bold"


def test_status_unconfigured():
    """status_summary() on unconfigured instance returns configured=False."""
    summary = PrismEmail().status_summary()
    assert summary.get("configured") is False


def test_draft_reply_no_llm():
    """draft_reply without an LLM router returns DraftReply with subject starting 'Re:'."""
    pe = PrismEmail()
    original = EmailMessage(
        msg_id="<abc123>",
        subject="Meeting tomorrow",
        sender="boss@example.com",
        to=["me@example.com"],
        date="Mon, 1 Jan 2024 10:00:00 +0000",
        body="Can we meet tomorrow at 3pm?",
    )
    draft = pe.draft_reply(original, "decline politely, suggest next week")
    assert isinstance(draft, DraftReply)
    assert draft.subject.startswith("Re:")


def test_summarise_empty():
    """summarise_inbox with empty list returns 'No unread emails.'"""
    assert PrismEmail().summarise_inbox([]) == "No unread emails."
