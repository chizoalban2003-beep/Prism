from __future__ import annotations

import pytest

from prism_instructions import Instruction, PrismInstructions


@pytest.fixture
def instr(tmp_path):
    return PrismInstructions(db_path=str(tmp_path / "instructions.db"))


def test_add_returns_instruction(instr):
    result = instr.add("always confirm before deleting files")
    assert isinstance(result, Instruction)
    assert result.text == "always confirm before deleting files"


def test_relevant_email(instr):
    instr.add("never reply to emails after 6pm", trigger="email")
    results = instr.relevant_for("check inbox for new messages")
    assert any("never reply" in i.text for i in results)


def test_always_returned(instr):
    instr.add("ask for confirmation before any action", trigger="always")
    results = instr.relevant_for("what time is my next meeting")
    assert any("confirmation" in i.text for i in results)


def test_parse_remember(instr):
    result = instr.parse_from_chat("remember: never schedule before 9am")
    assert isinstance(result, Instruction)
    assert "never schedule before 9am" in result.text


def test_parse_ignores_normal(instr):
    result = instr.parse_from_chat("what time is the meeting")
    assert result is None


def test_remove_deletes(instr):
    added = instr.add("temporary rule for testing")
    assert instr.remove(added.instr_id)
    assert all(i.instr_id != added.instr_id for i in instr.all_active())


# ── parse_fact: personal-fact extraction ──────────────────────────────────
# Issue #26 bug 3: "remember that my favourite colour is blue" was stored
# verbatim in the standing-rule store, so "what is my favourite colour"
# couldn't retrieve the value. parse_fact peels off the imperative and
# returns the (key, value) pair so the chat prelude can route it into
# PrismMemory instead.

def test_parse_fact_basic():
    out = PrismInstructions.parse_fact("remember that my favourite colour is blue")
    assert out == ("favourite colour", "blue")


def test_parse_fact_without_remember_prefix():
    out = PrismInstructions.parse_fact("my partner's name is Sam")
    assert out == ("partner's name", "Sam")


def test_parse_fact_handles_trailing_punctuation():
    out = PrismInstructions.parse_fact("remember my birthday is 4 March.")
    assert out == ("birthday", "4 March")


def test_parse_fact_rejects_non_fact():
    assert PrismInstructions.parse_fact("what time is my next meeting") is None
    assert PrismInstructions.parse_fact("always confirm before deleting files") is None
    assert PrismInstructions.parse_fact("") is None


def test_parse_from_chat_skips_fact(instr):
    """Personal facts must NOT land in the standing-rule store — the chat
    prelude routes them into PrismMemory instead."""
    assert instr.parse_from_chat("remember that my favourite colour is blue") is None
    assert all("favourite colour" not in i.text for i in instr.all_active())


# ── parse_from_chat: imperative prefix stripping ─────────────────────────
# Issue #26 bug 3 part 2: stored rule text echoed the imperative ("remember
# that ..."), so when the LLM was asked to apply it the meta-command leaked
# into responses. Strip pure markers; preserve always/never/whenever which
# carry the rule's quantifier.

def test_parse_strips_remember_that(instr):
    rule = instr.parse_from_chat("remember that I prefer short emails")
    assert rule is not None
    assert not rule.text.lower().startswith("remember"), rule.text


def test_parse_strips_from_now_on(instr):
    rule = instr.parse_from_chat("from now on, use metric units")
    assert rule is not None
    assert not rule.text.lower().startswith("from now on"), rule.text


def test_parse_preserves_never(instr):
    """`never` carries the rule's quantifier — stripping it would change
    meaning ('never log in' vs 'log in')."""
    rule = instr.parse_from_chat("never schedule meetings before 9am")
    assert rule is not None
    assert rule.text.lower().startswith("never"), rule.text


def test_parse_preserves_always(instr):
    rule = instr.parse_from_chat("always confirm before sending")
    assert rule is not None
    assert rule.text.lower().startswith("always"), rule.text
