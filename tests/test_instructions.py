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
