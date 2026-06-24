"""Instruction-vs-note prefix fix for issue #28 bug 11 — "note:" stored as a rule.

Live test: sending ``note: testing the note path`` returned ``Instruction
stored: ✓ Remembered: testing the note path`` — the standing-instruction
parser intercepted it before the intent router could route it to
note_append.

Cause: ``PrismInstructions.parse_from_chat`` listed ``"note:"`` in its
prefix detection set. That set is checked in ``PrismAgent.chat`` before
intent routing, so ``"note: X"`` was always rerouted away from the
PrismNotes store.

Fix: drop ``"note:"`` from the prefix list. ``parse_from_chat`` now
returns ``None`` for note-style messages, letting the note_append intent
take over. Other instruction prefixes (always, never, remember,
rule:, etc.) still work.
"""
from __future__ import annotations

import pytest

from prism_instructions import PrismInstructions


@pytest.fixture
def instr(tmp_path):
    return PrismInstructions(db_path=str(tmp_path / "instructions.db"))


class TestNotePrefixNoLongerStoresInstruction:
    def test_note_prefix_returns_none(self, instr):
        assert instr.parse_from_chat("note: testing the note path") is None

    def test_capitalised_note_prefix_returns_none(self, instr):
        assert instr.parse_from_chat("Note: buy milk on the way home") is None

    def test_note_without_colon_returns_none(self, instr):
        # Bare "note" without the colon was never a prefix match — confirm
        # we didn't accidentally widen detection while fixing the colon case.
        assert instr.parse_from_chat("note buy milk") is None


class TestOtherPrefixesStillWork:
    def test_always_prefix_still_detected(self, instr):
        result = instr.parse_from_chat("always confirm before deleting files")
        assert result is not None
        assert "confirm" in result.text

    def test_never_prefix_still_detected(self, instr):
        result = instr.parse_from_chat("never use Uber")
        assert result is not None

    def test_remember_prefix_still_detected(self, instr):
        result = instr.parse_from_chat("remember: meetings start on time")
        assert result is not None

    def test_rule_prefix_still_detected(self, instr):
        # "rule:" was kept — confirm the regression doesn't widen.
        result = instr.parse_from_chat("rule: prefer dark mode")
        assert result is not None

    def test_personal_fact_still_deferred_to_memory(self, instr):
        # "remember that my X is Y" must still defer to the fact store.
        assert instr.parse_from_chat(
            "remember that my favourite colour is teal"
        ) is None
