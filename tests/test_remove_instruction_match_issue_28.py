"""remove_instruction match logic for issue #28 bug 51.

Live test: with three stored instructions —

  • "never mind"
  • "I prefer short emails"
  • "testing the note path"

— the user typing ``remove the never mind instruction`` resulted in
``Removed: I prefer short emails``. The old matcher used
``any(w in message.lower() for w in instr.text.lower().split()[:3])`` —
substring containment on the first three words. A one-letter token
like ``"i"`` from "I prefer short emails" matches inside "instructIon",
so the first-iterated instruction always won regardless of intent.

Fix: tokenize on word boundaries, drop stopwords (and the keyword
"instruction"/"rule" itself, since they appear in every removal
phrasing), then pick the instruction with the highest count of
significant overlapping tokens. Only delete if a real overlap exists.
"""
from __future__ import annotations

import prism_pa_intents


class _FakeInstr:
    __slots__ = ("instr_id", "text", "trigger")

    def __init__(self, instr_id: str, text: str, trigger: str = "always"):
        self.instr_id = instr_id
        self.text = text
        self.trigger = trigger


class _FakeStore:
    """Drop-in stand-in for ``PrismInstructions`` for these tests."""

    def __init__(self, instrs: list[_FakeInstr]):
        self._instrs = list(instrs)
        self.removed: list[str] = []

    def all_active(self) -> list[_FakeInstr]:
        return list(self._instrs)

    def remove(self, instr_id: str) -> bool:
        before = len(self._instrs)
        self._instrs = [i for i in self._instrs if i.instr_id != instr_id]
        self.removed.append(instr_id)
        return len(self._instrs) != before


class _FakeAgent:
    def __init__(self, instrs: list[_FakeInstr]):
        self._instructions = _FakeStore(instrs)


def _run(agent: _FakeAgent, message: str):
    return prism_pa_intents.handle_pa_intent(
        agent, "remove_instruction", message, {}
    )


class TestPicksTheNamedInstruction:
    """The reported scenario: pick the instruction the user actually named."""

    def test_remove_never_mind_picks_never_mind(self):
        agent = _FakeAgent([
            _FakeInstr("a", "never mind"),
            _FakeInstr("b", "I prefer short emails"),
            _FakeInstr("c", "testing the note path"),
        ])
        card = _run(agent, "remove the never mind instruction")
        assert card.title == "Instruction removed"
        assert "never mind" in card.body
        assert agent._instructions.removed == ["a"]

    def test_dark_mode_picks_dark_mode(self):
        agent = _FakeAgent([
            _FakeInstr("x", "use dark mode in the UI"),
            _FakeInstr("y", "I like spicy food"),
        ])
        card = _run(agent, "forget the dark mode instruction")
        assert "dark mode" in card.body
        assert agent._instructions.removed == ["x"]

    def test_uber_rule_picks_uber_rule(self):
        agent = _FakeAgent([
            _FakeInstr("p", "always use Uber for rides"),
            _FakeInstr("q", "always check weather before commute"),
        ])
        card = _run(agent, "delete the rule about uber")
        assert "uber" in card.body.lower()
        assert agent._instructions.removed == ["p"]


class TestDoesNotRemoveOnNoOverlap:
    """If nothing meaningful overlaps, do NOT silently delete something."""

    def test_no_overlap_returns_not_found(self):
        agent = _FakeAgent([
            _FakeInstr("a", "I prefer short emails"),
            _FakeInstr("b", "testing the note path"),
        ])
        card = _run(agent, "remove the never mind instruction")
        # Neither instruction shares any significant token with the
        # message — must report "couldn't find" rather than delete one
        # at random.
        assert "Couldn't find" in card.body or "couldn't find" in card.body.lower()
        assert agent._instructions.removed == []

    def test_empty_store_returns_not_found(self):
        agent = _FakeAgent([])
        card = _run(agent, "remove the never mind instruction")
        assert "Couldn't find" in card.body or "couldn't find" in card.body.lower()
        assert agent._instructions.removed == []


class TestIgnoresStopwordsAndKeywords:
    """``the``, ``a``, ``instruction``, ``rule`` must not be the only
    thing that pins a match — otherwise every removal request would
    fuzzy-match every instruction."""

    def test_the_alone_does_not_match(self):
        # Two instructions both contain "the" — without stopword
        # filtering, "remove the rule" would pick one of them by chance.
        agent = _FakeAgent([
            _FakeInstr("a", "use the metric system"),
            _FakeInstr("b", "check the inbox hourly"),
        ])
        card = _run(agent, "remove the instruction")
        # No significant overlap, must not delete.
        assert agent._instructions.removed == []
        assert "Couldn't find" in card.body or "couldn't find" in card.body.lower()


class TestTieGoesToFirstMatch:
    """When two instructions tie on score, the first one wins —
    deterministic and predictable behaviour."""

    def test_first_tie_winner(self):
        agent = _FakeAgent([
            _FakeInstr("a", "use dark mode"),
            _FakeInstr("b", "set dark mode"),
        ])
        card = _run(agent, "forget the dark mode instruction")
        # Both share "dark" + "mode" → tie. First instruction wins.
        assert agent._instructions.removed == ["a"]
        assert "dark mode" in card.body
