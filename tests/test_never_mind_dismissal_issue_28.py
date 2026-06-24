"""Never-mind dismissal fix for issue #28 bug 16 — "never mind" was stored as a rule.

Live test: ``never mind`` returned ``Instruction stored: ✓ Remembered:
never mind``. The instruction prefix list includes ``never `` so any
message starting with "never" got persisted as a standing rule. Spoken
conversation routinely uses "never mind" as a dismissal — it should be a
no-op, not a forever-rule.

Fix: reject a known set of dismissal phrases in parse_from_chat before
the prefix check runs.
"""
from __future__ import annotations

import tempfile

from prism_instructions import PrismInstructions


def _store() -> PrismInstructions:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()
    return PrismInstructions(db_path=tmp.name)


class TestDismissalsNotStored:
    def test_never_mind_is_not_stored(self):
        s = _store()
        assert s.parse_from_chat("never mind") is None
        assert s.all_active() == []

    def test_never_mind_capitalised(self):
        s = _store()
        assert s.parse_from_chat("Never mind") is None

    def test_never_mind_with_period(self):
        s = _store()
        assert s.parse_from_chat("never mind.") is None

    def test_nevermind_one_word(self):
        s = _store()
        assert s.parse_from_chat("nevermind") is None

    def test_never_mind_that(self):
        s = _store()
        assert s.parse_from_chat("never mind that") is None


class TestRealRulesStillStored:
    """Make sure the dismissal guard didn't eat legitimate "never" rules."""

    def test_never_call_me_steve(self):
        s = _store()
        result = s.parse_from_chat("never call me Steve")
        assert result is not None
        assert "Steve" in result.text

    def test_never_send_after_9pm(self):
        s = _store()
        result = s.parse_from_chat("never send me notifications after 9pm")
        assert result is not None
