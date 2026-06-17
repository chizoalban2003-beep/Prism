"""M12c — denial → DPO pair ingestion in PrismLoraTrainer.

Covers the three pair shapes (standing category, task-slug, always),
empty rejection, and end-to-end collection with both outcomes.db and
instructions.db present.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from prism_instructions import Instruction, PrismInstructions
from prism_lora_trainer import PrismLoraTrainer

# ---------------------------------------------------------------------------
# _denial_to_dpo_pair — pure transform
# ---------------------------------------------------------------------------

def _make_rule(text: str, trigger: str) -> Instruction:
    return Instruction(instr_id="x", text=text, trigger=trigger)


def test_denial_to_dpo_standing_email():
    rule = _make_rule("never send work emails after 8pm", "email")
    pair = PrismLoraTrainer._denial_to_dpo_pair(rule)
    assert pair is not None
    assert "email" in pair["prompt"].lower()
    assert pair["chosen"] == "never send work emails after 8pm"
    assert "rule" in pair["rejected"].lower()


def test_denial_to_dpo_task_slug():
    rule = _make_rule("on 'send_email': not after hours", "send_email")
    pair = PrismLoraTrainer._denial_to_dpo_pair(rule)
    assert pair is not None
    assert "send email" in pair["prompt"].lower()
    assert "proceed" in pair["prompt"].lower()
    assert pair["rejected"].lower().startswith("yes")


def test_denial_to_dpo_always():
    rule = _make_rule("confirm any irreversible action", "always")
    pair = PrismLoraTrainer._denial_to_dpo_pair(rule)
    assert pair is not None
    assert "always" in pair["prompt"].lower()
    assert pair["chosen"] == "confirm any irreversible action"


def test_denial_to_dpo_empty_text_returns_none():
    assert PrismLoraTrainer._denial_to_dpo_pair(_make_rule("", "email")) is None
    assert PrismLoraTrainer._denial_to_dpo_pair(_make_rule("   ", "email")) is None


# ---------------------------------------------------------------------------
# _pairs_from_instructions — iterates a live instructions DB
# ---------------------------------------------------------------------------

@pytest.fixture
def trainer(tmp_path):
    return PrismLoraTrainer(work_dir=str(tmp_path / "lora"))


def test_pairs_from_instructions_mix(trainer, tmp_path):
    instr = PrismInstructions(db_path=str(tmp_path / "instr.db"))
    instr.add("never send work emails after 8pm", trigger="email")
    instr.add("on 'send_email': boundary", trigger="send_email")
    instr.add("always confirm irreversible work", trigger="always")

    pairs = trainer._pairs_from_instructions(instr)
    assert len(pairs) == 3
    prompts = " || ".join(p["prompt"].lower() for p in pairs)
    assert "email requests" in prompts
    assert "send email" in prompts
    assert "always follow" in prompts


def test_pairs_from_instructions_empty(trainer, tmp_path):
    instr = PrismInstructions(db_path=str(tmp_path / "instr.db"))
    assert trainer._pairs_from_instructions(instr) == []


# ---------------------------------------------------------------------------
# _collect_dpo_pairs — full path with both DBs mocked into tmp paths
# ---------------------------------------------------------------------------

def _seed_outcomes_db(path: Path) -> None:
    """Create a minimal outcomes.db OutcomeTracker can read."""
    from prism_outcome_tracker import OutcomeTracker
    tracker = OutcomeTracker(db_path=str(path))
    tracker.record(
        chain_id="test-chain-001",
        goal="answer the question",
        outcome="user_corrected",
        final_answer="initial guess",
        correction="the better answer",
    )


def test_collect_dpo_pairs_aggregates_both_sources(trainer, tmp_path, monkeypatch):
    outcomes_path = tmp_path / "outcomes.db"
    instr_path    = tmp_path / "instructions.db"
    _seed_outcomes_db(outcomes_path)

    # Seed instructions DB.
    instr = PrismInstructions(db_path=str(instr_path))
    instr.add("never send work emails after 8pm", trigger="email")
    instr.add("on 'send_email': boundary", trigger="send_email")

    # Redirect the hard-coded ~/.prism paths to the temp files via a
    # Path.expanduser monkeypatch keyed on the two strings of interest.
    real_expand = Path.expanduser

    def fake_expand(self: Path) -> Path:
        s = str(self)
        if s == "~/.prism/outcomes.db":
            return outcomes_path
        if s == "~/.prism/instructions.db":
            return instr_path
        return real_expand(self)

    monkeypatch.setattr(Path, "expanduser", fake_expand)

    pairs = trainer._collect_dpo_pairs()
    chosens = [p["chosen"] for p in pairs]
    assert "the better answer" in chosens             # from outcomes
    assert "never send work emails after 8pm" in chosens  # standing rule
    assert any("boundary" in c for c in chosens)      # task-slug denial


def test_collect_dpo_pairs_missing_dbs_is_silent(trainer, monkeypatch, tmp_path):
    """No DBs on disk → empty list, no exception."""
    real_expand = Path.expanduser

    def fake_expand(self: Path) -> Path:
        s = str(self)
        if s in ("~/.prism/outcomes.db", "~/.prism/instructions.db"):
            return tmp_path / "does-not-exist.db"
        return real_expand(self)

    monkeypatch.setattr(Path, "expanduser", fake_expand)
    assert trainer._collect_dpo_pairs() == []
