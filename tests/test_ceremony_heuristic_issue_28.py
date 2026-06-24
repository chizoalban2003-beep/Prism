"""Heuristic-fallback fix for issue #28 — single-blob soul snapshot.

Live test showed that a user whose 7 ceremony answers lacked sentence
terminators ended up with ``stated_values = ["<entire concatenated transcript>"]``
because the old heuristic joined every answer with a space and split on
``[.!?;]`` — separator-less answers collapsed into one giant "sentence".

The fix extracts each soul-seed field from its semantically-aligned
ceremony question:

    values     → stated_values
    success    → stated_goals
    boundaries → stated_constraints

and splits within an answer using ``[,;.!?\\n]``. These tests pin that
contract so it can't silently regress.

The LLM path (``EXTRACTION_PROMPT``) is untouched; this only governs the
fallback when the router is missing or its JSON output fails to parse.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from prism_identity_ceremony import IdentityCeremony
from prism_soul import PrismSoul


def _soul(tmp_path: Path) -> PrismSoul:
    return PrismSoul(db_path=str(tmp_path / "soul.db"))


def _heuristic_answers() -> dict[str, str]:
    """Answers WITHOUT sentence terminators — the exact shape that broke
    the old heuristic (one blob per key, no '.' '?' or ';')."""
    return {
        "identity":     "engineer building systems for humans",
        "decisions":    "architectural trade-offs and prioritisation",
        "values":       "honesty, craftsmanship, deep work, learning",
        "obstacles":    "procrastination, distraction by shiny things",
        "success":      "shipped two products, improved focus, reduced anxiety",
        "misunderstand":"people assume I'm an extrovert",
        "boundaries":   "ask before sharing personal notes",
    }


class TestPerKeyExtraction:
    def test_values_come_from_values_key(self, tmp_path):
        ceremony = IdentityCeremony(soul=_soul(tmp_path))
        seed = ceremony.run_from_answers(_heuristic_answers())
        joined = " ".join(seed.stated_values).lower()
        # Should contain tokens from the values answer, not the success answer.
        assert "honesty" in joined or "craftsmanship" in joined or "deep work" in joined
        assert "shipped" not in joined and "anxiety" not in joined

    def test_goals_come_from_success_key(self, tmp_path):
        ceremony = IdentityCeremony(soul=_soul(tmp_path))
        seed = ceremony.run_from_answers(_heuristic_answers())
        joined = " ".join(seed.stated_goals).lower()
        assert "shipped" in joined or "products" in joined or "focus" in joined
        assert "honesty" not in joined

    def test_constraints_come_from_boundaries_key(self, tmp_path):
        ceremony = IdentityCeremony(soul=_soul(tmp_path))
        seed = ceremony.run_from_answers(_heuristic_answers())
        joined = " ".join(seed.stated_constraints).lower()
        assert "ask" in joined or "personal notes" in joined or "sharing" in joined
        assert "honesty" not in joined and "shipped" not in joined


class TestNoSingleBlob:
    """The headline regression — separator-less answers must NOT collapse
    into one giant string."""

    def test_values_split_on_commas(self, tmp_path):
        ceremony = IdentityCeremony(soul=_soul(tmp_path))
        seed = ceremony.run_from_answers(_heuristic_answers())
        # The values answer "honesty, craftsmanship, deep work, learning"
        # should produce ≥3 separate values, not one blob.
        assert len(seed.stated_values) >= 3, (
            f"expected multiple values, got {seed.stated_values!r}"
        )

    def test_no_value_is_the_whole_concatenated_transcript(self, tmp_path):
        ceremony = IdentityCeremony(soul=_soul(tmp_path))
        answers = _heuristic_answers()
        seed = ceremony.run_from_answers(answers)
        full_blob = " ".join(answers.values())
        for v in seed.stated_values:
            assert v != full_blob, (
                "a stated_value equals the entire concatenated transcript "
                "— the old single-blob bug has returned"
            )
            # And no value should be absurdly long either
            assert len(v) < 200, f"value too long, looks like a blob: {v!r}"

    def test_goals_are_separate_phrases(self, tmp_path):
        ceremony = IdentityCeremony(soul=_soul(tmp_path))
        seed = ceremony.run_from_answers(_heuristic_answers())
        assert len(seed.stated_goals) >= 2


class TestLLMPathStillWorks:
    """The per-key heuristic only runs when LLM extraction fails. A
    successful LLM extraction must still take precedence."""

    def test_llm_output_used_when_available(self, tmp_path):
        router = MagicMock()
        router.chat.return_value = (
            '{"stated_values": ["from llm"], '
            '"stated_goals": ["llm goal"], '
            '"stated_constraints": ["llm constraint"], '
            '"initial_beliefs": [], '
            '"suggested_lenses": []}'
        )
        ceremony = IdentityCeremony(soul=_soul(tmp_path), llm_router=router)
        seed = ceremony.run_from_answers(_heuristic_answers())
        assert seed.stated_values == ["from llm"]
        assert seed.stated_goals == ["llm goal"]
        assert seed.stated_constraints == ["llm constraint"]

    def test_llm_failure_falls_back_to_per_key_heuristic(self, tmp_path):
        router = MagicMock()
        router.chat.side_effect = RuntimeError("boom")
        ceremony = IdentityCeremony(soul=_soul(tmp_path), llm_router=router)
        seed = ceremony.run_from_answers(_heuristic_answers())
        # Must hit the new per-key path, not the old all-blob join.
        joined = " ".join(seed.stated_values).lower()
        assert "honesty" in joined or "craftsmanship" in joined
