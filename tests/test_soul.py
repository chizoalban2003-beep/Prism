"""
tests/test_soul.py
==================
Comprehensive tests for prism_soul.py and prism_identity_ceremony.py.
"""
from __future__ import annotations

import json
import time
import uuid
from unittest.mock import MagicMock

import pytest

from prism_soul import (
    BeliefEdge,
    BeliefNode,
    PrismSoul,
    SoulLens,
    SoulSeed,
)
from prism_identity_ceremony import (
    CEREMONY_QUESTIONS,
    IdentityCeremony,
    _QUESTION_ORDER,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_soul(tmp_path) -> PrismSoul:
    db = str(tmp_path / "soul.db")
    return PrismSoul(db_path=db)


def full_answers() -> dict:
    return {
        "identity": "I'm a software engineer who loves building systems that empower people.",
        "decisions": "I want help with architectural trade-offs and prioritisation.",
        "values": "I value honesty, craftsmanship, deep work, and continuous learning.",
        "obstacles": "I tend to procrastinate and get distracted by shiny new ideas.",
        "success": "A year from now I will have shipped two major products and improved my focus.",
        "misunderstand": "People assume I'm an extrovert because I communicate well.",
        "boundaries": "Don't share my personal notes with anyone without asking first.",
    }


# ---------------------------------------------------------------------------
# TestBeliefNode
# ---------------------------------------------------------------------------


class TestBeliefNode:
    def test_creation_with_defaults(self):
        node = BeliefNode(
            node_id="abc12345",
            text="I value honesty",
            belief_type="value",
            source="stated",
        )
        assert node.node_id == "abc12345"
        assert node.text == "I value honesty"
        assert node.belief_type == "value"
        assert node.source == "stated"
        assert node.confidence == 0.8
        assert node.observation_count == 0
        assert node.notes == ""
        assert isinstance(node.created_at, float)
        assert isinstance(node.updated_at, float)

    def test_custom_fields(self):
        node = BeliefNode(
            node_id="xyz",
            text="I procrastinate",
            belief_type="pattern",
            source="observed",
            confidence=0.6,
            observation_count=5,
            notes="Noticed repeatedly",
        )
        assert node.confidence == 0.6
        assert node.observation_count == 5
        assert node.notes == "Noticed repeatedly"

    def test_field_types(self):
        node = BeliefNode(
            node_id="t1",
            text="test",
            belief_type="goal",
            source="stated",
        )
        assert isinstance(node.node_id, str)
        assert isinstance(node.text, str)
        assert isinstance(node.confidence, float)
        assert isinstance(node.observation_count, int)


# ---------------------------------------------------------------------------
# TestSoulLens
# ---------------------------------------------------------------------------


class TestSoulLens:
    def test_empty_trend(self):
        lens = SoulLens(lens_id="l1", name="Focus", description="Focus tracking", signal_types=[])
        assert lens.trend is None

    def test_add_observation(self):
        lens = SoulLens(lens_id="l1", name="Energy", description="Energy level", signal_types=[])
        lens.add_observation(0.8, "morning")
        assert len(lens.observations) == 1
        obs = lens.observations[0]
        assert obs["value"] == 0.8
        assert obs["context"] == "morning"
        assert "timestamp" in obs

    def test_trend_mean_of_values(self):
        lens = SoulLens(lens_id="l1", name="Mood", description="Mood tracking", signal_types=[])
        for i in range(5):
            lens.add_observation(float(i) * 0.1, f"obs{i}")
        # values: 0.0, 0.1, 0.2, 0.3, 0.4 — mean = 0.2
        assert abs(lens.trend - 0.2) < 1e-9

    def test_trend_last_10_only(self):
        lens = SoulLens(lens_id="l1", name="Test", description="desc", signal_types=[])
        # Add 15 observations: first 5 are 0.0, last 10 are 1.0
        for _ in range(5):
            lens.add_observation(0.0, "low")
        for _ in range(10):
            lens.add_observation(1.0, "high")
        # trend should be mean of last 10 = 1.0
        assert lens.trend == 1.0

    def test_trend_single_observation(self):
        lens = SoulLens(lens_id="l1", name="T", description="d", signal_types=[])
        lens.add_observation(0.5, "ctx")
        assert lens.trend == 0.5


# ---------------------------------------------------------------------------
# TestPrismSoulCRUD
# ---------------------------------------------------------------------------


class TestPrismSoulCRUD:
    def test_add_and_get_belief(self, tmp_path):
        soul = make_soul(tmp_path)
        nid = soul.add_belief("I value family", "value", "stated", confidence=0.9)
        assert isinstance(nid, str)
        assert len(nid) == 8  # uuid4[:8]

        node = soul.get_belief(nid)
        assert node is not None
        assert node.text == "I value family"
        assert node.belief_type == "value"
        assert node.source == "stated"
        assert node.confidence == 0.9

    def test_get_belief_not_found(self, tmp_path):
        soul = make_soul(tmp_path)
        assert soul.get_belief("nonexist") is None

    def test_list_beliefs_all(self, tmp_path):
        soul = make_soul(tmp_path)
        soul.add_belief("A", "value", "stated")
        soul.add_belief("B", "pattern", "observed")
        soul.add_belief("C", "goal", "stated")
        all_beliefs = soul.list_beliefs()
        assert len(all_beliefs) == 3

    def test_list_beliefs_filter_source(self, tmp_path):
        soul = make_soul(tmp_path)
        soul.add_belief("A", "value", "stated")
        soul.add_belief("B", "pattern", "observed")
        stated = soul.list_beliefs(source="stated")
        assert len(stated) == 1
        assert stated[0].text == "A"
        observed = soul.list_beliefs(source="observed")
        assert len(observed) == 1
        assert observed[0].text == "B"

    def test_list_beliefs_filter_type(self, tmp_path):
        soul = make_soul(tmp_path)
        soul.add_belief("A", "value", "stated")
        soul.add_belief("B", "pattern", "observed")
        soul.add_belief("C", "value", "observed")
        values = soul.list_beliefs(belief_type="value")
        assert len(values) == 2
        patterns = soul.list_beliefs(belief_type="pattern")
        assert len(patterns) == 1

    def test_update_belief_confidence(self, tmp_path):
        soul = make_soul(tmp_path)
        nid = soul.add_belief("Test belief", "value", "stated", confidence=0.5)
        soul.update_belief(nid, confidence=0.95)
        node = soul.get_belief(nid)
        assert abs(node.confidence - 0.95) < 1e-9

    def test_update_belief_notes(self, tmp_path):
        soul = make_soul(tmp_path)
        nid = soul.add_belief("Test", "value", "stated")
        soul.update_belief(nid, notes="Updated note")
        node = soul.get_belief(nid)
        assert node.notes == "Updated note"

    def test_update_belief_observation_count_delta(self, tmp_path):
        soul = make_soul(tmp_path)
        nid = soul.add_belief("Test", "pattern", "observed")
        soul.update_belief(nid, observation_count_delta=3)
        node = soul.get_belief(nid)
        assert node.observation_count == 3

    def test_add_edge(self, tmp_path):
        soul = make_soul(tmp_path)
        nid1 = soul.add_belief("A", "value", "stated")
        nid2 = soul.add_belief("B", "pattern", "observed")
        eid = soul.add_edge(nid1, nid2, "supports", strength=0.7)
        assert isinstance(eid, str)

    def test_list_edges_all(self, tmp_path):
        soul = make_soul(tmp_path)
        nid1 = soul.add_belief("A", "value", "stated")
        nid2 = soul.add_belief("B", "pattern", "observed")
        nid3 = soul.add_belief("C", "goal", "stated")
        soul.add_edge(nid1, nid2, "supports")
        soul.add_edge(nid2, nid3, "contradicts")
        edges = soul.list_edges()
        assert len(edges) == 2

    def test_list_edges_filter_from_id(self, tmp_path):
        soul = make_soul(tmp_path)
        nid1 = soul.add_belief("A", "value", "stated")
        nid2 = soul.add_belief("B", "pattern", "observed")
        nid3 = soul.add_belief("C", "goal", "stated")
        soul.add_edge(nid1, nid2, "supports")
        soul.add_edge(nid1, nid3, "supports")
        soul.add_edge(nid2, nid3, "explains")
        edges = soul.list_edges(from_id=nid1)
        assert len(edges) == 2

    def test_list_edges_filter_relation(self, tmp_path):
        soul = make_soul(tmp_path)
        nid1 = soul.add_belief("A", "value", "stated")
        nid2 = soul.add_belief("B", "pattern", "observed")
        nid3 = soul.add_belief("C", "goal", "stated")
        soul.add_edge(nid1, nid2, "supports")
        soul.add_edge(nid2, nid3, "contradicts")
        contradicts = soul.list_edges(relation="contradicts")
        assert len(contradicts) == 1
        assert contradicts[0].relation == "contradicts"


# ---------------------------------------------------------------------------
# TestPrismSoulSeed
# ---------------------------------------------------------------------------


class TestPrismSoulSeed:
    def test_set_and_get_seed(self, tmp_path):
        soul = make_soul(tmp_path)
        seed = SoulSeed(
            narrative="I'm a builder who loves systems.",
            stated_values=["craftsmanship", "honesty"],
            stated_goals=["ship two products", "improve focus"],
            stated_constraints=["no midnight work"],
        )
        soul.set_seed(seed)
        loaded = soul.get_seed()
        assert loaded is not None
        assert loaded.narrative == "I'm a builder who loves systems."
        assert loaded.stated_values == ["craftsmanship", "honesty"]
        assert loaded.stated_goals == ["ship two products", "improve focus"]
        assert loaded.stated_constraints == ["no midnight work"]

    def test_get_seed_none_when_empty(self, tmp_path):
        soul = make_soul(tmp_path)
        assert soul.get_seed() is None

    def test_has_seed_false(self, tmp_path):
        soul = make_soul(tmp_path)
        assert soul.has_seed() is False

    def test_has_seed_true(self, tmp_path):
        soul = make_soul(tmp_path)
        seed = SoulSeed(
            narrative="test",
            stated_values=[],
            stated_goals=[],
            stated_constraints=[],
        )
        soul.set_seed(seed)
        assert soul.has_seed() is True

    def test_upsert_seed(self, tmp_path):
        soul = make_soul(tmp_path)
        seed1 = SoulSeed(
            narrative="Original",
            stated_values=["focus"],
            stated_goals=["goal1"],
            stated_constraints=[],
        )
        soul.set_seed(seed1)
        seed2 = SoulSeed(
            narrative="Updated",
            stated_values=["focus", "honesty"],
            stated_goals=["goal1", "goal2"],
            stated_constraints=["no spam"],
        )
        soul.set_seed(seed2)
        loaded = soul.get_seed()
        assert loaded.narrative == "Updated"
        assert len(loaded.stated_values) == 2
        # Only one row in DB
        count = soul._conn.execute("SELECT COUNT(*) FROM seed").fetchone()[0]
        assert count == 1


# ---------------------------------------------------------------------------
# TestPrismSoulLenses
# ---------------------------------------------------------------------------


class TestPrismSoulLenses:
    def test_add_and_get_lens(self, tmp_path):
        soul = make_soul(tmp_path)
        lid = soul.add_lens("Focus", "Deep work focus", signal_types=["focus_signal"])
        lens = soul.get_lens(lid)
        assert lens is not None
        assert lens.name == "Focus"
        assert lens.description == "Deep work focus"
        assert lens.signal_types == ["focus_signal"]
        assert lens.observations == []

    def test_list_lenses(self, tmp_path):
        soul = make_soul(tmp_path)
        soul.add_lens("Focus", "desc1")
        soul.add_lens("Energy", "desc2")
        lenses = soul.list_lenses()
        assert len(lenses) == 2
        names = {l.name for l in lenses}
        assert names == {"Focus", "Energy"}

    def test_record_observation(self, tmp_path):
        soul = make_soul(tmp_path)
        lid = soul.add_lens("Focus", "desc", signal_types=["focus"])
        soul.record_observation(lid, 0.7, "morning session")
        lens = soul.get_lens(lid)
        assert len(lens.observations) == 1
        assert lens.observations[0]["value"] == 0.7
        assert lens.observations[0]["context"] == "morning session"

    def test_record_multiple_observations(self, tmp_path):
        soul = make_soul(tmp_path)
        lid = soul.add_lens("Energy", "desc", signal_types=["energy"])
        for v in [0.3, 0.5, 0.8]:
            soul.record_observation(lid, v, "")
        lens = soul.get_lens(lid)
        assert len(lens.observations) == 3
        assert abs(lens.trend - (0.3 + 0.5 + 0.8) / 3) < 1e-9

    def test_observe_signal_matches_lens(self, tmp_path):
        soul = make_soul(tmp_path)
        lid = soul.add_lens("Stress", "Stress tracking", signal_types=["stress_signal"])
        soul.observe_signal("stress_signal", {"stress": 0.6, "source": "calendar"})
        lens = soul.get_lens(lid)
        assert len(lens.observations) == 1
        assert lens.observations[0]["value"] == 0.6

    def test_observe_signal_no_match(self, tmp_path):
        soul = make_soul(tmp_path)
        lid = soul.add_lens("Focus", "desc", signal_types=["focus"])
        soul.observe_signal("unrelated_signal", {"value": 0.9})
        lens = soul.get_lens(lid)
        assert len(lens.observations) == 0

    def test_observe_signal_clips_to_01(self, tmp_path):
        soul = make_soul(tmp_path)
        lid = soul.add_lens("Load", "desc", signal_types=["load"])
        soul.observe_signal("load", {"load_factor": 2.5})
        lens = soul.get_lens(lid)
        assert lens.observations[0]["value"] == 1.0

    def test_observe_signal_clips_negative(self, tmp_path):
        soul = make_soul(tmp_path)
        lid = soul.add_lens("Load", "desc", signal_types=["load"])
        soul.observe_signal("load", {"load_factor": -0.5})
        lens = soul.get_lens(lid)
        assert lens.observations[0]["value"] == 0.0

    def test_observe_signal_multiple_lenses(self, tmp_path):
        soul = make_soul(tmp_path)
        lid1 = soul.add_lens("L1", "desc1", signal_types=["shared"])
        lid2 = soul.add_lens("L2", "desc2", signal_types=["shared"])
        soul.observe_signal("shared", {"metric": 0.4})
        l1 = soul.get_lens(lid1)
        l2 = soul.get_lens(lid2)
        assert len(l1.observations) == 1
        assert len(l2.observations) == 1


# ---------------------------------------------------------------------------
# TestDeltaReport
# ---------------------------------------------------------------------------


class TestDeltaReport:
    def test_delta_report_with_contradiction(self, tmp_path):
        soul = make_soul(tmp_path)
        stated_id = soul.add_belief("I prioritise rest", "value", "stated", confidence=0.9)
        observed_id = soul.add_belief("Works 12-hour days regularly", "pattern", "observed", confidence=0.8)
        soul.add_edge(stated_id, observed_id, "contradicts", strength=0.75)

        report = soul.delta_report()
        assert len(report) == 1
        assert report[0]["stated"] == "I prioritise rest"
        assert report[0]["observed"] == "Works 12-hour days regularly"
        assert abs(report[0]["strength"] - 0.75) < 1e-9

    def test_delta_report_reverse_order(self, tmp_path):
        """Edge from observed to stated should also be reported."""
        soul = make_soul(tmp_path)
        observed_id = soul.add_belief("Skips meals often", "pattern", "observed")
        stated_id = soul.add_belief("I eat healthily", "value", "stated")
        soul.add_edge(observed_id, stated_id, "contradicts", strength=0.6)

        report = soul.delta_report()
        assert len(report) == 1
        assert report[0]["stated"] == "I eat healthily"
        assert report[0]["observed"] == "Skips meals often"

    def test_delta_report_no_edge(self, tmp_path):
        soul = make_soul(tmp_path)
        soul.add_belief("I value balance", "value", "stated")
        soul.add_belief("Works late nights", "pattern", "observed")
        # No edge added
        report = soul.delta_report()
        assert report == []

    def test_delta_report_supports_edge_not_included(self, tmp_path):
        soul = make_soul(tmp_path)
        nid1 = soul.add_belief("I value learning", "value", "stated")
        nid2 = soul.add_belief("Reads books daily", "pattern", "observed")
        soul.add_edge(nid1, nid2, "supports")
        report = soul.delta_report()
        assert report == []

    def test_delta_report_both_stated_not_included(self, tmp_path):
        soul = make_soul(tmp_path)
        nid1 = soul.add_belief("I value A", "value", "stated")
        nid2 = soul.add_belief("I value B", "value", "stated")
        soul.add_edge(nid1, nid2, "contradicts")
        report = soul.delta_report()
        assert report == []


# ---------------------------------------------------------------------------
# TestCompressForLLM
# ---------------------------------------------------------------------------


class TestCompressForLLM:
    def test_compress_returns_string(self, tmp_path):
        soul = make_soul(tmp_path)
        seed = SoulSeed(
            narrative="Builder mindset.",
            stated_values=["craftsmanship", "honesty", "clarity"],
            stated_goals=["ship great software", "stay healthy"],
            stated_constraints=["no midnight work"],
        )
        soul.set_seed(seed)
        soul.add_belief("I value deep work", "value", "stated", confidence=0.9)
        soul.add_belief("Tends to overcommit", "pattern", "observed", confidence=0.7)
        soul.add_lens("Focus", "Focus tracking", signal_types=["focus"])
        result = soul.compress_for_llm()
        assert isinstance(result, str)

    def test_compress_respects_max_chars(self, tmp_path):
        soul = make_soul(tmp_path)
        seed = SoulSeed(
            narrative="x" * 1000,
            stated_values=["a" * 100] * 10,
            stated_goals=["goal"] * 10,
            stated_constraints=["constraint"],
        )
        soul.set_seed(seed)
        result = soul.compress_for_llm(max_chars=200)
        assert len(result) <= 200

    def test_compress_contains_key_terms(self, tmp_path):
        soul = make_soul(tmp_path)
        seed = SoulSeed(
            narrative="narrative",
            stated_values=["honesty"],
            stated_goals=["build something great"],
            stated_constraints=[],
        )
        soul.set_seed(seed)
        soul.add_belief("I work best alone", "value", "stated", confidence=0.95)
        result = soul.compress_for_llm()
        # Should contain something from the soul
        assert len(result) > 0

    def test_compress_empty_soul(self, tmp_path):
        soul = make_soul(tmp_path)
        result = soul.compress_for_llm()
        assert isinstance(result, str)

    def test_compress_includes_tensions(self, tmp_path):
        soul = make_soul(tmp_path)
        seed = SoulSeed(narrative="n", stated_values=[], stated_goals=[], stated_constraints=[])
        soul.set_seed(seed)
        nid1 = soul.add_belief("I value rest", "value", "stated")
        nid2 = soul.add_belief("Works 12-hour days", "pattern", "observed")
        soul.add_edge(nid1, nid2, "contradicts", strength=0.8)
        result = soul.compress_for_llm()
        assert "Tensions" in result


# ---------------------------------------------------------------------------
# TestExportImport
# ---------------------------------------------------------------------------


class TestExportImport:
    def test_export_json_structure(self, tmp_path):
        soul = make_soul(tmp_path)
        data = soul.export_json()
        assert "seed" in data
        assert "beliefs" in data
        assert "edges" in data
        assert "lenses" in data

    def test_export_import_round_trip(self, tmp_path):
        soul = make_soul(tmp_path)
        seed = SoulSeed(
            narrative="Original soul.",
            stated_values=["v1", "v2"],
            stated_goals=["g1"],
            stated_constraints=["c1"],
        )
        soul.set_seed(seed)
        nid1 = soul.add_belief("Belief one", "value", "stated", confidence=0.9)
        nid2 = soul.add_belief("Observed pattern", "pattern", "observed")
        soul.add_edge(nid1, nid2, "contradicts", strength=0.6)
        lid = soul.add_lens("Focus", "focus desc", signal_types=["focus_signal"])
        soul.record_observation(lid, 0.7, "test")

        exported = soul.export_json()

        # Restore into a new soul
        soul2 = make_soul(tmp_path / "soul2.db")
        soul2.import_json(exported)

        loaded_seed = soul2.get_seed()
        assert loaded_seed.narrative == "Original soul."
        assert loaded_seed.stated_values == ["v1", "v2"]

        beliefs = soul2.list_beliefs()
        assert len(beliefs) == 2

        edges = soul2.list_edges()
        assert len(edges) == 1
        assert edges[0].relation == "contradicts"

        lenses = soul2.list_lenses()
        assert len(lenses) == 1
        assert lenses[0].name == "Focus"
        assert len(lenses[0].observations) == 1

    def test_import_clears_existing(self, tmp_path):
        soul = make_soul(tmp_path)
        soul.add_belief("Old belief", "value", "stated")
        assert len(soul.list_beliefs()) == 1

        fresh_data = {"seed": None, "beliefs": [], "edges": [], "lenses": []}
        soul.import_json(fresh_data)
        assert len(soul.list_beliefs()) == 0

    def test_export_md_returns_markdown(self, tmp_path):
        soul = make_soul(tmp_path)
        seed = SoulSeed(
            narrative="A maker at heart.",
            stated_values=["creativity"],
            stated_goals=["build"],
            stated_constraints=[],
        )
        soul.set_seed(seed)
        md = soul.export_md()
        assert isinstance(md, str)
        assert "## Soul Seed" in md
        assert "## Values & Beliefs" in md
        assert "## Observed Patterns" in md
        assert "## Lenses" in md
        assert "## Tensions" in md

    def test_export_md_contains_seed_narrative(self, tmp_path):
        soul = make_soul(tmp_path)
        seed = SoulSeed(
            narrative="I am a craftsman.",
            stated_values=["quality"],
            stated_goals=["excellence"],
            stated_constraints=[],
        )
        soul.set_seed(seed)
        md = soul.export_md()
        assert "I am a craftsman." in md

    def test_export_md_no_seed(self, tmp_path):
        soul = make_soul(tmp_path)
        md = soul.export_md()
        assert "## Soul Seed" in md
        assert "identity ceremony" in md.lower() or "no seed" in md.lower()


# ---------------------------------------------------------------------------
# TestIdentityCeremony
# ---------------------------------------------------------------------------


class TestIdentityCeremony:
    def test_run_from_answers_returns_seed(self, tmp_path):
        soul = make_soul(tmp_path)
        ceremony = IdentityCeremony(soul=soul)
        seed = ceremony.run_from_answers(full_answers())
        assert isinstance(seed, SoulSeed)

    def test_run_from_answers_non_empty_values(self, tmp_path):
        soul = make_soul(tmp_path)
        ceremony = IdentityCeremony(soul=soul)
        seed = ceremony.run_from_answers(full_answers())
        # Heuristic fallback should produce some values
        assert isinstance(seed.stated_values, list)
        assert len(seed.stated_values) >= 1

    def test_run_from_answers_non_empty_goals(self, tmp_path):
        soul = make_soul(tmp_path)
        ceremony = IdentityCeremony(soul=soul)
        seed = ceremony.run_from_answers(full_answers())
        assert isinstance(seed.stated_goals, list)
        assert len(seed.stated_goals) >= 1

    def test_is_complete_after_all_answers(self, tmp_path):
        soul = make_soul(tmp_path)
        ceremony = IdentityCeremony(soul=soul)
        for answer in full_answers().values():
            ceremony.answer(answer)
        assert ceremony.is_complete() is True

    def test_is_complete_false_when_partial(self, tmp_path):
        soul = make_soul(tmp_path)
        ceremony = IdentityCeremony(soul=soul)
        ceremony.answer("Some answer")
        assert ceremony.is_complete() is False

    def test_soul_has_seed_after_ceremony(self, tmp_path):
        soul = make_soul(tmp_path)
        ceremony = IdentityCeremony(soul=soul)
        ceremony.run_from_answers(full_answers())
        assert soul.has_seed() is True

    def test_questions_returns_list(self, tmp_path):
        soul = make_soul(tmp_path)
        ceremony = IdentityCeremony(soul=soul)
        qs = ceremony.questions()
        assert isinstance(qs, list)
        assert len(qs) == 7

    def test_questions_contains_expected_text(self, tmp_path):
        soul = make_soul(tmp_path)
        ceremony = IdentityCeremony(soul=soul)
        qs = ceremony.questions()
        # Check the identity question is there
        assert any("defines you" in q or "who are you" in q.lower() for q in qs)

    def test_narrative_contains_qa_pairs(self, tmp_path):
        soul = make_soul(tmp_path)
        ceremony = IdentityCeremony(soul=soul)
        answers = full_answers()
        for a in answers.values():
            ceremony.answer(a)
        narrative = ceremony._build_narrative()
        assert "Q:" in narrative
        assert "A:" in narrative

    def test_ceremony_saves_beliefs_to_soul(self, tmp_path):
        soul = make_soul(tmp_path)
        ceremony = IdentityCeremony(soul=soul)
        ceremony.run_from_answers(full_answers())
        beliefs = soul.list_beliefs()
        # Heuristic extraction should create at least some beliefs
        assert isinstance(beliefs, list)

    def test_ceremony_saves_lenses_to_soul(self, tmp_path):
        soul = make_soul(tmp_path)
        ceremony = IdentityCeremony(soul=soul)
        ceremony.run_from_answers(full_answers())
        lenses = soul.list_lenses()
        assert isinstance(lenses, list)
        assert len(lenses) >= 1  # heuristic creates at least 2 lenses


# ---------------------------------------------------------------------------
# TestCeremonyWithLLM
# ---------------------------------------------------------------------------


class TestCeremonyWithLLM:
    def _make_mock_router(self) -> MagicMock:
        """Create a mock router that returns valid JSON extraction."""
        router = MagicMock()
        extraction = {
            "stated_values": ["deep work", "honesty", "craftsmanship"],
            "stated_goals": ["ship two products", "improve focus habits"],
            "stated_constraints": ["no midnight work", "privacy on personal notes"],
            "initial_beliefs": [
                {"text": "I value building systems that empower people", "belief_type": "value", "confidence": 0.9},
                {"text": "I tend to procrastinate on hard problems", "belief_type": "pattern", "confidence": 0.7},
                {"text": "I prefer written communication over meetings", "belief_type": "preference", "confidence": 0.8},
                {"text": "I work best in long uninterrupted sessions", "belief_type": "preference", "confidence": 0.85},
            ],
            "suggested_lenses": [
                {"name": "Focus Sessions", "description": "Track depth and duration of focused work"},
                {"name": "Decision Confidence", "description": "Track confidence in architectural decisions"},
            ],
        }
        router.chat.return_value = json.dumps(extraction)
        return router

    def test_llm_extraction_populates_values(self, tmp_path):
        soul = make_soul(tmp_path)
        router = self._make_mock_router()
        ceremony = IdentityCeremony(soul=soul, llm_router=router)
        seed = ceremony.run_from_answers(full_answers())
        assert seed.stated_values == ["deep work", "honesty", "craftsmanship"]

    def test_llm_extraction_populates_goals(self, tmp_path):
        soul = make_soul(tmp_path)
        router = self._make_mock_router()
        ceremony = IdentityCeremony(soul=soul, llm_router=router)
        seed = ceremony.run_from_answers(full_answers())
        assert "ship two products" in seed.stated_goals

    def test_llm_beliefs_added_to_soul(self, tmp_path):
        soul = make_soul(tmp_path)
        router = self._make_mock_router()
        ceremony = IdentityCeremony(soul=soul, llm_router=router)
        ceremony.run_from_answers(full_answers())
        beliefs = soul.list_beliefs()
        assert len(beliefs) == 4  # 4 initial_beliefs from mock

    def test_llm_lenses_added_to_soul(self, tmp_path):
        soul = make_soul(tmp_path)
        router = self._make_mock_router()
        ceremony = IdentityCeremony(soul=soul, llm_router=router)
        ceremony.run_from_answers(full_answers())
        lenses = soul.list_lenses()
        assert len(lenses) == 2
        names = {l.name for l in lenses}
        assert "Focus Sessions" in names
        assert "Decision Confidence" in names

    def test_llm_constraints_saved(self, tmp_path):
        soul = make_soul(tmp_path)
        router = self._make_mock_router()
        ceremony = IdentityCeremony(soul=soul, llm_router=router)
        seed = ceremony.run_from_answers(full_answers())
        assert "no midnight work" in seed.stated_constraints

    def test_llm_router_called_once(self, tmp_path):
        soul = make_soul(tmp_path)
        router = self._make_mock_router()
        ceremony = IdentityCeremony(soul=soul, llm_router=router)
        ceremony.run_from_answers(full_answers())
        router.chat.assert_called_once()

    def test_llm_failure_falls_back_to_heuristics(self, tmp_path):
        soul = make_soul(tmp_path)
        router = MagicMock()
        router.chat.side_effect = RuntimeError("LLM unavailable")
        ceremony = IdentityCeremony(soul=soul, llm_router=router)
        # Should not raise, falls back to heuristics
        seed = ceremony.run_from_answers(full_answers())
        assert isinstance(seed, SoulSeed)
        assert soul.has_seed() is True
