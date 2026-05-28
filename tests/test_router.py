"""
tests/test_router.py
====================
Unit tests for ksa_router.py — MasterFulcrum, RouteResult, IntentPattern.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ksa_lever import ThreeBarSystem
from ksa_registry import SnapshotRegistry
from ksa_router import MasterFulcrum, RouteResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry(tmp_path):
    return SnapshotRegistry(str(tmp_path / "test.db"))


@pytest.fixture
def router(registry):
    r = MasterFulcrum(registry)
    r.register_intent(
        task_name   = "file_index_stealth",
        keywords    = ["index", "scan", "files", "directory", "folder",
                       "stealth", "background", "quiet", "silently"],
        aliases     = ["index"],
        description = "Background file indexing",
    )
    r.register_intent(
        task_name   = "local_search",
        keywords    = ["search", "find", "locate", "grep", "query", "lookup"],
        aliases     = ["search", "find"],
        description = "Local file search",
    )
    r.register_intent(
        task_name   = "code_gen_assist",
        keywords    = ["write", "generate", "code", "function", "class",
                       "script", "implement", "draft"],
        aliases     = ["codegen"],
        description = "Code generation",
    )
    return r


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_intent_adds_pattern(self, router):
        intents = router.list_intents()
        names = [i["task_name"] for i in intents]
        assert "file_index_stealth" in names
        assert "local_search" in names
        assert "code_gen_assist" in names

    def test_unregister_intent(self, router):
        router.unregister_intent("local_search")
        names = [i["task_name"] for i in router.list_intents()]
        assert "local_search" not in names

    def test_unregister_returns_false_for_unknown(self, router):
        result = router.unregister_intent("nonexistent_task")
        assert result is False


# ---------------------------------------------------------------------------
# Routing tests
# ---------------------------------------------------------------------------

class TestRouting:
    def test_keyword_match_file_index(self, router):
        result = router.route("quietly scan my project directory in the background")
        assert result.task_name == "file_index_stealth"
        assert result.method == "keyword"
        assert result.confidence > 0

    def test_keyword_match_local_search(self, router):
        # Use multiple keywords to exceed the confidence floor (0.25)
        result = router.route("search and locate the query string")
        assert result.task_name == "local_search"
        assert result.method in ("keyword", "alias")

    def test_keyword_match_code_gen(self, router):
        result = router.route("write a Python function to parse JSON logs")
        assert result.task_name == "code_gen_assist"
        assert result.method == "keyword"

    def test_alias_exact_match(self, router):
        result = router.route("index")
        assert result.task_name == "file_index_stealth"
        assert result.method == "alias"
        assert result.confidence == pytest.approx(1.0)

    def test_bootstrap_on_unknown_prompt(self, router):
        result = router.route("do something completely unrecognised and unusual")
        assert result.method == "bootstrap"
        assert result.confidence == pytest.approx(0.0)

    def test_route_result_has_system(self, router):
        result = router.route("scan my folder")
        assert isinstance(result.system, ThreeBarSystem)

    def test_route_result_has_positive_elapsed(self, router):
        result = router.route("scan my folder")
        assert result.elapsed_ms >= 0.0

    def test_route_result_stores_raw_prompt(self, router):
        prompt = "find all log files"
        result = router.route(prompt)
        assert result.prompt_raw == prompt

    def test_confidence_floor_respected(self, registry):
        """If confidence_floor is very high, keyword matching should miss."""
        r = MasterFulcrum(registry, confidence_floor=0.99)
        r.register_intent("some_task", keywords=["scan", "files", "directory"])
        result = r.route("scan one file")   # only 1/3 keywords match → conf ~0.33
        # Should fall through to bootstrap since 0.33 < 0.99
        assert result.method in ("bootstrap", "llm")


# ---------------------------------------------------------------------------
# _normalise tests
# ---------------------------------------------------------------------------

class TestNormalise:
    def test_lowercases(self):
        assert MasterFulcrum._normalise("HELLO WORLD") == "hello world"

    def test_strips_punctuation(self):
        assert "," not in MasterFulcrum._normalise("hello, world!")

    def test_collapses_whitespace(self):
        result = MasterFulcrum._normalise("  hello   world  ")
        assert "  " not in result

    def test_empty_string(self):
        assert MasterFulcrum._normalise("") == ""


# ---------------------------------------------------------------------------
# _infer_task_name tests
# ---------------------------------------------------------------------------

class TestInferTaskName:
    def test_produces_snake_case(self):
        name = MasterFulcrum._infer_task_name("quietly backup my files")
        assert " " not in name
        assert "-" not in name

    def test_excludes_stop_words(self):
        name = MasterFulcrum._infer_task_name("can you run the script")
        assert "can" not in name
        assert "you" not in name

    def test_max_4_words(self):
        name = MasterFulcrum._infer_task_name(
            "alpha beta gamma delta epsilon zeta"
        )
        assert len(name.split("_")) <= 4

    def test_empty_prompt(self):
        name = MasterFulcrum._infer_task_name("")
        assert name == "unknown_task"
