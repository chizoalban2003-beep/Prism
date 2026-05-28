"""
tests/test_jarvis.py
====================
Unit tests for ksa_jarvis.py — JarvisAgent, ThinkResult, ActResult, Artifact.
"""

import json
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ksa_executor import FileIndexExecutor, LocalSearchExecutor, ShellExecutor
from ksa_jarvis import (
    Artifact,
    ActResult,
    JarvisAgent,
    ThinkResult,
    _infer_content_type,
    _row_to_artifact,
)
from ksa_lever import ThreeBarSystem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent(tmp_path):
    """A JarvisAgent backed by a temp DB with three built-in tasks registered."""
    db = str(tmp_path / "jarvis.db")
    a  = JarvisAgent(db_path=db, dry_run=True, auto_optimise=False)
    a.register(
        task_name   = "file_index_stealth",
        keywords    = ["index", "scan", "files", "folder", "stealth",
                       "background", "quiet", "silently"],
        executor    = FileIndexExecutor(),
        aliases     = ["index"],
        description = "Background file indexing",
    )
    a.register(
        task_name   = "local_search",
        keywords    = ["search", "find", "locate", "grep", "query"],
        executor    = LocalSearchExecutor(),
        aliases     = ["search", "find"],
        description = "Local file/content search",
    )
    a.register(
        task_name   = "shell_generic",
        keywords    = ["run", "exec", "execute", "shell", "command"],
        executor    = ShellExecutor(),
        aliases     = ["shell"],
        description = "Generic shell command execution",
    )
    return a


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_agent_is_created(self, agent):
        assert agent is not None

    def test_repr_contains_dry_run(self, agent):
        r = repr(agent)
        assert "dry_run=True" in r

    def test_db_file_created(self, tmp_path):
        db = str(tmp_path / "sub" / "jarvis.db")
        JarvisAgent(db_path=db, dry_run=True, auto_optimise=False)
        assert os.path.exists(db)

    def test_jarvis_tables_exist(self, agent):
        import sqlite3
        conn = sqlite3.connect(agent.db_path)
        tables = {
            r[0] for r in
            conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        conn.close()
        assert "artifacts" in tables
        assert "profiles"  in tables


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegister:
    def test_registered_tasks_appear_in_intents(self, agent):
        names = [i["task_name"] for i in agent.router.list_intents()]
        assert "file_index_stealth" in names
        assert "local_search"       in names
        assert "shell_generic"      in names

    def test_registered_executors_appear_in_registry(self, agent):
        assert "file_index_stealth" in agent.executor_registry.list_executors()

    def test_custom_default_system_is_forwarded(self, tmp_path):
        db     = str(tmp_path / "j2.db")
        a      = JarvisAgent(db_path=db, dry_run=True, auto_optimise=False)
        custom = ThreeBarSystem.from_defaults()
        custom.levers[0].set_weights(left=9.0, right=1.0)
        a.register(
            task_name      = "custom_task",
            keywords       = ["custom", "special"],
            executor       = ShellExecutor(),
            default_system = custom,
        )
        result = a.think("custom special command")
        assert result.task_name == "custom_task"


# ---------------------------------------------------------------------------
# think()
# ---------------------------------------------------------------------------

class TestThink:
    def test_returns_think_result(self, agent):
        result = agent.think("quietly scan my project directory")
        assert isinstance(result, ThinkResult)

    def test_task_name_is_set(self, agent):
        result = agent.think("scan files in background")
        assert result.task_name == "file_index_stealth"

    def test_confidence_between_0_and_1(self, agent):
        result = agent.think("search for python files")
        assert 0.0 <= result.confidence <= 1.0

    def test_decision_is_valid_action(self, agent):
        result = agent.think("index my files")
        assert result.decision in ("primary", "secondary", "safe")

    def test_rationale_is_non_empty(self, agent):
        result = agent.think("run ls")
        assert result.rationale != ""

    def test_route_is_attached(self, agent):
        result = agent.think("search for logs")
        assert result.route is not None
        assert isinstance(result.route.system, ThreeBarSystem)

    def test_think_does_not_create_artifact(self, agent):
        agent.think("scan files")
        artifacts = agent.remember("file_index_stealth")
        assert len(artifacts) == 0


# ---------------------------------------------------------------------------
# act()
# ---------------------------------------------------------------------------

class TestAct:
    def test_returns_act_result(self, agent):
        result = agent.act("scan my project folder")
        assert isinstance(result, ActResult)

    def test_think_embedded_in_act(self, agent):
        result = agent.act("quietly index my files")
        assert isinstance(result.think, ThinkResult)

    def test_outcome_has_expected_fields(self, agent):
        result = agent.act("scan my project folder")
        o = result.outcome
        assert o.task_name  == "file_index_stealth"
        assert o.return_code >= 0
        assert o.metrics is not None

    def test_successful_act_saves_artifact(self, agent):
        agent.act("scan my project folder quietly")
        arts = agent.remember("file_index_stealth")
        assert len(arts) >= 1

    def test_artifact_has_correct_task_name(self, agent):
        result = agent.act("index files silently")
        if result.artifact is not None:
            assert result.artifact.task_name == "file_index_stealth"

    def test_artifact_content_with_explicit_payload(self, agent):
        payload = {"files": ["a.py", "b.py"]}
        result  = agent.act("scan my project", artifact_content=payload)
        if result.artifact is not None:
            assert result.artifact.content_type == "config"

    def test_improved_flag_is_bool(self, agent):
        result = agent.act("scan files")
        assert isinstance(result.improved, bool)

    def test_act_multiple_times_accumulates_artifacts(self, agent):
        agent.act("scan files quietly")
        agent.act("background index scan")
        arts = agent.remember("file_index_stealth")
        assert len(arts) >= 2

    def test_bootstrap_prompt_still_returns_act_result(self, agent):
        result = agent.act("xyzzy unrecognised gibberish prompt")
        assert isinstance(result, ActResult)


# ---------------------------------------------------------------------------
# remember()
# ---------------------------------------------------------------------------

class TestRemember:
    def test_empty_when_no_history(self, agent):
        arts = agent.remember("file_index_stealth")
        assert arts == []

    def test_returns_artifacts_after_act(self, agent):
        agent.act("scan files background")
        arts = agent.remember("file_index_stealth")
        assert len(arts) >= 1

    def test_sorted_by_score_descending(self, agent):
        for _ in range(3):
            agent.act("scan files")
        arts = agent.remember("file_index_stealth", n=10)
        scores = [a.score for a in arts]
        assert scores == sorted(scores, reverse=True)

    def test_n_limits_results(self, agent):
        for _ in range(5):
            agent.act("scan files quietly")
        arts = agent.remember("file_index_stealth", n=2)
        assert len(arts) <= 2

    def test_min_score_filter(self, agent):
        agent.act("scan files")
        arts = agent.remember("file_index_stealth", min_score=9999.0)
        assert arts == []

    def test_artifact_fields_are_populated(self, agent):
        agent.act("scan project folder")
        arts = agent.remember("file_index_stealth")
        if arts:
            a = arts[0]
            assert a.artifact_id  != ""
            assert a.task_name    == "file_index_stealth"
            assert a.created_at   != ""
            assert a.score        >= 0.0


# ---------------------------------------------------------------------------
# reflect()
# ---------------------------------------------------------------------------

class TestReflect:
    def test_returns_dict_with_tasks_key(self, agent):
        r = agent.reflect()
        assert "tasks"           in r
        assert "total_artifacts" in r

    def test_tasks_contain_registered_entries(self, agent):
        agent.act("scan files")
        r     = agent.reflect()
        names = [t["task_name"] for t in r["tasks"]]
        assert "file_index_stealth" in names

    def test_task_entry_has_required_fields(self, agent):
        agent.act("scan files background")
        r = agent.reflect()
        for task in r["tasks"]:
            assert "task_name"       in task
            assert "current_version" in task
            assert "total_versions"  in task
            assert "fixed_fulcrum"   in task
            assert "artifact_count"  in task
            assert "best_score"      in task

    def test_total_artifacts_counts_all(self, agent):
        agent.act("scan files")
        agent.act("run ls")
        r = agent.reflect()
        # At least 1 artifact (local_search safe with no index = failure = no artifact)
        assert r["total_artifacts"] >= 1

    def test_reflect_when_empty(self, agent):
        r = agent.reflect()
        assert isinstance(r["tasks"], list)
        assert r["total_artifacts"] == 0


# ---------------------------------------------------------------------------
# status()
# ---------------------------------------------------------------------------

class TestStatus:
    def test_status_has_expected_keys(self, agent):
        s = agent.status()
        assert "tasks"     in s
        assert "intents"   in s
        assert "artifacts" in s

    def test_artifacts_count_is_int(self, agent):
        s = agent.status()
        assert isinstance(s["artifacts"], int)


# ---------------------------------------------------------------------------
# _drift_fixed_fulcrum()
# ---------------------------------------------------------------------------

class TestDriftFixedFulcrum:
    def test_profile_created_after_first_drift(self, agent):
        agent._drift_fixed_fulcrum("demo_task", 1.0)
        import sqlite3
        conn = sqlite3.connect(agent.db_path)
        row  = conn.execute(
            "SELECT fixed_fulcrum FROM profiles WHERE task_name = ?",
            ("demo_task",),
        ).fetchone()
        conn.close()
        assert row is not None

    def test_high_score_pushes_fulcrum_up(self, agent):
        # Start from neutral
        agent._drift_fixed_fulcrum("task_x", 0.0)  # score=0 → target ~0.0
        # Many good runs should push fulcrum toward 1.0
        for _ in range(50):
            agent._drift_fixed_fulcrum("task_x", 1000.0)  # target ~1.0

        import sqlite3
        conn = sqlite3.connect(agent.db_path)
        row  = conn.execute(
            "SELECT fixed_fulcrum FROM profiles WHERE task_name = ?",
            ("task_x",),
        ).fetchone()
        conn.close()
        assert row[0] > 0.5

    def test_drift_is_ema_bounded(self, agent):
        for _ in range(1000):
            agent._drift_fixed_fulcrum("bounded_task", 9999.0)

        import sqlite3
        conn = sqlite3.connect(agent.db_path)
        row  = conn.execute(
            "SELECT fixed_fulcrum FROM profiles WHERE task_name = ?",
            ("bounded_task",),
        ).fetchone()
        conn.close()
        assert 0.0 <= row[0] <= 1.0

    def test_zero_score_pushes_fulcrum_down(self, agent):
        # First push it up
        for _ in range(50):
            agent._drift_fixed_fulcrum("task_y", 1000.0)
        # Then push it down
        for _ in range(200):
            agent._drift_fixed_fulcrum("task_y", 0.0)

        import sqlite3
        conn = sqlite3.connect(agent.db_path)
        row  = conn.execute(
            "SELECT fixed_fulcrum FROM profiles WHERE task_name = ?",
            ("task_y",),
        ).fetchone()
        conn.close()
        assert row[0] < 0.95  # drifted back down from ~1.0


# ---------------------------------------------------------------------------
# _save_artifact() directly
# ---------------------------------------------------------------------------

class TestSaveArtifact:
    def test_artifact_is_returned(self, agent):
        a = agent._save_artifact(
            task_name    = "demo",
            version      = 1,
            content      = "hello world",
            content_type = "text",
            score        = 42.0,
            tags         = ["test"],
        )
        assert isinstance(a, Artifact)
        assert a.artifact_id != ""
        assert a.score == pytest.approx(42.0)

    def test_artifact_persisted_in_db(self, agent):
        a = agent._save_artifact(
            task_name    = "demo",
            version      = 1,
            content      = {"key": "value"},
            content_type = "config",
            score        = 5.0,
            tags         = [],
        )
        arts = agent.remember("demo")
        assert any(x.artifact_id == a.artifact_id for x in arts)

    def test_artifact_content_survives_roundtrip(self, agent):
        original = {"nested": [1, 2, 3]}
        a = agent._save_artifact(
            task_name    = "demo",
            version      = 1,
            content      = original,
            content_type = "config",
            score        = 1.0,
            tags         = [],
        )
        arts = agent.remember("demo")
        match = next((x for x in arts if x.artifact_id == a.artifact_id), None)
        assert match is not None
        assert match.content == original

    def test_tags_persisted(self, agent):
        tags = ["alpha", "beta"]
        a    = agent._save_artifact(
            task_name    = "demo",
            version      = 1,
            content      = "data",
            content_type = "text",
            score        = 1.0,
            tags         = tags,
        )
        arts = agent.remember("demo")
        match = next((x for x in arts if x.artifact_id == a.artifact_id), None)
        assert match is not None
        assert match.tags == tags


# ---------------------------------------------------------------------------
# _infer_content_type() helper
# ---------------------------------------------------------------------------

class TestInferContentType:
    def test_code_string_detected(self):
        src = "def foo():\n    return 42\nimport os"
        assert _infer_content_type(src) == "code"

    def test_file_path_detected(self):
        assert _infer_content_type("/home/user/file.py")    == "file_path"
        assert _infer_content_type("./relative/path.json")  == "file_path"

    def test_plain_text(self):
        assert _infer_content_type("some plain text here") == "text"

    def test_dict_is_config(self):
        assert _infer_content_type({"key": "val"}) == "config"

    def test_list_is_search_result(self):
        assert _infer_content_type(["a", "b"]) == "search_result"

    def test_none_falls_back_to_text(self):
        assert _infer_content_type(None) == "text"
