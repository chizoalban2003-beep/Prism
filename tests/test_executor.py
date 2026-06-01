"""
tests/test_executor.py
======================
Unit tests for ksa_executor.py — ExecutorRegistry, TaskExecutor,
FileIndexExecutor, LocalSearchExecutor, ShellExecutor.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ksa_lever import ThreeBarSystem, TiltDirection, EquilibriumResult, LeverState
from ksa_registry import SnapshotRegistry
from ksa_executor import (
    ExecutionContext,
    ExecutorRegistry,
    FileIndexExecutor,
    LocalSearchExecutor,
    ShellExecutor,
    TaskExecutor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_eq(tilt: TiltDirection, override: bool = False) -> EquilibriumResult:
    """Build a minimal EquilibriumResult for testing."""
    state = LeverState(
        lever_id       = 2,
        net_torque     = 1.0 if tilt == TiltDirection.LEFT else -1.0,
        tilt           = tilt,
        tilt_magnitude = 1.0,
        is_locked      = False,
    )
    return EquilibriumResult(
        states          = [state, state, state],
        final_tilt      = tilt,
        override_active = override,
        confidence      = 0.8,
    )


@pytest.fixture
def registry(tmp_path):
    reg = SnapshotRegistry(str(tmp_path / "test.db"))
    system = ThreeBarSystem.from_defaults()
    reg.save("file_index_stealth", system)
    reg.save("local_search", system)
    reg.save("shell_generic", system)
    return reg


@pytest.fixture
def executor_registry(registry):
    ereg = ExecutorRegistry(registry)
    ereg.register(FileIndexExecutor())
    ereg.register(LocalSearchExecutor())
    ereg.register(ShellExecutor())
    return ereg


# ---------------------------------------------------------------------------
# ExecutorRegistry
# ---------------------------------------------------------------------------

class TestExecutorRegistry:
    def test_register_and_list(self, registry):
        ereg = ExecutorRegistry(registry)
        ereg.register(FileIndexExecutor())
        assert "file_index_stealth" in ereg.list_executors()

    def test_register_without_task_name_raises(self, registry):
        ereg = ExecutorRegistry(registry)
        exc = TaskExecutor()  # task_name = ""
        with pytest.raises(ValueError):
            ereg.register(exc)

    def test_no_executor_returns_safe_noop(self, registry):
        ereg    = ExecutorRegistry(registry)
        eq      = _make_eq(TiltDirection.LEFT)
        ctx     = ExecutionContext("unknown_task", 1, eq, dry_run=True)
        outcome = ereg.execute(ctx)
        assert outcome.action_taken == "safe"
        assert outcome.return_code  == 0

    def test_left_tilt_routes_to_primary(self, executor_registry):
        eq      = _make_eq(TiltDirection.LEFT)
        ctx     = ExecutionContext("file_index_stealth", 1, eq, dry_run=True)
        outcome = executor_registry.execute(ctx)
        assert outcome.action_taken == "primary"

    def test_right_tilt_routes_to_secondary(self, executor_registry):
        eq      = _make_eq(TiltDirection.RIGHT)
        ctx     = ExecutionContext("file_index_stealth", 1, eq, dry_run=True)
        outcome = executor_registry.execute(ctx)
        assert outcome.action_taken == "secondary"

    def test_balanced_tilt_routes_to_safe(self, executor_registry):
        eq      = _make_eq(TiltDirection.BALANCED)
        ctx     = ExecutionContext("file_index_stealth", 1, eq, dry_run=True)
        outcome = executor_registry.execute(ctx)
        assert outcome.action_taken == "safe"

    def test_override_active_forces_safe(self, executor_registry):
        eq      = _make_eq(TiltDirection.LEFT, override=True)
        ctx     = ExecutionContext("file_index_stealth", 1, eq, dry_run=True)
        outcome = executor_registry.execute(ctx)
        assert outcome.action_taken == "safe"


# ---------------------------------------------------------------------------
# FileIndexExecutor
# ---------------------------------------------------------------------------

class TestFileIndexExecutor:
    def _make_ctx(self, action, dry_run=True):
        if action == "primary":
            tilt = TiltDirection.LEFT
        elif action == "secondary":
            tilt = TiltDirection.RIGHT
        else:
            tilt = TiltDirection.BALANCED
        eq = _make_eq(tilt)
        return ExecutionContext("file_index_stealth", 1, eq, dry_run=dry_run)

    def test_primary_dry_run_succeeds(self):
        exe     = FileIndexExecutor()
        ctx     = self._make_ctx("primary", dry_run=True)
        outcome = exe.primary(ctx)
        assert outcome.return_code  == 0
        assert outcome.action_taken == "primary"
        assert "dry-run" in outcome.stdout.lower()

    def test_secondary_dry_run_succeeds(self):
        exe     = FileIndexExecutor()
        ctx     = self._make_ctx("secondary", dry_run=True)
        outcome = exe.secondary(ctx)
        assert outcome.return_code  == 0
        assert outcome.action_taken == "secondary"

    def test_safe_is_noop(self):
        exe     = FileIndexExecutor()
        ctx     = self._make_ctx("safe", dry_run=True)
        outcome = exe.safe(ctx)
        assert outcome.return_code  == 0
        assert outcome.action_taken == "safe"

    def test_metrics_populated(self):
        exe     = FileIndexExecutor()
        ctx     = self._make_ctx("primary", dry_run=True)
        outcome = exe.primary(ctx)
        assert outcome.metrics is not None
        assert outcome.elapsed_ms >= 0.0


# ---------------------------------------------------------------------------
# LocalSearchExecutor
# ---------------------------------------------------------------------------

class TestLocalSearchExecutor:
    def _ctx(self, tilt, query="", dry_run=True):
        eq  = _make_eq(tilt)
        ctx = ExecutionContext(
            "local_search", 1, eq,
            dry_run = dry_run,
            payload = {"query": query} if query else {},
        )
        return ctx

    def test_primary_dry_run_succeeds(self):
        exe     = LocalSearchExecutor()
        ctx     = self._ctx(TiltDirection.LEFT, query="TODO", dry_run=True)
        outcome = exe.primary(ctx)
        assert outcome.return_code  == 0
        assert outcome.action_taken == "primary"

    def test_secondary_dry_run_succeeds(self):
        exe     = LocalSearchExecutor()
        ctx     = self._ctx(TiltDirection.RIGHT, query="TODO", dry_run=True)
        outcome = exe.secondary(ctx)
        assert outcome.return_code  == 0
        assert outcome.action_taken == "secondary"

    def test_safe_falls_back_to_index_file(self, tmp_path):
        index = tmp_path / ".ksa_index.txt"
        index.write_text("file1.py\nfile2.py\n")

        exe = LocalSearchExecutor()
        eq  = _make_eq(TiltDirection.BALANCED)
        ctx = ExecutionContext(
            "local_search", 1, eq,
            working_dir = str(tmp_path),
            dry_run     = False,
        )
        outcome = exe.safe(ctx)
        assert outcome.return_code  == 0
        assert "file1.py" in outcome.stdout

    def test_safe_missing_index_returns_error(self, tmp_path):
        exe = LocalSearchExecutor()
        eq  = _make_eq(TiltDirection.BALANCED)
        ctx = ExecutionContext(
            "local_search", 1, eq,
            working_dir = str(tmp_path),
            dry_run     = False,
        )
        outcome = exe.safe(ctx)
        assert outcome.return_code != 0

    def test_primary_no_query_returns_error(self):
        exe = LocalSearchExecutor()
        ctx = self._ctx(TiltDirection.LEFT, query="", dry_run=False)
        outcome = exe.primary(ctx)
        assert outcome.return_code != 0


# ---------------------------------------------------------------------------
# ShellExecutor
# ---------------------------------------------------------------------------

class TestShellExecutor:
    def _ctx(self, tilt, command="echo hello", dry_run=True):
        eq  = _make_eq(tilt)
        ctx = ExecutionContext(
            "shell_generic", 1, eq,
            dry_run = dry_run,
            payload = {"command": command},
        )
        return ctx

    def test_primary_dry_run(self):
        exe     = ShellExecutor()
        ctx     = self._ctx(TiltDirection.LEFT, dry_run=True)
        outcome = exe.primary(ctx)
        assert outcome.return_code  == 0
        assert outcome.action_taken == "primary"

    def test_secondary_dry_run(self):
        exe     = ShellExecutor()
        ctx     = self._ctx(TiltDirection.RIGHT, dry_run=True)
        outcome = exe.secondary(ctx)
        assert outcome.return_code  == 0
        assert outcome.action_taken == "secondary"

    def test_safe_is_noop(self):
        exe     = ShellExecutor()
        ctx     = self._ctx(TiltDirection.BALANCED, dry_run=True)
        outcome = exe.safe(ctx)
        assert outcome.return_code  == 0
        assert outcome.action_taken == "safe"

    def test_primary_no_command_returns_error(self):
        exe = ShellExecutor()
        eq  = _make_eq(TiltDirection.LEFT)
        ctx = ExecutionContext("shell_generic", 1, eq, dry_run=False, payload={})
        outcome = exe.primary(ctx)
        assert outcome.return_code != 0

    def test_primary_runs_real_command(self):
        exe = ShellExecutor()
        eq  = _make_eq(TiltDirection.LEFT)
        ctx = ExecutionContext(
            "shell_generic", 1, eq,
            dry_run = False,
            payload = {"command": "echo ksa_test"},
        )
        outcome = exe.primary(ctx)
        assert outcome.return_code == 0
        assert "ksa_test" in outcome.stdout
