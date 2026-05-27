"""
ksa_executor.py
===============
Kinetic State Agent — Executor Layer

Converts an EquilibriumResult tilt into a concrete OS action.
This is the ONLY layer that touches the host filesystem or shell.

Tilt routing rules:
    LEFT tilt              -> executor.primary(ctx)
    RIGHT tilt             -> executor.secondary(ctx)
    BALANCED tilt          -> executor.safe(ctx)
    override_active = True -> executor.safe(ctx) regardless of tilt

Three built-in executors:
    FileIndexExecutor   (task_name = "file_index_stealth")
    LocalSearchExecutor (task_name = "local_search")
    ShellExecutor       (task_name = "shell_generic")

Usage:
    reg  = SnapshotRegistry("ksa_state.db")
    ereg = ExecutorRegistry(reg)
    ereg.register(FileIndexExecutor())

    ctx     = ExecutionContext("file_index_stealth", 1, eq_result, working_dir=".")
    outcome = ereg.execute(ctx)
    print(outcome.return_code, outcome.stdout)
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import psutil

from ksa_lever import EquilibriumResult, TiltDirection
from ksa_registry import PerformanceMetrics, SnapshotRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resource sampler (background thread)
# ---------------------------------------------------------------------------

class _ResourceSampler:
    """Samples peak CPU and RAM usage in a daemon thread."""

    def __init__(self) -> None:
        self.cpu_peak:    float = 0.0
        self.ram_peak_mb: float = 0.0
        self._stop = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._sample, daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _sample(self) -> None:
        proc = psutil.Process()
        while not self._stop.is_set():
            self.cpu_peak    = max(self.cpu_peak, psutil.cpu_percent(interval=None))
            self.ram_peak_mb = max(
                self.ram_peak_mb,
                proc.memory_info().rss / 1024 / 1024,
            )
            time.sleep(0.5)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExecutionContext:
    """All inputs required to run a task executor."""
    task_name:   str
    version:     int
    result:      EquilibriumResult
    working_dir: str  = "."
    dry_run:     bool = False
    # Optional free-form payload (e.g. search query, command string)
    payload:     dict = field(default_factory=dict)


@dataclass
class ExecutionOutcome:
    """The full record of one executor invocation."""
    task_name:    str
    version:      int
    action_taken: str               # "primary" | "secondary" | "safe"
    return_code:  int
    stdout:       str
    stderr:       str
    metrics:      PerformanceMetrics
    elapsed_ms:   float

    def __str__(self) -> str:
        return (
            f"ExecutionOutcome("
            f"task='{self.task_name}', v{self.version}, "
            f"action={self.action_taken}, rc={self.return_code}, "
            f"{self.elapsed_ms:.1f}ms)"
        )


# ---------------------------------------------------------------------------
# Base TaskExecutor
# ---------------------------------------------------------------------------

class TaskExecutor:
    """
    Abstract base for task executors.

    Subclasses must set ``task_name`` to match a registered router intent
    and implement ``primary``, ``secondary``, and ``safe``.
    """

    task_name: str = ""

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _run_shell(
        cmd: str,
        working_dir: str = ".",
        timeout: int = 60,
    ) -> tuple[int, str, str]:
        """
        Run a shell command, return (return_code, stdout, stderr).
        Never raises on non-zero exit codes.
        """
        try:
            completed = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=working_dir,
                timeout=timeout,
            )
            return completed.returncode, completed.stdout, completed.stderr
        except subprocess.TimeoutExpired:
            return 124, "", "Command timed out"
        except Exception as exc:
            return 1, "", str(exc)

    @staticmethod
    def _make_outcome(
        ctx:          ExecutionContext,
        action_taken: str,
        return_code:  int,
        stdout:       str,
        stderr:       str,
        elapsed_ms:   float,
        sampler:      _ResourceSampler,
    ) -> ExecutionOutcome:
        metrics = PerformanceMetrics(
            execution_time_ms = elapsed_ms,
            cpu_peak_pct      = sampler.cpu_peak,
            ram_peak_mb       = sampler.ram_peak_mb,
            success           = return_code == 0,
            override_fired    = ctx.result.override_active,
            notes             = f"action={action_taken}",
        )
        return ExecutionOutcome(
            task_name    = ctx.task_name,
            version      = ctx.version,
            action_taken = action_taken,
            return_code  = return_code,
            stdout       = stdout,
            stderr       = stderr,
            metrics      = metrics,
            elapsed_ms   = elapsed_ms,
        )

    # ── interface ─────────────────────────────────────────────────────────────

    def primary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        raise NotImplementedError

    def secondary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        raise NotImplementedError

    def safe(self, ctx: ExecutionContext) -> ExecutionOutcome:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Executor Registry
# ---------------------------------------------------------------------------

class ExecutorRegistry:
    """
    Holds a mapping of task_name → TaskExecutor.
    Routes an ExecutionContext to the right executor based on tilt,
    then records the outcome in the SnapshotRegistry automatically.
    """

    def __init__(self, registry: SnapshotRegistry) -> None:
        self.registry  = registry
        self._executors: dict[str, TaskExecutor] = {}

    def register(self, executor: TaskExecutor) -> None:
        """Register an executor under its task_name."""
        if not executor.task_name:
            raise ValueError("TaskExecutor.task_name must be set before registering.")
        self._executors[executor.task_name] = executor
        logger.debug("Registered executor for task '%s'", executor.task_name)

    def execute(self, ctx: ExecutionContext) -> ExecutionOutcome:
        """
        Route tilt → primary / secondary / safe, run the executor,
        record the outcome in the registry, and return the result.

        If no executor is registered for the task, falls back to a no-op safe
        outcome with return_code=0.
        """
        executor = self._executors.get(ctx.task_name)
        if executor is None:
            logger.warning(
                "No executor registered for task '%s'; returning no-op safe outcome.",
                ctx.task_name,
            )
            metrics = PerformanceMetrics(
                execution_time_ms=0.0,
                success=True,
                override_fired=ctx.result.override_active,
                notes="no executor registered",
            )
            return ExecutionOutcome(
                task_name    = ctx.task_name,
                version      = ctx.version,
                action_taken = "safe",
                return_code  = 0,
                stdout       = "",
                stderr       = "",
                metrics      = metrics,
                elapsed_ms   = 0.0,
            )

        # Determine which handler to call
        if ctx.result.override_active or ctx.result.final_tilt == TiltDirection.BALANCED:
            handler_name = "safe"
            handler      = executor.safe
        elif ctx.result.final_tilt == TiltDirection.LEFT:
            handler_name = "primary"
            handler      = executor.primary
        else:  # RIGHT
            handler_name = "secondary"
            handler      = executor.secondary

        logger.info(
            "Executing task '%s' v%d via %s (tilt=%s, override=%s)",
            ctx.task_name,
            ctx.version,
            handler_name,
            ctx.result.final_tilt.value,
            ctx.result.override_active,
        )

        outcome = handler(ctx)

        # Auto-record in registry
        try:
            self.registry.record_outcome(ctx.task_name, ctx.version, outcome.metrics)
        except KeyError:
            logger.warning(
                "Could not record outcome for '%s' v%d: snapshot not in registry.",
                ctx.task_name,
                ctx.version,
            )

        return outcome

    def list_executors(self) -> list[str]:
        return list(self._executors.keys())


# ---------------------------------------------------------------------------
# Concrete Executors
# ---------------------------------------------------------------------------

class FileIndexExecutor(TaskExecutor):
    """
    task_name = "file_index_stealth"

    primary:   nice -n 19 find {working_dir} -type f > .ksa_index.txt
    secondary: find . -maxdepth 1 -type f > .ksa_index.txt
    safe:      log warning, no-op
    """

    task_name = "file_index_stealth"
    INDEX_FILE = ".ksa_index.txt"

    def primary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0  = time.perf_counter()

        if ctx.dry_run:
            rc, out, err = 0, "[dry-run] would run nice find", ""
        else:
            cmd = (
                f"nice -n 19 find {ctx.working_dir} -type f "
                f"> {self.INDEX_FILE}"
            )
            rc, out, err = self._run_shell(cmd, ctx.working_dir)

        elapsed = (time.perf_counter() - t0) * 1000
        sampler.stop()
        return self._make_outcome(ctx, "primary", rc, out, err, elapsed, sampler)

    def secondary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()

        if ctx.dry_run:
            rc, out, err = 0, "[dry-run] would run shallow find", ""
        else:
            cmd = f"find . -maxdepth 1 -type f > {self.INDEX_FILE}"
            rc, out, err = self._run_shell(cmd, ctx.working_dir)

        elapsed = (time.perf_counter() - t0) * 1000
        sampler.stop()
        return self._make_outcome(ctx, "secondary", rc, out, err, elapsed, sampler)

    def safe(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        logger.warning(
            "FileIndexExecutor: safe action triggered for task '%s' — no-op.",
            ctx.task_name,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        sampler.stop()
        return self._make_outcome(ctx, "safe", 0, "", "", elapsed, sampler)


class LocalSearchExecutor(TaskExecutor):
    """
    task_name = "local_search"

    primary:   grep -rl {query} {working_dir}
    secondary: find . -name "*{query}*"
    safe:      return cached .ksa_index.txt contents if available
    """

    task_name  = "local_search"
    INDEX_FILE = ".ksa_index.txt"

    def _query(self, ctx: ExecutionContext) -> str:
        return ctx.payload.get("query", "")

    def primary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0    = time.perf_counter()
        query = self._query(ctx)

        if ctx.dry_run:
            rc, out, err = 0, f"[dry-run] would grep -rl '{query}'", ""
        elif not query:
            rc, out, err = 1, "", "No query provided in ctx.payload['query']"
        else:
            cmd = f"grep -rl {query!r} {ctx.working_dir}"
            rc, out, err = self._run_shell(cmd, ctx.working_dir)

        elapsed = (time.perf_counter() - t0) * 1000
        sampler.stop()
        return self._make_outcome(ctx, "primary", rc, out, err, elapsed, sampler)

    def secondary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0    = time.perf_counter()
        query = self._query(ctx)

        if ctx.dry_run:
            rc, out, err = 0, f"[dry-run] would find -name '*{query}*'", ""
        elif not query:
            rc, out, err = 1, "", "No query provided in ctx.payload['query']"
        else:
            cmd = f"find . -name '*{query}*'"
            rc, out, err = self._run_shell(cmd, ctx.working_dir)

        elapsed = (time.perf_counter() - t0) * 1000
        sampler.stop()
        return self._make_outcome(ctx, "secondary", rc, out, err, elapsed, sampler)

    def safe(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()

        index_path = os.path.join(ctx.working_dir, self.INDEX_FILE)
        if os.path.exists(index_path):
            with open(index_path, encoding="utf-8", errors="replace") as fh:
                out = fh.read()
            rc, err = 0, ""
            logger.info("LocalSearchExecutor: returned cached index (%d bytes)", len(out))
        else:
            out = ""
            err = f"No cached index at {index_path}"
            rc  = 1
            logger.warning("LocalSearchExecutor: safe action — no cached index found.")

        elapsed = (time.perf_counter() - t0) * 1000
        sampler.stop()
        return self._make_outcome(ctx, "safe", rc, out, err, elapsed, sampler)


class ShellExecutor(TaskExecutor):
    """
    task_name = "shell_generic"

    primary:   run command string as-is            (from ctx.payload["command"])
    secondary: prepend "timeout 5" to the command
    safe:      no-op, return_code=0
    """

    task_name = "shell_generic"

    def _command(self, ctx: ExecutionContext) -> str:
        return ctx.payload.get("command", "")

    def primary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0  = time.perf_counter()
        cmd = self._command(ctx)

        if ctx.dry_run:
            rc, out, err = 0, f"[dry-run] would run: {cmd}", ""
        elif not cmd:
            rc, out, err = 1, "", "No command provided in ctx.payload['command']"
        else:
            rc, out, err = self._run_shell(cmd, ctx.working_dir)

        elapsed = (time.perf_counter() - t0) * 1000
        sampler.stop()
        return self._make_outcome(ctx, "primary", rc, out, err, elapsed, sampler)

    def secondary(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0  = time.perf_counter()
        cmd = self._command(ctx)

        if ctx.dry_run:
            rc, out, err = 0, f"[dry-run] would run: timeout 5 {cmd}", ""
        elif not cmd:
            rc, out, err = 1, "", "No command provided in ctx.payload['command']"
        else:
            rc, out, err = self._run_shell(f"timeout 5 {cmd}", ctx.working_dir)

        elapsed = (time.perf_counter() - t0) * 1000
        sampler.stop()
        return self._make_outcome(ctx, "secondary", rc, out, err, elapsed, sampler)

    def safe(self, ctx: ExecutionContext) -> ExecutionOutcome:
        sampler = _ResourceSampler()
        sampler.start()
        t0 = time.perf_counter()
        logger.info(
            "ShellExecutor: safe action — no-op for task '%s'.", ctx.task_name
        )
        elapsed = (time.perf_counter() - t0) * 1000
        sampler.stop()
        return self._make_outcome(ctx, "safe", 0, "", "", elapsed, sampler)


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    print("=== KSA Executor Demo ===\n")

    from ksa_lever import ThreeBarSystem
    from ksa_registry import SnapshotRegistry
    from ksa_router import MasterFulcrum

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        reg    = SnapshotRegistry(db_path)
        router = MasterFulcrum(reg)
        router.register_intent(
            "file_index_stealth",
            keywords=["index", "scan", "files"],
            aliases=["index"],
        )

        ereg = ExecutorRegistry(reg)
        ereg.register(FileIndexExecutor())
        ereg.register(LocalSearchExecutor())
        ereg.register(ShellExecutor())

        route  = router.route("scan my project folder quietly")
        eq     = route.system.simulate()

        ctx     = ExecutionContext(
            task_name   = route.task_name,
            version     = route.version,
            result      = eq,
            working_dir = ".",
            dry_run     = True,
        )
        outcome = ereg.execute(ctx)
        print(outcome)
        print("Metrics score:", outcome.metrics.score())
    finally:
        os.unlink(db_path)
        print("\nTemp DB cleaned up. ✓")
