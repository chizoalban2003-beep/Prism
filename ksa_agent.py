"""
ksa_agent.py
============
Kinetic State Agent — Top-Level Orchestrator

KSAgent is the single entry point for running the full KSA pipeline:

    1. Route a prompt to a task via MasterFulcrum
    2. Simulate the loaded ThreeBarSystem
    3. Execute the resolved task via ExecutorRegistry
    4. Record the outcome in SnapshotRegistry
    5. Optionally optimise lever geometry via KineticOptimizer

Usage:
    agent = KSAgent(db_path="~/.ksa/state.db", auto_optimise=True)

    agent.register(
        task_name   = "file_index_stealth",
        keywords    = ["index", "scan", "files", "folder"],
        executor    = FileIndexExecutor(),
        aliases     = ["index"],
        description = "Background file indexing",
    )

    outcome = agent.run("quietly scan my project folder")
    print(outcome)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from ksa_executor import (
    ExecutionContext,
    ExecutionOutcome,
    ExecutorRegistry,
    TaskExecutor,
)
from ksa_lever import ThreeBarSystem
from ksa_optimizer import KineticOptimizer
from ksa_registry import SnapshotRegistry
from ksa_router import MasterFulcrum

logger = logging.getLogger(__name__)


class KSAgent:
    """
    Top-level orchestrator that wires together the router, executor registry,
    snapshot registry, and optimizer into a single ``run()`` call.

    Parameters
    ----------
    db_path:
        Path to the SQLite state database.  ``~`` is expanded automatically.
        The parent directory is created if it does not exist.
    working_dir:
        Default working directory passed to every ExecutionContext.
    ollama_model:
        If set, a local Ollama LLM is used as a fallback resolver in the
        router when keyword confidence is below the floor.  Pass ``None``
        (default) to disable.
    ollama_host:
        Base URL of the Ollama API server.
    auto_optimise:
        When True, ``KineticOptimizer.maybe_improve()`` is called after
        every successful task run.
    dry_run:
        When True, executors perform no real OS actions.
    """

    def __init__(
        self,
        db_path:       str  = "ksa_state.db",
        working_dir:   str  = ".",
        ollama_model:  Optional[str] = None,
        ollama_host:   str  = "http://localhost:11434",
        auto_optimise: bool = True,
        dry_run:       bool = False,
    ) -> None:
        db_path        = os.path.expanduser(db_path)
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        self.working_dir   = working_dir
        self.dry_run       = dry_run
        self.auto_optimise = auto_optimise

        # Core components
        self.registry           = SnapshotRegistry(db_path)
        self.executor_registry  = ExecutorRegistry(self.registry)
        self.optimizer          = KineticOptimizer(self.registry)

        # Set up optional LLM resolver
        llm_resolver = None
        if ollama_model:
            llm_resolver = MasterFulcrum.ollama_resolver(
                model = ollama_model,
                host  = ollama_host,
            )

        self.router = MasterFulcrum(self.registry, llm_resolver=llm_resolver)

        logger.info(
            "KSAgent initialised: db=%s, dry_run=%s, auto_optimise=%s",
            db_path,
            dry_run,
            auto_optimise,
        )

    # ── Registration ──────────────────────────────────────────────────────────

    def register(
        self,
        task_name:      str,
        keywords:       list[str],
        executor:       TaskExecutor,
        aliases:        Optional[list[str]]       = None,
        default_system: Optional[ThreeBarSystem]  = None,
        description:    str                       = "",
    ) -> None:
        """
        Register a task intent with the router AND register the executor.

        Both registrations use the same ``task_name`` as the key.
        """
        self.router.register_intent(
            task_name      = task_name,
            keywords       = keywords,
            aliases        = aliases,
            default_system = default_system,
            description    = description,
        )
        # Ensure the executor's task_name matches
        executor.task_name = task_name
        self.executor_registry.register(executor)
        logger.debug("Registered task '%s'", task_name)

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def run(self, prompt: str) -> ExecutionOutcome:
        """
        Execute the full KSA pipeline for a prompt string.

        Steps:
            1. Route the prompt to a task + ThreeBarSystem snapshot.
            2. Simulate the system to obtain an EquilibriumResult.
            3. Build an ExecutionContext and dispatch to the executor.
               (ExecutorRegistry.execute also calls registry.record_outcome)
            4. Optionally run the optimizer.
            5. Return the ExecutionOutcome.
        """
        logger.info("KSAgent.run: %r", prompt)

        # 1. Route
        route_result = self.router.route(prompt)
        logger.debug("Routed to '%s' via %s (conf=%.0f%%)",
                     route_result.task_name, route_result.method,
                     route_result.confidence * 100)

        # 2. Simulate
        eq_result = route_result.system.simulate()
        logger.debug(
            "Simulation: tilt=%s, override=%s, conf=%.2f",
            eq_result.final_tilt.value,
            eq_result.override_active,
            eq_result.confidence,
        )

        # 3. Execute
        # Thread the prompt into the executor payload so search/shell executors
        # actually receive their input. Without this, LocalSearchExecutor and
        # ShellExecutor always saw an empty payload and returned
        # "No query/command provided in ctx.payload[...]".
        ctx = ExecutionContext(
            task_name   = route_result.task_name,
            version     = route_result.version,
            result      = eq_result,
            working_dir = self.working_dir,
            dry_run     = self.dry_run,
            payload     = {"query": prompt, "command": prompt},
        )
        outcome = self.executor_registry.execute(ctx)
        logger.info("Outcome: %s", outcome)

        # 4. Optimise (optional)
        if self.auto_optimise:
            try:
                new_ver = self.optimizer.maybe_improve(
                    task_name = outcome.task_name,
                    version   = outcome.version,
                    outcome   = outcome,
                )
                if new_ver:
                    logger.info(
                        "Optimizer improved '%s' to v%d", outcome.task_name, new_ver
                    )
            except Exception as exc:
                logger.warning("Optimizer error (non-fatal): %s", exc)

        return outcome

    # ── Inspection ────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """
        Return a summary dict with registered tasks and router intents.

        Suitable for display in a CLI ``status`` command.
        """
        return {
            "tasks":   self.registry.list_tasks(),
            "intents": self.router.list_intents(),
        }

    def __repr__(self) -> str:
        return (
            f"KSAgent("
            f"dry_run={self.dry_run}, "
            f"auto_optimise={self.auto_optimise}, "
            f"tasks={len(self.registry.list_tasks())})"
        )


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.WARNING)
    print("=== KSA Agent Demo ===\n")

    from ksa_executor import FileIndexExecutor, LocalSearchExecutor, ShellExecutor

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        agent = KSAgent(db_path=db_path, dry_run=True, auto_optimise=True)

        agent.register(
            task_name   = "file_index_stealth",
            keywords    = ["index", "scan", "files", "folder", "stealth",
                           "background", "quiet", "silently"],
            executor    = FileIndexExecutor(),
            aliases     = ["index"],
            description = "Background file indexing without UI interference",
        )
        agent.register(
            task_name   = "local_search",
            keywords    = ["search", "find", "locate", "grep", "query"],
            executor    = LocalSearchExecutor(),
            aliases     = ["search", "find"],
            description = "Low-priority local file/content search",
        )
        agent.register(
            task_name   = "shell_generic",
            keywords    = ["run", "exec", "execute", "shell", "command"],
            executor    = ShellExecutor(),
            aliases     = ["shell"],
            description = "Generic shell command execution",
        )

        prompts = [
            "quietly scan my project directory in the background",
            "find all TODO comments in my codebase",
            "run ls -la",
        ]

        for p in prompts:
            print(f"Prompt : {p!r}")
            outcome = agent.run(p)
            print(f"Outcome: {outcome}\n")

        print("Agent status (task count):", len(agent.status()["tasks"]))

    finally:
        os.unlink(db_path)
        print("Temp DB cleaned up. ✓")
