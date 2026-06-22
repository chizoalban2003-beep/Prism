"""
ksa_jarvis.py
=============
Kinetic State Agent — Jarvis Agent

The full Jarvis-like local agent.  Extends the KSA pipeline with:

  - Long-term memory: task_name → fixed_fulcrum calibration that drifts
    toward the user's actual preferences over hundreds of interactions
    (exponential moving average, α = 0.05).

  - Artifact storage: every successful task saves a structured artifact
    (code snippet, file path, config fragment, search result) tagged to
    the task + snapshot version that produced it.

  - Context injection: injects time-of-day, recent task history, and
    hardware state as movable-fulcrum weights before every decision.

  - Multi-LLM routing: uses LLM only for NLU (intent parsing) and content
    generation, never for routing decisions or resource allocation.

Usage:
    agent = JarvisAgent(db_path="~/.ksa/jarvis.db", dry_run=True)

    agent.register(
        task_name   = "file_index_stealth",
        keywords    = ["index", "scan", "files"],
        executor    = FileIndexExecutor(),
        description = "Background file indexing",
    )

    result = agent.act("quietly scan my project folder")
    print(result.outcome)
    print(agent.reflect())
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Optional

import psutil

from ksa_executor import (
    ExecutionContext,
    ExecutionOutcome,
    ExecutorRegistry,
    TaskExecutor,
)
from ksa_lever import ThreeBarSystem
from ksa_optimizer import KineticOptimizer
from ksa_registry import SnapshotRegistry
from ksa_router import MasterFulcrum, RouteResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Artifact:
    """A structured result produced by a successful task execution."""
    artifact_id:  str        # uuid4 hex
    task_name:    str
    version:      int
    created_at:   str        # ISO-8601
    content:      Any        # str, dict, list — whatever the task produced
    content_type: str        # "text" | "code" | "file_path" | "config" | "search_result"
    score:        float      # PerformanceMetrics.score() at time of creation
    tags:         list[str] = field(default_factory=list)


@dataclass
class ThinkResult:
    """Output of think() — route + simulate, no execution."""
    task_name:  str
    decision:   str     # primary plank name: "primary" | "secondary" | "safe"
    confidence: float
    rationale:  str     # one-line explanation
    route:      RouteResult


@dataclass
class ActResult:
    """Output of act() — full pipeline including execution and learning."""
    think:    ThinkResult
    outcome:  ExecutionOutcome
    artifact: Optional[Artifact]
    improved: bool           # did optimizer create a new snapshot version?


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _hardware_pressure() -> tuple[float, float]:
    """Return (cpu_pct, ram_pct) as 0-100 floats."""
    cpu  = psutil.cpu_percent(interval=None)
    mem  = psutil.virtual_memory()
    ram  = mem.percent
    return cpu, ram


# ---------------------------------------------------------------------------
# JarvisAgent
# ---------------------------------------------------------------------------

class JarvisAgent:
    """
    Full Jarvis-style local agent.

    Pipeline per act() call:
      1. router.route(prompt)              → RouteResult
      2. _inject_hardware_weights()        → mutates system in place
      3. system.simulate()                 → EquilibriumResult
      4. executor_registry.execute()       → ExecutionOutcome
      5. registry.record_outcome()
      6. optimizer.maybe_improve()         → Optional new snapshot version
      7. _save_artifact()                  → Artifact saved to SQLite
      8. _drift_fixed_fulcrum()            → slowly calibrate base to user's pattern
      9. return ActResult
    """

    # EMA learning rate for fixed-fulcrum drift
    _DRIFT_ALPHA: float = 0.05

    def __init__(
        self,
        db_path:       str           = "~/.ksa/jarvis.db",
        working_dir:   str           = ".",
        ollama_model:  Optional[str] = None,
        ollama_host:   str           = "http://localhost:11434",
        auto_optimise: bool          = True,
        dry_run:       bool          = False,
    ) -> None:
        db_path      = os.path.expanduser(db_path)
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        self.db_path       = db_path
        self.working_dir   = working_dir
        self.dry_run       = dry_run
        self.auto_optimise = auto_optimise

        # Core KSA stack
        self.registry           = SnapshotRegistry(db_path)
        self.executor_registry  = ExecutorRegistry(self.registry)
        self.optimizer          = KineticOptimizer(self.registry)

        llm_resolver = None
        if ollama_model:
            llm_resolver = MasterFulcrum.ollama_resolver(
                model = ollama_model,
                host  = ollama_host,
            )

        self.router = MasterFulcrum(self.registry, llm_resolver=llm_resolver)

        # Jarvis-specific SQLite tables (artifacts + profiles)
        self._init_jarvis_db()

        logger.info(
            "JarvisAgent initialised: db=%s, dry_run=%s, auto_optimise=%s",
            db_path, dry_run, auto_optimise,
        )

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_jarvis_db(self) -> None:
        """Create the artifacts and profiles tables if they don't exist."""
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    id           TEXT PRIMARY KEY,
                    task_name    TEXT NOT NULL,
                    version      INTEGER NOT NULL,
                    created_at   TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    score        REAL NOT NULL,
                    tags_json    TEXT NOT NULL DEFAULT '[]'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS profiles (
                    task_name    TEXT PRIMARY KEY,
                    fixed_fulcrum REAL NOT NULL,
                    updated_at   TEXT NOT NULL
                )
            """)
            conn.commit()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            yield conn
        finally:
            conn.close()

    # ── Public registration ───────────────────────────────────────────────────

    def register(
        self,
        task_name:      str,
        keywords:       list[str],
        executor:       TaskExecutor,
        aliases:        Optional[list[str]]      = None,
        default_system: Optional[ThreeBarSystem] = None,
        description:    str                      = "",
    ) -> None:
        """Register a task intent with the router AND register the executor."""
        self.router.register_intent(
            task_name      = task_name,
            keywords       = keywords,
            aliases        = aliases,
            default_system = default_system,
            description    = description,
        )
        executor.task_name = task_name
        self.executor_registry.register(executor)
        logger.debug("JarvisAgent registered task '%s'", task_name)

    # ── Public pipeline ───────────────────────────────────────────────────────

    def think(self, prompt: str) -> ThinkResult:
        """
        Route and simulate without executing — for preview / dry-run.

        Returns a ThinkResult describing which action would be taken
        and why, without touching the filesystem or shell.
        """
        route = self.router.route(prompt)

        # Apply learned fixed-fulcrum bias to the system
        self._apply_profile_bias(route.task_name, route.system)

        # Inject real-time hardware weights
        self._inject_hardware_weights(route.system)

        eq       = route.system.simulate()
        # Map tilt values to action names
        decision_name = {"left": "primary", "right": "secondary", "balanced": "safe"}.get(
            eq.final_tilt.value, "safe"
        )
        if eq.override_active:
            decision_name = "safe"

        rationale = (
            f"routed via {route.method} (conf={route.confidence:.0%}), "
            f"tilt={eq.final_tilt.value}, override={eq.override_active}"
        )

        return ThinkResult(
            task_name  = route.task_name,
            decision   = decision_name,
            confidence = route.confidence,
            rationale  = rationale,
            route      = route,
        )

    def act(
        self,
        prompt:           str,
        artifact_content: Optional[Any] = None,
    ) -> ActResult:
        """
        Execute the full JarvisAgent pipeline.

        artifact_content: optional pre-computed result to store.  When
        None the executor's stdout is stored as a "text" artifact.
        """
        # 1. Route
        route = self.router.route(prompt)

        # 2. Apply learned bias + hardware injection, then simulate
        self._apply_profile_bias(route.task_name, route.system)
        self._inject_hardware_weights(route.system)
        eq = route.system.simulate()

        # Build ThinkResult
        decision_name = {"left": "primary", "right": "secondary", "balanced": "safe"}.get(
            eq.final_tilt.value, "safe"
        )
        if eq.override_active:
            decision_name = "safe"

        think = ThinkResult(
            task_name  = route.task_name,
            decision   = decision_name,
            confidence = route.confidence,
            rationale  = (
                f"routed via {route.method} (conf={route.confidence:.0%}), "
                f"tilt={eq.final_tilt.value}, override={eq.override_active}"
            ),
            route = route,
        )

        # 3. Execute
        ctx = ExecutionContext(
            task_name   = route.task_name,
            version     = route.version,
            result      = eq,
            working_dir = self.working_dir,
            dry_run     = self.dry_run,
        )
        outcome = self.executor_registry.execute(ctx)

        # 4. Record outcome (ExecutorRegistry already calls record_outcome,
        #    but only if the task had a registered snapshot version.)

        # 5. Optimise
        improved = False
        if self.auto_optimise:
            try:
                new_ver = self.optimizer.maybe_improve(
                    task_name = outcome.task_name,
                    version   = outcome.version,
                    outcome   = outcome,
                )
                if new_ver is not None:
                    improved = True
                    logger.info(
                        "Optimizer improved '%s' to v%d",
                        outcome.task_name, new_ver,
                    )
            except Exception as exc:
                logger.warning("Optimizer error (non-fatal): %s", exc)

        # 6. Save artifact
        score    = outcome.metrics.score()
        content  = artifact_content if artifact_content is not None else outcome.stdout
        c_type   = "text" if artifact_content is None else _infer_content_type(artifact_content)
        artifact = None
        if outcome.metrics.success:
            artifact = self._save_artifact(
                task_name    = outcome.task_name,
                version      = outcome.version,
                content      = content,
                content_type = c_type,
                score        = score,
                tags         = [route.method],
            )

        # 7. Drift fixed fulcrum
        self._drift_fixed_fulcrum(outcome.task_name, score)

        return ActResult(
            think    = think,
            outcome  = outcome,
            artifact = artifact,
            improved = improved,
        )

    # ── Memory retrieval ──────────────────────────────────────────────────────

    def remember(
        self,
        task_name: str,
        n:         int   = 5,
        min_score: float = 0.0,
    ) -> list[Artifact]:
        """
        Retrieve top-n artifacts for task_name, sorted by score descending.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, task_name, version, created_at,
                       content_json, content_type, score, tags_json
                FROM artifacts
                WHERE task_name = ? AND score >= ?
                ORDER BY score DESC
                LIMIT ?
                """,
                (task_name, min_score, n),
            ).fetchall()

        return [_row_to_artifact(r) for r in rows]

    # ── Self-reflection ───────────────────────────────────────────────────────

    def reflect(self) -> dict:
        """
        Return a summary of all tasks, their current fixed_fulcrum positions,
        version counts, and best artifact scores.

        This is how Jarvis "knows itself" — its current learned state.
        """
        tasks = self.registry.list_tasks()

        with self._conn() as conn:
            profiles = {
                row[0]: row[1]
                for row in conn.execute(
                    "SELECT task_name, fixed_fulcrum FROM profiles"
                ).fetchall()
            }
            artifact_counts = {
                row[0]: row[1]
                for row in conn.execute(
                    "SELECT task_name, COUNT(*) FROM artifacts GROUP BY task_name"
                ).fetchall()
            }
            best_scores = {
                row[0]: row[1]
                for row in conn.execute(
                    "SELECT task_name, MAX(score) FROM artifacts GROUP BY task_name"
                ).fetchall()
            }

        summary = []
        for t in tasks:
            name = t["task_name"]
            summary.append({
                "task_name":       name,
                "current_version": t["current_version"],
                "total_versions":  t["total_versions"],
                "fixed_fulcrum":   profiles.get(name, 0.5),
                "artifact_count":  artifact_counts.get(name, 0),
                "best_score":      best_scores.get(name),
            })

        return {
            "tasks":           summary,
            "total_artifacts": sum(artifact_counts.values()),
        }

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """
        {"tasks": registry.list_tasks(), "intents": router.list_intents(),
         "artifacts": count of stored artifacts}
        """
        with self._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM artifacts"
            ).fetchone()[0]
        return {
            "tasks":     self.registry.list_tasks(),
            "intents":   self.router.list_intents(),
            "artifacts": count,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _inject_hardware_weights(self, system: ThreeBarSystem) -> None:
        """
        Read live CPU and RAM usage and inject them as weights on Lever 0.

        High CPU → more weight on the right arm (conservative / secondary).
        High RAM → also pushes right (resource pressure → safe path).
        """
        cpu, ram = _hardware_pressure()
        # Normalise to 0-10 range so they're commensurate with typical lever weights
        cpu_weight = cpu / 10.0
        ram_weight = ram / 10.0
        system.levers[0].add_weight("right", cpu_weight + ram_weight)
        logger.debug(
            "_inject_hardware_weights: cpu=%.1f%% ram=%.1f%% → right +%.2f",
            cpu, ram, cpu_weight + ram_weight,
        )

    def _apply_profile_bias(self, task_name: str, system: ThreeBarSystem) -> None:
        """
        Load the learned fixed_fulcrum for this task and apply it as an
        additional fulcrum_bias on Lever 1.  This makes the system lean
        toward the user's historically preferred decision style.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT fixed_fulcrum FROM profiles WHERE task_name = ?",
                (task_name,),
            ).fetchone()

        if row is None:
            return  # no history yet; keep defaults

        # fixed_fulcrum is in [0,1] with 0.5 = neutral.
        # Map to [-1, +1] bias range and add to Lever 1.
        bias_delta = (row[0] - 0.5) * 2.0
        system.levers[1].fulcrum_bias += bias_delta
        logger.debug(
            "_apply_profile_bias: task=%s, fulcrum=%.3f → Lever1 bias delta=%.3f",
            task_name, row[0], bias_delta,
        )

    def _save_artifact(
        self,
        task_name:    str,
        version:      int,
        content:      Any,
        content_type: str,
        score:        float,
        tags:         list[str],
    ) -> Artifact:
        """Persist an artifact to the SQLite artifacts table."""
        artifact_id = uuid.uuid4().hex
        created_at  = _now_iso()

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO artifacts
                    (id, task_name, version, created_at, content_json,
                     content_type, score, tags_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    task_name,
                    version,
                    created_at,
                    json.dumps(content, default=str),
                    content_type,
                    score,
                    json.dumps(tags),
                ),
            )
            conn.commit()

        logger.debug(
            "_save_artifact: %s for '%s' v%d (score=%.4f)",
            artifact_id, task_name, version, score,
        )

        return Artifact(
            artifact_id  = artifact_id,
            task_name    = task_name,
            version      = version,
            created_at   = created_at,
            content      = content,
            content_type = content_type,
            score        = score,
            tags         = tags,
        )

    def _drift_fixed_fulcrum(self, task_name: str, outcome_score: float) -> None:
        """
        Slowly drift the stored fixed_fulcrum for task_name toward the
        fulcrum position that produced the best recent outcomes.

        Uses exponential moving average:
            new_base = (1 - α) * old_base + α * best_fulcrum

        where α = 0.05 and best_fulcrum is derived from outcome_score:
            best_fulcrum = clamp(outcome_score / (1 + outcome_score), 0, 1)

        A very successful run (high score) pushes the fulcrum toward 1.0
        (aggressive / primary preferred); poor runs let it drift back.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT fixed_fulcrum FROM profiles WHERE task_name = ?",
                (task_name,),
            ).fetchone()

            old_base = row[0] if row else 0.5

        # Map score to a [0,1] target fulcrum
        best_fulcrum = min(1.0, max(0.0, outcome_score / (1.0 + outcome_score)))

        new_base = (1.0 - self._DRIFT_ALPHA) * old_base + self._DRIFT_ALPHA * best_fulcrum
        now      = _now_iso()

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO profiles (task_name, fixed_fulcrum, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(task_name) DO UPDATE SET
                    fixed_fulcrum = excluded.fixed_fulcrum,
                    updated_at    = excluded.updated_at
                """,
                (task_name, new_base, now),
            )
            conn.commit()

        logger.debug(
            "_drift_fixed_fulcrum: task=%s, %.4f → %.4f (target=%.4f)",
            task_name, old_base, new_base, best_fulcrum,
        )

    def __repr__(self) -> str:
        return (
            f"JarvisAgent("
            f"dry_run={self.dry_run}, "
            f"auto_optimise={self.auto_optimise}, "
            f"db={self.db_path!r})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_content_type(content: Any) -> str:
    if isinstance(content, str):
        if "\n" in content and any(
            kw in content for kw in ("def ", "class ", "import ", "return ")
        ):
            return "code"
        if content.startswith(("/", "./", "~/")) or content.endswith(
            (".py", ".txt", ".json", ".toml")
        ):
            return "file_path"
        return "text"
    if isinstance(content, dict):
        return "config"
    if isinstance(content, list):
        return "search_result"
    return "text"


def _row_to_artifact(row: tuple) -> Artifact:
    (artifact_id, task_name, version, created_at,
     content_json, content_type, score, tags_json) = row
    return Artifact(
        artifact_id  = artifact_id,
        task_name    = task_name,
        version      = version,
        created_at   = created_at,
        content      = json.loads(content_json),
        content_type = content_type,
        score        = score,
        tags         = json.loads(tags_json),
    )
