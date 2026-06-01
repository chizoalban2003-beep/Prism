# GitHub Copilot Build Prompt — Kinetic State Agent (KSA)
#
# INSTRUCTIONS FOR COPILOT:
#   1. Read this entire file before generating any code.
#   2. Treat the existing code in Section 3 as ground truth for all interfaces.
#   3. Do not rewrite existing files unless explicitly asked to.
#   4. Build each TODO module in the order listed in Section 8.

---

## 1. Project philosophy

The KSA is a local-first, hardware-native AI agent that uses a
**physics simulation metaphor** (levers, fulcrums, torque) to make
routing and resource-allocation decisions without a neural network.

Key principles:
- Transparency: every decision maps to an inspectable vector (W, F, L).
- Low footprint: the decision core is pure math, not LLM inference.
- Self-optimisation: after each task run, the lever geometry is updated
  if the run outperformed the stored snapshot.
- Surgical LLM use: a local Ollama model is called ONLY when keyword
  confidence is below threshold, never for every decision.

---

## 2. Repo file structure (target state)

```
ksa/
|-- ksa_lever.py        DONE -- 3-bar lever system + snapshots
|-- ksa_registry.py     DONE -- SQLite snapshot registry
|-- ksa_router.py       DONE -- Master Fulcrum intent router
|-- ksa_executor.py     TODO -- Hardware Execution Layer
|-- ksa_optimizer.py    TODO -- Self-optimising loop
|-- ksa_agent.py        TODO -- Orchestrator (wires all layers)
|-- ksa_cli.py          TODO -- CLI entry point
|-- ksa_config.py       TODO -- Config loader (TOML/JSON)
|-- intents.json        TODO -- Serialised intent registry
`-- tests/
    |-- test_lever.py       TODO
    |-- test_registry.py    TODO
    |-- test_router.py      TODO
    |-- test_executor.py    TODO
    `-- test_optimizer.py   TODO
```

---

## 3. Existing code (DO NOT REWRITE)

### 3A. ksa_lever.py

```python
"""
ksa_lever.py
============
Kinetic State Agent — Core 3-Bar Lever System

Models a cascade of three linked levers that simulate torque-based
decision-making. Each lever's tilt can propagate weight into the next
lever in the chain, producing a final equilibrium state that maps to
a concrete action or routing decision.

Architecture:
    Lever 0 (Input Lever)    — receives raw weighted inputs
    Lever 1 (Logic Lever)    — applies constraint bias via fulcrum offset
    Lever 2 (Balancer Bar)   — safety/health monitor; can override cascade

Usage:
    system = ThreeBarSystem.from_defaults()
    system.levers[0].set_weights(left=8.0, right=3.0)
    result = system.simulate()
    print(result)

    snapshot = system.snapshot()        # save state
    system2 = ThreeBarSystem()
    system2.hydrate(snapshot)           # restore state
"""

from __future__ import annotations

import json
import math
import copy
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums & Result Types
# ---------------------------------------------------------------------------

class TiltDirection(str, Enum):
    LEFT      = "left"
    RIGHT     = "right"
    BALANCED  = "balanced"


@dataclass
class LeverState:
    """The computed outcome of a single lever after torque simulation."""
    lever_id:        int
    net_torque:      float          # positive = left-heavy, negative = right-heavy
    tilt:            TiltDirection
    tilt_magnitude:  float          # absolute torque, useful for propagation scaling
    is_locked:       bool = False   # True if the Linkage Matrix suppressed this lever


@dataclass
class EquilibriumResult:
    """Final output after the full 3-bar cascade has been simulated."""
    states:          list[LeverState]
    final_tilt:      TiltDirection   # Lever 2 (Balancer) outcome
    override_active: bool            # True if Balancer damped an upstream instability
    confidence:      float           # 0.0–1.0, derived from tilt_magnitude ratio

    def __str__(self) -> str:
        lines = ["── Equilibrium Result ──────────────────"]
        for s in self.states:
            lock = " [LOCKED]" if s.is_locked else ""
            lines.append(
                f"  Lever {s.lever_id}: {s.tilt.value:8s} | "
                f"torque={s.net_torque:+.3f}{lock}"
            )
        lines.append(f"  Final Decision : {self.final_tilt.value.upper()}")
        lines.append(f"  Override Active: {self.override_active}")
        lines.append(f"  Confidence     : {self.confidence:.2%}")
        lines.append("────────────────────────────────────────")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Lever
# ---------------------------------------------------------------------------

class Lever:
    """
    A single mechanical lever with a fulcrum, two arm lengths,
    and independent left/right weights.

    Torque model:
        torque_left  = left_weight  * left_arm_length
        torque_right = right_weight * right_arm_length
        net_torque   = torque_left - torque_right

        net_torque > +threshold  → tilts LEFT
        net_torque < -threshold  → tilts RIGHT
        otherwise                → BALANCED
    """

    # Below this absolute net torque the lever is considered balanced
    BALANCE_THRESHOLD: float = 0.01

    def __init__(
        self,
        lever_id:         int,
        left_arm_length:  float = 1.0,
        right_arm_length: float = 1.0,
        fulcrum_bias:     float = 0.0,
        left_weight:      float = 0.0,
        right_weight:     float = 0.0,
    ):
        """
        Args:
            lever_id:         Identifier (0, 1, or 2 in a 3-bar system).
            left_arm_length:  Distance from fulcrum to left end (increases sensitivity).
            right_arm_length: Distance from fulcrum to right end.
            fulcrum_bias:     A constant offset added to net_torque, representing
                              the lever's built-in operational predisposition
                              (positive = biased left, negative = biased right).
            left_weight:      Current load on the left arm.
            right_weight:     Current load on the right arm.
        """
        self.lever_id         = lever_id
        self.left_arm_length  = left_arm_length
        self.right_arm_length = right_arm_length
        self.fulcrum_bias     = fulcrum_bias
        self.left_weight      = left_weight
        self.right_weight     = right_weight

    # ── Weight API ──────────────────────────────────────────────────────────

    def set_weights(self, left: float, right: float) -> None:
        """Set both weights at once."""
        self.left_weight  = left
        self.right_weight = right

    def add_weight(self, side: str, amount: float) -> None:
        """Add weight to 'left' or 'right' side (used by cascade propagation)."""
        if side == "left":
            self.left_weight  = max(0.0, self.left_weight  + amount)
        elif side == "right":
            self.right_weight = max(0.0, self.right_weight + amount)
        else:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    # ── Physics ─────────────────────────────────────────────────────────────

    def compute_torque(self) -> float:
        """
        Returns signed net torque.
        Positive  → left arm is heavier (tilts left).
        Negative  → right arm is heavier (tilts right).
        """
        t_left  = self.left_weight  * self.left_arm_length
        t_right = self.right_weight * self.right_arm_length
        return (t_left - t_right) + self.fulcrum_bias

    def evaluate(self) -> LeverState:
        """Compute torque and return a LeverState snapshot."""
        net = self.compute_torque()
        if net > self.BALANCE_THRESHOLD:
            tilt = TiltDirection.LEFT
        elif net < -self.BALANCE_THRESHOLD:
            tilt = TiltDirection.RIGHT
        else:
            tilt = TiltDirection.BALANCED
        return LeverState(
            lever_id       = self.lever_id,
            net_torque     = net,
            tilt           = tilt,
            tilt_magnitude = abs(net),
        )

    # ── Serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "lever_id":         self.lever_id,
            "left_arm_length":  self.left_arm_length,
            "right_arm_length": self.right_arm_length,
            "fulcrum_bias":     self.fulcrum_bias,
            "left_weight":      self.left_weight,
            "right_weight":     self.right_weight,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Lever":
        return cls(**d)

    def __repr__(self) -> str:
        return (
            f"Lever(id={self.lever_id}, "
            f"L={self.left_weight}×{self.left_arm_length}, "
            f"R={self.right_weight}×{self.right_arm_length}, "
            f"bias={self.fulcrum_bias:+.3f})"
        )


# ---------------------------------------------------------------------------
# Three-Bar System
# ---------------------------------------------------------------------------

class ThreeBarSystem:
    """
    A cascade of three levers:
        Lever 0 → Lever 1 → Lever 2 (Balancer)

    Linkage Matrix (3×3):
        L[i][j] = how much of Lever i's tilt_magnitude is forwarded
                  onto Lever j.

        Positive value → adds weight to the LEFT side of lever j.
        Negative value → adds weight to the RIGHT side of lever j.
        Zero           → no coupling.

        Cascade runs left-to-right (0 → 1 → 2).
        L[2] (Balancer row) only affects an override, not further levers.

    Balancer Override Logic:
        If Lever 2's tilt_magnitude exceeds `balancer_threshold`, the
        Balancer is considered to have "damped" the system: the final
        tilt is forced to BALANCED and override_active is set True.
        This models the safety weight sliding out to absorb instability.
    """

    BALANCER_OVERRIDE_THRESHOLD: float = 5.0

    def __init__(
        self,
        levers:           Optional[list[Lever]]   = None,
        linkage_matrix:   Optional[list[list[float]]] = None,
        balancer_threshold: float = BALANCER_OVERRIDE_THRESHOLD,
    ):
        self.levers = levers if levers is not None else [
            Lever(lever_id=0),
            Lever(lever_id=1),
            Lever(lever_id=2),
        ]
        # Default: no coupling between levers
        self.linkage_matrix: list[list[float]] = linkage_matrix if linkage_matrix else [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
        self.balancer_threshold = balancer_threshold

        if len(self.levers) != 3:
            raise ValueError("ThreeBarSystem requires exactly 3 levers.")

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_defaults(cls) -> "ThreeBarSystem":
        """
        A sensible starting configuration:
          - Lever 0: equal arms, no bias (pure input lever)
          - Lever 1: right arm slightly longer (mildly conservative bias)
          - Lever 2: Balancer — long arms for high sensitivity, slight left bias
                     so it fires early on instability
        Coupling: Lever 0 → Lever 1 at 50% strength
        """
        levers = [
            Lever(lever_id=0, left_arm_length=1.0, right_arm_length=1.0, fulcrum_bias=0.0),
            Lever(lever_id=1, left_arm_length=1.0, right_arm_length=1.2, fulcrum_bias=0.0),
            Lever(lever_id=2, left_arm_length=2.0, right_arm_length=2.0, fulcrum_bias=0.5),
        ]
        linkage_matrix = [
            [0.0, 0.5, 0.0],   # Lever 0 feeds 50% of its magnitude into Lever 1
            [0.0, 0.0, 0.3],   # Lever 1 feeds 30% of its magnitude into Lever 2
            [0.0, 0.0, 0.0],   # Lever 2 (Balancer) does not feed forward
        ]
        return cls(levers=levers, linkage_matrix=linkage_matrix, balancer_threshold=5.0)

    # ── Simulation ───────────────────────────────────────────────────────────

    def simulate(self) -> EquilibriumResult:
        """
        Run the full 3-bar cascade simulation.

        Steps:
            1. Evaluate Lever 0, propagate its tilt into downstream levers.
            2. Evaluate Lever 1, propagate its tilt into Lever 2.
            3. Evaluate Lever 2 (Balancer).
            4. Check Balancer threshold — apply override if breached.
            5. Compute confidence from final lever's tilt_magnitude.
        """
        states: list[LeverState]  = []
        # Work on temporary weight copies so simulate() is non-destructive
        working = copy.deepcopy(self.levers)

        for i, lever in enumerate(working):
            state = lever.evaluate()
            states.append(state)

            # Propagate into downstream levers via linkage matrix
            for j in range(i + 1, 3):
                coupling = self.linkage_matrix[i][j]
                if coupling == 0.0 or state.is_locked:
                    continue
                propagated_weight = state.tilt_magnitude * abs(coupling)
                # Sign of coupling determines which side gains weight
                if coupling > 0:
                    target_side = "left"  if state.tilt == TiltDirection.LEFT  else "right"
                else:
                    target_side = "right" if state.tilt == TiltDirection.LEFT  else "left"
                working[j].add_weight(target_side, propagated_weight)

        # ── Balancer override check ──────────────────────────────────────────
        balancer_state   = states[2]
        override_active  = balancer_state.tilt_magnitude > self.balancer_threshold

        if override_active:
            final_tilt = TiltDirection.BALANCED
        else:
            final_tilt = balancer_state.tilt

        # ── Confidence: how decisively did the Balancer tilt? ───────────────
        # Normalised with a soft sigmoid so extreme torques asymptote to 1.0
        confidence = self._sigmoid_confidence(balancer_state.tilt_magnitude)

        return EquilibriumResult(
            states          = states,
            final_tilt      = final_tilt,
            override_active = override_active,
            confidence      = confidence,
        )

    @staticmethod
    def _sigmoid_confidence(magnitude: float, scale: float = 3.0) -> float:
        """Map tilt magnitude to a 0–1 confidence score via sigmoid."""
        return 1.0 / (1.0 + math.exp(-magnitude / scale))

    # ── Snapshot (State Saving) ───────────────────────────────────────────────

    def snapshot(self) -> dict:
        """
        Serialise the current system state into a KineticSnapshot dict.

        Corresponds to the blueprint's:
            S = [ W⃗ (weights), F⃗ (fulcrum biases), L (linkage matrix) ]
        """
        return {
            "W": [
                {"left": l.left_weight, "right": l.right_weight}
                for l in self.levers
            ],
            "F": [l.fulcrum_bias for l in self.levers],
            "L": copy.deepcopy(self.linkage_matrix),
            "arm_lengths": [
                {"left": l.left_arm_length, "right": l.right_arm_length}
                for l in self.levers
            ],
            "balancer_threshold": self.balancer_threshold,
        }

    def hydrate(self, s: dict) -> None:
        """
        Restore system state from a KineticSnapshot dict.
        Instantly snaps levers to their historical optimal orientation.
        """
        for i, lever in enumerate(self.levers):
            lever.left_weight      = s["W"][i]["left"]
            lever.right_weight     = s["W"][i]["right"]
            lever.fulcrum_bias     = s["F"][i]
            lever.left_arm_length  = s["arm_lengths"][i]["left"]
            lever.right_arm_length = s["arm_lengths"][i]["right"]
        self.linkage_matrix    = copy.deepcopy(s["L"])
        self.balancer_threshold = s.get("balancer_threshold", self.BALANCER_OVERRIDE_THRESHOLD)

    def save_snapshot(self, path: str) -> None:
        """Write snapshot JSON to disk."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.snapshot(), f, indent=2)

    @classmethod
    def load_snapshot(cls, path: str) -> "ThreeBarSystem":
        """Reconstruct a ThreeBarSystem from a saved JSON snapshot."""
        with open(path, encoding="utf-8") as f:
            s = json.load(f)
        system = cls()
        system.hydrate(s)
        return system

    def __repr__(self) -> str:
        return (
            f"ThreeBarSystem(\n"
            + "\n".join(f"  {l}" for l in self.levers)
            + f"\n  linkage={self.linkage_matrix}\n)"
        )


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== KSA — 3-Bar Lever Demo ===\n")

    # ── Scenario 1: Heavy left input → expect LEFT decision ─────────────────
    print("Scenario 1: Heavy left input")
    sys1 = ThreeBarSystem.from_defaults()
    sys1.levers[0].set_weights(left=8.0, right=2.0)
    result1 = sys1.simulate()
    print(result1)

    # ── Scenario 2: Heavy right input → expect RIGHT decision ───────────────
    print("Scenario 2: Heavy right input")
    sys2 = ThreeBarSystem.from_defaults()
    sys2.levers[0].set_weights(left=1.0, right=9.0)
    result2 = sys2.simulate()
    print(result2)

    # ── Scenario 3: Extreme overload — Balancer should fire override ─────────
    print("Scenario 3: Extreme cascade — Balancer override expected")
    sys3 = ThreeBarSystem.from_defaults()
    sys3.levers[0].set_weights(left=50.0, right=0.0)
    sys3.levers[1].set_weights(left=20.0, right=0.0)
    result3 = sys3.simulate()
    print(result3)

    # ── Snapshot round-trip ──────────────────────────────────────────────────
    print("Snapshot round-trip:")
    snap = sys1.snapshot()
    sys4 = ThreeBarSystem()
    sys4.hydrate(snap)
    result4 = sys4.simulate()
    print(f"  Hydrated result matches original: "
          f"{result4.final_tilt == result1.final_tilt}")
    print("\nSnapshot JSON:")
    print(json.dumps(snap, indent=2))
```

### 3B. ksa_registry.py

```python
"""
ksa_registry.py
===============
Kinetic State Agent — Snapshot Registry

A SQLite-backed registry that stores, versions, and retrieves KineticSnapshot
matrices keyed by task name. Every successful task run can persist its lever
configuration here. The self-optimising loop reads from here to hot-swap the
best known configuration for a given task.

Schema summary:
    snapshots       — one row per snapshot version per task
    task_metrics    — per-run performance telemetry (feeds optimisation loop)

Public API:
    registry = SnapshotRegistry("ksa_state.db")

    registry.save(task_name, system)             → version int
    registry.load(task_name) → ThreeBarSystem    (current best version)
    registry.record_outcome(task_name, version, metrics)
    registry.promote(task_name, version)         → make a version "current"
    registry.rollback(task_name)                 → revert to previous version
    registry.best_version(task_name)             → version with best score
    registry.list_tasks()                        → summary of all tasks
    registry.history(task_name)                  → all versions + metrics
    registry.prune(task_name, keep=5)            → trim old versions
    registry.delete_task(task_name)              → remove all versions
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Import the lever system from the same package
# (assumes ksa_lever.py is on sys.path or in the same directory)
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from ksa_lever import ThreeBarSystem


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PerformanceMetrics:
    """
    Telemetry captured after a task run.
    Used by the optimisation loop to decide which snapshot version is best.
    """
    execution_time_ms: float  = 0.0   # wall-clock time for the task
    cpu_peak_pct:      float  = 0.0   # peak CPU % during execution
    ram_peak_mb:       float  = 0.0   # peak RAM usage in MB
    success:           bool   = True  # did the task complete without error?
    override_fired:    bool   = False # did the Balancer override trigger?
    notes:             str    = ""    # free-text annotation

    def score(self) -> float:
        """
        Composite performance score (higher = better).

        Formula balances three signals:
          - Speed:    inverse of execution time (faster = better)
          - Stability: penalty if override fired (instability = bad)
          - Success:  hard multiplier of 0 if the task failed

        Returns a float in [0, ∞). Intended for relative ranking only.
        """
        if not self.success:
            return 0.0
        speed_score     = 1000.0 / max(self.execution_time_ms, 1.0)
        stability_bonus = 0.0 if self.override_fired else 1.0
        return round((speed_score + stability_bonus) * 1.0, 6)

    def to_dict(self) -> dict:
        return {
            "execution_time_ms": self.execution_time_ms,
            "cpu_peak_pct":      self.cpu_peak_pct,
            "ram_peak_mb":       self.ram_peak_mb,
            "success":           self.success,
            "override_fired":    self.override_fired,
            "notes":             self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PerformanceMetrics":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SnapshotRecord:
    """A single row from the snapshots table, fully hydrated."""
    id:            int
    task_name:     str
    version:       int
    snapshot:      dict              # the raw S = {W, F, L, arm_lengths, ...}
    created_at:    str
    is_current:    bool
    metrics:       Optional[PerformanceMetrics] = None
    score:         Optional[float]              = None

    def to_system(self) -> ThreeBarSystem:
        """Instantiate and hydrate a ThreeBarSystem from this record."""
        sys = ThreeBarSystem()
        sys.hydrate(self.snapshot)
        return sys


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class SnapshotRegistry:
    """
    Manages all KineticSnapshot storage, versioning, and retrieval.

    Thread-safety: SQLite WAL mode is enabled. Each public method opens
    and closes its own connection, so the registry is safe to use from
    multiple threads as long as they share the same db path.
    """

    SCHEMA = """
    PRAGMA journal_mode = WAL;
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS snapshots (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        task_name     TEXT    NOT NULL,
        version       INTEGER NOT NULL,
        snapshot_json TEXT    NOT NULL,
        created_at    TEXT    NOT NULL,
        is_current    INTEGER NOT NULL DEFAULT 1,
        UNIQUE (task_name, version)
    );

    CREATE TABLE IF NOT EXISTS task_metrics (
        id                 INTEGER PRIMARY KEY AUTOINCREMENT,
        task_name          TEXT    NOT NULL,
        version            INTEGER NOT NULL,
        recorded_at        TEXT    NOT NULL,
        execution_time_ms  REAL    NOT NULL DEFAULT 0,
        cpu_peak_pct       REAL    NOT NULL DEFAULT 0,
        ram_peak_mb        REAL    NOT NULL DEFAULT 0,
        success            INTEGER NOT NULL DEFAULT 1,
        override_fired     INTEGER NOT NULL DEFAULT 0,
        score              REAL    NOT NULL DEFAULT 0,
        notes              TEXT    NOT NULL DEFAULT '',
        FOREIGN KEY (task_name, version)
            REFERENCES snapshots (task_name, version)
            ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_snap_task
        ON snapshots (task_name, is_current);
    CREATE INDEX IF NOT EXISTS idx_metrics_task
        ON task_metrics (task_name, version);
    """

    def __init__(self, db_path: str = "ksa_state.db"):
        self.db_path = Path(db_path)
        self._init_db()

    # ── Internal helpers ────────────────────────────────────────────────────

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(self.SCHEMA)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _next_version(self, conn: sqlite3.Connection, task_name: str) -> int:
        row = conn.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 FROM snapshots WHERE task_name = ?",
            (task_name,)
        ).fetchone()
        return row[0]

    def _get_record(
        self,
        conn: sqlite3.Connection,
        task_name: str,
        version: Optional[int] = None
    ) -> Optional[sqlite3.Row]:
        if version is None:
            return conn.execute(
                "SELECT * FROM snapshots WHERE task_name = ? AND is_current = 1",
                (task_name,)
            ).fetchone()
        return conn.execute(
            "SELECT * FROM snapshots WHERE task_name = ? AND version = ?",
            (task_name, version)
        ).fetchone()

    def _best_metrics(
        self,
        conn: sqlite3.Connection,
        task_name: str,
        version: int
    ) -> Optional[PerformanceMetrics]:
        """Return the most recent metrics row for a given task/version."""
        row = conn.execute(
            """SELECT * FROM task_metrics
               WHERE task_name = ? AND version = ?
               ORDER BY recorded_at DESC LIMIT 1""",
            (task_name, version)
        ).fetchone()
        if row is None:
            return None
        return PerformanceMetrics(
            execution_time_ms = row["execution_time_ms"],
            cpu_peak_pct      = row["cpu_peak_pct"],
            ram_peak_mb       = row["ram_peak_mb"],
            success           = bool(row["success"]),
            override_fired    = bool(row["override_fired"]),
            notes             = row["notes"],
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def save(self, task_name: str, system: ThreeBarSystem) -> int:
        """
        Serialise the system's current state and save it as a new snapshot
        version for the given task. Automatically marks this version as
        `is_current`, demoting any previous current version.

        Returns the new version number.
        """
        snapshot_json = json.dumps(system.snapshot())
        with self._conn() as conn:
            version = self._next_version(conn, task_name)
            # Demote all existing current versions for this task
            conn.execute(
                "UPDATE snapshots SET is_current = 0 WHERE task_name = ?",
                (task_name,)
            )
            conn.execute(
                """INSERT INTO snapshots
                   (task_name, version, snapshot_json, created_at, is_current)
                   VALUES (?, ?, ?, ?, 1)""",
                (task_name, version, snapshot_json, self._now())
            )
        return version

    def load(self, task_name: str, version: Optional[int] = None) -> ThreeBarSystem:
        """
        Hydrate and return a ThreeBarSystem from the registry.

        If version is None, loads the current (most recently promoted) version.
        Raises KeyError if the task or version is not found.
        """
        with self._conn() as conn:
            row = self._get_record(conn, task_name, version)
        if row is None:
            label = f"version {version}" if version else "current version"
            raise KeyError(f"No snapshot found for task '{task_name}' ({label})")
        snapshot = json.loads(row["snapshot_json"])
        sys = ThreeBarSystem()
        sys.hydrate(snapshot)
        return sys

    def record_outcome(
        self,
        task_name:  str,
        version:    int,
        metrics:    PerformanceMetrics,
    ) -> None:
        """
        Store a performance telemetry row for a given task/version.
        Can be called multiple times; all runs are retained for trend analysis.
        """
        with self._conn() as conn:
            # Verify the snapshot exists
            row = self._get_record(conn, task_name, version)
            if row is None:
                raise KeyError(
                    f"Cannot record outcome: snapshot '{task_name}' v{version} not found."
                )
            conn.execute(
                """INSERT INTO task_metrics
                   (task_name, version, recorded_at,
                    execution_time_ms, cpu_peak_pct, ram_peak_mb,
                    success, override_fired, score, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    task_name, version, self._now(),
                    metrics.execution_time_ms,
                    metrics.cpu_peak_pct,
                    metrics.ram_peak_mb,
                    int(metrics.success),
                    int(metrics.override_fired),
                    metrics.score(),
                    metrics.notes,
                )
            )

    def promote(self, task_name: str, version: int) -> None:
        """
        Explicitly mark a specific version as `is_current`.
        Useful after manual inspection or A/B comparison.
        """
        with self._conn() as conn:
            row = self._get_record(conn, task_name, version)
            if row is None:
                raise KeyError(f"Snapshot '{task_name}' v{version} not found.")
            conn.execute(
                "UPDATE snapshots SET is_current = 0 WHERE task_name = ?",
                (task_name,)
            )
            conn.execute(
                "UPDATE snapshots SET is_current = 1 WHERE task_name = ? AND version = ?",
                (task_name, version)
            )

    def rollback(self, task_name: str) -> int:
        """
        Revert to the previous version (current_version - 1).
        Returns the version number now marked as current.
        Raises ValueError if there is no previous version.
        """
        with self._conn() as conn:
            current = self._get_record(conn, task_name)
            if current is None:
                raise KeyError(f"No snapshot found for task '{task_name}'.")
            prev_version = current["version"] - 1
            prev = self._get_record(conn, task_name, prev_version)
            if prev is None:
                raise ValueError(
                    f"Cannot rollback '{task_name}': already at version 1."
                )
            conn.execute(
                "UPDATE snapshots SET is_current = 0 WHERE task_name = ?",
                (task_name,)
            )
            conn.execute(
                "UPDATE snapshots SET is_current = 1 "
                "WHERE task_name = ? AND version = ?",
                (task_name, prev_version)
            )
        return prev_version

    def best_version(self, task_name: str) -> Optional[int]:
        """
        Return the version number with the highest average performance score
        across all recorded outcomes. Returns None if no metrics exist yet.

        This is the core feed for the self-optimising loop:
            v = registry.best_version(task_name)
            if v:
                registry.promote(task_name, v)
        """
        with self._conn() as conn:
            row = conn.execute(
                """SELECT version, AVG(score) as avg_score
                   FROM task_metrics
                   WHERE task_name = ? AND success = 1
                   GROUP BY version
                   ORDER BY avg_score DESC
                   LIMIT 1""",
                (task_name,)
            ).fetchone()
        return int(row["version"]) if row else None

    def auto_promote_best(self, task_name: str) -> Optional[int]:
        """
        Convenience: find the best version and promote it as current.
        Returns the promoted version, or None if no metrics exist.
        This is the one-liner for the self-optimisation loop.
        """
        best = self.best_version(task_name)
        if best is not None:
            self.promote(task_name, best)
        return best

    def list_tasks(self) -> list[dict]:
        """
        Return a summary list of all tasks in the registry:
            task_name, current_version, total_versions, last_updated, best_score
        """
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT
                     s.task_name,
                     s.version              AS current_version,
                     s.created_at           AS last_updated,
                     COUNT(s2.version)      AS total_versions,
                     COALESCE(MAX(m.score), NULL) AS best_score
                   FROM snapshots s
                   JOIN snapshots s2 ON s2.task_name = s.task_name
                   LEFT JOIN task_metrics m ON m.task_name = s.task_name
                   WHERE s.is_current = 1
                   GROUP BY s.task_name
                   ORDER BY s.task_name"""
            ).fetchall()
        return [dict(r) for r in rows]

    def history(self, task_name: str) -> list[SnapshotRecord]:
        """
        Return all snapshot versions for a task, newest first,
        each annotated with its best metrics.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM snapshots WHERE task_name = ? ORDER BY version DESC",
                (task_name,)
            ).fetchall()
            records = []
            for row in rows:
                metrics = self._best_metrics(conn, task_name, row["version"])
                records.append(SnapshotRecord(
                    id         = row["id"],
                    task_name  = row["task_name"],
                    version    = row["version"],
                    snapshot   = json.loads(row["snapshot_json"]),
                    created_at = row["created_at"],
                    is_current = bool(row["is_current"]),
                    metrics    = metrics,
                    score      = metrics.score() if metrics else None,
                ))
        return records

    def prune(self, task_name: str, keep: int = 5) -> int:
        """
        Delete old versions for a task, keeping only the `keep` most recent.
        Never deletes the current version. Returns the number of rows deleted.
        """
        with self._conn() as conn:
            # Get IDs to keep (newest `keep` versions, always include current)
            rows = conn.execute(
                """SELECT id FROM snapshots WHERE task_name = ?
                   ORDER BY version DESC LIMIT ?""",
                (task_name, keep)
            ).fetchall()
            keep_ids = [r["id"] for r in rows]
            if not keep_ids:
                return 0
            placeholders = ",".join("?" * len(keep_ids))
            result = conn.execute(
                f"DELETE FROM snapshots WHERE task_name = ? AND id NOT IN ({placeholders})",
                [task_name] + keep_ids
            )
        return result.rowcount

    def delete_task(self, task_name: str) -> None:
        """Remove all snapshots and metrics for a task. Irreversible."""
        with self._conn() as conn:
            conn.execute("DELETE FROM snapshots WHERE task_name = ?", (task_name,))
            conn.execute("DELETE FROM task_metrics WHERE task_name = ?", (task_name,))

    # ── Debug / inspection ──────────────────────────────────────────────────

    def __repr__(self) -> str:
        tasks = self.list_tasks()
        return (
            f"SnapshotRegistry(db='{self.db_path}', "
            f"tasks={len(tasks)}, "
            f"task_names={[t['task_name'] for t in tasks]})"
        )


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, os

    print("=== KSA Registry Demo ===\n")

    # Use a temp DB so the demo is self-contained
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        registry = SnapshotRegistry(db_path)

        # ── 1. Save initial snapshots for two tasks ─────────────────────────
        print("1. Saving initial snapshots…")

        sys_index = ThreeBarSystem.from_defaults()
        sys_index.levers[0].set_weights(left=7.0, right=2.0)
        v1 = registry.save("file_index_stealth", sys_index)
        print(f"   file_index_stealth  → v{v1}")

        sys_search = ThreeBarSystem.from_defaults()
        sys_search.levers[0].set_weights(left=3.0, right=6.0)
        v1s = registry.save("local_search", sys_search)
        print(f"   local_search        → v{v1s}")

        # ── 2. Record outcomes ───────────────────────────────────────────────
        print("\n2. Recording outcomes…")

        m1 = PerformanceMetrics(
            execution_time_ms=420.0, cpu_peak_pct=18.0,
            ram_peak_mb=112.0, success=True, override_fired=False,
            notes="First run, slight lag"
        )
        registry.record_outcome("file_index_stealth", v1, m1)
        print(f"   v1 score: {m1.score():.4f}")

        # ── 3. Save an improved version ──────────────────────────────────────
        print("\n3. Saving improved snapshot (longer left arm, tuned bias)…")
        sys_index.levers[1].left_arm_length = 1.4
        sys_index.levers[2].fulcrum_bias    = 0.8
        v2 = registry.save("file_index_stealth", sys_index)
        print(f"   file_index_stealth  → v{v2}")

        m2 = PerformanceMetrics(
            execution_time_ms=210.0, cpu_peak_pct=12.0,
            ram_peak_mb=98.0, success=True, override_fired=False,
            notes="Tuned arms — faster, lower CPU"
        )
        registry.record_outcome("file_index_stealth", v2, m2)
        print(f"   v2 score: {m2.score():.4f}")

        # ── 4. Auto-promote best ─────────────────────────────────────────────
        print("\n4. Auto-promoting best version…")
        best = registry.auto_promote_best("file_index_stealth")
        print(f"   Best version promoted: v{best}")

        # ── 5. Load and simulate ─────────────────────────────────────────────
        print("\n5. Loading current snapshot and simulating…")
        loaded = registry.load("file_index_stealth")
        result = loaded.simulate()
        print(result)

        # ── 6. Rollback ──────────────────────────────────────────────────────
        print("6. Rolling back to previous version…")
        rolled_to = registry.rollback("file_index_stealth")
        print(f"   Now at: v{rolled_to}")

        # ── 7. History ───────────────────────────────────────────────────────
        print("\n7. Version history for 'file_index_stealth':")
        for rec in registry.history("file_index_stealth"):
            current_marker = " ◀ current" if rec.is_current else ""
            score_str = f"score={rec.score:.4f}" if rec.score else "no metrics yet"
            print(f"   v{rec.version}  {score_str}{current_marker}")

        # ── 8. List all tasks ────────────────────────────────────────────────
        print("\n8. All tasks in registry:")
        for t in registry.list_tasks():
            print(
                f"   {t['task_name']:30s} "
                f"current=v{t['current_version']}  "
                f"total_versions={t['total_versions']}  "
                f"best_score={t['best_score']}"
            )

        # ── 9. Prune ─────────────────────────────────────────────────────────
        print("\n9. Pruning (keep=1)…")
        deleted = registry.prune("file_index_stealth", keep=1)
        print(f"   Deleted {deleted} old version(s)")
        print(f"   Versions remaining: "
              f"{[r.version for r in registry.history('file_index_stealth')]}")

        print(f"\n{registry}")

    finally:
        os.unlink(db_path)
        print("\nTemp DB cleaned up. ✓")
```

### 3C. ksa_router.py

```python
"""
ksa_router.py
=============
Kinetic State Agent — Master Fulcrum (Router)

The entry point of the KSA system. Parses a user prompt or OS event string,
resolves it to a registered task name, then loads and returns the best-known
ThreeBarSystem for that task from the SnapshotRegistry.

Resolution strategy (fast-path first):
    1. Keyword scoring  — zero-LLM, sub-millisecond, deterministic
    2. LLM resolver     — optional Ollama slot, only called when keyword
                          confidence is below threshold
    3. Bootstrap        — if no match at all, create a default system,
                          save it to the registry, and return it

Usage:
    router = MasterFulcrum(registry)
    router.register_intent(
        task_name = "file_index_stealth",
        keywords  = ["index", "scan", "files", "directory", "stealth", "background"],
    )

    result = router.route("quietly index my project folder")
    print(result.task_name)       # "file_index_stealth"
    print(result.method)          # "keyword"
    result.system.simulate()
"""

from __future__ import annotations

import re
import sys
import os
import time
import json
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(__file__))
from ksa_lever import ThreeBarSystem
from ksa_registry import SnapshotRegistry, PerformanceMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IntentPattern:
    """
    A registered mapping from a task name to a set of keywords and
    an optional pre-configured default ThreeBarSystem to bootstrap with.
    """
    task_name:      str
    keywords:       list[str]           # matched case-insensitively
    aliases:        list[str] = field(default_factory=list)  # exact task-name aliases
    default_system: Optional[ThreeBarSystem] = None
    description:    str = ""


@dataclass
class RouteResult:
    """
    The fully resolved output of a route() call.
    Contains everything needed to start execution.
    """
    task_name:   str
    system:      ThreeBarSystem
    version:     int
    confidence:  float              # 0.0–1.0 keyword match ratio
    method:      str                # "keyword" | "llm" | "bootstrap" | "alias"
    elapsed_ms:  float
    prompt_raw:  str

    def __str__(self) -> str:
        return (
            f"RouteResult("
            f"task='{self.task_name}', "
            f"v{self.version}, "
            f"method={self.method}, "
            f"conf={self.confidence:.0%}, "
            f"{self.elapsed_ms:.1f}ms)"
        )


# ---------------------------------------------------------------------------
# MasterFulcrum
# ---------------------------------------------------------------------------

class MasterFulcrum:
    """
    The top-level lever of the KSA cascade.

    Receives raw intent strings and returns a ready-to-simulate ThreeBarSystem
    loaded with the best snapshot for the resolved task. Acts as the single
    choke-point between the outside world and the physics engine.

    Keyword scoring:
        Each registered keyword that appears in the normalised prompt adds
        1 point. The task with the most points wins. Ties are broken by
        pattern registration order (first registered wins). Confidence is
        (matched_keywords / total_keywords_in_pattern).

    LLM slot (optional):
        Set llm_resolver to any callable that accepts a prompt str and
        returns a task_name str (or None). The built-in Ollama helper is
        provided as MasterFulcrum.ollama_resolver(model, host).

    Bootstrap:
        If no pattern matches and no LLM result, a default ThreeBarSystem
        is created for the inferred task name (snake_cased prompt), saved
        to the registry, and returned. Confidence = 0.0, method = "bootstrap".
    """

    # Keyword confidence must meet this to skip the LLM slot
    KEYWORD_CONFIDENCE_THRESHOLD: float = 0.25

    def __init__(
        self,
        registry:        SnapshotRegistry,
        llm_resolver:    Optional[Callable[[str], Optional[str]]] = None,
        confidence_floor: float = KEYWORD_CONFIDENCE_THRESHOLD,
    ):
        self.registry          = registry
        self.llm_resolver      = llm_resolver
        self.confidence_floor  = confidence_floor
        self._patterns:  list[IntentPattern] = []
        self._alias_map: dict[str, str]      = {}  # alias → canonical task_name

    # ── Pattern registration ─────────────────────────────────────────────────

    def register_intent(
        self,
        task_name:      str,
        keywords:       list[str],
        aliases:        Optional[list[str]]       = None,
        default_system: Optional[ThreeBarSystem]  = None,
        description:    str                       = "",
    ) -> None:
        """
        Register a task name with its intent keywords.

        Args:
            task_name:      Canonical name used as the registry key.
            keywords:       Words/phrases that signal this task in a prompt.
                            Matched case-insensitively, whole-word preferred.
            aliases:        Exact synonyms for task_name (e.g. short names).
            default_system: ThreeBarSystem to save if no snapshot exists yet.
                            If None, ThreeBarSystem.from_defaults() is used.
            description:    Human-readable purpose of this task.
        """
        pattern = IntentPattern(
            task_name      = task_name,
            keywords       = [k.lower() for k in keywords],
            aliases        = [a.lower() for a in (aliases or [])],
            default_system = default_system,
            description    = description,
        )
        self._patterns.append(pattern)
        for alias in pattern.aliases:
            self._alias_map[alias] = task_name
        logger.debug("Registered intent: %s (%d keywords)", task_name, len(keywords))

    def unregister_intent(self, task_name: str) -> bool:
        """Remove a registered intent pattern. Returns True if found."""
        before = len(self._patterns)
        self._patterns = [p for p in self._patterns if p.task_name != task_name]
        self._alias_map = {k: v for k, v in self._alias_map.items() if v != task_name}
        return len(self._patterns) < before

    # ── Routing ──────────────────────────────────────────────────────────────

    def route(self, prompt: str) -> RouteResult:
        """
        Resolve a prompt to a task and return a hot-swapped RouteResult.

        Resolution order:
            1. Alias exact match   (instant, O(1))
            2. Keyword scoring     (O(patterns × keywords))
            3. LLM resolver        (only if keyword confidence < floor)
            4. Bootstrap           (last resort)
        """
        t0 = time.perf_counter()
        normalised = self._normalise(prompt)

        # ── 1. Alias check ───────────────────────────────────────────────────
        for token in normalised.split():
            if token in self._alias_map:
                task_name = self._alias_map[token]
                system, version = self._load_or_bootstrap(task_name)
                return RouteResult(
                    task_name  = task_name,
                    system     = system,
                    version    = version,
                    confidence = 1.0,
                    method     = "alias",
                    elapsed_ms = (time.perf_counter() - t0) * 1000,
                    prompt_raw = prompt,
                )

        # ── 2. Keyword scoring ───────────────────────────────────────────────
        best_task, best_conf = self._keyword_score(normalised)

        if best_task and best_conf >= self.confidence_floor:
            system, version = self._load_or_bootstrap(best_task)
            return RouteResult(
                task_name  = best_task,
                system     = system,
                version    = version,
                confidence = best_conf,
                method     = "keyword",
                elapsed_ms = (time.perf_counter() - t0) * 1000,
                prompt_raw = prompt,
            )

        # ── 3. LLM resolver (optional slot) ─────────────────────────────────
        if self.llm_resolver is not None:
            try:
                llm_task = self.llm_resolver(prompt)
                if llm_task:
                    system, version = self._load_or_bootstrap(llm_task)
                    return RouteResult(
                        task_name  = llm_task,
                        system     = system,
                        version    = version,
                        confidence = 0.6,   # LLM result is treated as moderate confidence
                        method     = "llm",
                        elapsed_ms = (time.perf_counter() - t0) * 1000,
                        prompt_raw = prompt,
                    )
            except Exception as exc:
                logger.warning("LLM resolver failed: %s — falling back to bootstrap", exc)

        # ── 4. Bootstrap: infer task name from prompt, create default ────────
        inferred_name = self._infer_task_name(normalised)
        system, version = self._load_or_bootstrap(inferred_name)
        return RouteResult(
            task_name  = inferred_name,
            system     = system,
            version    = version,
            confidence = 0.0,
            method     = "bootstrap",
            elapsed_ms = (time.perf_counter() - t0) * 1000,
            prompt_raw = prompt,
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _normalise(text: str) -> str:
        """Lowercase, strip punctuation, collapse whitespace."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _keyword_score(self, normalised: str) -> tuple[Optional[str], float]:
        """
        Score every registered pattern against the normalised prompt.
        Returns (best_task_name, confidence) or (None, 0.0).

        Confidence = matched_count / total_keywords_in_pattern
        (capped at 1.0 so patterns with many keywords aren't penalised).
        """
        tokens = set(normalised.split())
        best_task  = None
        best_score = 0.0
        best_conf  = 0.0

        for pattern in self._patterns:
            if not pattern.keywords:
                continue
            matched = sum(
                1 for kw in pattern.keywords
                if kw in tokens or kw in normalised   # whole-word + substring
            )
            if matched == 0:
                continue
            score = matched                                   # raw match count
            conf  = matched / len(pattern.keywords)           # normalised confidence
            if score > best_score or (score == best_score and conf > best_conf):
                best_score = score
                best_conf  = conf
                best_task  = pattern.task_name

        return best_task, best_conf

    def _load_or_bootstrap(self, task_name: str) -> tuple[ThreeBarSystem, int]:
        """
        Try to load the current snapshot for task_name from the registry.
        If not found, create a default system, save it, and return it.
        """
        try:
            system = self.registry.load(task_name)
            # Retrieve the current version number
            tasks = {t["task_name"]: t for t in self.registry.list_tasks()}
            version = tasks[task_name]["current_version"] if task_name in tasks else 1
            return system, version
        except KeyError:
            return self._bootstrap(task_name)

    def _bootstrap(self, task_name: str) -> tuple[ThreeBarSystem, int]:
        """
        No snapshot exists. Find the registered default_system for this task
        (if any), otherwise use ThreeBarSystem.from_defaults(). Save and return.
        """
        default = None
        for pattern in self._patterns:
            if pattern.task_name == task_name and pattern.default_system is not None:
                default = pattern.default_system
                break
        system  = default if default is not None else ThreeBarSystem.from_defaults()
        version = self.registry.save(task_name, system)
        logger.info("Bootstrapped new snapshot for task '%s' at v%d", task_name, version)
        return system, version

    @staticmethod
    def _infer_task_name(normalised: str) -> str:
        """
        Derive a snake_case task name from the first 4 significant words
        of the normalised prompt. Used as a last-resort registry key.
        """
        stop_words = {"the", "a", "an", "my", "me", "i", "it", "is", "in",
                      "on", "of", "to", "for", "and", "or", "please", "can",
                      "you", "with", "at", "this", "that", "do", "run"}
        words = [w for w in normalised.split() if w not in stop_words][:4]
        return "_".join(words) if words else "unknown_task"

    # ── Built-in LLM resolver: Ollama ────────────────────────────────────────

    @staticmethod
    def ollama_resolver(
        model: str = "mistral",
        host:  str = "http://localhost:11434",
        registered_tasks: Optional[list[str]] = None,
    ) -> Callable[[str], Optional[str]]:
        """
        Factory that returns an LLM resolver using a local Ollama instance.

        The resolver sends the prompt + registered task list to Ollama and
        asks it to return only the single best-matching task name.

        Args:
            model:             Ollama model tag (e.g. "mistral", "llama3").
            host:              Ollama server URL.
            registered_tasks:  If provided, included in the system prompt so
                               the LLM can pick from the known list.

        Returns a callable: prompt:str → task_name:str | None
        """
        try:
            import urllib.request
        except ImportError:
            raise RuntimeError("urllib.request not available.")

        task_list_str = (
            "\n".join(f"  - {t}" for t in registered_tasks)
            if registered_tasks else "  (none registered yet)"
        )

        def resolver(prompt: str) -> Optional[str]:
            system_msg = (
                "You are a task classifier for a local AI agent. "
                "Given a user prompt, return ONLY the single most appropriate "
                "task name from the list below, as a bare snake_case string. "
                "No explanation, no punctuation, no quotes — just the task name.\n\n"
                f"Registered tasks:\n{task_list_str}"
            )
            payload = json.dumps({
                "model":  model,
                "prompt": f"{system_msg}\n\nUser prompt: {prompt}",
                "stream": False,
            }).encode()
            req = urllib.request.Request(
                f"{host}/api/generate",
                data    = payload,
                headers = {"Content-Type": "application/json"},
                method  = "POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data     = json.loads(resp.read())
                raw      = data.get("response", "").strip().lower()
                cleaned  = re.sub(r"[^\w]", "_", raw).strip("_")
                return cleaned if cleaned else None

        return resolver

    # ── Inspection ───────────────────────────────────────────────────────────

    def list_intents(self) -> list[dict]:
        """Return a summary of all registered intents."""
        return [
            {
                "task_name":   p.task_name,
                "keywords":    p.keywords,
                "aliases":     p.aliases,
                "description": p.description,
            }
            for p in self._patterns
        ]

    def __repr__(self) -> str:
        llm = "Ollama" if self.llm_resolver else "none"
        return (
            f"MasterFulcrum("
            f"patterns={len(self._patterns)}, "
            f"llm={llm}, "
            f"floor={self.confidence_floor:.0%})"
        )


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, os

    print("=== KSA Master Fulcrum Demo ===\n")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        registry = SnapshotRegistry(db_path)
        router   = MasterFulcrum(registry)

        # ── Register task intents ────────────────────────────────────────────
        router.register_intent(
            task_name   = "file_index_stealth",
            keywords    = ["index", "scan", "files", "directory", "folder",
                           "stealth", "background", "quiet", "silently"],
            aliases     = ["index"],
            description = "Background file indexing without UI interference",
        )
        router.register_intent(
            task_name   = "local_search",
            keywords    = ["search", "find", "locate", "grep", "query", "lookup"],
            aliases     = ["search", "find"],
            description = "Low-priority local file/content search",
        )
        router.register_intent(
            task_name   = "code_gen_assist",
            keywords    = ["write", "generate", "code", "function", "class",
                           "script", "implement", "draft"],
            aliases     = ["codegen"],
            description = "Code generation with local LLM assist",
        )

        print(f"Router: {router}\n")

        # ── Route several prompts ────────────────────────────────────────────
        prompts = [
            "quietly scan my project directory in the background",
            "find all TODO comments in my codebase",
            "write a Python function to parse JSON logs",
            "index",                                            # alias exact match
            "do something completely unrecognised please",      # bootstrap path
        ]

        for p in prompts:
            result = router.route(p)
            print(f"  Prompt : {p!r}")
            print(f"  Result : {result}")
            eq = result.system.simulate()
            print(f"  Decision: {eq.final_tilt.value.upper()} "
                  f"(confidence {eq.confidence:.0%})\n")

        # ── Inspect registry after routing ───────────────────────────────────
        print("Registry tasks after routing:")
        for t in registry.list_tasks():
            print(f"  {t['task_name']:35s} v{t['current_version']}")

    finally:
        os.unlink(db_path)
        print("\nTemp DB cleaned up. ✓")
```

---

## 4. Modules to build

---

### 4A. ksa_executor.py — Hardware Execution Layer

**Purpose:** Receives an `EquilibriumResult` from the physics engine and
converts the final tilt + confidence into a concrete local system action.
This is the only KSA layer that touches the host OS.

**Routing rules:**
- `TiltDirection.LEFT`     -> call `executor.primary(ctx)`
- `TiltDirection.RIGHT`    -> call `executor.secondary(ctx)`
- `TiltDirection.BALANCED` -> call `executor.safe(ctx)`
- `override_active=True`   -> always call `executor.safe(ctx)` regardless of tilt

**Every executor action must be non-blocking** (subprocess or thread).
CPU and RAM must be sampled with psutil during execution and returned
as a `PerformanceMetrics` object.

**Required signatures:**

```python
from __future__ import annotations
import subprocess, threading, time, logging
from dataclasses import dataclass
from typing import Optional
import psutil
from ksa_lever import EquilibriumResult, TiltDirection
from ksa_registry import SnapshotRegistry, PerformanceMetrics

@dataclass
class ExecutionContext:
    task_name:   str
    version:     int
    result:      EquilibriumResult
    working_dir: str  = "."
    dry_run:     bool = False   # log actions but do not execute

@dataclass
class ExecutionOutcome:
    task_name:    str
    version:      int
    action_taken: str            # "primary" | "secondary" | "safe"
    return_code:  int
    stdout:       str
    stderr:       str
    metrics:      PerformanceMetrics
    elapsed_ms:   float

class TaskExecutor:
    task_name: str               # must match router intent task_name

    def primary(self, ctx: ExecutionContext) -> ExecutionOutcome: ...
    def secondary(self, ctx: ExecutionContext) -> ExecutionOutcome: ...
    def safe(self, ctx: ExecutionContext) -> ExecutionOutcome: ...

class ExecutorRegistry:
    def __init__(self, registry: SnapshotRegistry): ...
    def register(self, executor: TaskExecutor) -> None: ...
    def execute(self, ctx: ExecutionContext) -> ExecutionOutcome:
        # routes ctx.result -> primary/secondary/safe
        # records outcome to SnapshotRegistry after execution
        ...
```

**Concrete executors to implement:**

1. `FileIndexExecutor` (task_name = "file_index_stealth")
   - primary:   `nice -n 19 find {working_dir} -type f > .ksa_index.txt`
   - secondary: index current directory only, not recursive
   - safe:      skip, log a warning

2. `LocalSearchExecutor` (task_name = "local_search")
   - primary:   `grep -rl {query} {working_dir}` (query extracted from context)
   - secondary: filename-only search (`find . -name "*{query}*"`)
   - safe:      return contents of `.ksa_index.txt` if it exists

3. `ShellExecutor` (task_name = "shell_generic", generic fallback)
   - Reads a shell command string from ctx (passed via working_dir convention)
   - primary:   run the command as-is
   - secondary: run with `timeout 5` prefix
   - safe:      no-op, return empty outcome

**psutil sampling thread pattern:**

```python
class _ResourceSampler:
    def __init__(self):
        self.cpu_peak = 0.0
        self.ram_peak_mb = 0.0
        self._stop = threading.Event()

    def start(self):
        threading.Thread(target=self._sample, daemon=True).start()

    def stop(self):
        self._stop.set()

    def _sample(self):
        proc = psutil.Process()
        while not self._stop.is_set():
            self.cpu_peak = max(self.cpu_peak, psutil.cpu_percent(interval=None))
            self.ram_peak_mb = max(
                self.ram_peak_mb,
                proc.memory_info().rss / 1024 / 1024
            )
            time.sleep(0.5)
```

---

### 4B. ksa_optimizer.py — Self-Optimising Loop

**Purpose:** After a task run, if the new `PerformanceMetrics.score()`
improves on the best historical score by at least 2%, perturb the lever
arm lengths and biases using Gaussian noise and save the improved
snapshot as a new registry version.

**Rules:**
- Gradient-free only. No backpropagation.
- Arm lengths must stay in [0.5, 4.0]. Biases in [-2.0, 2.0].
- Never mutate Lever 2 (the Balancer) — its geometry is fixed.
- Only mutate on SUCCESS runs where `override_active` was False.
- After saving, call `registry.auto_promote_best(task_name)`.

**Required signatures:**

```python
from __future__ import annotations
import copy, random, logging
from typing import Optional
from ksa_lever import ThreeBarSystem
from ksa_registry import SnapshotRegistry
from ksa_executor import ExecutionOutcome

class KineticOptimizer:
    def __init__(
        self,
        registry:       SnapshotRegistry,
        step_size:      float = 0.05,
        max_arm_length: float = 4.0,
        min_arm_length: float = 0.5,
        max_bias:       float = 2.0,
        min_bias:       float = -2.0,
        improvement_threshold: float = 0.02,  # 2% minimum improvement
    ): ...

    def maybe_improve(
        self,
        task_name: str,
        version:   int,
        outcome:   ExecutionOutcome,
    ) -> Optional[int]:
        """
        If outcome.metrics.score() exceeds historical best by improvement_threshold,
        perturb lever 0 and lever 1 geometry using Gaussian noise (mean=0,
        std=step_size), clamp to bounds, save new snapshot, auto-promote.
        Returns new version number or None.
        """
        ...

    def hill_climb(
        self,
        task_name: str,
        n_trials:  int  = 5,
        dry_run:   bool = False,
    ) -> ThreeBarSystem:
        """
        Generate n_trials random mutations of the current best snapshot,
        simulate each (no executor), return the one with highest
        EquilibriumResult.confidence. Save winner to registry unless dry_run.
        """
        ...
```

---

### 4C. ksa_agent.py — Orchestrator

**Purpose:** Single top-level object that wires the full pipeline:
router -> physics engine -> executor -> optimizer.

**Required signatures:**

```python
from __future__ import annotations
from typing import Optional
from ksa_lever import ThreeBarSystem
from ksa_registry import SnapshotRegistry
from ksa_router import MasterFulcrum
from ksa_executor import TaskExecutor, ExecutorRegistry, ExecutionOutcome, ExecutionContext
from ksa_optimizer import KineticOptimizer

class KSAgent:
    def __init__(
        self,
        db_path:       str  = "ksa_state.db",
        working_dir:   str  = ".",
        ollama_model:  Optional[str] = None,
        ollama_host:   str  = "http://localhost:11434",
        auto_optimise: bool = True,
        dry_run:       bool = False,
    ): ...

    def register(
        self,
        task_name:      str,
        keywords:       list[str],
        executor:       TaskExecutor,
        aliases:        Optional[list[str]]      = None,
        default_system: Optional[ThreeBarSystem] = None,
        description:    str = "",
    ) -> None:
        """Register both intent (router) and executor in one call."""
        ...

    def run(self, prompt: str) -> ExecutionOutcome:
        """
        Full pipeline:
          1. self.router.route(prompt)         -> RouteResult
          2. result.system.simulate()          -> EquilibriumResult
          3. self.executor_registry.execute()  -> ExecutionOutcome
          4. self.registry.record_outcome()
          5. if auto_optimise: self.optimizer.maybe_improve()
          6. return outcome
        """
        ...

    def status(self) -> dict:
        """Return registry task list and router intent summary."""
        ...
```

**Example usage to put in module docstring:**

```python
agent = KSAgent(db_path="~/.ksa/state.db", auto_optimise=True)

agent.register(
    task_name   = "file_index_stealth",
    keywords    = ["index", "scan", "files", "directory", "background"],
    executor    = FileIndexExecutor(),
    aliases     = ["index"],
    description = "Background file indexing without UI lag",
)

outcome = agent.run("quietly index my project folder in the background")
print(outcome.action_taken)     # "primary"
print(outcome.metrics.score())  # e.g. 4.76
```

---

### 4D. ksa_cli.py — CLI Entry Point

**Purpose:** `python -m ksa` or `ksa` CLI command. Use `argparse`.

**Commands:**

```
ksa run "<prompt>"               Run a prompt through the full pipeline
ksa list                         List all tasks + current versions
ksa history <task_name>          Show version history + scores
ksa rollback <task_name>         Revert to previous snapshot version
ksa prune <task_name> [--keep N] Prune old versions (default N=5)
ksa hill-climb <task_name> [--trials N]  Run hill-climb optimiser
ksa export <task_name>           Print snapshot JSON to stdout
ksa import <task_name> <file>    Load snapshot JSON from file
```

**Global flags:**

```
--db <path>    Path to ksa_state.db (default: ~/.ksa/state.db)
--dry-run      Log but do not execute
--verbose      Set logging to DEBUG
--config <path>  Load intents from ksa_config.toml
```

---

### 4E. ksa_config.py — Config Loader

**Purpose:** Load agent configuration from `ksa_config.toml` or
`ksa_config.json` without requiring code changes.

**Config schema (TOML):**

```toml
[agent]
db_path       = "~/.ksa/state.db"
working_dir   = "~/projects"
auto_optimise = true
ollama_model  = "mistral"   # omit to disable LLM routing

[[intents]]
task_name   = "file_index_stealth"
keywords    = ["index", "scan", "files", "directory", "background"]
aliases     = ["index"]
executor    = "FileIndexExecutor"
description = "Background file indexing"

[[intents]]
task_name   = "local_search"
keywords    = ["search", "find", "locate", "grep"]
aliases     = ["search"]
executor    = "LocalSearchExecutor"
description = "Low-priority content search"
```

**Required functions:**

```python
def load_config(path: str) -> dict:
    """Load and validate a TOML or JSON config file. Return dict."""
    ...

def build_agent_from_config(path: str) -> KSAgent:
    """
    Build a fully configured KSAgent from a config file.
    Map executor strings (e.g. 'FileIndexExecutor') to concrete classes
    in ksa_executor.py.
    """
    ...
```

---

## 5. Test suite

Write pytest tests. Use `tmp_path` for all file/DB operations.
Never touch the real `~/.ksa/` directory. Run with `pytest tests/ -v`.

### tests/test_lever.py
- test_tilt_left: left_weight > right_weight -> TiltDirection.LEFT
- test_tilt_right: right > left -> RIGHT
- test_balanced: equal weights -> BALANCED
- test_balancer_override: extreme weights -> override_active=True, BALANCED
- test_snapshot_roundtrip: snapshot() + hydrate() -> same simulate() result
- test_linkage_propagation: coupling value propagates weight to correct side of downstream lever

### tests/test_registry.py
- test_save_and_load: save a system, load it, simulate(), result matches
- test_versioning: second save -> version=2 is_current, version=1 demoted
- test_auto_promote_best: two versions with different scores -> best promoted
- test_rollback: rollback after v2 -> v1 is_current=True
- test_prune: after prune(keep=1), only 1 version row remains
- test_record_outcome_score: faster run produces higher score

### tests/test_router.py
- test_keyword_match: matching keyword prompt -> correct task_name, method="keyword"
- test_alias_match: single alias word -> task_name, method="alias"
- test_bootstrap: unrecognised prompt -> method="bootstrap", new snapshot saved
- test_confidence_floor: low-confidence prompt falls to bootstrap when no LLM
- test_normalisation: punctuation/uppercase do not affect routing result

### tests/test_executor.py
- test_dry_run_no_side_effects: dry_run=True -> no files created, return_code=0
- test_primary_on_left_tilt: LEFT EquilibriumResult -> primary() called
- test_safe_on_override: override_active=True -> safe() called regardless of tilt
- test_metrics_populated: after real execution, metrics.execution_time_ms > 0
- test_psutil_peak_captured: cpu_peak_pct and ram_peak_mb are non-negative floats

### tests/test_optimizer.py
- test_no_mutation_below_threshold: score improvement < 2% -> None returned
- test_mutation_creates_new_version: clear improvement -> new version int returned
- test_lever2_never_mutated: after maybe_improve, lever 2 arm lengths unchanged
- test_arm_lengths_clamped: after many mutations, all arm lengths in [0.5, 4.0]
- test_hill_climb_returns_system: hill_climb() returns ThreeBarSystem with simulate() working

---

## 6. Dependencies

Standard library only, except:
- `psutil`  -- CPU/RAM sampling in ksa_executor.py
- `tomllib` (Python 3.11+ stdlib) or `tomli` -- TOML config in ksa_config.py

Install: `pip install psutil tomli`

Do NOT add: numpy, scipy, pandas, langchain, openai, anthropic, or any ML
framework. The LLM integration uses only `urllib.request` (stdlib) to POST
to a local Ollama server.

---

## 7. Coding standards

- Python 3.11+. All files start with: `from __future__ import annotations`
- Type hints on all public functions and class attributes.
- Docstrings on all public classes and methods (one-line or Google style).
- No global mutable state outside class instances.
- All file/DB I/O in try/except with informative error messages.
- Logging via `logging.getLogger(__name__)`. Never `print()` in library code.
- `print()` only in `if __name__ == "__main__":` demo blocks and ksa_cli.py.
- Each module must be runnable standalone as `python ksa_xxx.py` with a
  working `__main__` demo block.

---

## 8. Build order (recommended for Copilot session)

Build files in this exact order so each can import the previous:

  Step 1: ksa_executor.py   (imports ksa_lever, ksa_registry)
  Step 2: ksa_optimizer.py  (imports ksa_lever, ksa_registry, ksa_executor)
  Step 3: ksa_agent.py      (imports all above + ksa_router)
  Step 4: ksa_config.py     (imports ksa_agent)
  Step 5: ksa_cli.py        (imports ksa_agent, ksa_config)
  Step 6: tests/            (imports everything)

At the start of each step, re-read the files in Section 3 to confirm
all interface names and signatures before writing new code.
