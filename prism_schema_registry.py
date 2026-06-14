"""
prism_schema_registry.py — Centralized SQLite schema documentation.

All 22 PRISM databases under ~/.prism/ are listed here with their schemas,
migration notes, and owning module.  This is a single source of truth for
schema discovery and future migration tooling.

Migration approach per module:
  - prism_soul.py, prism_memory.py  → PRAGMA user_version + ALTER TABLE
  - most others                      → CREATE TABLE IF NOT EXISTS (additive-only)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ColumnDef:
    name:    str
    type:    str
    constraints: str = ""


@dataclass
class TableDef:
    name:    str
    columns: list[ColumnDef]
    notes:   str = ""


@dataclass
class DbSchema:
    db_path:  str               # relative to ~/.prism/
    owner:    str               # module that owns (and migrates) this DB
    tables:   list[TableDef]
    version:  int = 0           # PRAGMA user_version; 0 = unversioned
    notes:    str = ""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: list[DbSchema] = [

    DbSchema(
        db_path="memory.db",
        owner="prism_memory",
        version=2,
        tables=[
            TableDef("memories", [
                ColumnDef("id",        "INTEGER", "PRIMARY KEY AUTOINCREMENT"),
                ColumnDef("text",      "TEXT",    "NOT NULL"),
                ColumnDef("embedding", "BLOB"),
                ColumnDef("source",    "TEXT"),
                ColumnDef("ts",        "REAL"),
                ColumnDef("meta",      "TEXT"),
            ]),
        ],
        notes="Ollama nomic-embed-text embeddings; BM25 fallback when Ollama offline.",
    ),

    DbSchema(
        db_path="soul.db",
        owner="prism_soul",
        version=3,
        tables=[
            TableDef("beliefs", [
                ColumnDef("id",          "TEXT",    "PRIMARY KEY"),
                ColumnDef("text",        "TEXT",    "NOT NULL"),
                ColumnDef("belief_type", "TEXT"),
                ColumnDef("confidence",  "REAL"),
                ColumnDef("source",      "TEXT"),
                ColumnDef("created_at",  "REAL"),
                ColumnDef("updated_at",  "REAL"),
            ]),
            TableDef("edges", [
                ColumnDef("from_id",      "TEXT", "REFERENCES beliefs(id)"),
                ColumnDef("to_id",        "TEXT", "REFERENCES beliefs(id)"),
                ColumnDef("relation",     "TEXT"),
                ColumnDef("strength",     "REAL"),
            ]),
            TableDef("lenses", [
                ColumnDef("lens_id",     "TEXT",    "PRIMARY KEY"),
                ColumnDef("name",        "TEXT"),
                ColumnDef("filters",     "TEXT"),
                ColumnDef("created_at",  "REAL"),
            ]),
            TableDef("seed", [
                ColumnDef("key",   "TEXT", "PRIMARY KEY"),
                ColumnDef("value", "TEXT"),
            ]),
        ],
        notes="Uses PRAGMA user_version for migrations. Bayesian confidence updates via OutcomeTracker.",
    ),

    DbSchema(
        db_path="chains.db",
        owner="prism_chain",
        tables=[
            TableDef("chain_runs", [
                ColumnDef("run_id",     "TEXT", "PRIMARY KEY"),
                ColumnDef("goal",       "TEXT"),
                ColumnDef("steps",      "TEXT"),
                ColumnDef("result",     "TEXT"),
                ColumnDef("created_at", "REAL"),
            ]),
        ],
    ),

    DbSchema(
        db_path="chains_expert.db",
        owner="prism_chain_expert",
        tables=[
            TableDef("expert_runs", [
                ColumnDef("run_id",     "TEXT", "PRIMARY KEY"),
                ColumnDef("goal",       "TEXT"),
                ColumnDef("result",     "TEXT"),
                ColumnDef("created_at", "REAL"),
            ]),
        ],
    ),

    DbSchema(
        db_path="orchestrator.db",
        owner="prism_orchestrator",
        tables=[
            TableDef("graphs", [
                ColumnDef("graph_id",   "TEXT", "PRIMARY KEY"),
                ColumnDef("goal",       "TEXT"),
                ColumnDef("status",     "TEXT"),
                ColumnDef("nodes",      "TEXT"),
                ColumnDef("created_at", "REAL"),
                ColumnDef("updated_at", "REAL"),
            ]),
        ],
        notes="Serializes TaskGraph DAGs. resume_waiting() loads paused graphs on startup.",
    ),

    DbSchema(
        db_path="outcomes.db",
        owner="prism_outcome_tracker",
        tables=[
            TableDef("outcomes", [
                ColumnDef("outcome_id",  "TEXT",    "PRIMARY KEY"),
                ColumnDef("goal",        "TEXT"),
                ColumnDef("final_answer","TEXT"),
                ColumnDef("correction",  "TEXT"),
                ColumnDef("status",      "TEXT"),
                ColumnDef("created_at",  "REAL"),
            ]),
        ],
        notes="Source for DPO training pairs (correction field). Feeds soul Bayesian updates.",
    ),

    DbSchema(
        db_path="horizon.db",
        owner="prism_horizon",
        tables=[
            TableDef("goals", [
                ColumnDef("goal_id",          "TEXT", "PRIMARY KEY"),
                ColumnDef("trigger_condition", "TEXT"),
                ColumnDef("action",            "TEXT"),
                ColumnDef("status",            "TEXT"),
                ColumnDef("accumulated_context","TEXT"),
                ColumnDef("created_at",        "REAL"),
                ColumnDef("updated_at",        "REAL"),
            ]),
            TableDef("steps", [
                ColumnDef("step_id",   "TEXT", "PRIMARY KEY"),
                ColumnDef("goal_id",   "TEXT", "REFERENCES goals(goal_id)"),
                ColumnDef("summary",   "TEXT"),
                ColumnDef("created_at","REAL"),
            ]),
        ],
    ),

    DbSchema(
        db_path="policy.db",
        owner="prism_policy",
        tables=[
            TableDef("policies", [
                ColumnDef("organ",   "TEXT", "PRIMARY KEY"),
                ColumnDef("policy",  "TEXT"),
                ColumnDef("updated_at","REAL"),
            ]),
        ],
    ),

    DbSchema(
        db_path="policy_audit.db",
        owner="prism_policy",
        tables=[
            TableDef("audit_log", [
                ColumnDef("id",        "INTEGER", "PRIMARY KEY AUTOINCREMENT"),
                ColumnDef("organ",     "TEXT"),
                ColumnDef("action",    "TEXT"),
                ColumnDef("approved",  "INTEGER"),
                ColumnDef("reason",    "TEXT"),
                ColumnDef("created_at","REAL"),
            ]),
        ],
    ),

    DbSchema(
        db_path="organ_bus.db",
        owner="prism_organ_bus",
        tables=[
            TableDef("signals", [
                ColumnDef("signal_id",   "TEXT", "PRIMARY KEY"),
                ColumnDef("source",      "TEXT", "NOT NULL"),
                ColumnDef("signal_type", "TEXT", "NOT NULL"),
                ColumnDef("payload",     "TEXT"),
                ColumnDef("priority",    "INTEGER"),
                ColumnDef("ts",          "REAL"),
            ]),
            TableDef("deliveries", [
                ColumnDef("delivery_id", "TEXT", "PRIMARY KEY"),
                ColumnDef("signal_id",   "TEXT", "REFERENCES signals(signal_id)"),
                ColumnDef("organ_name",  "TEXT"),
                ColumnDef("translated",  "TEXT"),
                ColumnDef("delivered_at","REAL"),
            ]),
        ],
        notes="SQLite replay buffer. LLM translation cache lives in-memory with 300s TTL.",
    ),

    DbSchema(
        db_path="federation.db",
        owner="prism_federation",
        tables=[
            TableDef("peers", [
                ColumnDef("peer_id",      "TEXT", "PRIMARY KEY"),
                ColumnDef("url",          "TEXT"),
                ColumnDef("clock",        "INTEGER"),
                ColumnDef("last_seen",    "REAL"),
            ]),
            TableDef("state_log", [
                ColumnDef("entry_id",  "TEXT", "PRIMARY KEY"),
                ColumnDef("peer_id",   "TEXT"),
                ColumnDef("payload",   "TEXT"),
                ColumnDef("clock",     "INTEGER"),
                ColumnDef("merged_at", "REAL"),
            ]),
        ],
        notes="Lamport vector clock. Additive-only merge — no deletions propagate.",
    ),

    DbSchema(
        db_path="identity.db",
        owner="digital_identity",
        tables=[
            TableDef("profiles", [
                ColumnDef("profile_id", "TEXT", "PRIMARY KEY"),
                ColumnDef("data",       "TEXT"),
                ColumnDef("created_at", "REAL"),
            ]),
        ],
    ),

    DbSchema(
        db_path="identity_bus.db",
        owner="identity_bus",
        tables=[
            TableDef("events", [
                ColumnDef("event_id",  "TEXT",    "PRIMARY KEY"),
                ColumnDef("type",      "TEXT"),
                ColumnDef("payload",   "TEXT"),
                ColumnDef("ts",        "REAL"),
            ]),
        ],
    ),

    DbSchema(
        db_path="artifacts.db",
        owner="artifact_store",
        tables=[
            TableDef("artifacts", [
                ColumnDef("artifact_id", "TEXT", "PRIMARY KEY"),
                ColumnDef("kind",        "TEXT"),
                ColumnDef("data",        "TEXT"),
                ColumnDef("created_at",  "REAL"),
            ]),
        ],
    ),

    DbSchema(
        db_path="bus.db",
        owner="identity_bus",
        tables=[
            TableDef("messages", [
                ColumnDef("msg_id",    "TEXT", "PRIMARY KEY"),
                ColumnDef("topic",     "TEXT"),
                ColumnDef("payload",   "TEXT"),
                ColumnDef("ts",        "REAL"),
            ]),
        ],
    ),

    DbSchema(
        db_path="tasks.db",
        owner="prism_tasks",
        tables=[
            TableDef("tasks", [
                ColumnDef("task_id",    "TEXT", "PRIMARY KEY"),
                ColumnDef("title",      "TEXT"),
                ColumnDef("status",     "TEXT"),
                ColumnDef("due",        "REAL"),
                ColumnDef("created_at", "REAL"),
            ]),
        ],
    ),

    DbSchema(
        db_path="tools.db",
        owner="prism_executor_agent",
        tables=[
            TableDef("tools", [
                ColumnDef("tool_id",     "TEXT", "PRIMARY KEY"),
                ColumnDef("name",        "TEXT"),
                ColumnDef("description", "TEXT"),
                ColumnDef("schema",      "TEXT"),
                ColumnDef("created_at",  "REAL"),
            ]),
        ],
    ),

    DbSchema(
        db_path="executions.db",
        owner="prism_executor_agent",
        tables=[
            TableDef("executions", [
                ColumnDef("exec_id",    "TEXT", "PRIMARY KEY"),
                ColumnDef("tool_id",    "TEXT"),
                ColumnDef("input",      "TEXT"),
                ColumnDef("output",     "TEXT"),
                ColumnDef("status",     "TEXT"),
                ColumnDef("created_at", "REAL"),
            ]),
        ],
    ),

    DbSchema(
        db_path="causality.db",
        owner="prism_causality",
        tables=[
            TableDef("causal_edges", [
                ColumnDef("edge_id",    "TEXT", "PRIMARY KEY"),
                ColumnDef("cause",      "TEXT"),
                ColumnDef("effect",     "TEXT"),
                ColumnDef("strength",   "REAL"),
                ColumnDef("created_at", "REAL"),
            ]),
        ],
    ),

    DbSchema(
        db_path="lora_registry.db",
        owner="prism_lora_registry",
        tables=[
            TableDef("adapters", [
                ColumnDef("adapter_id",  "TEXT", "PRIMARY KEY"),
                ColumnDef("gguf_path",   "TEXT"),
                ColumnDef("ollama_name", "TEXT"),
                ColumnDef("created_at",  "REAL"),
            ]),
        ],
        notes="Trained LoRA adapters registered after Unsloth QLoRA → GGUF export.",
    ),

    DbSchema(
        db_path="prism.db",
        owner="prism_agent",
        tables=[
            TableDef("sessions", [
                ColumnDef("session_id",  "TEXT", "PRIMARY KEY"),
                ColumnDef("user",        "TEXT"),
                ColumnDef("created_at",  "REAL"),
                ColumnDef("ended_at",    "REAL"),
            ]),
        ],
        notes="General-purpose agent state. Schemas evolve additive-only.",
    ),

    DbSchema(
        db_path="session_manager.db",
        owner="prism_session_manager",
        tables=[
            TableDef("chat_sessions", [
                ColumnDef("session_id", "TEXT", "PRIMARY KEY"),
                ColumnDef("history",    "TEXT"),
                ColumnDef("created_at", "REAL"),
                ColumnDef("updated_at", "REAL"),
            ]),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get(db_path: str) -> Optional[DbSchema]:
    """Return the DbSchema for a given db_path (relative to ~/.prism/), or None."""
    for schema in REGISTRY:
        if schema.db_path == db_path:
            return schema
    return None


def all_paths() -> list[str]:
    """Return all registered database paths (relative to ~/.prism/)."""
    return [s.db_path for s in REGISTRY]


def validate_present(prism_dir: Optional[Path] = None) -> dict[str, bool]:
    """
    Check which registered databases actually exist on disk.
    Returns {db_path: exists} mapping.
    """
    base = (prism_dir or Path.home() / ".prism").expanduser()
    return {s.db_path: (base / s.db_path).exists() for s in REGISTRY}
