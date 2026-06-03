"""
prism_soul.py
=============
Living identity infrastructure — the user's digital soul.

Three layers
------------
Soul Seed
    The output of the identity ceremony. A short narrative (free text) plus
    a structured dict of stated values, goals, and constraints. User-editable.
    Stored in ~/.prism/soul.md (human-readable) and soul.db (queryable).

Belief Graph
    Nodes: beliefs, values, patterns, preferences. Each has:
      - text: the belief in plain English
      - belief_type: "value" | "pattern" | "preference" | "goal" | "constraint"
      - source: "stated" (user said it) or "observed" (system inferred it)
      - confidence: 0.0–1.0
      - timestamp
    Edges: relationships between nodes:
      - "supports": belief A is evidence for belief B
      - "contradicts": belief A conflicts with belief B
      - "explains": belief A explains why belief B exists

Lenses
    User-defined observation dimensions. Each lens has:
      - name: what the user calls this dimension
      - description: what they want tracked
      - signal_types: OrganBus signal types that feed this lens
      - observations: list of (timestamp, value, context) tuples
      - trend: recent average (last 10 observations)

Delta report
    Compares stated beliefs vs observed patterns and surfaces contradictions.
    E.g. user states "I prioritise rest" but observed pattern shows 12-hour days.

Portability
    export_json() / import_json() for soul migration.
    export_md() writes ~/.prism/soul.md — human-readable, user-editable.
    import_md() reads back any edits the user made to soul.md.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BeliefNode:
    node_id: str
    text: str
    belief_type: str  # "value" | "pattern" | "preference" | "goal" | "constraint"
    source: str       # "stated" | "observed"
    confidence: float = 0.8
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    observation_count: int = 0
    notes: str = ""


@dataclass
class BeliefEdge:
    edge_id: str
    from_id: str
    to_id: str
    relation: str   # "supports" | "contradicts" | "explains"
    strength: float = 0.5
    created_at: float = field(default_factory=time.time)


@dataclass
class SoulLens:
    lens_id: str
    name: str
    description: str
    signal_types: List[str]
    observations: List[dict] = field(default_factory=list)
    user_created: bool = True

    @property
    def trend(self) -> Optional[float]:
        """Mean of the last 10 observation values, or None if empty."""
        if not self.observations:
            return None
        recent = self.observations[-10:]
        values = [o["value"] for o in recent if "value" in o]
        if not values:
            return None
        return sum(values) / len(values)

    def add_observation(self, value: float, context: str) -> None:
        """Append a new observation."""
        self.observations.append({
            "timestamp": time.time(),
            "value": value,
            "context": context,
        })


@dataclass
class SoulSeed:
    narrative: str
    stated_values: List[str]
    stated_goals: List[str]
    stated_constraints: List[str]
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# PrismSoul
# ---------------------------------------------------------------------------


class PrismSoul:
    """Living identity infrastructure for PRISM."""

    def __init__(self, db_path: str = "~/.prism/soul.db", llm_router=None):
        self._db_path = Path(db_path).expanduser()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._llm_router = llm_router
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._init_db()

    # ------------------------------------------------------------------
    # DB initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        c = self._conn
        c.execute("""
            CREATE TABLE IF NOT EXISTS seed (
                id INTEGER PRIMARY KEY,
                narrative TEXT,
                stated_values TEXT,
                stated_goals TEXT,
                stated_constraints TEXT,
                created_at REAL,
                updated_at REAL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS beliefs (
                node_id TEXT PRIMARY KEY,
                text TEXT,
                belief_type TEXT,
                source TEXT,
                confidence REAL,
                created_at REAL,
                updated_at REAL,
                observation_count INTEGER,
                notes TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                edge_id TEXT PRIMARY KEY,
                from_id TEXT,
                to_id TEXT,
                relation TEXT,
                strength REAL,
                created_at REAL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS lenses (
                lens_id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                signal_types TEXT,
                observations TEXT,
                user_created INTEGER
            )
        """)
        c.commit()

    # ------------------------------------------------------------------
    # Soul Seed
    # ------------------------------------------------------------------

    def set_seed(self, seed: SoulSeed) -> None:
        """Upsert the soul seed."""
        row = self._conn.execute("SELECT id FROM seed LIMIT 1").fetchone()
        now = time.time()
        if row:
            self._conn.execute(
                """UPDATE seed SET narrative=?, stated_values=?, stated_goals=?,
                   stated_constraints=?, updated_at=? WHERE id=?""",
                (
                    seed.narrative,
                    json.dumps(seed.stated_values),
                    json.dumps(seed.stated_goals),
                    json.dumps(seed.stated_constraints),
                    now,
                    row[0],
                ),
            )
        else:
            self._conn.execute(
                """INSERT INTO seed (narrative, stated_values, stated_goals,
                   stated_constraints, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    seed.narrative,
                    json.dumps(seed.stated_values),
                    json.dumps(seed.stated_goals),
                    json.dumps(seed.stated_constraints),
                    seed.created_at,
                    now,
                ),
            )
        self._conn.commit()

    def get_seed(self) -> Optional[SoulSeed]:
        """Load seed from DB."""
        row = self._conn.execute(
            "SELECT narrative, stated_values, stated_goals, stated_constraints,"
            " created_at, updated_at FROM seed LIMIT 1"
        ).fetchone()
        if not row:
            return None
        return SoulSeed(
            narrative=row[0],
            stated_values=json.loads(row[1]),
            stated_goals=json.loads(row[2]),
            stated_constraints=json.loads(row[3]),
            created_at=row[4],
            updated_at=row[5],
        )

    def has_seed(self) -> bool:
        """True if a seed exists."""
        return self._conn.execute("SELECT COUNT(*) FROM seed").fetchone()[0] > 0

    # ------------------------------------------------------------------
    # Beliefs
    # ------------------------------------------------------------------

    def add_belief(
        self,
        text: str,
        belief_type: str,
        source: str,
        confidence: float = 0.8,
        notes: str = "",
    ) -> str:
        """Insert a BeliefNode, return node_id."""
        node_id = str(uuid.uuid4())[:8]
        now = time.time()
        self._conn.execute(
            """INSERT INTO beliefs (node_id, text, belief_type, source, confidence,
               created_at, updated_at, observation_count, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (node_id, text, belief_type, source, confidence, now, now, 0, notes),
        )
        self._conn.commit()
        return node_id

    def get_belief(self, node_id: str) -> Optional[BeliefNode]:
        """Load from DB."""
        row = self._conn.execute(
            "SELECT node_id, text, belief_type, source, confidence,"
            " created_at, updated_at, observation_count, notes FROM beliefs WHERE node_id=?",
            (node_id,),
        ).fetchone()
        if not row:
            return None
        return BeliefNode(*row)

    def list_beliefs(
        self, source: Optional[str] = None, belief_type: Optional[str] = None
    ) -> List[BeliefNode]:
        """Filter query."""
        query = (
            "SELECT node_id, text, belief_type, source, confidence,"
            " created_at, updated_at, observation_count, notes FROM beliefs WHERE 1=1"
        )
        params: List[Any] = []
        if source is not None:
            query += " AND source=?"
            params.append(source)
        if belief_type is not None:
            query += " AND belief_type=?"
            params.append(belief_type)
        rows = self._conn.execute(query, params).fetchall()
        return [BeliefNode(*r) for r in rows]

    def update_belief(
        self,
        node_id: str,
        confidence: Optional[float] = None,
        notes: Optional[str] = None,
        observation_count_delta: int = 0,
    ) -> None:
        """Partial update."""
        updates: List[str] = ["updated_at=?"]
        params: List[Any] = [time.time()]
        if confidence is not None:
            updates.append("confidence=?")
            params.append(confidence)
        if notes is not None:
            updates.append("notes=?")
            params.append(notes)
        if observation_count_delta:
            updates.append("observation_count=observation_count+?")
            params.append(observation_count_delta)
        params.append(node_id)
        self._conn.execute(
            f"UPDATE beliefs SET {', '.join(updates)} WHERE node_id=?", params
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    def add_edge(
        self, from_id: str, to_id: str, relation: str, strength: float = 0.5
    ) -> str:
        """Insert BeliefEdge, return edge_id."""
        edge_id = str(uuid.uuid4())[:8]
        now = time.time()
        self._conn.execute(
            "INSERT INTO edges (edge_id, from_id, to_id, relation, strength, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (edge_id, from_id, to_id, relation, strength, now),
        )
        self._conn.commit()
        return edge_id

    def list_edges(
        self, from_id: Optional[str] = None, relation: Optional[str] = None
    ) -> List[BeliefEdge]:
        """Filter query."""
        query = "SELECT edge_id, from_id, to_id, relation, strength, created_at FROM edges WHERE 1=1"
        params: List[Any] = []
        if from_id is not None:
            query += " AND from_id=?"
            params.append(from_id)
        if relation is not None:
            query += " AND relation=?"
            params.append(relation)
        rows = self._conn.execute(query, params).fetchall()
        return [BeliefEdge(*r) for r in rows]

    # ------------------------------------------------------------------
    # Lenses
    # ------------------------------------------------------------------

    def add_lens(
        self,
        name: str,
        description: str,
        signal_types: Optional[List[str]] = None,
    ) -> str:
        """Insert SoulLens, return lens_id."""
        lens_id = str(uuid.uuid4())[:8]
        self._conn.execute(
            "INSERT INTO lenses"
            " (lens_id, name, description, signal_types, observations, user_created)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                lens_id,
                name,
                description,
                json.dumps(signal_types or []),
                json.dumps([]),
                1,
            ),
        )
        self._conn.commit()
        return lens_id

    def get_lens(self, lens_id: str) -> Optional[SoulLens]:
        """Load with observations from DB."""
        row = self._conn.execute(
            "SELECT lens_id, name, description, signal_types, observations, user_created FROM lenses WHERE lens_id=?",
            (lens_id,),
        ).fetchone()
        if not row:
            return None
        return SoulLens(
            lens_id=row[0],
            name=row[1],
            description=row[2],
            signal_types=json.loads(row[3]),
            observations=json.loads(row[4]),
            user_created=bool(row[5]),
        )

    def list_lenses(self) -> List[SoulLens]:
        """All lenses."""
        rows = self._conn.execute(
            "SELECT lens_id, name, description, signal_types, observations, user_created FROM lenses"
        ).fetchall()
        return [
            SoulLens(
                lens_id=r[0],
                name=r[1],
                description=r[2],
                signal_types=json.loads(r[3]),
                observations=json.loads(r[4]),
                user_created=bool(r[5]),
            )
            for r in rows
        ]

    def record_observation(self, lens_id: str, value: float, context: str = "") -> None:
        """Append observation to lens, update lens row in DB."""
        lens = self.get_lens(lens_id)
        if lens is None:
            logger.warning("record_observation: lens %s not found", lens_id)
            return
        lens.add_observation(value, context)
        self._conn.execute(
            "UPDATE lenses SET observations=? WHERE lens_id=?",
            (json.dumps(lens.observations), lens_id),
        )
        self._conn.commit()

    def observe_signal(
        self, signal_type: str, payload: dict, context: str = ""
    ) -> None:
        """Route a signal to all matching lenses."""
        lenses = self.list_lenses()
        for lens in lenses:
            if signal_type in lens.signal_types:
                # derive float value from payload: first numeric value, clipped to 0-1
                value = 0.5
                for v in payload.values():
                    if isinstance(v, (int, float)):
                        value = float(max(0.0, min(1.0, v)))
                        break
                self.record_observation(lens.lens_id, value, context)

    # ------------------------------------------------------------------
    # Delta report
    # ------------------------------------------------------------------

    def delta_report(self) -> List[dict]:
        """
        Find stated/observed pairs connected by a 'contradicts' edge.
        Returns list of {"stated": text, "observed": text, "strength": float}.
        """
        contradicts_edges = self.list_edges(relation="contradicts")
        results = []
        for edge in contradicts_edges:
            a = self.get_belief(edge.from_id)
            b = self.get_belief(edge.to_id)
            if a is None or b is None:
                continue
            # one must be stated, other observed
            if a.source == "stated" and b.source == "observed":
                results.append({"stated": a.text, "observed": b.text, "strength": edge.strength})
            elif a.source == "observed" and b.source == "stated":
                results.append({"stated": b.text, "observed": a.text, "strength": edge.strength})
        return results

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def compress_for_llm(self, max_chars: int = 600) -> str:
        """Concise soul summary for LLM system prompts."""
        seed = self.get_seed()
        beliefs = self.list_beliefs()
        lenses = self.list_lenses()
        tensions = self.delta_report()

        # Build components
        seed_line = ""
        if seed:
            vals = seed.stated_values[:3]
            goals = seed.stated_goals[:2]
            seed_line = f"Soul: {vals} | Goals: {goals}"

        stated_high = sorted(
            [b for b in beliefs if b.source == "stated"],
            key=lambda b: b.confidence,
            reverse=True,
        )[:3]
        values_line = "Values: " + "; ".join(b.text for b in stated_high)

        observed = sorted(
            [b for b in beliefs if b.source == "observed"],
            key=lambda b: b.confidence,
            reverse=True,
        )[:3]
        patterns_line = "Patterns: " + "; ".join(
            f"{b.text} ({b.confidence:.1f})" for b in observed
        )

        lens_parts = []
        for lens in lenses:
            t = lens.trend
            lens_parts.append(f"{lens.name}: {t:.2f}" if t is not None else f"{lens.name}: no data")
        lenses_line = "Lenses: " + (", ".join(lens_parts) if lens_parts else "none")

        tension_parts = []
        for t in tensions:
            tension_parts.append(f"{t['stated']!r} vs {t['observed']!r}")
        tensions_line = "Tensions: " + ("; ".join(tension_parts) if tension_parts else "none")

        summary = "\n".join(
            filter(None, [seed_line, values_line, patterns_line, lenses_line, tensions_line])
        )
        return summary[:max_chars]

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------

    def export_json(self) -> dict:
        """Serialize everything."""
        seed = self.get_seed()
        seed_data = None
        if seed:
            seed_data = {
                "narrative": seed.narrative,
                "stated_values": seed.stated_values,
                "stated_goals": seed.stated_goals,
                "stated_constraints": seed.stated_constraints,
                "created_at": seed.created_at,
                "updated_at": seed.updated_at,
            }
        beliefs = [
            {
                "node_id": b.node_id,
                "text": b.text,
                "belief_type": b.belief_type,
                "source": b.source,
                "confidence": b.confidence,
                "created_at": b.created_at,
                "updated_at": b.updated_at,
                "observation_count": b.observation_count,
                "notes": b.notes,
            }
            for b in self.list_beliefs()
        ]
        edges = [
            {
                "edge_id": e.edge_id,
                "from_id": e.from_id,
                "to_id": e.to_id,
                "relation": e.relation,
                "strength": e.strength,
                "created_at": e.created_at,
            }
            for e in self.list_edges()
        ]
        lenses = [
            {
                "lens_id": ln.lens_id,
                "name": ln.name,
                "description": ln.description,
                "signal_types": ln.signal_types,
                "observations": ln.observations,
                "user_created": ln.user_created,
            }
            for ln in self.list_lenses()
        ]
        return {"seed": seed_data, "beliefs": beliefs, "edges": edges, "lenses": lenses}

    def import_json(self, data: dict) -> None:
        """Restore from export. Clears existing data first."""
        self._conn.execute("DELETE FROM seed")
        self._conn.execute("DELETE FROM beliefs")
        self._conn.execute("DELETE FROM edges")
        self._conn.execute("DELETE FROM lenses")
        self._conn.commit()

        if data.get("seed"):
            s = data["seed"]
            seed = SoulSeed(
                narrative=s["narrative"],
                stated_values=s["stated_values"],
                stated_goals=s["stated_goals"],
                stated_constraints=s["stated_constraints"],
                created_at=s.get("created_at", time.time()),
                updated_at=s.get("updated_at", time.time()),
            )
            self.set_seed(seed)

        for b in data.get("beliefs", []):
            self._conn.execute(
                """INSERT INTO beliefs (node_id, text, belief_type, source, confidence,
                   created_at, updated_at, observation_count, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    b["node_id"], b["text"], b["belief_type"], b["source"],
                    b["confidence"], b["created_at"], b["updated_at"],
                    b["observation_count"], b["notes"],
                ),
            )

        for e in data.get("edges", []):
            self._conn.execute(
                "INSERT INTO edges (edge_id, from_id, to_id, relation, strength, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (e["edge_id"], e["from_id"], e["to_id"], e["relation"], e["strength"], e["created_at"]),
            )

        for ln in data.get("lenses", []):
            self._conn.execute(
                "INSERT INTO lenses"
            " (lens_id, name, description, signal_types, observations, user_created)"
            " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    ln["lens_id"], ln["name"], ln["description"],
                    json.dumps(ln["signal_types"]),
                    json.dumps(ln["observations"]),
                    int(ln["user_created"]),
                ),
            )

        self._conn.commit()

    def export_md(self) -> str:
        """Human-readable markdown. Also writes to ~/.prism/soul.md."""
        seed = self.get_seed()
        beliefs = self.list_beliefs()
        lenses = self.list_lenses()
        tensions = self.delta_report()

        lines = ["# PRISM Soul\n"]

        lines.append("## Soul Seed\n")
        if seed:
            lines.append(seed.narrative)
            lines.append("\n**Values:** " + ", ".join(seed.stated_values))
            lines.append("\n**Goals:** " + ", ".join(seed.stated_goals))
            lines.append("\n**Constraints:** " + ", ".join(seed.stated_constraints))
        else:
            lines.append("_No seed yet — run the identity ceremony._")

        lines.append("\n\n## Values & Beliefs\n")
        stated = [b for b in beliefs if b.source == "stated"]
        if stated:
            for b in stated:
                lines.append(f"- [{b.belief_type}] {b.text} (confidence: {b.confidence:.2f})")
        else:
            lines.append("_None yet._")

        lines.append("\n\n## Observed Patterns\n")
        observed = [b for b in beliefs if b.source == "observed"]
        if observed:
            for b in observed:
                lines.append(
                    f"- [{b.belief_type}] {b.text}"
                    f" (confidence: {b.confidence:.2f}, obs: {b.observation_count})"
                )
        else:
            lines.append("_None yet._")

        lines.append("\n\n## Lenses\n")
        if lenses:
            for lens in lenses:
                t = lens.trend
                trend_str = f"{t:.3f}" if t is not None else "no data"
                lines.append(f"### {lens.name}")
                lines.append(f"_{lens.description}_")
                lines.append(f"Trend: {trend_str} | Observations: {len(lens.observations)}")
                lines.append(f"Signal types: {', '.join(lens.signal_types) or 'none'}")
        else:
            lines.append("_No lenses configured._")

        lines.append("\n\n## Tensions\n")
        if tensions:
            for t in tensions:
                lines.append(f"- **Stated:** {t['stated']}")
                lines.append(f"  **Observed:** {t['observed']} (strength: {t['strength']:.2f})")
        else:
            lines.append("_No contradictions detected._")

        md = "\n".join(lines) + "\n"

        soul_md = Path("~/.prism/soul.md").expanduser()
        soul_md.parent.mkdir(parents=True, exist_ok=True)
        soul_md.write_text(md)

        return md

    # ------------------------------------------------------------------
    # OrganBus integration
    # ------------------------------------------------------------------

    def _bus_handler(self, payload: dict) -> None:
        """Handler for OrganBus signals."""
        signal_type = payload.get("signal_type", "")
        if not signal_type:
            # Try all lenses that have any subscription
            return
        self.observe_signal(signal_type, payload)

    def register_with_bus(self, organ_bus) -> None:
        """Register with OrganBus for all signal types across all lenses."""
        lenses = self.list_lenses()
        all_signal_types: List[str] = []
        for lens in lenses:
            for st in lens.signal_types:
                if st not in all_signal_types:
                    all_signal_types.append(st)

        if not all_signal_types:
            all_signal_types = ["*"]

        organ_bus.register(
            organ_name="prism_soul",
            signal_types=all_signal_types,
            handler=self._bus_handler,
            vocabulary=(
                "Accepts any numeric-valued signal. Extracts 0-1 float values "
                "and records them as observations on matching lenses."
            ),
        )
