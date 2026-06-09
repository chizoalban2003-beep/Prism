"""
prism_causality.py
==================
Causal Reasoning layer for PRISM.

Two components
--------------
CausalGraph
    A directed acyclic graph (DAG) of causal relationships between belief nodes.
    Persisted to SQLite (table: causal_edges).
    Provides DFS-based causal chain traversal, cycle detection, and edge queries.

CausalReasoner
    Generates natural-language explanations and counterfactual ("what-if") answers
    by combining the CausalGraph with a PrismSoul belief graph.
    Falls back to heuristic explanations when no LLM router is available.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CausalEdge:
    cause_id: str
    effect_id: str
    strength: float          # 0–1 (how strongly cause → effect)
    direction: str           # "positive" | "negative" | "unknown"
    evidence_count: int      # how many observations support this edge
    created_at: float


@dataclass
class CounterfactualResult:
    query: str
    original_outcome: str
    counterfactual_outcome: str
    changed_beliefs: list[str]   # which beliefs would change
    confidence: float
    explanation: str


# ---------------------------------------------------------------------------
# CausalGraph
# ---------------------------------------------------------------------------


class CausalGraph:
    """Directed acyclic graph of causal relationships between beliefs.

    Storage: SQLite table ``causal_edges`` at *db_path*.
    """

    def __init__(self, db_path: str = "~/.prism/causality.db") -> None:
        self._db_path = Path(db_path).expanduser()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._init_db()

    # ------------------------------------------------------------------
    # DB initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS causal_edges (
                cause_id       TEXT NOT NULL,
                effect_id      TEXT NOT NULL,
                strength       REAL NOT NULL DEFAULT 0.5,
                direction      TEXT NOT NULL DEFAULT 'positive',
                evidence_count INTEGER NOT NULL DEFAULT 1,
                created_at     REAL NOT NULL,
                PRIMARY KEY (cause_id, effect_id)
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_ce_cause  ON causal_edges(cause_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_ce_effect ON causal_edges(effect_id)"
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _row_to_edge(self, row: tuple) -> CausalEdge:
        cause_id, effect_id, strength, direction, evidence_count, created_at = row
        return CausalEdge(
            cause_id=cause_id,
            effect_id=effect_id,
            strength=float(strength),
            direction=direction,
            evidence_count=int(evidence_count),
            created_at=float(created_at),
        )

    # ------------------------------------------------------------------
    # Mutating API
    # ------------------------------------------------------------------

    def add_edge(
        self,
        cause_id: str,
        effect_id: str,
        strength: float = 0.5,
        direction: str = "positive",
    ) -> CausalEdge:
        """Insert or update a causal edge.  Returns the resulting CausalEdge."""
        now = time.time()
        strength = max(0.0, min(1.0, float(strength)))
        if direction not in ("positive", "negative", "unknown"):
            direction = "unknown"

        # Upsert: increment evidence_count if edge already exists
        existing = self._conn.execute(
            "SELECT evidence_count FROM causal_edges WHERE cause_id=? AND effect_id=?",
            (cause_id, effect_id),
        ).fetchone()

        if existing:
            new_count = existing[0] + 1
            self._conn.execute(
                """UPDATE causal_edges
                   SET strength=?, direction=?, evidence_count=?
                   WHERE cause_id=? AND effect_id=?""",
                (strength, direction, new_count, cause_id, effect_id),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT cause_id, effect_id, strength, direction, evidence_count, created_at"
                " FROM causal_edges WHERE cause_id=? AND effect_id=?",
                (cause_id, effect_id),
            ).fetchone()
        else:
            self._conn.execute(
                """INSERT INTO causal_edges
                   (cause_id, effect_id, strength, direction, evidence_count, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (cause_id, effect_id, strength, direction, 1, now),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT cause_id, effect_id, strength, direction, evidence_count, created_at"
                " FROM causal_edges WHERE cause_id=? AND effect_id=?",
                (cause_id, effect_id),
            ).fetchone()

        return self._row_to_edge(row)

    def remove_edge(self, cause_id: str, effect_id: str) -> bool:
        """Delete an edge.  Returns True if the edge existed."""
        cursor = self._conn.execute(
            "DELETE FROM causal_edges WHERE cause_id=? AND effect_id=?",
            (cause_id, effect_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_causes(self, node_id: str) -> list[CausalEdge]:
        """Return all edges where *node_id* is the effect (what causes this?)."""
        rows = self._conn.execute(
            "SELECT cause_id, effect_id, strength, direction, evidence_count, created_at"
            " FROM causal_edges WHERE effect_id=? ORDER BY strength DESC",
            (node_id,),
        ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def get_effects(self, node_id: str) -> list[CausalEdge]:
        """Return all edges where *node_id* is the cause (what does this cause?)."""
        rows = self._conn.execute(
            "SELECT cause_id, effect_id, strength, direction, evidence_count, created_at"
            " FROM causal_edges WHERE cause_id=? ORDER BY strength DESC",
            (node_id,),
        ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def all_edges(self) -> list[CausalEdge]:
        """Return every edge in the graph."""
        rows = self._conn.execute(
            "SELECT cause_id, effect_id, strength, direction, evidence_count, created_at"
            " FROM causal_edges ORDER BY created_at"
        ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def causal_chain(self, root_id: str, depth: int = 3) -> list[list[str]]:
        """DFS from *root_id* following cause→effect edges.

        Returns a list of paths (each path is a list of node IDs starting with
        *root_id*).  Depth-limited to *depth* hops to guard against very large
        graphs.  Cycle-safe: each node is visited at most once per path.
        """
        results: list[list[str]] = []

        def _dfs(current: str, path: list[str], remaining: int) -> None:
            effects = self.get_effects(current)
            if not effects or remaining == 0:
                if len(path) > 1:
                    results.append(list(path))
                return
            extended = False
            for edge in effects:
                nxt = edge.effect_id
                if nxt in path:       # avoid revisiting in current path
                    continue
                extended = True
                path.append(nxt)
                _dfs(nxt, path, remaining - 1)
                path.pop()
            if not extended and len(path) > 1:
                results.append(list(path))

        _dfs(root_id, [root_id], depth)
        return results

    def strongest_causes(self, effect_id: str, top_n: int = 5) -> list[CausalEdge]:
        """Return the *top_n* strongest causal edges that lead to *effect_id*."""
        rows = self._conn.execute(
            "SELECT cause_id, effect_id, strength, direction, evidence_count, created_at"
            " FROM causal_edges WHERE effect_id=? ORDER BY strength DESC LIMIT ?",
            (effect_id, top_n),
        ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def detect_cycles(self) -> list[list[str]]:
        """Return a list of cycle paths found in the graph.

        A well-formed causal DAG should return an empty list.
        Uses DFS with a recursion stack to detect back-edges.
        """
        # Build adjacency list
        adj: dict[str, list[str]] = defaultdict(list)
        for edge in self.all_edges():
            adj[edge.cause_id].append(edge.effect_id)

        all_nodes: set[str] = set(adj.keys())
        for edges in adj.values():
            all_nodes.update(edges)

        visited: set[str] = set()
        rec_stack: set[str] = set()
        cycles: list[list[str]] = []

        def _dfs(node: str, path: list[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            for neighbour in adj.get(node, []):
                if neighbour not in visited:
                    _dfs(neighbour, path)
                elif neighbour in rec_stack:
                    # Found a cycle: extract the cycle portion of path
                    idx = path.index(neighbour)
                    cycles.append(path[idx:] + [neighbour])
            path.pop()
            rec_stack.discard(node)

        for node in all_nodes:
            if node not in visited:
                _dfs(node, [])

        return cycles


# ---------------------------------------------------------------------------
# CausalReasoner
# ---------------------------------------------------------------------------


class CausalReasoner:
    """Generates counterfactual explanations using CausalGraph + PrismSoul."""

    def __init__(
        self,
        graph: CausalGraph,
        soul: Optional[Any] = None,
        llm_router: Optional[Any] = None,
    ) -> None:
        self._graph = graph
        self._soul = soul
        self._llm_router = llm_router

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _belief_text(self, node_id: str) -> str:
        """Try to fetch belief text from the soul; fall back to the node ID."""
        if self._soul is not None:
            try:
                belief = self._soul.get_belief(node_id)
                if belief is not None:
                    return belief.text
            except Exception:
                pass
        return node_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(self, belief_id: str) -> str:
        """Return a human-readable explanation of why a belief exists.

        E.g. "Belief X exists because A (strength 0.8) and B (strength 0.6)
        caused it."
        Falls back to heuristic text when no LLM is available.
        """
        causes = self._graph.get_causes(belief_id)
        belief_text = self._belief_text(belief_id)

        if not causes:
            return (
                f'Belief "{belief_text}" has no registered causal predecessors '
                f"in the causal graph."
            )

        cause_parts = []
        for edge in causes:
            ct = self._belief_text(edge.cause_id)
            cause_parts.append(f'"{ct}" (strength {edge.strength:.2f})')

        cause_str = " and ".join(cause_parts)
        heuristic = f'Belief "{belief_text}" exists because {cause_str} caused it.'

        if self._llm_router is not None:
            prompt = (
                f"Explain in one clear sentence why the following belief holds, "
                f"given these causal predecessors.\n\n"
                f"Belief: {belief_text}\n"
                f"Caused by: {cause_str}\n\n"
                f"Reply with a single sentence starting with 'This belief'."
            )
            try:
                response, _ = self._llm_router.call(prompt)
                if response and len(response.strip()) > 10:
                    return response.strip()
            except Exception as exc:
                logger.warning("CausalReasoner.explain LLM error: %s", exc)

        return heuristic

    def counterfactual(
        self, query: str, remove_belief_id: str
    ) -> CounterfactualResult:
        """Simulate removing *remove_belief_id* and predict downstream effects.

        Traces the full causal chain from the removed belief, collects all
        downstream belief IDs, and returns a CounterfactualResult.
        When an LLM router is available, it narrates the outcome.
        """
        original_text = self._belief_text(remove_belief_id)

        # Collect all downstream effects via BFS
        chains = self._graph.causal_chain(remove_belief_id, depth=5)
        changed: list[str] = []
        seen: set[str] = {remove_belief_id}
        for path in chains:
            for nid in path[1:]:   # skip the root (the removed belief)
                if nid not in seen:
                    seen.add(nid)
                    changed.append(nid)

        original_outcome = (
            f'Belief "{original_text}" is present; '
            f"{len(changed)} downstream belief(s) are active."
        )
        if not changed:
            counterfactual_outcome = (
                f'Removing "{original_text}" has no registered downstream effects.'
            )
            confidence = 0.3
        else:
            changed_texts = [self._belief_text(b) for b in changed]
            counterfactual_outcome = (
                f'Removing "{original_text}" would cascade to affect: '
                + ", ".join(f'"{t}"' for t in changed_texts[:5])
                + (f" and {len(changed_texts) - 5} more" if len(changed_texts) > 5 else "")
                + "."
            )
            # Confidence scales with average edge strength along chain
            all_cause_edges = [
                e
                for nid in changed
                for e in self._graph.get_causes(nid)
                if e.cause_id in seen
            ]
            if all_cause_edges:
                confidence = sum(e.strength for e in all_cause_edges) / len(all_cause_edges)
            else:
                confidence = 0.5

        heuristic_explanation = (
            f"If '{original_text}' were removed, the causal chain analysis "
            f"predicts {len(changed)} belief(s) would be affected: "
            + (", ".join(f'"{self._belief_text(b)}"' for b in changed[:3])
               if changed else "none")
            + ("..." if len(changed) > 3 else "")
            + f" (confidence: {confidence:.2f})."
        )

        if self._llm_router is not None:
            prompt = (
                f"Counterfactual query: {query}\n\n"
                f"Belief being removed: {original_text}\n"
                f"Downstream beliefs that would change: "
                + ", ".join(self._belief_text(b) for b in changed[:8])
                + "\n\nIn 2-3 sentences, explain what would change and why."
            )
            try:
                response, _ = self._llm_router.call(prompt)
                if response and len(response.strip()) > 20:
                    heuristic_explanation = response.strip()
            except Exception as exc:
                logger.warning("CausalReasoner.counterfactual LLM error: %s", exc)

        return CounterfactualResult(
            query=query,
            original_outcome=original_outcome,
            counterfactual_outcome=counterfactual_outcome,
            changed_beliefs=changed,
            confidence=round(confidence, 4),
            explanation=heuristic_explanation,
        )

    def infer_edges_from_soul(self, soul: Any) -> int:
        """Extract causal edges from the soul's belief contradiction/support edges.

        Maps soul edge relations to causal edges:
          - "supports"   → positive causal edge (cause → effect)
          - "explains"   → positive causal edge (cause → effect)
          - "contradicts"→ negative causal edge (cause → effect)

        Returns the count of new edges added.
        """
        if soul is None:
            return 0

        added = 0
        try:
            soul_edges = soul.list_edges()
        except Exception as exc:
            logger.warning("infer_edges_from_soul: could not list edges: %s", exc)
            return 0

        for se in soul_edges:
            relation = getattr(se, "relation", "")
            strength = float(getattr(se, "strength", 0.5))
            from_id  = getattr(se, "from_id", None)
            to_id    = getattr(se, "to_id", None)
            if not from_id or not to_id:
                continue

            if relation in ("supports", "explains"):
                direction = "positive"
            elif relation == "contradicts":
                direction = "negative"
            else:
                continue  # skip unknown relations

            # Check if edge already exists (by looking for it directly)
            existing = self._graph._conn.execute(
                "SELECT 1 FROM causal_edges WHERE cause_id=? AND effect_id=?",
                (from_id, to_id),
            ).fetchone()
            if existing:
                continue

            self._graph.add_edge(from_id, to_id, strength=strength, direction=direction)
            added += 1

        return added

    def build_explanation_tree(self, belief_id: str) -> dict[str, Any]:
        """Return a nested dict describing a belief's causal context.

        Structure::

            {
              "belief_id": "...",
              "text": "...",
              "causes":  [{"belief_id": ..., "text": ..., "strength": ..., ...}],
              "effects": [{"belief_id": ..., "text": ..., "strength": ..., ...}],
            }
        """
        text = self._belief_text(belief_id)

        causes_data = []
        for edge in self._graph.get_causes(belief_id):
            causes_data.append({
                "belief_id": edge.cause_id,
                "text": self._belief_text(edge.cause_id),
                "strength": edge.strength,
                "direction": edge.direction,
                "evidence_count": edge.evidence_count,
            })

        effects_data = []
        for edge in self._graph.get_effects(belief_id):
            effects_data.append({
                "belief_id": edge.effect_id,
                "text": self._belief_text(edge.effect_id),
                "strength": edge.strength,
                "direction": edge.direction,
                "evidence_count": edge.evidence_count,
            })

        return {
            "belief_id": belief_id,
            "text": text,
            "causes": causes_data,
            "effects": effects_data,
        }
