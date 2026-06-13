"""
PRISM Layered Memory Graph

Write path:  write_node/write_edge  → WAL append → hot buffer
Commit path: commit_pending()       → WAL drain  → cold SQLite (atomic tx)
Read path:   MemoryAggregator       → merges cold + hot (hot wins on collision)
Recovery:    replay_wal()           → re-hydrates hot buffer from uncommitted WAL entries
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from prism_wal import PrismWAL

_DEFAULT_DB  = Path.home() / ".prism" / "memory_graph.db"
_DEFAULT_WAL = Path.home() / ".prism" / "wal.db"


# ── Domain objects ────────────────────────────────────────────────────────────

@dataclass
class GraphNode:
    node_id:   str
    node_type: str        # entity | fact | context | session | observation
    value:     dict[str, Any]
    ts:        float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"node_id": self.node_id, "node_type": self.node_type,
                "value": self.value, "ts": self.ts}

    @classmethod
    def from_row(cls, row: tuple) -> GraphNode:
        node_id, node_type, value_json, ts = row
        return cls(node_id=node_id, node_type=node_type,
                   value=json.loads(value_json), ts=ts)


@dataclass
class GraphEdge:
    src:      str
    dst:      str
    relation: str
    weight:   float = 1.0
    ts:       float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"src": self.src, "dst": self.dst, "relation": self.relation,
                "weight": self.weight, "ts": self.ts}


# ── Cold layer ────────────────────────────────────────────────────────────────

class _ColdLayer:
    """SQLite-backed persistent graph. All writes are transactional."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._setup()

    def _setup(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS graph_nodes (
                node_id   TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                value     TEXT NOT NULL,
                ts        REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_gn_type ON graph_nodes(node_type);
            CREATE INDEX IF NOT EXISTS ix_gn_ts   ON graph_nodes(ts);

            CREATE TABLE IF NOT EXISTS graph_edges (
                src      TEXT NOT NULL,
                dst      TEXT NOT NULL,
                relation TEXT NOT NULL,
                weight   REAL NOT NULL DEFAULT 1.0,
                ts       REAL NOT NULL,
                PRIMARY KEY (src, dst, relation)
            );
            CREATE INDEX IF NOT EXISTS ix_ge_src ON graph_edges(src);
            CREATE INDEX IF NOT EXISTS ix_ge_dst ON graph_edges(dst);
        """)
        self._conn.commit()

    def upsert_node(self, node: GraphNode) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO graph_nodes(node_id, node_type, value, ts)"
            " VALUES (?,?,?,?)",
            (node.node_id, node.node_type, json.dumps(node.value), node.ts),
        )
        self._conn.commit()

    def upsert_nodes_batch(self, nodes: list[GraphNode]) -> None:
        """Insert/replace multiple nodes in a single transaction."""
        if not nodes:
            return
        self._conn.executemany(
            "INSERT OR REPLACE INTO graph_nodes(node_id, node_type, value, ts) VALUES (?,?,?,?)",
            [(n.node_id, n.node_type, json.dumps(n.value), n.ts) for n in nodes],
        )
        self._conn.commit()

    def upsert_edge(self, edge: GraphEdge) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO graph_edges(src, dst, relation, weight, ts)"
            " VALUES (?,?,?,?,?)",
            (edge.src, edge.dst, edge.relation, edge.weight, edge.ts),
        )
        self._conn.commit()

    def upsert_edges_batch(self, edges: list[GraphEdge]) -> None:
        """Insert/replace multiple edges in a single transaction."""
        if not edges:
            return
        self._conn.executemany(
            "INSERT OR REPLACE INTO graph_edges(src, dst, relation, weight, ts) VALUES (?,?,?,?,?)",
            [(e.src, e.dst, e.relation, e.weight, e.ts) for e in edges],
        )
        self._conn.commit()

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        row = self._conn.execute(
            "SELECT node_id, node_type, value, ts FROM graph_nodes WHERE node_id=?",
            (node_id,),
        ).fetchone()
        return GraphNode.from_row(row) if row else None

    def query_nodes(self, node_type: str | None = None, limit: int = 50) -> list[GraphNode]:
        if node_type:
            rows = self._conn.execute(
                "SELECT node_id, node_type, value, ts FROM graph_nodes"
                " WHERE node_type=? ORDER BY ts DESC LIMIT ?",
                (node_type, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT node_id, node_type, value, ts FROM graph_nodes"
                " ORDER BY ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [GraphNode.from_row(r) for r in rows]

    def search(self, query: str, limit: int = 20) -> list[GraphNode]:
        rows = self._conn.execute(
            "SELECT node_id, node_type, value, ts FROM graph_nodes"
            " WHERE value LIKE ? ORDER BY ts DESC LIMIT ?",
            (f"%{query}%", limit),
        ).fetchall()
        return [GraphNode.from_row(r) for r in rows]

    def edges_for(self, node_id: str) -> list[GraphEdge]:
        rows = self._conn.execute(
            "SELECT src, dst, relation, weight, ts FROM graph_edges"
            " WHERE src=? OR dst=?",
            (node_id, node_id),
        ).fetchall()
        return [GraphEdge(r[0], r[1], r[2], r[3], r[4]) for r in rows]

    def node_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0]

    def close(self) -> None:
        self._conn.close()


# ── Hot buffer ────────────────────────────────────────────────────────────────

class _HotBuffer:
    """Thread-safe in-process uncommitted buffer."""

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, list[GraphEdge]] = {}
        self._lock = threading.Lock()

    def upsert_node(self, node: GraphNode) -> None:
        with self._lock:
            self._nodes[node.node_id] = node

    def upsert_edge(self, edge: GraphEdge) -> None:
        with self._lock:
            bucket = self._edges.setdefault(edge.src, [])
            self._edges[edge.src] = [
                e for e in bucket
                if not (e.dst == edge.dst and e.relation == edge.relation)
            ]
            self._edges[edge.src].append(edge)

    def flush_node(self, node_id: str) -> None:
        with self._lock:
            self._nodes.pop(node_id, None)

    def flush_edge(self, src: str, dst: str, relation: str) -> None:
        with self._lock:
            if src in self._edges:
                self._edges[src] = [
                    e for e in self._edges[src]
                    if not (e.dst == dst and e.relation == relation)
                ]

    def snapshot(self) -> tuple[list[GraphNode], list[GraphEdge]]:
        with self._lock:
            nodes = list(self._nodes.values())
            edges = [e for es in self._edges.values() for e in es]
        return nodes, edges

    def size(self) -> int:
        with self._lock:
            return len(self._nodes)


# ── MemoryAggregator ──────────────────────────────────────────────────────────

class MemoryAggregator:
    """
    Conflict-aware merge of cold + hot layers.
    Hot buffer always wins on node_id collision (overwrite principle).
    """

    def __init__(self, cold: _ColdLayer, hot: _HotBuffer) -> None:
        self._cold = cold
        self._hot  = hot

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        with self._hot._lock:
            if node_id in self._hot._nodes:
                return self._hot._nodes[node_id]
        return self._cold.get_node(node_id)

    def query_nodes(self, node_type: str | None = None, limit: int = 50) -> list[GraphNode]:
        cold_map = {n.node_id: n for n in self._cold.query_nodes(node_type=node_type, limit=limit)}
        with self._hot._lock:
            hot_map = {
                nid: n for nid, n in self._hot._nodes.items()
                if node_type is None or n.node_type == node_type
            }
        merged = {**cold_map, **hot_map}
        return sorted(merged.values(), key=lambda n: n.ts, reverse=True)[:limit]

    def search(self, query: str, limit: int = 10) -> list[GraphNode]:
        q = query.lower()
        results: list[GraphNode] = []
        with self._hot._lock:
            hot_ids = set(self._hot._nodes.keys())
            for node in self._hot._nodes.values():
                if q in json.dumps(node.value).lower():
                    results.append(node)
        for node in self._cold.search(query, limit=limit * 2):
            if node.node_id not in hot_ids:
                results.append(node)
        return results[:limit]

    def edges_for(self, node_id: str) -> list[GraphEdge]:
        cold_edges = self._cold.edges_for(node_id)
        with self._hot._lock:
            hot_edges = list(self._hot._edges.get(node_id, []))
        seen: set[tuple] = set()
        merged: list[GraphEdge] = []
        for e in hot_edges + cold_edges:
            key = (e.src, e.dst, e.relation)
            if key not in seen:
                seen.add(key)
                merged.append(e)
        return merged


# ── PrismMemoryGraph (public API) ─────────────────────────────────────────────

class PrismMemoryGraph:
    """
    Layered memory graph for PRISM.

    Usage::

        g = PrismMemoryGraph()
        g.replay_wal()                   # on startup — recover uncommitted entries

        seq = g.write_node(GraphNode("u1", "entity", {"name": "Alice"}))
        g.write_edge(GraphEdge("u1", "u2", "knows"))

        node = g.get_node("u1")          # reads hot buffer first
        g.commit_pending()               # flush hot → cold (also called by ShadowPipeline)
        g.close()
    """

    def __init__(
        self,
        db_path:  Path | str = _DEFAULT_DB,
        wal_path: Path | str = _DEFAULT_WAL,
    ) -> None:
        self._cold = _ColdLayer(Path(db_path))
        self._hot  = _HotBuffer()
        self._wal  = PrismWAL(wal_path)
        self.aggregator = MemoryAggregator(self._cold, self._hot)
        self._commit_lock = threading.Lock()

    # ── Write API (hot path) ──────────────────────────────────────────────────

    def write_node(self, node: GraphNode) -> str:
        """WAL-log and buffer a node. Returns seq_id."""
        seq_id = self._wal.append("upsert_node", node.to_dict())
        self._hot.upsert_node(node)
        return seq_id

    def write_edge(self, edge: GraphEdge) -> str:
        """WAL-log and buffer an edge. Returns seq_id."""
        seq_id = self._wal.append("upsert_edge", edge.to_dict())
        self._hot.upsert_edge(edge)
        return seq_id

    def write_nodes_batch(self, nodes: list[GraphNode]) -> list[str]:
        """WAL-log and buffer multiple nodes in a single transaction. Returns seq_ids."""
        entries = [("upsert_node", n.to_dict()) for n in nodes]
        seq_ids = self._wal.append_batch(entries)
        for node in nodes:
            self._hot.upsert_node(node)
        return seq_ids

    def write_edges_batch(self, edges: list[GraphEdge]) -> list[str]:
        """WAL-log and buffer multiple edges in a single transaction. Returns seq_ids."""
        entries = [("upsert_edge", e.to_dict()) for e in edges]
        seq_ids = self._wal.append_batch(entries)
        for edge in edges:
            self._hot.upsert_edge(edge)
        return seq_ids

    # ── Commit API (called by ShadowPipeline) ─────────────────────────────────

    def commit_pending(self) -> int:
        """
        Atomically drain pending WAL entries into the cold layer.
        Returns number of entries committed.
        Idempotent: safe to call after a crash (WAL seq_ids have UNIQUE constraint).
        Uses batch upserts (executemany) for ~100x lower latency vs per-row commits.
        """
        pending = self._wal.pending()
        if not pending:
            return 0
        with self._commit_lock:
            nodes: list[GraphNode] = []
            edges: list[GraphEdge] = []
            node_ids: list[str] = []
            edge_keys: list[tuple[str, str, str]] = []
            seq_ids: list[str] = []

            for entry in pending:
                op = entry["op"]
                p  = entry["payload"]
                try:
                    if op == "upsert_node":
                        nodes.append(GraphNode(node_id=p["node_id"], node_type=p["node_type"],
                                               value=p["value"], ts=p["ts"]))
                        node_ids.append(p["node_id"])
                    elif op == "upsert_edge":
                        edges.append(GraphEdge(src=p["src"], dst=p["dst"], relation=p["relation"],
                                               weight=p["weight"], ts=p["ts"]))
                        edge_keys.append((p["src"], p["dst"], p["relation"]))
                    seq_ids.append(entry["seq_id"])
                except (KeyError, TypeError):
                    # Malformed payload — skip entry; leave it pending for manual inspection
                    continue

            try:
                self._cold.upsert_nodes_batch(nodes)
                self._cold.upsert_edges_batch(edges)
                for nid in node_ids:
                    self._hot.flush_node(nid)
                for src, dst, rel in edge_keys:
                    self._hot.flush_edge(src, dst, rel)
                self._wal.mark_committed_batch(seq_ids)
            except Exception:
                return 0  # leave all entries pending; crash-safe replay on next call

        return len(seq_ids)

    # ── Crash recovery ────────────────────────────────────────────────────────

    def replay_wal(self) -> int:
        """
        Re-hydrate hot buffer from uncommitted WAL entries.
        Call once on startup before the ShadowPipeline starts.
        """
        pending = self._wal.pending()
        for entry in pending:
            p  = entry["payload"]
            op = entry["op"]
            if op == "upsert_node":
                self._hot.upsert_node(
                    GraphNode(node_id=p["node_id"], node_type=p["node_type"],
                              value=p["value"], ts=p["ts"])
                )
            elif op == "upsert_edge":
                self._hot.upsert_edge(
                    GraphEdge(src=p["src"], dst=p["dst"], relation=p["relation"],
                              weight=p["weight"], ts=p["ts"])
                )
        return len(pending)

    # ── Consistency metric ────────────────────────────────────────────────────

    def consistency_psi(self) -> int:
        """Ψ = pending WAL entries. 0 = equilibrium, >0 = mutations in flight."""
        return self._wal.pending_count()

    # ── Read API (pass-through to aggregator) ─────────────────────────────────

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self.aggregator.get_node(node_id)

    def query_nodes(self, node_type: str | None = None, limit: int = 50) -> list[GraphNode]:
        return self.aggregator.query_nodes(node_type=node_type, limit=limit)

    def search(self, query: str, limit: int = 10) -> list[GraphNode]:
        return self.aggregator.search(query, limit=limit)

    def edges_for(self, node_id: str) -> list[GraphEdge]:
        return self.aggregator.edges_for(node_id)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        self.commit_pending()
        self._cold.close()
        self._wal.close()
