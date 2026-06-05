"""Tests for PrismMemoryGraph — cold/hot layers, aggregator, WAL, crash recovery."""
import time

import pytest

from prism_memory_graph import GraphEdge, GraphNode, PrismMemoryGraph


@pytest.fixture()
def graph(tmp_path):
    g = PrismMemoryGraph(
        db_path=tmp_path / "graph.db",
        wal_path=tmp_path / "wal.db",
    )
    yield g
    g.close()


def _node(nid, ntype="entity", **kw) -> GraphNode:
    return GraphNode(node_id=nid, node_type=ntype, value=kw or {}, ts=time.time())


def _edge(src, dst, rel="knows") -> GraphEdge:
    return GraphEdge(src=src, dst=dst, relation=rel)


# ── Write + read from hot buffer ──────────────────────────────────────────────

class TestHotBuffer:
    def test_write_node_visible_immediately(self, graph):
        graph.write_node(_node("n1", name="Alice"))
        node = graph.get_node("n1")
        assert node is not None
        assert node.value["name"] == "Alice"

    def test_write_edge_visible_immediately(self, graph):
        graph.write_node(_node("a"))
        graph.write_node(_node("b"))
        graph.write_edge(_edge("a", "b"))
        edges = graph.edges_for("a")
        assert any(e.dst == "b" for e in edges)

    def test_hot_buffer_size(self, graph):
        for i in range(5):
            graph.write_node(_node(f"n{i}"))
        assert graph._hot.size() == 5

    def test_wal_pending_increments_on_write(self, graph):
        graph.write_node(_node("n1"))
        graph.write_node(_node("n2"))
        assert graph.consistency_psi() == 2


# ── commit_pending: hot → cold ────────────────────────────────────────────────

class TestCommitPending:
    def test_commit_moves_nodes_to_cold(self, graph):
        graph.write_node(_node("n1", name="Bob"))
        assert graph.commit_pending() == 1
        # Now read directly from cold layer
        node = graph._cold.get_node("n1")
        assert node is not None and node.value["name"] == "Bob"

    def test_commit_flushes_hot_buffer(self, graph):
        graph.write_node(_node("n1"))
        graph.commit_pending()
        assert graph._hot.size() == 0

    def test_commit_clears_wal(self, graph):
        graph.write_node(_node("n1"))
        graph.commit_pending()
        assert graph.consistency_psi() == 0

    def test_commit_edges(self, graph):
        graph.write_node(_node("a"))
        graph.write_node(_node("b"))
        graph.write_edge(_edge("a", "b", "follows"))
        graph.commit_pending()
        edges = graph._cold.edges_for("a")
        assert any(e.relation == "follows" for e in edges)

    def test_commit_idempotent(self, graph):
        graph.write_node(_node("n1"))
        graph.commit_pending()
        n = graph.commit_pending()
        assert n == 0  # nothing left to commit

    def test_commit_returns_count(self, graph):
        for i in range(4):
            graph.write_node(_node(f"n{i}"))
        assert graph.commit_pending() == 4


# ── MemoryAggregator: hot wins on collision ───────────────────────────────────

class TestMemoryAggregatorOverwrite:
    def test_hot_wins_over_cold(self, graph):
        # Write to cold directly
        graph._cold.upsert_node(_node("n1", version="cold"))
        # Write hot version
        graph.write_node(_node("n1", version="hot"))
        node = graph.get_node("n1")
        assert node.value["version"] == "hot"

    def test_after_commit_cold_is_authoritative(self, graph):
        graph._cold.upsert_node(_node("n1", version="cold"))
        graph.write_node(_node("n1", version="hot"))
        graph.commit_pending()
        node = graph._cold.get_node("n1")
        assert node.value["version"] == "hot"

    def test_query_nodes_deduplicates(self, graph):
        graph._cold.upsert_node(_node("n1", name="cold-alice"))
        graph.write_node(_node("n1", name="hot-alice"))
        results = graph.query_nodes()
        ids = [n.node_id for n in results]
        assert ids.count("n1") == 1
        assert next(n for n in results if n.node_id == "n1").value["name"] == "hot-alice"

    def test_search_includes_hot_nodes(self, graph):
        graph.write_node(_node("n1", label="unique_search_token"))
        results = graph.search("unique_search_token")
        assert any(n.node_id == "n1" for n in results)

    def test_search_not_duplicated_after_partial_commit(self, graph):
        graph.write_node(_node("n1", label="tok"))
        graph.commit_pending()
        graph.write_node(_node("n1", label="tok"))  # re-write same node (hot again)
        results = graph.search("tok")
        ids = [n.node_id for n in results]
        assert ids.count("n1") == 1

    def test_edges_merged_no_duplicates(self, graph):
        # Same edge in cold and hot
        graph._cold.upsert_edge(_edge("a", "b", "likes"))
        graph.write_edge(_edge("a", "b", "likes"))
        edges = graph.edges_for("a")
        likes = [e for e in edges if e.relation == "likes"]
        assert len(likes) == 1


# ── WAL replay (crash recovery) ───────────────────────────────────────────────

class TestWALReplay:
    def test_replay_rehydrates_hot_buffer(self, tmp_path):
        g1 = PrismMemoryGraph(tmp_path / "g.db", tmp_path / "w.db")
        g1.write_node(_node("n1", name="persistent"))
        # Simulate crash: close without committing
        g1._cold.close()
        g1._wal.close()

        # Reopen
        g2 = PrismMemoryGraph(tmp_path / "g.db", tmp_path / "w.db")
        replayed = g2.replay_wal()
        assert replayed == 1
        node = g2.get_node("n1")
        assert node is not None and node.value["name"] == "persistent"
        g2.close()

    def test_replay_then_commit_works(self, tmp_path):
        g1 = PrismMemoryGraph(tmp_path / "g.db", tmp_path / "w.db")
        g1.write_node(_node("n1"))
        g1._cold.close()
        g1._wal.close()

        g2 = PrismMemoryGraph(tmp_path / "g.db", tmp_path / "w.db")
        g2.replay_wal()
        committed = g2.commit_pending()
        assert committed == 1
        assert g2._cold.get_node("n1") is not None
        g2.close()


# ── Consistency metric ────────────────────────────────────────────────────────

class TestConsistencyPsi:
    def test_psi_zero_on_empty(self, graph):
        assert graph.consistency_psi() == 0

    def test_psi_equals_pending_writes(self, graph):
        graph.write_node(_node("n1"))
        graph.write_node(_node("n2"))
        assert graph.consistency_psi() == 2

    def test_psi_zero_after_commit(self, graph):
        graph.write_node(_node("n1"))
        graph.commit_pending()
        assert graph.consistency_psi() == 0


# ── query_nodes filtering ─────────────────────────────────────────────────────

class TestQueryNodes:
    def test_filter_by_type(self, graph):
        graph.write_node(_node("e1", "entity"))
        graph.write_node(_node("f1", "fact"))
        graph.commit_pending()
        entities = graph.query_nodes(node_type="entity")
        assert all(n.node_type == "entity" for n in entities)

    def test_limit_respected(self, graph):
        for i in range(10):
            graph.write_node(_node(f"n{i}"))
        graph.commit_pending()
        results = graph.query_nodes(limit=3)
        assert len(results) <= 3
