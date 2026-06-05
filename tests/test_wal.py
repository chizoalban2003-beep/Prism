"""Tests for PrismWAL."""

import pytest

from prism_wal import PrismWAL


@pytest.fixture()
def wal(tmp_path):
    w = PrismWAL(tmp_path / "wal.db")
    yield w
    w.close()


class TestWALBasicOps:
    def test_append_returns_seq_id(self, wal):
        seq = wal.append("upsert_node", {"node_id": "n1"})
        assert isinstance(seq, str) and len(seq) == 32

    def test_pending_count_increments(self, wal):
        assert wal.pending_count() == 0
        wal.append("upsert_node", {"node_id": "n1"})
        wal.append("upsert_node", {"node_id": "n2"})
        assert wal.pending_count() == 2

    def test_pending_returns_correct_entries(self, wal):
        seq = wal.append("upsert_edge", {"src": "a", "dst": "b"})
        entries = wal.pending()
        assert len(entries) == 1
        assert entries[0]["seq_id"] == seq
        assert entries[0]["op"] == "upsert_edge"
        assert entries[0]["payload"] == {"src": "a", "dst": "b"}

    def test_mark_committed_removes_from_pending(self, wal):
        seq = wal.append("upsert_node", {"node_id": "n1"})
        wal.mark_committed(seq)
        assert wal.pending_count() == 0

    def test_committed_not_in_pending(self, wal):
        s1 = wal.append("upsert_node", {"node_id": "n1"})
        s2 = wal.append("upsert_node", {"node_id": "n2"})
        wal.mark_committed(s1)
        pending = wal.pending()
        assert len(pending) == 1
        assert pending[0]["seq_id"] == s2

    def test_drain_committed_removes_old_entries(self, wal):
        import time
        seq = wal.append("upsert_node", {"node_id": "n1"})
        wal.mark_committed(seq)
        # Force timestamp to be old
        wal._conn.execute("UPDATE wal SET ts=? WHERE seq_id=?", (time.time() - 10 * 86400, seq))
        wal._conn.commit()
        removed = wal.drain_committed(older_than_days=7)
        assert removed == 1

    def test_drain_committed_preserves_recent(self, wal):
        seq = wal.append("upsert_node", {"node_id": "n1"})
        wal.mark_committed(seq)
        removed = wal.drain_committed(older_than_days=7)
        assert removed == 0


class TestWALIdempotency:
    def test_seq_id_is_unique_per_append(self, wal):
        seqs = {wal.append("upsert_node", {"node_id": f"n{i}"}) for i in range(20)}
        assert len(seqs) == 20

    def test_mark_committed_idempotent(self, wal):
        seq = wal.append("upsert_node", {"node_id": "n1"})
        wal.mark_committed(seq)
        wal.mark_committed(seq)  # second call must not raise
        assert wal.pending_count() == 0

    def test_payload_roundtrip(self, wal):
        payload = {"node_id": "abc", "node_type": "entity", "value": {"name": "test"}, "ts": 1.0}
        wal.append("upsert_node", payload)
        entry = wal.pending()[0]
        assert entry["payload"] == payload
