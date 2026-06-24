from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MemoryEntry:
    entry_id:    str
    content:     str          # the text content
    source:      str          # "email"|"note"|"document"|"conversation"|"artifact"
    title:       str = ""
    timestamp:   float = field(default_factory=time.time)
    tags:        list[str] = field(default_factory=list)
    embedding:   list[float] = field(default_factory=list)   # optional

@dataclass
class MemoryResult:
    entry:     MemoryEntry
    score:     float
    excerpt:   str            # relevant excerpt (max 300 chars)

class PrismMemory:
    """
    Local semantic memory. Stores and retrieves any content by meaning.

    Ingest: email, document, note, conversation turn, artifact output.
    Retrieve: find relevant past content for the current query.

    Embedding strategy (in priority order):
      1. Ollama embeddings API (/api/embeddings) — semantic search
      2. BM25 keyword scoring — good approximate fallback
      3. Simple TF-IDF — always available, no dependencies

    Storage: single SQLite file. All content stays on device.
    """

    def __init__(self, db_path: str = "~/.prism/memory.db",
                 ollama_host: str = "http://localhost:11434",
                 embed_model: str = "nomic-embed-text"):
        self._db       = Path(db_path).expanduser()
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._ollama   = ollama_host
        self._embed_m  = embed_model
        self._init_db()

    def ingest(self, content: str, source: str = "note",
               title: str = "", tags: list[str] | None = None) -> str:
        """Store content in memory. Returns entry_id."""
        entry_id  = hashlib.sha256(
            f"{content[:100]}{time.time()}".encode()).hexdigest()[:12]
        embedding = self._embed(content[:2000])
        with sqlite3.connect(self._db, timeout=30.0) as c:
            c.execute(
                "INSERT OR REPLACE INTO memory VALUES(?,?,?,?,?,?,?)",
                (entry_id, content, source, title or content[:60],
                 json.dumps(tags or []),
                 json.dumps(embedding) if embedding else "[]",
                 time.time()))
        return entry_id

    def search(self, query: str, top_n: int = 5,
               source_filter: str | None = None) -> list[MemoryResult]:
        """
        Find the most relevant memory entries for a query.
        Uses embedding cosine similarity when available, BM25 otherwise.
        """
        query_emb = self._embed(query)
        with sqlite3.connect(self._db, timeout=30.0) as c:
            sql  = "SELECT * FROM memory"
            args = []
            if source_filter:
                sql += " WHERE source=?"
                args.append(source_filter)
            sql += " ORDER BY timestamp DESC LIMIT 200"
            rows = c.execute(sql, args).fetchall()

        if not rows:
            return []

        scored = []
        for row in rows:
            content   = row[1]
            stored_emb = json.loads(row[5]) if row[5] else []
            if query_emb and stored_emb:
                score = self._cosine(query_emb, stored_emb)
            else:
                score = self._bm25(query, content)
            excerpt   = self._excerpt(query, content)
            entry     = MemoryEntry(
                entry_id = row[0], content=content, source=row[2],
                title=row[3], tags=json.loads(row[4]), timestamp=row[6])
            scored.append(MemoryResult(entry=entry, score=score,
                                        excerpt=excerpt))

        scored.sort(key=lambda r: r.score, reverse=True)
        return [r for r in scored[:top_n] if r.score > 0.15]

    def delete_by_tag(self, tag: str, source: str | None = None) -> int:
        """Delete all entries carrying *tag*, optionally limited to one source.

        Returns the number of rows removed. Used by the fact-store path to
        upsert by key: when the user says "my favourite colour is teal",
        any prior "my favourite colour is blue" entry must be removed
        before the new one is ingested — otherwise recall surfaces all
        historical values together as if they were equally true.

        Implementation note: tags are stored as a JSON array string in
        ``tags_json``, so we filter with ``LIKE`` over the serialised form.
        A false-positive substring match would need a tag that contains
        ``"<tag>"`` as a literal substring — vanishingly unlikely in
        practice and harmless if it happened (we'd just over-delete the
        same user's stored facts).
        """
        like = f'%"{tag}"%'
        with sqlite3.connect(self._db, timeout=30.0) as c:
            if source is not None:
                cur = c.execute(
                    "DELETE FROM memory WHERE tags_json LIKE ? AND source = ?",
                    (like, source),
                )
            else:
                cur = c.execute(
                    "DELETE FROM memory WHERE tags_json LIKE ?",
                    (like,),
                )
            return cur.rowcount or 0

    def ingest_conversation(self, role: str, content: str) -> str | None:
        """Store a single conversation turn. Returns the entry_id, or None if
        the turn was too short to store.

        The role is recorded as a ``role:<role>`` tag so recall sites can
        decide whether to surface the entry. Without this, PRISM's own
        past replies were being returned to the user as if they were
        stored facts (issue #28-6).
        """
        if len(content) > 50:   # skip very short turns
            return self.ingest(content, source="conversation",
                               title=f"{role}: {content[:40]}",
                               tags=[f"role:{role}"])
        return None

    # ── Helpers ───────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        try:
            import urllib.request as ur
            payload = json.dumps({"model":self._embed_m,
                                  "prompt":text}).encode()
            req = ur.Request(f"{self._ollama}/api/embeddings",
                data=payload,headers={"Content-Type":"application/json"})
            resp = ur.urlopen(req, timeout=5)
            return json.loads(resp.read()).get("embedding",[])
        except Exception:
            return []

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x*y for x,y in zip(a,b))
        na  = math.sqrt(sum(x*x for x in a))
        nb  = math.sqrt(sum(x*x for x in b))
        return dot / (na * nb + 1e-9)

    @staticmethod
    def _bm25(query: str, doc: str, k1: float=1.5, b: float=0.75) -> float:
        """Simplified single-document BM25 scoring."""
        q_terms  = query.lower().split()
        d_terms  = doc.lower().split()
        d_len    = len(d_terms)
        avg_len  = 100.0
        score    = 0.0
        for term in q_terms:
            tf = d_terms.count(term)
            if tf == 0:
                continue
            idf  = math.log(2.0)   # simplified: assume moderate rarity
            norm_tf = (tf*(k1+1)) / (tf + k1*(1-b+b*d_len/avg_len))
            score  += idf * norm_tf
        return min(1.0, score / max(len(q_terms),1))

    @staticmethod
    def _excerpt(query: str, content: str, max_len: int = 300) -> str:
        """Find the most relevant excerpt around query terms."""
        terms = query.lower().split()
        lower = content.lower()
        best_pos = 0
        best_count = 0
        for i in range(0, len(content)-50, 50):
            window = lower[i:i+200]
            count  = sum(1 for t in terms if t in window)
            if count > best_count:
                best_count = count
                best_pos = i
        start = max(0, best_pos)
        return content[start:start+max_len].strip()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db, timeout=30.0) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS memory(
                id TEXT PRIMARY KEY, content TEXT, source TEXT,
                title TEXT, tags_json TEXT, embedding_json TEXT,
                timestamp REAL)""")
            c.execute("CREATE INDEX IF NOT EXISTS ix_ts ON memory(timestamp)")
            c.execute("CREATE INDEX IF NOT EXISTS ix_src ON memory(source)")
            self._migrate(c)

    def _migrate(self, c) -> None:
        ver = c.execute("PRAGMA user_version").fetchone()[0]
        if ver < 1:
            c.execute("PRAGMA user_version = 1")
