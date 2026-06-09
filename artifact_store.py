from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from digital_identity import DigitalIdentity


@dataclass
class Artifact:
    """One saved decision artifact, tagged to an identity snapshot."""

    artifact_id: str
    user_name: str
    domain: str
    artifact_type: str
    title: str
    content: dict
    fulcrum_at_time: float
    identity_version: int
    created_at: float
    rating: Optional[float] = None


class ArtifactStore:
    """
    Persistent store for decision artifacts.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS artifacts (
        artifact_id TEXT PRIMARY KEY,
        user_name TEXT NOT NULL,
        domain TEXT NOT NULL,
        artifact_type TEXT NOT NULL,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        fulcrum_at_time REAL NOT NULL,
        identity_version INTEGER NOT NULL,
        created_at REAL NOT NULL,
        rating REAL
    );
    CREATE INDEX IF NOT EXISTS idx_artifacts_domain_created
    ON artifacts(domain, created_at DESC);
    """

    def __init__(self, db_path: str = "~/.prism/artifacts.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(self.SCHEMA)

    def _row_to_artifact(self, row: sqlite3.Row) -> Artifact:
        return Artifact(
            artifact_id=row["artifact_id"],
            user_name=row["user_name"],
            domain=row["domain"],
            artifact_type=row["artifact_type"],
            title=row["title"],
            content=json.loads(row["content"]),
            fulcrum_at_time=float(row["fulcrum_at_time"]),
            identity_version=int(row["identity_version"]),
            created_at=float(row["created_at"]),
            rating=None if row["rating"] is None else float(row["rating"]),
        )

    def save(self, artifact: Artifact) -> str:
        artifact_id = artifact.artifact_id or str(uuid.uuid4())
        created_at = float(artifact.created_at or time.time())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO artifacts (
                    artifact_id, user_name, domain, artifact_type, title, content,
                    fulcrum_at_time, identity_version, created_at, rating
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    artifact.user_name,
                    artifact.domain,
                    artifact.artifact_type,
                    artifact.title,
                    json.dumps(artifact.content),
                    float(artifact.fulcrum_at_time),
                    int(artifact.identity_version),
                    created_at,
                    artifact.rating,
                ),
            )
        return artifact_id

    def get(self, artifact_id: str) -> Optional[Artifact]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
        return self._row_to_artifact(row) if row else None

    def recent(self, domain: Optional[str] = None, n: int = 10) -> list[Artifact]:
        limit = max(1, int(n))
        with self._connect() as conn:
            if domain:
                rows = conn.execute(
                    """
                    SELECT * FROM artifacts
                    WHERE domain = ?
                    ORDER BY created_at DESC, artifact_id DESC
                    LIMIT ?
                    """,
                    (domain, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM artifacts
                    ORDER BY created_at DESC, artifact_id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [self._row_to_artifact(row) for row in rows]

    def rate(self, artifact_id: str, rating: float) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE artifacts SET rating = ? WHERE artifact_id = ?",
                (max(0.0, min(1.0, float(rating))), artifact_id),
            )

    def best_by_domain(self, domain: str, n: int = 5) -> list[Artifact]:
        """Return top-rated artifacts for a domain — used for identity validation."""
        limit = max(1, int(n))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM artifacts
                WHERE domain = ? AND rating IS NOT NULL
                ORDER BY rating DESC, created_at DESC, artifact_id DESC
                LIMIT ?
                """,
                (domain, limit),
            ).fetchall()
        return [self._row_to_artifact(row) for row in rows]

    def compose(
        self,
        domain: str,
        situation: dict,
        identity: "DigitalIdentity",
    ) -> dict:
        """
        Compose a new recommendation by finding the best-matching
        past artifact and adapting it to the current situation.
        """
        matches = self.best_by_domain(domain, n=1)
        if not matches:
            matches = self.recent(domain=domain, n=1)
        baseline = matches[0].content.copy() if matches else {}
        baseline.setdefault("title", f"{domain.title()} recommendation")
        baseline["domain"] = domain
        baseline["situation"] = situation
        baseline["identity_version"] = getattr(identity, "version", 1)
        baseline["identity_confidence"] = round(getattr(identity, "confidence", 0.0), 3)
        if matches:
            baseline["based_on_artifact_id"] = matches[0].artifact_id
        return baseline
