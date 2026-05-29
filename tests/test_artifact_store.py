from __future__ import annotations

import time

from artifact_store import Artifact, ArtifactStore


def _artifact_store(tmp_path):
    return ArtifactStore(db_path=str(tmp_path / "artifacts.db"))


def _artifact(domain: str = "sport", rating=None) -> Artifact:
    return Artifact(
        artifact_id="",
        user_name="TestUser",
        domain=domain,
        artifact_type="plan",
        title=f"{domain} artifact",
        content={"domain": domain},
        fulcrum_at_time=0.6,
        identity_version=2,
        created_at=time.time(),
        rating=rating,
    )


def test_save_returns_id(tmp_path):
    store = _artifact_store(tmp_path)
    artifact_id = store.save(_artifact())
    assert artifact_id


def test_recent_returns_list(tmp_path):
    store = _artifact_store(tmp_path)
    store.save(_artifact())
    recent = store.recent()
    assert isinstance(recent, list)
    assert recent


def test_rate_updates(tmp_path):
    store = _artifact_store(tmp_path)
    artifact_id = store.save(_artifact())
    store.rate(artifact_id, 0.9)
    artifact = store.get(artifact_id)
    assert artifact is not None
    assert artifact.rating == 0.9


def test_best_by_domain(tmp_path):
    store = _artifact_store(tmp_path)
    low_id = store.save(_artifact(domain="sport", rating=0.4))
    high_id = store.save(_artifact(domain="sport", rating=0.9))
    best = store.best_by_domain("sport")
    assert best
    assert best[0].artifact_id == high_id
