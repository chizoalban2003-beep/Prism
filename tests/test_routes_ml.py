"""
Tests for prism_routes_ml — /ml/* REST endpoints.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from prism_ml_assembler import MLAssembler
from prism_routes_ml import get_or_set_assembler, router


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(router)
    asm = MLAssembler()
    get_or_set_assembler(asm)
    return TestClient(app)


class TestMLStatus:
    def test_status_200(self, client):
        r = client.get("/ml/status")
        assert r.status_code == 200

    def test_status_has_ready(self, client):
        data = client.get("/ml/status").json()
        assert data["ready"] is True

    def test_status_has_thresholds(self, client):
        data = client.get("/ml/status").json()
        assert "thresholds" in data
        t = data["thresholds"]
        assert "linear_r" in t
        assert "heavy_n" in t
        assert "torch_n" in t


class TestMLRun:
    def test_run_missing_task_422(self, client):
        r = client.post("/ml/run", json={"X": [[1, 2]]})
        assert r.status_code == 422

    def test_run_missing_X_422(self, client):
        r = client.post("/ml/run", json={"task": "predict something"})
        assert r.status_code == 422

    def test_run_valid_labelled(self, client):
        pytest.importorskip("numpy")
        import numpy as np
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 3)).tolist()
        y = rng.standard_normal(30).tolist()
        payload = {"task": "predict score", "X": X, "y": y, "translate": False}
        r = client.post("/ml/run", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "algorithm" in data
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0

    def test_run_valid_unlabelled(self, client):
        pytest.importorskip("numpy")
        import numpy as np
        rng = np.random.default_rng(7)
        X = rng.standard_normal((30, 3)).tolist()
        payload = {"task": "cluster tasks", "X": X, "translate": False}
        r = client.post("/ml/run", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["algorithm"] in {"dbscan", "kmeans", "fallback_mean"}

    def test_run_returns_result_id(self, client):
        pytest.importorskip("numpy")
        import numpy as np
        X = np.random.randn(10, 2).tolist()
        y = np.random.randn(10).tolist()
        r = client.post("/ml/run", json={"task": "t", "X": X, "y": y, "translate": False})
        assert r.status_code == 200
        data = r.json()
        assert "result_id" in data
        assert len(data["result_id"]) > 0

    def test_run_prediction_is_list_or_scalar(self, client):
        pytest.importorskip("numpy")
        import numpy as np
        X = np.random.randn(15, 2).tolist()
        y = np.random.randn(15).tolist()
        r = client.post("/ml/run", json={"task": "t", "X": X, "y": y, "translate": False})
        data = r.json()
        pred = data["prediction"]
        assert isinstance(pred, (list, float, int))

    def test_sequential_flag_forwarded(self, client):
        """sequential=True in the body should be forwarded to the assembler."""
        pytest.importorskip("numpy")
        import numpy as np
        rng = np.random.default_rng(3)
        X = rng.random((40, 4)).tolist()
        y = (rng.random(40) * 2).tolist()
        r = client.post("/ml/run", json={"task": "t", "X": X, "y": y,
                                         "translate": False, "sequential": True})
        assert r.status_code == 200
        data = r.json()
        assert data["algorithm"] in {"lstm", "gru", "fallback_mean"}

    def test_run_no_numpy_fallback(self, client, monkeypatch):
        """Without numpy, assembler returns a fallback result — route still 200."""
        import builtins
        real_import = builtins.__import__

        def fail_numpy(name, *args, **kwargs):
            if name == "numpy":
                raise ImportError("no numpy")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fail_numpy)
        payload = {"task": "test", "X": [[1, 2], [3, 4]], "translate": False}
        r = client.post("/ml/run", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["algorithm"] == "fallback_mean"


class TestMLNightlySweep:
    def test_sweep_200_no_tracker(self, client):
        r = client.post("/ml/nightly_sweep")
        assert r.status_code == 200

    def test_sweep_returns_updated_dict(self, client):
        data = client.post("/ml/nightly_sweep").json()
        assert "updated" in data
        assert "algos_updated" in data

    def test_sweep_note_when_no_tracker(self, client):
        data = client.post("/ml/nightly_sweep").json()
        assert "note" in data
        assert "outcome_tracker" in data["note"]

    def test_sweep_with_tracker(self, client, monkeypatch):
        """With a mock tracker wired into _state, sweep returns algos_updated list."""
        from unittest.mock import MagicMock

        mock_tracker = MagicMock()
        mock_tracker.get_failed_outcomes.return_value = []

        import prism_state
        original_state = prism_state._state.copy()
        prism_state._state["outcome_tracker"] = mock_tracker
        try:
            r = client.post("/ml/nightly_sweep")
            assert r.status_code == 200
            data = r.json()
            assert "algos_updated" in data
            assert isinstance(data["algos_updated"], list)
        finally:
            prism_state._state.clear()
            prism_state._state.update(original_state)

    def test_sweep_algos_updated_is_list(self, client):
        data = client.post("/ml/nightly_sweep").json()
        assert isinstance(data["algos_updated"], list)
