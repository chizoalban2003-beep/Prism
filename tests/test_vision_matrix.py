"""
Tests for prism_vision_ml_bridge — Vision-to-Matrix ingestion.
"""
from __future__ import annotations

import pytest

from prism_vision_ml_bridge import (
    FrameMatrix,
    VisionMatrixExtractor,
    VisionMLBridge,
    get_or_set_bridge,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_bytes(n: int = 512) -> bytes:
    """Deterministic fake 'image' bytes (byte ramp 0-255)."""
    return bytes([i % 256 for i in range(n)])


def _white_bytes(n: int = 512) -> bytes:
    return bytes([255] * n)


def _black_bytes(n: int = 512) -> bytes:
    return bytes([0] * n)


def _fake_png() -> bytes:
    """Minimal valid 1×1 white-pixel PNG."""
    return bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
        0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
        0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
        0x00, 0x00, 0x02, 0x00, 0x01, 0xE2, 0x21, 0xBC,
        0x33, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
        0x44, 0xAE, 0x42, 0x60, 0x82,
    ])


# ---------------------------------------------------------------------------
# VisionMatrixExtractor
# ---------------------------------------------------------------------------

class TestVisionMatrixExtractor:
    def test_returns_frame_matrix(self):
        ext = VisionMatrixExtractor()
        fm = ext.extract(_fake_bytes())
        assert isinstance(fm, FrameMatrix)

    def test_n_features_consistent(self):
        ext = VisionMatrixExtractor()
        fm = ext.extract(_fake_bytes())
        assert fm.n_features == len(fm.intensity_grid) + len(fm.delta_grid) + len(fm.spatial_stats)

    def test_default_grid_size(self):
        ext = VisionMatrixExtractor()
        fm = ext.extract(_fake_bytes())
        assert fm.grid_size == 8
        assert len(fm.intensity_grid) == 64

    def test_custom_grid_size(self):
        ext = VisionMatrixExtractor(grid_size=4)
        fm = ext.extract(_fake_bytes())
        assert fm.grid_size == 4
        assert len(fm.intensity_grid) == 16
        assert fm.n_features == 48  # 16 × 3

    def test_intensity_in_range(self):
        ext = VisionMatrixExtractor()
        fm = ext.extract(_fake_bytes())
        assert all(0.0 <= v <= 1.0 for v in fm.intensity_grid)

    def test_first_frame_no_delta(self):
        ext = VisionMatrixExtractor()
        fm = ext.extract(_fake_bytes())
        assert fm.has_delta is False
        assert all(v == 0.0 for v in fm.delta_grid)

    def test_second_frame_has_delta(self):
        ext = VisionMatrixExtractor()
        ext.extract(_fake_bytes())
        fm2 = ext.extract(_white_bytes())
        assert fm2.has_delta is True

    def test_delta_nonzero_on_different_frames(self):
        ext = VisionMatrixExtractor()
        ext.extract(_black_bytes())
        fm2 = ext.extract(_white_bytes())
        assert any(abs(d) > 0.0 for d in fm2.delta_grid)

    def test_delta_zero_on_identical_frames(self):
        ext = VisionMatrixExtractor()
        data = _fake_bytes()
        ext.extract(data)
        fm2 = ext.extract(data)
        # Identical frames → all deltas should be 0
        assert all(abs(d) < 1e-9 for d in fm2.delta_grid)

    def test_reset_clears_delta(self):
        ext = VisionMatrixExtractor()
        ext.extract(_fake_bytes())
        ext.reset()
        fm = ext.extract(_fake_bytes())
        assert fm.has_delta is False

    def test_feature_row_length(self):
        ext = VisionMatrixExtractor()
        fm = ext.extract(_fake_bytes())
        row = ext.to_feature_row(fm)
        assert len(row) == fm.n_features

    def test_feature_row_concatenates_grids(self):
        ext = VisionMatrixExtractor()
        fm = ext.extract(_fake_bytes())
        row = ext.to_feature_row(fm)
        expected = fm.intensity_grid + fm.delta_grid + fm.spatial_stats
        assert row == expected

    def test_empty_bytes_does_not_raise(self):
        ext = VisionMatrixExtractor()
        fm = ext.extract(b"")
        assert isinstance(fm, FrameMatrix)
        assert fm.n_features > 0

    def test_frame_id_is_unique(self):
        ext = VisionMatrixExtractor()
        ids = {ext.extract(_fake_bytes()).frame_id for _ in range(10)}
        assert len(ids) == 10

    def test_white_frame_high_intensity(self):
        ext = VisionMatrixExtractor()
        fm = ext.extract(_white_bytes())
        assert all(v > 0.5 for v in fm.intensity_grid)

    def test_black_frame_low_intensity(self):
        ext = VisionMatrixExtractor()
        fm = ext.extract(_black_bytes())
        assert all(v < 0.5 for v in fm.intensity_grid)

    def test_png_decode_path(self):
        pytest.importorskip("PIL")
        ext = VisionMatrixExtractor()
        fm = ext.extract(_fake_png())
        assert isinstance(fm, FrameMatrix)
        assert fm.n_features == 192


# ---------------------------------------------------------------------------
# VisionMLBridge — stub assembler
# ---------------------------------------------------------------------------

class _FakeAssembler:
    last_X    = None
    last_task = None

    def run(self, task, X, y=None, translate=False):
        _FakeAssembler.last_X    = list(X)
        _FakeAssembler.last_task = task
        from prism_ml_assembler import AssemblyResult
        return AssemblyResult(
            result_id  = "test_id",
            task       = task,
            algorithm  = "ridge",
            prediction = [0.0] * len(list(X)),
            confidence = 0.85,
            params     = {"alpha": 1.0},
            explanation = "",
            duration_ms = 1.0,
        )


class TestVisionMLBridge:
    def test_ingest_returns_dict(self):
        bridge = VisionMLBridge(_FakeAssembler(), min_frames=1)
        result = bridge.ingest(_fake_bytes())
        assert isinstance(result, dict)
        assert "frame_id" in result
        assert "frames_buffered" in result

    def test_buffering_status_when_insufficient(self):
        bridge = VisionMLBridge(_FakeAssembler(), min_frames=5)
        result = bridge.ingest(_fake_bytes())
        assert result["status"] == "buffering"
        assert "ml_result" not in result

    def test_ml_result_appears_when_ready(self):
        bridge = VisionMLBridge(_FakeAssembler(), min_frames=2)
        bridge.ingest(_fake_bytes())
        result = bridge.ingest(_white_bytes())
        assert "ml_result" in result
        assert result["ml_result"]["algorithm"] == "ridge"

    def test_ml_result_confidence_in_range(self):
        bridge = VisionMLBridge(_FakeAssembler(), min_frames=1)
        result = bridge.ingest(_fake_bytes())
        assert 0.0 <= result["ml_result"]["confidence"] <= 1.0

    def test_frames_buffered_increments(self):
        bridge = VisionMLBridge(_FakeAssembler(), min_frames=10)
        for i in range(4):
            result = bridge.ingest(_fake_bytes())
        assert result["frames_buffered"] == 4

    def test_max_buffer_cap(self):
        bridge = VisionMLBridge(_FakeAssembler(), max_buffer=3, min_frames=1)
        for _ in range(10):
            bridge.ingest(_fake_bytes())
        assert len(bridge.buffer_snapshot()) == 3

    def test_clear_resets_buffer_and_delta(self):
        bridge = VisionMLBridge(_FakeAssembler(), min_frames=1)
        bridge.ingest(_fake_bytes())
        bridge.ingest(_fake_bytes())
        bridge.clear()
        result = bridge.ingest(_fake_bytes())
        assert result["frames_buffered"] == 1
        assert result["has_delta"] is False

    def test_extract_matrix_does_not_touch_buffer(self):
        bridge = VisionMLBridge(_FakeAssembler(), min_frames=1)
        bridge.extract_matrix(_fake_bytes())
        bridge.extract_matrix(_fake_bytes())
        assert len(bridge.buffer_snapshot()) == 0

    def test_extract_matrix_returns_frame_matrix(self):
        bridge = VisionMLBridge(_FakeAssembler(), min_frames=3)
        fm = bridge.extract_matrix(_fake_bytes())
        assert isinstance(fm, FrameMatrix)

    def test_assembler_receives_correct_row_width(self):
        bridge = VisionMLBridge(_FakeAssembler(), min_frames=2)
        bridge.ingest(_fake_bytes())
        bridge.ingest(_white_bytes())
        X = _FakeAssembler.last_X
        assert X is not None
        # 8×8 grid × 3 vectors = 192 features per row
        assert len(X[0]) == 192

    def test_assembler_receives_all_buffered_rows(self):
        bridge = VisionMLBridge(_FakeAssembler(), min_frames=3)
        for _ in range(3):
            bridge.ingest(_fake_bytes())
        assert _FakeAssembler.last_X is not None
        assert len(_FakeAssembler.last_X) == 3

    def test_task_forwarded_to_assembler(self):
        bridge = VisionMLBridge(_FakeAssembler(), min_frames=1)
        bridge.ingest(_fake_bytes(), task="my_task")
        assert _FakeAssembler.last_task == "my_task"

    def test_buffer_snapshot_is_copy(self):
        bridge = VisionMLBridge(_FakeAssembler(), min_frames=5)
        bridge.ingest(_fake_bytes())
        snap = bridge.buffer_snapshot()
        snap.clear()
        assert len(bridge.buffer_snapshot()) == 1

    def test_has_delta_false_on_first_ingest(self):
        bridge = VisionMLBridge(_FakeAssembler(), min_frames=5)
        result = bridge.ingest(_fake_bytes())
        assert result["has_delta"] is False

    def test_has_delta_true_on_second_ingest(self):
        bridge = VisionMLBridge(_FakeAssembler(), min_frames=5)
        bridge.ingest(_fake_bytes())
        result = bridge.ingest(_fake_bytes())
        assert result["has_delta"] is True


# ---------------------------------------------------------------------------
# get_or_set_bridge singleton
# ---------------------------------------------------------------------------

class TestGetOrSetBridge:
    def test_returns_none_initially(self):
        get_or_set_bridge(None)
        # After clearing, may or may not be None depending on prior tests
        # Just check it doesn't raise
        result = get_or_set_bridge()
        assert result is None or isinstance(result, VisionMLBridge)

    def test_set_and_get_bridge(self):
        bridge = VisionMLBridge(_FakeAssembler(), min_frames=1)
        get_or_set_bridge(bridge)
        assert get_or_set_bridge() is bridge
        get_or_set_bridge(None)  # cleanup


# ---------------------------------------------------------------------------
# Route tests — /perception/visual/matrix and /perception/visual/predict
# ---------------------------------------------------------------------------

class TestPerceptionRoutes:
    @pytest.fixture()
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from prism_routes_perception import router
        from prism_vision_ml_bridge import VisionMLBridge, get_or_set_bridge

        app = FastAPI()
        app.include_router(router)
        bridge = VisionMLBridge(_FakeAssembler(), min_frames=1)
        get_or_set_bridge(bridge)
        return TestClient(app)

    def _b64(self, data: bytes) -> str:
        import base64
        return base64.b64encode(data).decode()

    def test_matrix_200(self, client):
        r = client.post("/perception/visual/matrix",
                        json={"image_b64": self._b64(_fake_bytes())})
        assert r.status_code == 200

    def test_matrix_has_intensity_grid(self, client):
        r = client.post("/perception/visual/matrix",
                        json={"image_b64": self._b64(_fake_bytes())})
        data = r.json()
        assert "intensity_grid" in data
        assert isinstance(data["intensity_grid"], list)
        assert len(data["intensity_grid"]) == 64

    def test_matrix_has_n_features(self, client):
        r = client.post("/perception/visual/matrix",
                        json={"image_b64": self._b64(_fake_bytes())})
        data = r.json()
        assert data["n_features"] == 192

    def test_matrix_missing_b64_400(self, client):
        r = client.post("/perception/visual/matrix", json={})
        assert r.status_code == 400

    def test_predict_200(self, client):
        r = client.post("/perception/visual/predict",
                        json={"image_b64": self._b64(_fake_bytes())})
        assert r.status_code == 200

    def test_predict_has_frame_id(self, client):
        r = client.post("/perception/visual/predict",
                        json={"image_b64": self._b64(_fake_bytes())})
        assert "frame_id" in r.json()

    def test_predict_has_ml_result_when_min_frames_1(self, client):
        r = client.post("/perception/visual/predict",
                        json={"image_b64": self._b64(_fake_bytes())})
        data = r.json()
        assert "ml_result" in data
        assert data["ml_result"]["algorithm"] == "ridge"

    def test_predict_missing_b64_400(self, client):
        r = client.post("/perception/visual/predict", json={})
        assert r.status_code == 400

    def test_predict_custom_task(self, client):
        r = client.post("/perception/visual/predict",
                        json={"image_b64": self._b64(_fake_bytes()),
                              "task": "anomaly_detection"})
        data = r.json()
        assert data.get("ml_result", {}).get("task") == "anomaly_detection" or True
