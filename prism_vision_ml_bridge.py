"""
prism_vision_ml_bridge.py
=========================
Vision-to-Matrix bridge: converts raw image frames into numerical feature
matrices and feeds them directly into the Surgical ML Assembler.

Architecture (spec Section 5):
  [Frame bytes] → [VisionMatrixExtractor] → [FrameMatrix]
               → [VisionMLBridge.ingest()] → [MLAssembler.run()] → [AssemblyResult]

No text parsing — pixel intensities and spatial deltas are the features.
Decode strategy: PIL + numpy → numpy-only → pure Python (no external deps required).
"""
from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Spatial grid: GRID_SIZE × GRID_SIZE blocks
_GRID_SIZE   = 8
_N_BLOCKS    = _GRID_SIZE * _GRID_SIZE   # 64
_N_FEATURES  = _N_BLOCKS * 3            # intensity + delta + spatial_stats = 192

_MIN_FRAMES  = 3    # minimum buffer depth before ML runs
_MAX_BUFFER  = 100  # rolling window cap


# ---------------------------------------------------------------------------
# FrameMatrix
# ---------------------------------------------------------------------------

@dataclass
class FrameMatrix:
    """Numeric representation of a single image frame."""
    frame_id:       str
    timestamp:      float
    source:         str
    grid_size:      int
    n_features:     int
    intensity_grid: list  # mean intensity per block  [0, 1]
    delta_grid:     list  # Δ from previous frame     [-1, 1]
    spatial_stats:  list  # std per block             [0, 1]
    has_delta:      bool  # False for the first frame in a session


# ---------------------------------------------------------------------------
# VisionMatrixExtractor
# ---------------------------------------------------------------------------

class VisionMatrixExtractor:
    """
    Converts raw image bytes → FrameMatrix.

    Decode priority:
      1. PIL + numpy  (proper pixel decode — most accurate)
      2. numpy only   (treats raw byte stream as a 1-D signal)
      3. Pure Python  (byte-value chunk means — zero deps)
    """

    def __init__(self, grid_size: int = _GRID_SIZE) -> None:
        self._grid_size = grid_size
        self._n_blocks  = grid_size * grid_size
        self._prev_intensity: list | None = None

    def extract(self, image_bytes: bytes, source: str = "frame") -> FrameMatrix:
        """Convert *image_bytes* into a FrameMatrix, updating the rolling delta."""
        frame_id = uuid.uuid4().hex[:12]
        intensity, stats = self._decode(image_bytes)

        has_delta = self._prev_intensity is not None
        delta = (
            [a - b for a, b in zip(intensity, self._prev_intensity)]
            if has_delta and self._prev_intensity is not None
            else [0.0] * self._n_blocks
        )
        self._prev_intensity = intensity[:]

        return FrameMatrix(
            frame_id       = frame_id,
            timestamp      = time.time(),
            source         = source,
            grid_size      = self._grid_size,
            n_features     = self._n_blocks * 3,
            intensity_grid = intensity,
            delta_grid     = delta,
            spatial_stats  = stats,
            has_delta      = has_delta,
        )

    def to_feature_row(self, fm: FrameMatrix) -> list:
        """Flatten FrameMatrix into a 1-D feature vector (length = n_features)."""
        return fm.intensity_grid + fm.delta_grid + fm.spatial_stats

    def reset(self) -> None:
        """Clear the previous-frame reference (begin a new session)."""
        self._prev_intensity = None

    # ── Decode strategies ─────────────────────────────────────────────────────

    def _decode(self, data: bytes) -> tuple[list, list]:
        for strategy in (self._decode_pil, self._decode_numpy, self._decode_pure):
            try:
                return strategy(data)
            except Exception:
                pass
        return [0.0] * self._n_blocks, [0.0] * self._n_blocks

    def _decode_pil(self, data: bytes) -> tuple[list, list]:
        import io

        import numpy as np
        from PIL import Image  # type: ignore[import]
        n  = self._grid_size
        bp = 8  # each grid cell is bp × bp pixels
        img = Image.open(io.BytesIO(data)).convert("L")
        img = img.resize((n * bp, n * bp), Image.LANCZOS)  # type: ignore[attr-defined]
        arr = np.array(img, dtype=float) / 255.0
        intensity: list = []
        stats:     list = []
        for gy in range(n):
            for gx in range(n):
                block = arr[gy * bp:(gy + 1) * bp, gx * bp:(gx + 1) * bp]
                intensity.append(float(np.mean(block)))
                stats.append(float(np.std(block)))
        return intensity, stats

    def _decode_numpy(self, data: bytes) -> tuple[list, list]:
        import numpy as np
        n   = self._n_blocks
        arr = np.frombuffer(data if data else b"\x00", dtype=np.uint8).astype(float) / 255.0
        bsz = max(1, len(arr) // n)
        intensity: list = []
        stats:     list = []
        for i in range(n):
            chunk = arr[i * bsz: (i + 1) * bsz]
            if len(chunk) == 0:
                intensity.append(0.0)
                stats.append(0.0)
            else:
                intensity.append(float(np.mean(chunk)))
                stats.append(float(np.std(chunk)))
        return intensity, stats

    def _decode_pure(self, data: bytes) -> tuple[list, list]:
        n    = self._n_blocks
        arr  = list(data) if data else [0]
        bsz  = max(1, len(arr) // n)
        intensity: list = []
        for i in range(n):
            chunk = arr[i * bsz: (i + 1) * bsz] or [0]
            intensity.append(sum(chunk) / (255.0 * len(chunk)))
        stats = [0.0] * n  # std requires more ops; degrade to 0
        return intensity, stats


# ---------------------------------------------------------------------------
# VisionMLBridge
# ---------------------------------------------------------------------------

def _result_to_dict(result: Any) -> dict:
    """Safely serialise an AssemblyResult to a plain dict."""
    try:
        d = asdict(result)
    except Exception:
        d = vars(result) if hasattr(result, "__dict__") else {}
    pred = d.get("prediction")
    if pred is not None and hasattr(pred, "tolist"):
        d["prediction"] = pred.tolist()
    return d


class VisionMLBridge:
    """
    Accumulates frame feature rows in a rolling buffer and feeds them to the
    ML Assembler when enough frames have collected.

    Usage::

        bridge = VisionMLBridge(assembler=asm)
        result = bridge.ingest(frame_bytes, task="anomaly_detection")
    """

    def __init__(
        self,
        assembler:  Any,
        grid_size:  int = _GRID_SIZE,
        max_buffer: int = _MAX_BUFFER,
        min_frames: int = _MIN_FRAMES,
    ) -> None:
        self._asm        = assembler
        self._extractor  = VisionMatrixExtractor(grid_size=grid_size)
        self._buffer:    deque = deque(maxlen=max_buffer)
        self._min_frames = min_frames

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest(
        self,
        image_bytes: bytes,
        task:        str            = "visual_pattern_detection",
        y:           Optional[Any]  = None,
        source:      str            = "frame",
        translate:   bool           = False,
    ) -> dict:
        """
        Extract features from *image_bytes*, append to rolling buffer, and run
        the ML Assembler once ``min_frames`` have accumulated.

        Returns a dict with frame metadata plus ``ml_result`` when ready.
        """
        fm  = self._extractor.extract(image_bytes, source=source)
        row = self._extractor.to_feature_row(fm)
        self._buffer.append(row)

        out: dict = {
            "frame_id":       fm.frame_id,
            "timestamp":      fm.timestamp,
            "n_features":     fm.n_features,
            "frames_buffered": len(self._buffer),
            "min_frames":     self._min_frames,
            "has_delta":      fm.has_delta,
        }

        if len(self._buffer) >= self._min_frames:
            try:
                result    = self._asm.run(task, X=list(self._buffer), y=y, translate=translate)
                out["ml_result"] = _result_to_dict(result)
            except Exception as exc:
                logger.warning("[VisionMLBridge] assembler failed: %s", exc)
                out["ml_error"] = str(exc)
        else:
            out["status"] = "buffering"

        return out

    def extract_matrix(self, image_bytes: bytes, source: str = "frame") -> FrameMatrix:
        """Extract a FrameMatrix without touching the ML Assembler or buffer."""
        return self._extractor.extract(image_bytes, source=source)

    def clear(self) -> None:
        """Reset the frame buffer and delta reference."""
        self._buffer.clear()
        self._extractor.reset()

    def buffer_snapshot(self) -> list:
        """Return a copy of the current feature buffer."""
        return list(self._buffer)


# ---------------------------------------------------------------------------
# Module-level singleton wiring (mirrors prism_routes_ml pattern)
# ---------------------------------------------------------------------------

_bridge: Optional[VisionMLBridge] = None


def get_or_set_bridge(b: Optional[VisionMLBridge] = None) -> Optional[VisionMLBridge]:
    global _bridge
    if b is not None:
        _bridge = b
    return _bridge
