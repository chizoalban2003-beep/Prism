"""
prism_routes_ml.py
==================
REST endpoints for the Surgical ML Assembler.

Routes:
  POST /ml/run     → profile data + run algorithm + translate → AssemblyResult
  GET  /ml/status  → assembler health + nightly param overrides
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from prism_ml_assembler import MLAssembler

router = APIRouter()

_assembler: Optional[MLAssembler] = None


def get_or_set_assembler(assembler: Optional[MLAssembler] = None) -> MLAssembler:
    global _assembler  # noqa: PLW0603
    if assembler is not None:
        _assembler = assembler
    if _assembler is None:
        _assembler = MLAssembler()
    return _assembler


@router.get("/ml/status")
async def ml_status() -> JSONResponse:
    asm = get_or_set_assembler()
    return JSONResponse({
        "ready":          True,
        "nightly_params": asm._nightly_params,
        "thresholds": {
            "linear_r":        asm.LINEAR_R_THRESHOLD,
            "heavy_n":         asm.HEAVY_N_THRESHOLD,
            "torch_n":         asm.TORCH_N_THRESHOLD,
            "lasso_sparsity":  asm.LASSO_SPARSITY,
            "error_threshold": asm.ERROR_THRESHOLD,
        },
    })


@router.post("/ml/run")
async def ml_run(body: dict) -> JSONResponse:
    """
    Run the ML Assembler on provided data.

    Body:
      task          str             — human-readable task description
      X             list[list[float]] — feature matrix
      y             list[float]|null  — labels (null → unsupervised)
      feature_names list[str]|null
      translate     bool (default true)
    """
    try:
        task = str(body["task"])
        X = body["X"]
        y = body.get("y")
        feature_names = body.get("feature_names")
        translate = bool(body.get("translate", True))
        sequential = bool(body.get("sequential", False))
    except (KeyError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)

    asm = get_or_set_assembler()
    result = asm.run(
        task=task, X=X, y=y, feature_names=feature_names,
        translate=translate, sequential=sequential,
    )

    pred = result.prediction
    if hasattr(pred, "tolist"):
        pred = pred.tolist()
    elif hasattr(pred, "__iter__") and not isinstance(pred, (str, dict)):
        pred = list(pred)

    return JSONResponse({
        "result_id":   result.result_id,
        "algorithm":   result.algorithm,
        "confidence":  round(result.confidence, 4),
        "prediction":  pred,
        "explanation": result.explanation,
        "duration_ms": round(result.duration_ms, 2),
        "params":      result.params,
        "error":       result.error,
    })
