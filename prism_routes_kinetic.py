"""
prism_routes_kinetic.py
=======================
Diagnostic REST endpoints for the Kinetic compound signal engine.

These routes are read-only / diagnostic. Levers and links are configured
internally by the engine factory (KineticEngine.for_prism()) and by the
outcome tracker — not via REST. The signal ingestion endpoint exists for
testing and external integrations only.

Routes:
  GET  /kinetic/status   → {levers, windows_1h, links, compound_phi_delta}
  GET  /kinetic/windows  → {windows, count}  (?max_age=N seconds)
  POST /kinetic/signal   → ingest a PersonalSignal for testing → {windows}
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from prism_kinetic_engine import KineticEngine, PersonalSignal

router = APIRouter()

_engine: Optional[KineticEngine] = None


def _get_engine() -> KineticEngine:
    global _engine  # noqa: PLW0603
    if _engine is None:
        _engine = KineticEngine.for_prism()
    return _engine


def get_or_set_engine(engine: Optional[KineticEngine] = None) -> KineticEngine:
    """Called by prism_agent at startup to wire the shared engine instance."""
    global _engine  # noqa: PLW0603
    if engine is not None:
        _engine = engine
    return _get_engine()


def _window_dict(w: object) -> dict:
    from prism_kinetic_engine import ActionWindow
    if not isinstance(w, ActionWindow):
        return {}
    return {
        "window_id":   w.window_id,
        "lever_id":    w.lever_id,
        "domain":      w.source_signal.domain,
        "signal_type": w.source_signal.signal_type,
        "z_score":     round(w.source_signal.z_score, 4),
        "delta_a":     round(w.delta_a, 4),
        "is_crisis":   w.is_crisis,
        "triggered_at": w.triggered_at,
        "message":     w.to_proactive_message(),
    }


@router.get("/kinetic/status")
async def kinetic_status() -> JSONResponse:
    engine = _get_engine()
    return JSONResponse({
        "levers":            engine.lever_status(),
        "links":             engine.link_status(),
        "windows_1h":        len(engine.active_windows(3600.0)),
        "compound_phi_delta": round(engine.compound_phi_delta(), 4),
    })


@router.get("/kinetic/windows")
async def kinetic_windows(
    max_age: float = Query(default=3600.0, description="Age cutoff in seconds"),
) -> JSONResponse:
    engine = _get_engine()
    windows = engine.active_windows(max_age_seconds=max_age)
    return JSONResponse({
        "windows": [_window_dict(w) for w in windows],
        "count":   len(windows),
    })


@router.post("/kinetic/signal")
async def kinetic_ingest(body: dict) -> JSONResponse:
    """Ingest a personal signal. Primarily for testing and external integrations."""
    try:
        signal = PersonalSignal(
            domain=str(body["domain"]),
            signal_type=str(body["signal_type"]),
            raw_value=float(body["raw_value"]),
            mu=float(body["mu"]),
            sigma=float(body["sigma"]),
            impact=float(body.get("impact", 1.0)),
            confidence=float(body.get("confidence", 1.0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)

    new_windows = _get_engine().ingest(signal)
    return JSONResponse({"windows": [_window_dict(w) for w in new_windows]})
