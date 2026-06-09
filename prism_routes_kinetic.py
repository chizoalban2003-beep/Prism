"""
prism_routes_kinetic.py
=======================
FastAPI router for Project Kinetic — cross-domain stochastic decision engine.

Routes:
  GET  /kinetic/status              → {levers, windows_24h, links}
  POST /kinetic/signal              → {windows}
  GET  /kinetic/windows             → {windows, count}  (?max_age=N seconds)
  GET  /kinetic/levers              → lever_status list
  POST /kinetic/levers              → register a new DecisionLever → {ok}
  POST /kinetic/links               → register a CrossElasticityLink → {ok}
  PUT  /kinetic/links/confidence    → update Bayesian confidence → {ok}
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from prism_kinetic_engine import CrossElasticityLink, DecisionLever, DomainSignal, KineticEngine

router = APIRouter()

# ── Lazy singleton ────────────────────────────────────────────────────────────

_engine: Optional[KineticEngine] = None


def _get_engine() -> KineticEngine:
    global _engine  # noqa: PLW0603
    if _engine is None:
        _engine = KineticEngine()
    return _engine


# ── Helper ────────────────────────────────────────────────────────────────────


def _window_dict(w: Any) -> dict:
    return {
        "window_id": w.window_id,
        "lever_id": w.lever_id,
        "source_domain": w.source_signal.domain,
        "signal_type": w.source_signal.signal_type,
        "z_score": round(w.source_signal.z_score, 4),
        "u_potential": round(w.u_potential, 4),
        "u_current": round(w.u_current, 4),
        "c_friction": round(w.c_friction, 4),
        "delta_a": round(w.delta_a, 4),
        "is_black_swan": w.is_black_swan,
        "triggered_at": w.triggered_at,
    }


# ── Routes ────────────────────────────────────────────────────────────────────


@router.get("/kinetic/status")
async def kinetic_status() -> JSONResponse:
    """Overall engine status: levers, 24-hour window count, and links."""
    engine = _get_engine()
    windows_24h = engine.active_windows(max_age_seconds=86400.0)
    return JSONResponse(
        {
            "levers": engine.lever_status(),
            "windows_24h": len(windows_24h),
            "links": engine.link_status(),
        }
    )


@router.post("/kinetic/signal")
async def kinetic_ingest_signal(body: dict) -> JSONResponse:
    """
    Ingest a domain signal and return any newly triggered arbitrage windows.

    Body fields: domain, signal_type, raw_value, mu, sigma,
                 value_at_risk (opt), probability (opt).
    """
    try:
        signal = DomainSignal(
            domain=str(body["domain"]),
            signal_type=str(body["signal_type"]),
            raw_value=float(body["raw_value"]),
            mu=float(body["mu"]),
            sigma=float(body["sigma"]),
            value_at_risk=float(body.get("value_at_risk", 0.0)),
            probability=float(body.get("probability", 1.0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)

    engine = _get_engine()
    new_windows = engine.ingest(signal)
    return JSONResponse({"windows": [_window_dict(w) for w in new_windows]})


@router.get("/kinetic/windows")
async def kinetic_windows(
    max_age: float = Query(default=3600.0, description="Window age cutoff in seconds"),
) -> JSONResponse:
    """Return arbitrage windows younger than max_age seconds (default 1 hour)."""
    engine = _get_engine()
    windows = engine.active_windows(max_age_seconds=max_age)
    return JSONResponse({"windows": [_window_dict(w) for w in windows], "count": len(windows)})


@router.get("/kinetic/levers")
async def kinetic_levers() -> JSONResponse:
    """Return current torque state of all registered levers."""
    engine = _get_engine()
    return JSONResponse(engine.lever_status())


@router.post("/kinetic/levers")
async def kinetic_add_lever(body: dict) -> JSONResponse:
    """Register a new decision lever."""
    try:
        lever = DecisionLever(
            lever_id=str(body["lever_id"]),
            name=str(body["name"]),
            description=str(body.get("description", "")),
            activate_threshold=float(body.get("activate_threshold", 3.0)),
            deactivate_threshold=float(body.get("deactivate_threshold", 1.5)),
            dashpot_threshold=float(body.get("dashpot_threshold", 10.0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)

    _get_engine().add_lever(lever)
    return JSONResponse({"ok": True, "lever_id": lever.lever_id})


@router.post("/kinetic/links")
async def kinetic_add_link(body: dict) -> JSONResponse:
    """Register a cross-elasticity link between two domains."""
    try:
        link = CrossElasticityLink(
            source_domain=str(body["source_domain"]),
            target_domain=str(body["target_domain"]),
            lambda_base=float(body["lambda_base"]),
            confidence=float(body.get("confidence", 1.0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)

    _get_engine().add_link(link)
    return JSONResponse({"ok": True})


@router.put("/kinetic/links/confidence")
async def kinetic_update_confidence(body: dict) -> JSONResponse:
    """Update the Bayesian confidence for an existing source→target link."""
    try:
        source = str(body["source"])
        target = str(body["target"])
        confidence = float(body["confidence"])
    except (KeyError, TypeError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)

    _get_engine().update_link_confidence(source, target, confidence)
    return JSONResponse({"ok": True})
