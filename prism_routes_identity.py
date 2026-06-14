"""
prism_routes_identity.py
========================
FastAPI APIRouter — Identity Dashboard, Weekly Report, LIQUID Onboarding,
and Calibration Loop.

Routes
------
GET  /identity/ui                   — HTML visual identity dashboard
GET  /identity/dashboard            — JSON identity snapshot
GET  /identity/onboard              — HTML onboarding ceremony page
GET  /reports/weekly                — JSON latest reflection report
POST /reports/weekly/generate       — trigger immediate reflection run
GET  /onboarding/status             — ceremony completion status + current phase
POST /onboarding/start              — start (or restart) the identity ceremony
POST /onboarding/answer             — submit next ceremony answer
GET  /calibration/history           — calibration event history (with ?domain= ?n=)
GET  /calibration/summary           — aggregate stats + current VEAX state
POST /calibration/feedback          — programmatic calibration feedback
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from prism_state import _get_agent, _state

router = APIRouter()

_ONBOARDING_PATH = Path("~/.prism/onboarding_state.json").expanduser()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _soul():
    agent = _get_agent()
    return getattr(agent, "_soul", None) if agent else None


def _persona():
    agent = _get_agent()
    return getattr(agent, "_persona", None) if agent else None


def _phase_engine():
    agent = _get_agent()
    return getattr(agent, "_phase", None) if agent else None


def _kinetic_engine():
    agent = _get_agent()
    return getattr(agent, "_kinetic", None) if agent else None


def _reflection():
    agent = _get_agent()
    return getattr(agent, "_reflection", None) if agent else None


def _router_llm():
    agent = _get_agent()
    return getattr(agent, "_router", None) if agent else None


# ---------------------------------------------------------------------------
# GET /identity/dashboard — JSON snapshot
# ---------------------------------------------------------------------------


@router.get("/identity/dashboard")
async def identity_dashboard():
    """Full identity snapshot: phase, soul, persona, growth, tensions."""
    soul = _soul()
    persona = _persona()

    snapshot: dict[str, Any] = {"generated_at": time.time()}

    # Phase
    phase_engine = _phase_engine()
    if phase_engine is not None:
        try:
            reading = phase_engine.compute(soul, None, kinetic=_kinetic_engine())
            phase_val = reading.phase.value if hasattr(reading.phase, "value") else str(reading.phase)
            snapshot["phase"] = {
                "current": phase_val,
                "phi": round(reading.phi, 3),
                "delta_H": round(reading.delta_H, 3),
                "delta_K": round(reading.delta_K, 3),
            }
        except Exception:
            snapshot["phase"] = {"current": "UNKNOWN", "phi": 0.0, "delta_H": 0.0, "delta_K": 0.0}
    else:
        snapshot["phase"] = {"current": "UNKNOWN", "phi": 0.0, "delta_H": 0.0, "delta_K": 0.0}

    # Soul
    if soul is not None:
        try:
            seed = soul.get_seed()
            beliefs = soul.list_beliefs()
            lenses = soul.list_lenses()
            delta = soul.delta_report()
            snapshot["soul"] = {
                "has_seed": seed is not None,
                "stated_values": seed.stated_values if seed else [],
                "stated_goals": seed.stated_goals if seed else [],
                "stated_constraints": seed.stated_constraints if seed else [],
                "beliefs": [
                    {
                        "text": b.text,
                        "type": b.belief_type,
                        "source": b.source,
                        "confidence": round(b.confidence, 3),
                        "observations": b.observation_count,
                    }
                    for b in beliefs[:20]
                ],
                "lenses": [
                    {
                        "name": ln.name,
                        "description": ln.description,
                        "trend": round(ln.trend, 3) if hasattr(ln, "trend") else None,
                    }
                    for ln in lenses
                ],
                "tensions": [
                    {
                        "stated": t.get("stated_text", ""),
                        "observed": t.get("observed_text", ""),
                        "relation": t.get("relation", "contradicts"),
                    }
                    for t in (delta or [])[:10]
                ],
            }
        except Exception as exc:
            snapshot["soul"] = {"error": str(exc)}
    else:
        snapshot["soul"] = None

    # Persona
    if persona is not None:
        try:
            traits = persona.list_traits()
            growth = persona.growth_since(days=7)
            peak = persona.peak_hours()
            obs_count = sum(t.observation_count for t in traits) if traits else 0
            snapshot["persona"] = {
                "traits": [
                    {
                        "name": t.name,
                        "value": t.value,
                        "confidence": round(t.confidence, 3),
                        "source": t.source,
                        "observations": t.observation_count,
                    }
                    for t in traits
                ],
                "peak_hours": peak,
                "growth": growth,
                "total_observations": obs_count,
            }
        except Exception as exc:
            snapshot["persona"] = {"error": str(exc)}
    else:
        snapshot["persona"] = None

    # Crystallisation percentage — mean trait confidence
    try:
        trait_list = (snapshot.get("persona") or {}).get("traits", [])
        if trait_list:
            avg_conf = sum(t["confidence"] for t in trait_list) / len(trait_list)
            snapshot["crystallisation_pct"] = round(avg_conf * 100, 1)
        else:
            snapshot["crystallisation_pct"] = 0.0
    except Exception:
        snapshot["crystallisation_pct"] = 0.0

    return snapshot


# ---------------------------------------------------------------------------
# GET /identity — HTML visual dashboard
# ---------------------------------------------------------------------------


@router.get("/identity/ui", response_class=HTMLResponse)
async def identity_html():
    """Render the visual identity dashboard page."""
    return HTMLResponse(_identity_dashboard_html())


# ---------------------------------------------------------------------------
# GET /identity/onboard — HTML onboarding ceremony page
# ---------------------------------------------------------------------------


@router.get("/identity/onboard", response_class=HTMLResponse)
async def identity_onboard_html():
    """Serve the interactive identity ceremony onboarding page."""
    return HTMLResponse(_onboarding_html())


# ---------------------------------------------------------------------------
# GET /reports/weekly — latest reflection report
# ---------------------------------------------------------------------------


@router.get("/reports/weekly")
async def weekly_report():
    """Return the most recent weekly reflection report."""
    reflection = _reflection()
    if reflection is None:
        return JSONResponse(
            {"error": "Reflection engine not available", "status": 503},
            status_code=503,
        )

    cached = _state.get("_last_weekly_report")
    if cached is not None:
        return cached

    return {"info": "No report yet — POST /reports/weekly/generate to run one."}


# ---------------------------------------------------------------------------
# POST /reports/weekly/generate — trigger immediate reflection
# ---------------------------------------------------------------------------


@router.post("/reports/weekly/generate")
async def weekly_report_generate():
    """Trigger an immediate reflection run and cache the result."""
    reflection = _reflection()
    if reflection is None:
        return JSONResponse(
            {"error": "Reflection engine not available", "status": 503},
            status_code=503,
        )

    try:
        report = reflection.run()
        result = {
            "ran_at": getattr(report, "ran_at", time.time()),
            "summary": report.summary,
            "patterns": list(report.patterns or []),
            "belief_proposals": [
                {
                    "node_id": p.get("node_id"),
                    "text": p.get("text"),
                    "new_confidence": p.get("new_confidence"),
                    "rationale": p.get("rationale"),
                }
                for p in (report.belief_proposals or [])
            ],
            "unresolved_goals": list(report.unresolved_goals or []),
            "applied": report.applied,
            "error": report.error,
        }
        _state["_last_weekly_report"] = result
        return result
    except Exception as exc:
        return JSONResponse({"error": str(exc), "status": 500}, status_code=500)


# ---------------------------------------------------------------------------
# GET /onboarding/status
# ---------------------------------------------------------------------------


@router.get("/onboarding/status")
async def onboarding_status():
    """Return ceremony completion status, current phase, and observation count."""
    soul = _soul()
    has_seed = False
    if soul is not None:
        try:
            has_seed = bool(soul.has_seed())
        except Exception:
            pass

    obs_count = 0
    persona = _persona()
    if persona is not None:
        try:
            traits = persona.list_traits()
            obs_count = sum(t.observation_count for t in traits)
        except Exception:
            pass

    current_phase = "UNKNOWN"
    phase_engine = _phase_engine()
    if phase_engine is not None:
        try:
            current_phase = str(getattr(phase_engine, "current_phase", "UNKNOWN"))
        except Exception:
            pass

    state_data = _load_onboarding_state()

    if has_seed and current_phase in ("CRYSTAL", "STABLE"):
        message = "Your Prism is crystallised."
    elif has_seed:
        message = "Identity ceremony complete. Prism is learning from you."
    else:
        message = "Start the identity ceremony to personalise Prism."

    return {
        "ceremony_complete": has_seed,
        "answers_submitted": len(state_data.get("answers", {})),
        "questions_total": 7,
        "current_phase": current_phase,
        "observations": obs_count,
        "message": message,
    }


# ---------------------------------------------------------------------------
# POST /onboarding/start
# ---------------------------------------------------------------------------


@router.post("/onboarding/start")
async def onboarding_start():
    """
    Start (or restart) the identity ceremony.

    Returns the first question and total count.
    """
    try:
        from prism_identity_ceremony import CEREMONY_QUESTIONS
        questions = list(CEREMONY_QUESTIONS.items())
    except Exception as exc:
        return JSONResponse(
            {"error": f"Ceremony module unavailable: {exc}"},
            status_code=503,
        )

    _save_onboarding_state({"answers": {}, "started_at": time.time()})

    key, question = questions[0]
    return {
        "started": True,
        "question_index": 0,
        "question_key": key,
        "question": question,
        "total_questions": len(questions),
        "progress": f"1/{len(questions)}",
    }


# ---------------------------------------------------------------------------
# POST /onboarding/answer
# ---------------------------------------------------------------------------


@router.post("/onboarding/answer")
async def onboarding_answer(request: Request):
    """
    Submit the answer to the current ceremony question.

    Body: ``{"answer": "..."}``

    Returns the next question, or a completion summary when all are answered.
    """
    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    answer: str = (body.get("answer") or "").strip()
    if not answer:
        return JSONResponse({"error": "'answer' is required"}, status_code=400)

    try:
        from prism_identity_ceremony import CEREMONY_QUESTIONS
        questions = list(CEREMONY_QUESTIONS.items())
    except Exception as exc:
        return JSONResponse(
            {"error": f"Ceremony module unavailable: {exc}"},
            status_code=503,
        )

    state_data = _load_onboarding_state()
    answers: dict = state_data.get("answers", {})

    q_index = len(answers)
    if q_index >= len(questions):
        return JSONResponse({"error": "Ceremony already complete"}, status_code=400)

    key, _question = questions[q_index]
    answers[key] = answer
    state_data["answers"] = answers
    _save_onboarding_state(state_data)

    next_index = q_index + 1
    if next_index >= len(questions):
        soul = _soul()
        seed_result = None
        if soul is not None:
            try:
                from prism_identity_ceremony import IdentityCeremony
                ceremony = IdentityCeremony(soul=soul, llm_router=_router_llm())
                seed = ceremony.run_from_answers(answers)
                seed_result = {
                    "stated_values": list(seed.stated_values[:5]) if seed else [],
                    "stated_goals": list(seed.stated_goals[:3]) if seed else [],
                }
                _save_onboarding_state({
                    "answers": answers,
                    "completed_at": time.time(),
                })
            except Exception as exc:
                return JSONResponse(
                    {"error": f"Ceremony completion failed: {exc}"},
                    status_code=500,
                )
        return {
            "complete": True,
            "message": "Identity ceremony complete. Your Prism is initialised.",
            "seed": seed_result,
        }

    next_key, next_question = questions[next_index]
    return {
        "complete": False,
        "question_index": next_index,
        "question_key": next_key,
        "question": next_question,
        "total_questions": len(questions),
        "progress": f"{next_index + 1}/{len(questions)}",
    }


# ---------------------------------------------------------------------------
# Onboarding state persistence
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------


def _calibration():
    agent = _get_agent()
    return getattr(agent, "_calibration", None) if agent else None


# ---------------------------------------------------------------------------
# GET /calibration/history
# ---------------------------------------------------------------------------


@router.get("/calibration/history")
async def calibration_history(domain: str = "", n: int = 20):
    """Return recent calibration events, optionally filtered by domain."""
    cal = _calibration()
    if cal is None:
        return JSONResponse(
            {"error": "Calibration engine not available", "status": 503},
            status_code=503,
        )
    events = cal.history(domain=domain or None, n=n)
    return {
        "events": [
            {
                "event_id": e.event_id,
                "domain": e.domain,
                "direction": e.direction,
                "factor_id": e.factor_id,
                "adjustment": e.adjustment,
                "message": e.message,
                "timestamp": e.timestamp,
            }
            for e in events
        ],
        "count": len(events),
    }


# ---------------------------------------------------------------------------
# GET /calibration/summary
# ---------------------------------------------------------------------------


@router.get("/calibration/summary")
async def calibration_summary_route():
    """Return aggregate calibration stats + current VEAX state."""
    cal = _calibration()
    if cal is None:
        return JSONResponse(
            {"error": "Calibration engine not available", "status": 503},
            status_code=503,
        )

    summary_text = cal.summary()
    events = cal.history(n=50)

    from collections import Counter
    direction_counts = dict(Counter(e.direction for e in events))
    domain_counts = dict(Counter(e.domain for e in events))

    avg_adj = sum(abs(e.adjustment) for e in events) / max(len(events), 1)

    veax_state = None
    veax_render = None
    try:
        from prism_veax import get_current_gates, render_gates
        gates = get_current_gates()
        if gates is not None:
            veax_state = {"V": gates.V, "E": gates.E, "A": gates.A, "X": gates.X}
            veax_render = render_gates(gates)
    except Exception:
        pass

    return {
        "summary": summary_text,
        "total_events": len(events),
        "direction_counts": direction_counts,
        "domain_counts": domain_counts,
        "avg_adjustment_magnitude": round(avg_adj, 4),
        "veax": veax_state,
        "veax_render": veax_render,
    }


# ---------------------------------------------------------------------------
# POST /calibration/feedback — programmatic feedback
# ---------------------------------------------------------------------------


@router.post("/calibration/feedback")
async def calibration_feedback(request: Request):
    """
    Submit calibration feedback programmatically.

    Body::

        {
            "message":          "that was too aggressive",
            "domain":           "sport",         # optional
            "fulcrum_position": 0.72             # optional, 0-1
        }

    Returns the calibration event + before/after VEAX render.
    """
    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    message: str = (body.get("message") or "").strip()
    if not message:
        return JSONResponse({"error": "'message' is required"}, status_code=400)

    cal = _calibration()
    if cal is None:
        return JSONResponse(
            {"error": "Calibration engine not available", "status": 503},
            status_code=503,
        )

    direction = cal.detect(message)
    if not direction:
        return JSONResponse(
            {"error": "Message not recognised as feedback", "recognised_patterns": list(cal.FEEDBACK_PATTERNS.keys())},
            status_code=422,
        )

    last_decision = {
        "domain": body.get("domain", "general"),
        "fulcrum_position": float(body.get("fulcrum_position", 0.5)),
        "factors": {},
    }

    veax_before = None
    veax_after = None
    try:
        from prism_veax import (
            SpectrumGates,
            get_current_gates,
            render_gates,
            save_spectrum_state,
        )
        _VEAX_DELTAS: dict[str, dict[str, float]] = {
            "too_aggressive":   {"A": -0.05, "V": +0.03},
            "too_conservative": {"A": +0.05, "V": -0.03},
            "wrong":            {"V": +0.05},
            "correct":          {"X": +0.02},
        }
        gates = get_current_gates()
        if gates is not None:
            veax_before = {"V": gates.V, "E": gates.E, "A": gates.A, "X": gates.X}
            deltas = _VEAX_DELTAS.get(direction, {})
            if deltas:
                new_gates = SpectrumGates(
                    V=max(0.0, min(1.0, gates.V + deltas.get("V", 0.0))),
                    E=max(0.0, min(1.0, gates.E + deltas.get("E", 0.0))),
                    A=max(0.0, min(1.0, gates.A + deltas.get("A", 0.0))),
                    X=max(0.0, min(1.0, gates.X + deltas.get("X", 0.0))),
                )
                save_spectrum_state(new_gates)
                veax_after = {"V": new_gates.V, "E": new_gates.E, "A": new_gates.A, "X": new_gates.X}
                render_before = render_gates(gates)
                render_after = render_gates(new_gates)
            else:
                render_before = render_gates(gates)
                render_after = None
        else:
            render_before = None
            render_after = None
    except Exception:
        render_before = None
        render_after = None

    event = cal.process(
        message=message,
        direction=direction,
        last_decision=last_decision,
        beam=None,
        llm_router=_router_llm(),
    )

    return {
        "event": {
            "event_id": event.event_id,
            "domain": event.domain,
            "direction": event.direction,
            "factor_id": event.factor_id,
            "adjustment": event.adjustment,
        },
        "veax_before": veax_before,
        "veax_after": veax_after,
        "render_before": render_before,
        "render_after": render_after,
        "summary": cal.summary(),
    }


# ---------------------------------------------------------------------------
# Onboarding state persistence
# ---------------------------------------------------------------------------


def _load_onboarding_state() -> dict:
    try:
        if _ONBOARDING_PATH.exists():
            return json.loads(_ONBOARDING_PATH.read_text())
    except Exception:
        pass
    return {}


def _save_onboarding_state(data: dict) -> None:
    try:
        _ONBOARDING_PATH.parent.mkdir(parents=True, exist_ok=True)
        _ONBOARDING_PATH.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# HTML templates
# ---------------------------------------------------------------------------


def _identity_dashboard_html() -> str:  # noqa: E501
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Your Prism — Identity</title>
<style>
:root{--bg:#0d0d0d;--card:#161616;--border:#2a2a2a;--text:#e8e8e8;--muted:#888;--accent:#7c6ff7;--crystal:#5eead4;--liquid:#f97316;--stable:#60a5fa;--viscous:#facc15}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:14px;line-height:1.6;padding:24px;max-width:1200px;margin:0 auto}
h1{font-size:22px;font-weight:700;margin-bottom:4px}
.subtitle{color:var(--muted);font-size:13px;margin-bottom:28px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:20px}
.card h2{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin-bottom:14px}
.phase-badge{display:inline-flex;align-items:center;gap:8px;padding:6px 14px;border-radius:20px;font-size:16px;font-weight:700;margin-bottom:12px}
.phase-CRYSTAL{background:rgba(94,234,212,.12);color:var(--crystal);border:1px solid rgba(94,234,212,.3)}
.phase-STABLE{background:rgba(96,165,250,.12);color:var(--stable);border:1px solid rgba(96,165,250,.3)}
.phase-VISCOUS{background:rgba(250,204,21,.12);color:var(--viscous);border:1px solid rgba(250,204,21,.3)}
.phase-LIQUID{background:rgba(249,115,22,.12);color:var(--liquid);border:1px solid rgba(249,115,22,.3)}
.phase-UNKNOWN{background:rgba(136,136,136,.12);color:var(--muted);border:1px solid var(--border)}
.phi-bar{height:6px;background:var(--border);border-radius:3px;margin:10px 0 14px}
.phi-fill{height:100%;border-radius:3px;background:linear-gradient(90deg,var(--crystal),var(--accent));transition:width .8s ease}
.row{display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid var(--border);font-size:13px}
.row:last-child{border-bottom:none}
.row-label{color:var(--muted)}
.row-value{font-weight:600}
.trait-item{margin-bottom:12px}
.trait-header{display:flex;justify-content:space-between;margin-bottom:5px;font-size:13px}
.trait-name-text{font-weight:500}
.trait-meta{color:var(--muted)}
.trait-track{height:5px;background:var(--border);border-radius:3px}
.trait-fill{height:100%;border-radius:3px;background:var(--accent)}
.belief-item{padding:9px 0;border-bottom:1px solid var(--border)}
.belief-item:last-child{border-bottom:none}
.belief-text{font-size:13px;margin-bottom:4px}
.tags{display:flex;gap:5px;flex-wrap:wrap}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;background:var(--border);color:var(--muted)}
.tag-value{background:rgba(124,111,247,.15);color:var(--accent)}
.tag-observed{background:rgba(94,234,212,.15);color:var(--crystal)}
.tension-item{padding:9px 0;border-bottom:1px solid var(--border)}
.tension-item:last-child{border-bottom:none}
.tension-label{font-size:10px;font-weight:700;text-transform:uppercase;color:var(--liquid);margin-bottom:3px;letter-spacing:.05em}
.tension-text{font-size:13px}
.tension-observed{font-size:12px;color:var(--muted);margin-top:2px}
.value-pill{display:inline-block;padding:3px 12px;margin:3px;border-radius:12px;background:rgba(124,111,247,.15);color:var(--accent);font-size:12px}
.onboarding-cta{background:rgba(124,111,247,.08);border:1px solid rgba(124,111,247,.2);border-radius:10px;padding:20px;margin-bottom:20px;text-align:center;font-size:14px}
.onboarding-cta a{color:var(--accent);text-decoration:none;font-weight:600}
.onboarding-cta a:hover{text-decoration:underline}
.moat{text-align:center;color:var(--muted);font-size:12px;margin-top:28px;padding:16px;border-top:1px solid var(--border)}
.loading{color:var(--muted);font-style:italic;padding:40px;text-align:center}
.refresh-btn{float:right;background:none;border:1px solid var(--border);color:var(--muted);padding:4px 12px;border-radius:6px;cursor:pointer;font-size:12px}
.refresh-btn:hover{border-color:var(--accent);color:var(--accent)}
</style>
</head>
<body>
<h1>Your Prism <button class="refresh-btn" onclick="load()">Refresh</button></h1>
<p class="subtitle">Identity dashboard — the first AI that becomes a different system for every person who uses it.</p>
<div id="app"><p class="loading">Loading identity snapshot…</p></div>

<script>
async function load(){
  const app=document.getElementById('app');
  try{
    const r=await fetch('/identity/dashboard');
    if(!r.ok)throw new Error(r.status);
    const d=await r.json();
    app.innerHTML=render(d);
  }catch(e){
    app.innerHTML='<p style="color:var(--liquid);padding:20px">Could not load identity data: '+e.message+'</p>';
  }
}

function render(d){
  const phase=d.phase||{};const soul=d.soul||{};const persona=d.persona||{};
  const growth=persona.growth||{};const phaseName=phase.current||'UNKNOWN';
  const phi=phase.phi||0;const crystallPct=d.crystallisation_pct||0;
  let html='';

  if(!soul.has_seed){
    html+=`<div class="onboarding-cta">
      <strong>Your Prism hasn\'t met you yet.</strong><br><br>
      <a href="/identity/onboard">Start the identity ceremony</a> — 7 questions, 3 minutes, creates your digital soul.
    </div>`;
  }

  html+='<div class="grid">';

  // Phase card
  html+=`<div class="card"><h2>Crystallisation Phase</h2>
    <div class="phase-badge phase-${phaseName}">${phaseName}</div>
    <div class="phi-bar"><div class="phi-fill" style="width:${Math.min(100,Math.round(phi*100))}%"></div></div>
    <div class="row"><span class="row-label">Φ_melt (pressure)</span><span class="row-value">${phi.toFixed(3)}</span></div>
    <div class="row"><span class="row-label">ΔH (hardware)</span><span class="row-value">${(phase.delta_H||0).toFixed(3)}</span></div>
    <div class="row"><span class="row-label">ΔK (soul tension)</span><span class="row-value">${(phase.delta_K||0).toFixed(3)}</span></div>
    <div class="row"><span class="row-label">Identity confidence</span><span class="row-value">${crystallPct}%</span></div>
  </div>`;

  // Values + goals
  if(soul.stated_values&&soul.stated_values.length){
    html+=`<div class="card"><h2>Your Values</h2>
      <div style="margin-bottom:14px">${soul.stated_values.map(v=>`<span class="value-pill">${v}</span>`).join('')}</div>
      <h2>Your Goals</h2>
      <div>${(soul.stated_goals||[]).map(g=>`<span class="value-pill">${g}</span>`).join('')||'<span style="color:var(--muted)">None set</span>'}</div>
    </div>`;
  }

  // Traits
  const traits=persona.traits||[];
  if(traits.length){
    html+=`<div class="card"><h2>Crystallised Traits (${traits.length})</h2>`;
    traits.slice(0,8).forEach(t=>{
      const pct=Math.round(t.confidence*100);
      html+=`<div class="trait-item">
        <div class="trait-header"><span class="trait-name-text">${t.name.replace(/_/g,' ')}</span><span class="trait-meta">${t.value} · ${pct}%</span></div>
        <div class="trait-track"><div class="trait-fill" style="width:${pct}%"></div></div>
      </div>`;
    });
    html+='</div>';
  }

  // Growth
  if(persona.total_observations>0){
    const peakHours=(persona.peak_hours||[]).map(h=>h+':00').join(', ');
    html+=`<div class="card"><h2>7-Day Growth</h2>
      <div class="row"><span class="row-label">New traits</span><span class="row-value">+${growth.new_traits||0}</span></div>
      <div class="row"><span class="row-label">New patterns</span><span class="row-value">+${growth.new_patterns||0}</span></div>
      <div class="row"><span class="row-label">Avg confidence</span><span class="row-value">${((growth.confidence_avg||0)*100).toFixed(1)}%</span></div>
      <div class="row"><span class="row-label">Total observations</span><span class="row-value">${persona.total_observations}</span></div>
      ${peakHours?`<div class="row"><span class="row-label">Peak hours</span><span class="row-value">${peakHours}</span></div>`:''}
    </div>`;
  }

  // Beliefs
  const beliefs=(soul.beliefs||[]).slice(0,6);
  if(beliefs.length){
    html+=`<div class="card"><h2>Soul Beliefs (${(soul.beliefs||[]).length})</h2>`;
    beliefs.forEach(b=>{
      const tcls=b.source==='observed'?'tag-observed':'tag-value';
      html+=`<div class="belief-item"><div class="belief-text">${b.text}</div>
        <div class="tags"><span class="tag ${tcls}">${b.source}</span><span class="tag">${b.type}</span><span class="tag">${Math.round(b.confidence*100)}% conf</span></div>
      </div>`;
    });
    html+='</div>';
  }

  // Tensions
  const tensions=(soul.tensions||[]).slice(0,4);
  if(tensions.length){
    html+=`<div class="card"><h2>Active Tensions (${(soul.tensions||[]).length})</h2>
      <p style="font-size:12px;color:var(--muted);margin-bottom:12px">Stated values diverging from observed behaviour</p>`;
    tensions.forEach(t=>{
      html+=`<div class="tension-item">
        <div class="tension-label">contradiction</div>
        <div class="tension-text">${t.stated||'—'}</div>
        <div class="tension-observed">observed: ${t.observed||'—'}</div>
      </div>`;
    });
    html+='</div>';
  }

  html+='</div>';
  html+=`<div class="moat">Your decision history, identity model, and organ configuration stay on this device.<br>
    They cannot be read, trained on, or transferred without your consent. <strong>This is your Prism.</strong></div>`;
  return html;
}

load();
</script>
</body>
</html>"""


def _onboarding_html() -> str:  # noqa: E501
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Prism — Identity Ceremony</title>
<style>
:root{--bg:#0d0d0d;--card:#161616;--border:#2a2a2a;--text:#e8e8e8;--muted:#888;--accent:#7c6ff7}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:15px;line-height:1.7;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:24px}
.container{max-width:560px;width:100%}
.header{text-align:center;margin-bottom:40px}
.header h1{font-size:24px;font-weight:700;margin-bottom:8px}
.header p{color:var(--muted);font-size:14px}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:28px}
.progress-bar{height:3px;background:var(--border);border-radius:2px;margin-bottom:28px}
.progress-fill{height:100%;background:var(--accent);border-radius:2px;transition:width .4s ease}
.step-label{font-size:12px;color:var(--muted);margin-bottom:8px}
.question{font-size:18px;font-weight:600;margin-bottom:20px;line-height:1.4}
textarea{width:100%;background:#111;border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:14px;font-family:inherit;line-height:1.6;padding:12px;resize:vertical;min-height:100px}
textarea:focus{outline:none;border-color:var(--accent)}
.actions{display:flex;justify-content:flex-end;gap:12px;margin-top:16px}
button{padding:10px 24px;border-radius:8px;font-size:14px;font-weight:600;cursor:pointer;border:none}
.btn-next{background:var(--accent);color:#fff}
.btn-next:hover{opacity:.9}
.btn-skip{background:var(--border);color:var(--muted)}
.btn-skip:hover{color:var(--text)}
.complete{text-align:center;padding:20px 0}
.complete h2{font-size:22px;font-weight:700;margin-bottom:12px}
.complete p{color:var(--muted);margin-bottom:20px}
.values-list{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin:16px 0}
.value-pill{padding:4px 14px;border-radius:14px;background:rgba(124,111,247,.15);color:var(--accent);font-size:13px}
.btn-dashboard{background:var(--accent);color:#fff;padding:12px 28px;border-radius:8px;text-decoration:none;display:inline-block;font-weight:600;font-size:14px;margin-top:16px}
.error{color:#f97316;font-size:13px;margin-top:8px}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>Identity Ceremony</h1>
    <p>7 questions · 3 minutes · creates your digital soul<br>Your answers stay on this device.</p>
  </div>
  <div class="card" id="ceremony-card">
    <div class="progress-bar"><div class="progress-fill" id="prog" style="width:0%"></div></div>
    <p class="loading" style="color:var(--muted);text-align:center">Starting ceremony…</p>
  </div>
</div>

<script>
let totalQ=7;

async function start(){
  try{
    const r=await fetch('/onboarding/start',{method:'POST'});
    const d=await r.json();
    totalQ=d.total_questions;
    showQuestion(d);
  }catch(e){showError('Could not start ceremony: '+e.message);}
}

function showQuestion(d){
  const pct=Math.round((d.question_index/totalQ)*100);
  document.getElementById('ceremony-card').innerHTML=`
    <div class="progress-bar"><div class="progress-fill" id="prog" style="width:${pct}%"></div></div>
    <p class="step-label">Question ${d.question_index+1} of ${totalQ}</p>
    <p class="question">${d.question}</p>
    <textarea id="ans" placeholder="Type your answer…" autofocus></textarea>
    <p class="error" id="err" style="display:none"></p>
    <div class="actions">
      <button class="btn-skip" onclick="skip()">Skip</button>
      <button class="btn-next" onclick="submitAnswer()">Next →</button>
    </div>`;
  setTimeout(()=>document.getElementById('ans')&&document.getElementById('ans').focus(),100);
}

async function submitAnswer(skipped){
  const ansEl=document.getElementById('ans');
  const answer=skipped?'—':(ansEl?ansEl.value.trim():'');
  if(!skipped&&!answer){
    const err=document.getElementById('err');
    if(err){err.textContent='Please write something — or press Skip.';err.style.display='block';}
    return;
  }
  try{
    const r=await fetch('/onboarding/answer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({answer:answer||'—'})});
    const d=await r.json();
    if(d.complete){showComplete(d);}else{showQuestion(d);}
  }catch(e){showError('Error: '+e.message);}
}

function skip(){submitAnswer(true);}

function showComplete(d){
  const seed=d.seed||{};
  const values=(seed.stated_values||[]).map(v=>`<span class="value-pill">${v}</span>`).join('');
  document.getElementById('ceremony-card').innerHTML=`
    <div class="complete">
      <h2>Your Prism is ready.</h2>
      <p>Your digital soul has been created and stored locally.</p>
      ${values?`<div class="values-list">${values}</div>`:''}
      <br>
      <a href="/identity" class="btn-dashboard">View your identity dashboard →</a>
    </div>`;
}

function showError(msg){
  document.getElementById('ceremony-card').innerHTML=`<p style="color:#f97316;text-align:center">${msg}</p>`;
}

document.addEventListener('keydown',e=>{if(e.key==='Enter'&&(e.ctrlKey||e.metaKey))submitAnswer();});
start();
</script>
</body>
</html>"""
