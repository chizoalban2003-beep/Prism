"""
prism_routes_users.py
=====================
FastAPI router for multi-user home hub and household bus endpoints.

Routes
------
GET  /users                    — list all registered users
POST /users                    — register a new user {user_id, name, role}
DELETE /users/{user_id}        — remove a user
POST /users/{user_id}/activate — switch active user context in _state["agent"]
GET  /users/{user_id}/identity — per-user soul/persona snapshot
GET  /household/signals        — recent household bus signals (last N)
GET  /household/dashboard      — HTML household dashboard
GET  /household/analytics      — JSON household analytics
POST /household/broadcast      — emit signal to all users {signal_type, payload}
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from prism_state import _state, set_state

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _registry():
    return _state.get("user_registry")


def _household_bus():
    return _state.get("household_bus")


def _no_registry() -> JSONResponse:
    return JSONResponse(
        {"error": "UserRegistry not available — daemon not started", "status": 503},
        status_code=503,
    )


# ---------------------------------------------------------------------------
# GET /users
# ---------------------------------------------------------------------------


@router.get("/users")
async def list_users():
    reg = _registry()
    if reg is None:
        return _no_registry()

    profiles = reg.list_users()
    return {
        "total": len(profiles),
        "users": [p.to_dict() for p in profiles],
    }


# ---------------------------------------------------------------------------
# POST /users
# ---------------------------------------------------------------------------


@router.post("/users")
async def register_user(request: Request):
    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    reg = _registry()
    if reg is None:
        return _no_registry()

    user_id = body.get("user_id", "").strip()
    name = body.get("name", "").strip()
    role = body.get("role", "member").strip()

    if not user_id or not name:
        return JSONResponse(
            {"error": "'user_id' and 'name' are required", "status": 400},
            status_code=400,
        )

    try:
        profile = reg.register(user_id=user_id, name=name, role=role)
    except ValueError as exc:
        return JSONResponse(
            {"error": str(exc), "status": 409},
            status_code=409,
        )

    return {"ok": True, "user": profile.to_dict()}


# ---------------------------------------------------------------------------
# DELETE /users/{user_id}
# ---------------------------------------------------------------------------


@router.delete("/users/{user_id}")
async def remove_user(user_id: str):
    reg = _registry()
    if reg is None:
        return _no_registry()

    removed = reg.remove(user_id)
    if not removed:
        return JSONResponse(
            {"error": f"User {user_id!r} not found", "status": 404},
            status_code=404,
        )

    return {"ok": True, "user_id": user_id}


# ---------------------------------------------------------------------------
# POST /users/{user_id}/activate
# ---------------------------------------------------------------------------


@router.post("/users/{user_id}/activate")
async def activate_user(user_id: str):
    reg = _registry()
    if reg is None:
        return _no_registry()

    profile = reg.get(user_id)
    if profile is None:
        return JSONResponse(
            {"error": f"User {user_id!r} not found", "status": 404},
            status_code=404,
        )

    # Switch the active user context on the agent when available
    agent = _state.get("agent")
    if agent is not None:
        try:
            agent._active_user = user_id
        except Exception:
            pass

    # Store in state so other routers can see who is active
    set_state("active_user_id", user_id)

    reg.touch(user_id)
    return {"ok": True, "active_user_id": user_id, "profile": profile.to_dict()}


# ---------------------------------------------------------------------------
# GET /household/signals
# ---------------------------------------------------------------------------


@router.get("/household/signals")
async def household_signals(n: int = 20):
    bus = _household_bus()

    # Fallback to OrganBus history when HouseholdBus not yet wired
    if bus is None:
        reg = _registry()
        if reg is None:
            return _no_registry()
        return {"signals": [], "total": 0}

    signals = bus.signal_history(n=n)
    return {"signals": signals, "total": len(signals)}


# ---------------------------------------------------------------------------
# GET /users/{user_id}/identity
# ---------------------------------------------------------------------------


@router.get("/users/{user_id}/identity")
async def user_identity(user_id: str):
    """Return a per-user soul/persona/phase snapshot."""
    reg = _registry()
    if reg is None:
        return _no_registry()

    profile = reg.get(user_id)
    if profile is None:
        return JSONResponse(
            {"error": f"User {user_id!r} not found", "status": 404},
            status_code=404,
        )

    snapshot: dict[str, Any] = {
        "user_id": profile.user_id,
        "name": profile.name,
        "role": profile.role,
        "last_active": profile.last_active,
        "soul_beliefs": [],
        "phase": None,
    }

    # Soul beliefs — try registry.get_soul() first
    try:
        soul = reg.get_soul(user_id)
        if soul is not None:
            beliefs = soul.list_beliefs()
            snapshot["soul_beliefs"] = [
                {
                    "text": b.text,
                    "type": b.belief_type,
                    "confidence": round(b.confidence, 3),
                    "source": b.source,
                }
                for b in (beliefs or [])[:20]
            ]
    except Exception:
        pass

    # Phase — read the cached reading the daemon's phase-ticker maintains.
    # (Fall back to the process-wide singleton; PrismAgent has no _phase. Do
    # NOT call compute() in a GET — it would mutate the global phase.)
    try:
        from prism_state import _get_agent  # noqa: PLC0415

        agent = _get_agent()
        phase_engine = getattr(agent, "_phase", None) if agent else None
        if phase_engine is None:
            import prism_phase  # noqa: PLC0415
            phase_engine = prism_phase.get_engine()
        p = getattr(phase_engine, "current_phase", None)
        if p is not None:
            snapshot["phase"] = p.value if hasattr(p, "value") else str(p)
    except Exception:
        pass

    return snapshot


# ---------------------------------------------------------------------------
# GET /persona/policy — CEO-inspectable view of what the manager learned
# ---------------------------------------------------------------------------


@router.get("/persona/policy")
async def persona_policy():
    """Return what the crystalliser has learned about the user, formatted as
    a policy snapshot the CEO can review. Read-only: this exposes traits,
    behavioural patterns, peak hours, and the underlying source of each
    inference so the user can decide if the manager's model of them is
    accurate.

    Shape:
        {
          "traits":   [{name, value, confidence, source, observations}, ...],
          "patterns": [{description, frequency, examples}, ...],
          "peak_hours": [int, ...],
          "growth_7d": {new_traits, new_patterns, confidence_avg},
        }
    """
    try:
        from prism_state import _get_agent  # noqa: PLC0415
        agent = _get_agent()
    except Exception:
        agent = None
    persona = getattr(agent, "_persona", None) if agent is not None else None
    if persona is None:
        return JSONResponse(
            {"error": "PrismPersona not available", "status": 503},
            status_code=503,
        )

    try:
        traits = persona.list_traits()
        patterns = persona._top_patterns(20)
        peaks = persona.peak_hours()
        growth = persona.growth_since(days=7)
    except Exception as exc:
        return JSONResponse(
            {"error": f"persona read failed: {exc}", "status": 500},
            status_code=500,
        )

    return {
        "traits": [
            {
                "name":         t.name,
                "value":        t.value,
                "confidence":   round(t.confidence, 3),
                "source":       t.source,
                "observations": t.observation_count,
            }
            for t in traits
        ],
        "patterns": [
            {
                "description": p.description,
                "frequency":   p.frequency,
                "examples":    list(p.examples or [])[-3:],
            }
            for p in patterns
        ],
        "peak_hours": peaks,
        "growth_7d":  growth,
    }


# ---------------------------------------------------------------------------
# GET /household/dashboard — HTML
# ---------------------------------------------------------------------------


@router.get("/household/dashboard", response_class=HTMLResponse)
async def household_dashboard():
    """Render the household team dashboard as an HTML page."""
    return HTMLResponse(_household_dashboard_html())


# ---------------------------------------------------------------------------
# GET /household/analytics — JSON
# ---------------------------------------------------------------------------


@router.get("/household/analytics")
async def household_analytics():
    """Return JSON analytics for the household."""
    reg = _registry()
    if reg is None:
        return _no_registry()

    now = time.time()
    _24h = 86_400.0
    _7d = 7 * _24h

    profiles = reg.list_users()
    total_users = len(profiles)
    active_today = sum(1 for p in profiles if (now - p.last_active) <= _24h)
    active_this_week = sum(1 for p in profiles if (now - p.last_active) <= _7d)

    by_role: dict[str, int] = {"admin": 0, "member": 0, "guest": 0}
    users_list = []
    for p in profiles:
        by_role[p.role] = by_role.get(p.role, 0) + 1
        users_list.append(
            {
                "user_id": p.user_id,
                "name": p.name,
                "role": p.role,
                "last_active": p.last_active,
                "is_active_today": (now - p.last_active) <= _24h,
            }
        )

    recent_signals: list[dict[str, Any]] = []
    bus = _household_bus()
    if bus is not None:
        recent_signals = bus.signal_history(n=10)

    return {
        "total_users": total_users,
        "active_today": active_today,
        "active_this_week": active_this_week,
        "by_role": by_role,
        "recent_signals": recent_signals,
        "users": users_list,
    }


# ---------------------------------------------------------------------------
# POST /household/broadcast
# ---------------------------------------------------------------------------


@router.post("/household/broadcast")
async def household_broadcast(request: Request):
    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        pass

    reg = _registry()
    if reg is None:
        return _no_registry()

    signal_type = body.get("signal_type", "").strip()
    payload = body.get("payload", {})
    source = body.get("source", "household_api")

    if not signal_type:
        return JSONResponse(
            {"error": "'signal_type' is required", "status": 400},
            status_code=400,
        )

    if not isinstance(payload, dict):
        payload = {"value": payload}

    from prism_organ_bus import OrganSignal  # noqa: PLC0415

    signal = OrganSignal(
        source=source,
        signal_type=signal_type,
        payload=payload,
    )

    bus = _household_bus()
    if bus is None:
        # No HouseholdBus wired — record that we would have broadcast
        return {
            "ok": True,
            "signal_id": signal.signal_id,
            "signal_type": signal_type,
            "note": "HouseholdBus not wired — signal not delivered",
            "results": {},
        }

    results = bus.broadcast(signal)
    return {
        "ok": True,
        "signal_id": signal.signal_id,
        "signal_type": signal_type,
        "results": results,
    }


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------


def _household_dashboard_html() -> str:  # noqa: E501
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>PRISM Household</title>
<style>
:root{--bg:#0d0d0d;--card:#1a1a1a;--border:#2a2a2a;--text:#e8e8e8;--muted:#888;--accent:#7c6ff7;--gold:#facc15;--blue:#60a5fa;--gray:#9ca3af;--green:#4ade80;--orange:#f97316}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Courier New',Courier,monospace;font-size:14px;line-height:1.6;padding:24px;max-width:1280px;margin:0 auto}
h1{font-size:22px;font-weight:700;margin-bottom:4px;letter-spacing:-.01em}
.subtitle{color:var(--muted);font-size:13px;margin-bottom:28px}
.section-title{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);margin-bottom:14px}
.stats-bar{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:28px}
.stat-card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:14px 20px;flex:1;min-width:120px}
.stat-value{font-size:26px;font-weight:700;color:var(--accent);letter-spacing:-.02em}
.stat-label{font-size:11px;color:var(--muted);margin-top:2px;text-transform:uppercase;letter-spacing:.07em}
.users-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:14px;margin-bottom:28px}
.user-card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px}
.user-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
.user-name{font-size:15px;font-weight:600}
.role-badge{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;padding:2px 8px;border-radius:10px}
.role-admin{background:rgba(250,204,21,.15);color:var(--gold);border:1px solid rgba(250,204,21,.3)}
.role-member{background:rgba(96,165,250,.15);color:var(--blue);border:1px solid rgba(96,165,250,.3)}
.role-guest{background:rgba(156,163,175,.15);color:var(--gray);border:1px solid rgba(156,163,175,.3)}
.user-meta{font-size:12px;color:var(--muted)}
.user-id{font-size:11px;color:var(--muted);margin-top:6px;word-break:break-all}
.signal-table{width:100%;border-collapse:collapse}
.signal-table th{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);text-align:left;padding:6px 8px;border-bottom:1px solid var(--border);font-weight:600}
.signal-table td{padding:7px 8px;border-bottom:1px solid rgba(42,42,42,.6);font-size:12px;vertical-align:top}
.signal-table tr:last-child td{border-bottom:none}
.sig-type{color:var(--accent);font-weight:600}
.sig-source{color:var(--text)}
.sig-time{color:var(--muted)}
.sig-recipients{color:var(--muted);font-size:11px}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:20px;margin-bottom:24px}
.empty{color:var(--muted);font-style:italic;padding:16px 0;text-align:center;font-size:13px}
.refresh-btn{float:right;background:none;border:1px solid var(--border);color:var(--muted);padding:4px 12px;border-radius:6px;cursor:pointer;font-size:12px;font-family:inherit}
.refresh-btn:hover{border-color:var(--accent);color:var(--accent)}
.moat{text-align:center;color:var(--muted);font-size:11px;margin-top:28px;padding:16px;border-top:1px solid var(--border)}
.loading{color:var(--muted);font-style:italic;padding:40px;text-align:center}
</style>
</head>
<body>
<h1>PRISM Household <button class="refresh-btn" onclick="load()">Refresh</button></h1>
<p class="subtitle" id="active-count">Loading household data...</p>
<div id="app"><p class="loading">Fetching household data...</p></div>

<script>
function relTime(ts){
  const d=Math.round(Date.now()/1000-ts);
  if(d<60)return d+'s ago';
  if(d<3600)return Math.round(d/60)+'m ago';
  if(d<86400)return Math.round(d/3600)+'h ago';
  return Math.round(d/86400)+'d ago';
}

async function load(){
  const app=document.getElementById('app');
  try{
    const r=await fetch('/household/analytics');
    if(!r.ok)throw new Error(r.status);
    const d=await r.json();
    document.getElementById('active-count').textContent=
      'Household dashboard — '+d.active_today+' active today';
    app.innerHTML=render(d);
  }catch(e){
    app.innerHTML='<p style="color:var(--orange);padding:20px">Could not load household data: '+e.message+'</p>';
  }
}

function render(d){
  let html='';

  // Stats bar
  html+='<div class="stats-bar">';
  html+=stat(d.total_users,'Total Users');
  html+=stat(d.active_today,'Active Today');
  html+=stat(d.active_this_week,'Active This Week');
  html+=stat(d.by_role.admin||0,'Admins');
  html+=stat(d.by_role.member||0,'Members');
  html+=stat(d.by_role.guest||0,'Guests');
  html+='</div>';

  // User cards
  html+='<p class="section-title">Registered Users</p>';
  if(!d.users||!d.users.length){
    html+='<div class="card"><p class="empty">No users registered yet.</p></div>';
  }else{
    html+='<div class="users-grid">';
    d.users.forEach(u=>{
      const rc='role-'+u.role;
      html+=`<div class="user-card">
        <div class="user-header">
          <span class="user-name">${esc(u.name)}</span>
          <span class="role-badge ${rc}">${esc(u.role)}</span>
        </div>
        <div class="user-meta">Last active: ${relTime(u.last_active)}</div>
        <div class="user-id">${esc(u.user_id)}</div>
      </div>`;
    });
    html+='</div>';
  }

  // Signal feed
  html+='<p class="section-title">Recent Signals</p>';
  html+='<div class="card">';
  if(!d.recent_signals||!d.recent_signals.length){
    html+='<p class="empty">No signals recorded yet.</p>';
  }else{
    html+='<table class="signal-table"><thead><tr><th>Time</th><th>Type</th><th>Sender</th><th>Recipients</th></tr></thead><tbody>';
    d.recent_signals.forEach(s=>{
      const recipients=s.results?Object.keys(s.results).join(', '):'-';
      html+=`<tr>
        <td class="sig-time">${relTime(s.ts)}</td>
        <td class="sig-type">${esc(s.signal_type)}</td>
        <td class="sig-source">${esc(s.source)}</td>
        <td class="sig-recipients">${esc(recipients)}</td>
      </tr>`;
    });
    html+='</tbody></table>';
  }
  html+='</div>';

  html+='<div class="moat">All household data stays on this device. No cloud sync without your consent.</div>';
  return html;
}

function stat(val,label){
  return `<div class="stat-card"><div class="stat-value">${val}</div><div class="stat-label">${label}</div></div>`;
}

function esc(s){
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

load();
</script>
</body>
</html>"""
