"""
prism_routes_chain.py
=====================
FastAPI router for chain planner, organs, and organ-bus endpoints.

Routes:
  GET /chain/recent
  GET /chain/expert/recent
  GET /chain/status
  GET /organs
  GET /organ_bus/history
  GET /organ_bus/subscribers
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter

from prism_state import _state

router = APIRouter()


def _get_agent():
    return _state.get("agent")


# ---------------------------------------------------------------------------
# /chain
# ---------------------------------------------------------------------------

@router.get("/chain/recent")
async def chain_recent(n: int = 5):
    agent = _get_agent()
    if agent and hasattr(agent, "_chain"):
        chains = agent._chain.recent_chains(n=n)
        return {"chains": chains}
    return {"chains": []}


@router.get("/chain/expert/recent")
async def chain_expert_recent(n: int = 5):
    agent = _get_agent()
    if agent and hasattr(agent, "_chain_expert"):
        import sqlite3

        db = agent._chain_expert._db

        def _query():
            with sqlite3.connect(db) as c:
                return c.execute(
                    "SELECT chain_id,original,n_steps,n_llm_calls,"
                    "done,final_answer,avg_eval_score,created_at "
                    "FROM expert_chains ORDER BY created_at DESC LIMIT ?",
                    (n,),
                ).fetchall()

        rows = await asyncio.to_thread(_query)
        return {
            "chains": [
                {
                    "chain_id":       r[0],
                    "original":       r[1],
                    "n_steps":        r[2],
                    "n_llm_calls":    r[3],
                    "done":           bool(r[4]),
                    "summary":        r[5][:80] if r[5] else "",
                    "avg_eval_score": r[6],
                    "created_at":     r[7],
                }
                for r in rows
            ]
        }
    return {"chains": []}


@router.get("/chain/status")
async def chain_status():
    agent = _get_agent()
    if agent and hasattr(agent, "_chain"):
        return {
            "max_steps":    agent._chain.MAX_STEPS,
            "db":           str(agent._chain._db),
            "recent_count": len(agent._chain.recent_chains(50)),
        }
    return {"configured": False}


# ---------------------------------------------------------------------------
# /organs
# ---------------------------------------------------------------------------

@router.get("/organs")
async def organs():
    agent = _get_agent()
    ol    = getattr(agent, "_organ_loader", None) if agent else None
    if ol is None:
        return {"organs": {}}
    return {"organs": ol.known_intents(), "count": len(ol.list_organs())}


# ---------------------------------------------------------------------------
# /organ_bus
# ---------------------------------------------------------------------------

@router.get("/organ_bus/history")
async def organ_bus_history(n: int = 20):
    agent = _get_agent()
    ob    = getattr(agent, "_organ_bus", None) if agent else None
    if ob is None:
        return {"signals": [], "available": False}
    return {"signals": ob.history(n=n), "available": True}


@router.get("/organ_bus/subscribers")
async def organ_bus_subscribers():
    agent = _get_agent()
    ob    = getattr(agent, "_organ_bus", None) if agent else None
    if ob is None:
        return {"subscribers": [], "available": False}
    with ob._lock:
        subs = [
            {
                "organ":        s.organ_name,
                "signal_types": s.signal_types,
                "vocabulary":   s.vocabulary[:120],
            }
            for s in ob._subscribers
        ]
    return {"subscribers": subs, "count": len(subs), "available": True}
