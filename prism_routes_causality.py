"""
prism_routes_causality.py
=========================
FastAPI APIRouter for the Causal Reasoning endpoints.

Routes
------
GET    /causality/graph                    Return all causal edges
POST   /causality/edges                    Add a causal edge
DELETE /causality/edges/{cause}/{effect}   Remove an edge
GET    /causality/explain/{belief_id}      Explain why a belief exists
POST   /causality/counterfactual           Counterfactual "what-if" query
GET    /causality/chain/{belief_id}        Full causal chain from a root belief
GET    /causality/tree/{belief_id}         Explanation tree (causes + effects)
POST   /causality/infer                    Infer edges from soul beliefs

All routes return HTTP 503 when CausalReasoner is not available in _state.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from prism_state import _state

router = APIRouter()


def _reasoner():
    return _state.get("causal_reasoner")


def _503():
    return JSONResponse({"error": "causal_reasoner not configured"}, status_code=503)


# ---------------------------------------------------------------------------
# Graph-level endpoints
# ---------------------------------------------------------------------------

@router.get("/causality/graph")
async def causality_graph():
    """Return all causal edges as a list."""
    cr = _reasoner()
    if cr is None:
        return _503()
    edges = cr._graph.all_edges()
    return {
        "edges": [
            {
                "cause_id":       e.cause_id,
                "effect_id":      e.effect_id,
                "strength":       e.strength,
                "direction":      e.direction,
                "evidence_count": e.evidence_count,
                "created_at":     e.created_at,
            }
            for e in edges
        ]
    }


@router.post("/causality/edges")
async def add_causal_edge(body: dict[str, Any]):
    """Add (or strengthen) a causal edge.

    Body: ``{cause_id, effect_id, strength?, direction?}``
    """
    cr = _reasoner()
    if cr is None:
        return _503()
    cause_id  = str(body.get("cause_id", ""))
    effect_id = str(body.get("effect_id", ""))
    if not cause_id or not effect_id:
        return JSONResponse({"error": "cause_id and effect_id required"}, status_code=422)
    strength  = float(body.get("strength", 0.5))
    direction = str(body.get("direction", "positive"))
    edge = cr._graph.add_edge(cause_id, effect_id, strength=strength, direction=direction)
    return {
        "cause_id":       edge.cause_id,
        "effect_id":      edge.effect_id,
        "strength":       edge.strength,
        "direction":      edge.direction,
        "evidence_count": edge.evidence_count,
    }


@router.delete("/causality/edges/{cause_id}/{effect_id}")
async def remove_causal_edge(cause_id: str, effect_id: str):
    """Remove a causal edge.  Returns ``{removed: bool}``."""
    cr = _reasoner()
    if cr is None:
        return _503()
    removed = cr._graph.remove_edge(cause_id, effect_id)
    return {"removed": removed}


# ---------------------------------------------------------------------------
# Reasoning endpoints
# ---------------------------------------------------------------------------

@router.get("/causality/explain/{belief_id:path}")
async def explain_belief(belief_id: str):
    """Explain why a belief exists using its causal predecessors."""
    cr = _reasoner()
    if cr is None:
        return _503()
    return {"belief_id": belief_id, "explanation": cr.explain(belief_id)}


@router.post("/causality/counterfactual")
async def counterfactual(body: dict[str, Any]):
    """Simulate removing a belief and predict downstream effects.

    Body: ``{query, remove_belief_id}``
    """
    cr = _reasoner()
    if cr is None:
        return _503()
    query            = str(body.get("query", ""))
    remove_belief_id = str(body.get("remove_belief_id", ""))
    if not remove_belief_id:
        return JSONResponse({"error": "remove_belief_id required"}, status_code=422)
    result = cr.counterfactual(query, remove_belief_id)
    return {
        "query":                  result.query,
        "original_outcome":       result.original_outcome,
        "counterfactual_outcome": result.counterfactual_outcome,
        "changed_beliefs":        result.changed_beliefs,
        "confidence":             result.confidence,
        "explanation":            result.explanation,
    }


@router.get("/causality/chain/{belief_id:path}")
async def causal_chain(belief_id: str, depth: int = 3):
    """Return all causal chains starting from *belief_id* (DFS, depth-limited)."""
    cr = _reasoner()
    if cr is None:
        return _503()
    chains = cr._graph.causal_chain(belief_id, depth=min(depth, 6))
    return {"belief_id": belief_id, "chains": chains}


@router.get("/causality/tree/{belief_id:path}")
async def explanation_tree(belief_id: str):
    """Return causes and effects of a belief as a nested dict."""
    cr = _reasoner()
    if cr is None:
        return _503()
    return cr.build_explanation_tree(belief_id)


@router.post("/causality/infer")
async def infer_from_soul():
    """Trigger inference of causal edges from the soul's belief edges."""
    cr = _reasoner()
    if cr is None:
        return _503()
    soul  = _state.get("agent")
    soul  = getattr(soul, "_soul", None) if soul is not None else None
    added = cr.infer_edges_from_soul(soul)
    return {"edges_added": added}
