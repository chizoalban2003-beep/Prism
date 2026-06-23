"""
prism_routes_agents.py
======================
FastAPI router for the unified agent registry.

Route:
  GET  /agents?capability=...   — aggregated view of every agent surface
                                  (LLM / organ / MCP / mesh peer).

Returns the same shape ``prism_agent_registry.inventory`` produces.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter

from prism_agent_registry import inventory
from prism_state import _state

router = APIRouter()


@router.get("/agents")
async def agents_list(capability: Optional[str] = None):
    """Return the unified inventory across LLM / organ / MCP / peer.

    Optional ``capability`` filters entries whose ``capabilities`` list
    contains the tag (case-insensitive substring match).
    """
    return inventory(_state, capability=capability)
