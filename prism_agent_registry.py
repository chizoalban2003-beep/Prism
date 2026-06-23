"""
prism_agent_registry.py
=======================
Pure aggregator that gathers PRISM's four agent-surface registries into
a single normalized inventory:

* **LLM providers**  — from ``LLMRouter.status_summary()``
* **Organs**         — from ``OrganLoader._organs`` (intent → ORGAN_META/POLICY)
* **MCP tools**      — from ``MCPManager.status()`` + ``list_tools()``
* **Mesh peers**     — from ``PrismMesh.list_peers()``

The aggregator does not make routing decisions. It answers *"what agents
do I have?"* in one schema so the chat surface (``agents_inventory``
organ) and the HTTP API (``GET /agents``) can both render the same view.

Each entry is a dict with this common shape:

    {
        "kind":         "llm" | "organ" | "mcp" | "peer",
        "name":         str,           # display name
        "status":       "ready" | "loaded" | "online" | "offline" | ...
        "capabilities": list[str],     # normalized capability tags
        # optional, kind-specific:
        "latency_ms":   float | None,  # llm only
        "provider":     str | None,    # llm only
        "risk":         str | None,    # organ only ("low"/"medium"/"high")
        "server":       str | None,    # mcp only
        "host":         str | None,    # peer only
    }

The function takes a ``state`` mapping for testability — pass the
``_state`` dict (or a stub with the same keys: ``agent``, ``llm_router``,
``mcp``, ``organ_loader``). Each source is wrapped in try/except so one
broken surface does not poison the others.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional


def _llm_entries(llm_router: Any) -> list[dict]:
    if llm_router is None:
        return []
    try:
        summary = llm_router.status_summary()
    except Exception:
        return []
    out: list[dict] = []
    for opt in summary.get("available", []) or []:
        provider = opt.get("provider", "")
        model    = opt.get("model", "")
        out.append({
            "kind":         "llm",
            "name":         f"{provider}/{model}" if provider and model else (provider or model),
            "status":       "ready" if opt.get("available") else "offline",
            "capabilities": ["reasoning"] if opt.get("available") else [],
            "latency_ms":   opt.get("latency_ms"),
            "provider":     provider,
            "capability_rank": opt.get("capability"),
        })
    return out


def _organ_entries(organ_loader: Any) -> list[dict]:
    if organ_loader is None:
        return []
    out: list[dict] = []
    try:
        organs = getattr(organ_loader, "_organs", {}) or {}
        disabled = getattr(organ_loader, "_disabled", set()) or set()
    except Exception:
        return []
    for intent, entry in organs.items():
        try:
            fn, meta = entry
        except Exception:
            continue
        caps = list((meta or {}).get("capabilities") or [])
        policy = getattr(fn, "_organ_policy", {}) or {}
        out.append({
            "kind":         "organ",
            "name":         intent,
            "status":       "disabled" if intent in disabled else "loaded",
            "capabilities": caps,
            "risk":         policy.get("risk_level"),
            "description":  (meta or {}).get("description", ""),
        })
    return out


def _mcp_entries(mcp: Any) -> list[dict]:
    if mcp is None:
        return []
    out: list[dict] = []
    try:
        servers = mcp.status()
    except Exception:
        return []
    try:
        tools = mcp.list_tools()
    except Exception:
        tools = []
    tools_by_server: dict[str, list[str]] = {}
    for t in tools or []:
        sname = t.get("server") or ""
        tname = t.get("name") or ""
        if tname:
            tools_by_server.setdefault(sname, []).append(tname)
    for s in servers or []:
        sname = s.get("name", "")
        caps  = tools_by_server.get(sname, [])
        out.append({
            "kind":         "mcp",
            "name":         sname,
            "status":       "ready" if s.get("alive") and s.get("initialized") else "offline",
            "capabilities": caps,
            "server":       sname,
            "tool_count":   s.get("tool_count", 0),
        })
    return out


def _peer_entries() -> list[dict]:
    try:
        from prism_mesh import get_mesh
        mesh = get_mesh()
        peers = mesh.list_peers()
    except Exception:
        return []
    out: list[dict] = []
    for p in peers or []:
        caps_raw = getattr(p, "capabilities", {}) or {}
        if isinstance(caps_raw, dict):
            caps = [k for k, v in caps_raw.items() if v]
        else:
            caps = list(caps_raw)
        out.append({
            "kind":         "peer",
            "name":         getattr(p, "name", ""),
            "status":       "online" if getattr(p, "last_seen", 0) > 0 else "offline",
            "capabilities": caps,
            "host":         f"{getattr(p, 'host', '')}:{getattr(p, 'port', '')}",
        })
    return out


def inventory(
    state: Optional[Mapping[str, Any]] = None,
    *,
    capability: Optional[str] = None,
) -> dict:
    """Aggregate the four registries into a single inventory.

    Parameters
    ----------
    state : Mapping, optional
        Mapping like ``prism_state._state`` providing ``agent``,
        ``llm_router``, ``mcp``, ``organ_loader`` keys. Missing keys
        contribute empty lists.
    capability : str, optional
        If set, return only entries whose ``capabilities`` contain this
        tag (case-insensitive substring match).
    """
    state = state or {}
    agent = state.get("agent")
    llm_router   = state.get("llm_router")
    mcp          = state.get("mcp")
    organ_loader = getattr(agent, "_organ_loader", None) or state.get("organ_loader")

    agents: list[dict] = []
    agents.extend(_llm_entries(llm_router))
    agents.extend(_organ_entries(organ_loader))
    agents.extend(_mcp_entries(mcp))
    agents.extend(_peer_entries())

    if capability:
        needle = capability.lower()
        agents = [
            a for a in agents
            if any(needle in (c or "").lower() for c in a.get("capabilities", []))
        ]

    summary = {
        "llm":   sum(1 for a in agents if a["kind"] == "llm"),
        "organ": sum(1 for a in agents if a["kind"] == "organ"),
        "mcp":   sum(1 for a in agents if a["kind"] == "mcp"),
        "peer":  sum(1 for a in agents if a["kind"] == "peer"),
        "ready": sum(1 for a in agents if a["status"] in ("ready", "loaded", "online")),
        "total": len(agents),
    }
    return {"agents": agents, "summary": summary}
