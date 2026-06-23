"""Bundled organ: agents_inventory — show every agent surface in one view.

Aggregates the four registries PRISM already maintains (LLM providers,
organs, MCP tool servers, mesh peers) into a single chat-routable
summary. Backed by ``prism_agent_registry.inventory``.
"""
ORGAN_META = {
    "intent":      "agents_inventory",
    "description": "List every agent PRISM has access to — LLM providers, organs, MCP servers, mesh peers",
    "version":     "1.0",
    "capabilities": [],
    "inputs": {
        "capability": "str",
    },
    "outputs": {
        "agents":  "list[{kind:str,name:str,status:str,capabilities:list[str]}]",
        "summary": "dict",
    },
}

ORGAN_POLICY = {
    "risk_level":        "low",
    "requires_approval": False,
    "irreversible":      False,
    "max_per_session":   None,
}


def _extract_capability(message: str) -> str:
    import re
    for pat in [
        r'(?:with|for|that\s+(?:can|do|support))\s+([a-z_][\w\s]*?)(?:\s+capability)?[\s?.]*$',
        r'capability[:\s]+([a-z_][\w]+)',
        r'agents?\s+for\s+([a-z_][\w]+)',
    ]:
        m = re.search(pat, message, re.IGNORECASE)
        if m:
            return m.group(1).strip().split()[0]
    return ""


def execute(intent: str, message: str, ctx: dict):
    from prism_agent_registry import inventory
    from prism_responses import text_card
    from prism_state import _state

    capability = _extract_capability(message) or None
    state = ctx.get("state") or _state
    inv = inventory(state, capability=capability)

    agents = inv["agents"]
    summary = inv["summary"]

    if not agents:
        suffix = f" with capability '{capability}'" if capability else ""
        return text_card(f"No agents registered{suffix}.", "Agents")

    header = (
        f"Agents — {summary['total']} total "
        f"({summary['ready']} ready) · "
        f"llm={summary['llm']} organ={summary['organ']} "
        f"mcp={summary['mcp']} peer={summary['peer']}"
    )
    if capability:
        header += f"\nFiltered by capability: {capability}"

    sections: dict[str, list[str]] = {"llm": [], "organ": [], "mcp": [], "peer": []}
    for a in agents:
        kind = a["kind"]
        name = a["name"]
        status = a["status"]
        caps = ", ".join(a.get("capabilities") or []) or "—"
        if kind == "llm":
            lat = a.get("latency_ms")
            lat_s = f" · {lat:.0f}ms" if isinstance(lat, (int, float)) else ""
            sections["llm"].append(f"  {name:30s}  {status:<8}{lat_s}")
        elif kind == "organ":
            risk = a.get("risk") or "low"
            sections["organ"].append(f"  {name:30s}  {status:<8}  risk={risk:<6}  caps: {caps}")
        elif kind == "mcp":
            count = a.get("tool_count", 0)
            sections["mcp"].append(f"  {name:30s}  {status:<8}  {count} tool(s): {caps}")
        elif kind == "peer":
            host = a.get("host", "")
            sections["peer"].append(f"  {name:30s}  {status:<8}  {host}  caps: {caps}")

    lines = [header, ""]
    labels = {"llm": "LLM providers", "organ": "Organs", "mcp": "MCP servers", "peer": "Mesh peers"}
    for kind in ("llm", "mcp", "peer", "organ"):
        if sections[kind]:
            lines.append(f"── {labels[kind]} ──")
            cap = 15 if kind == "organ" else len(sections[kind])
            lines.extend(sections[kind][:cap])
            if kind == "organ" and len(sections["organ"]) > cap:
                lines.append(f"  … and {len(sections['organ']) - cap} more")
            lines.append("")

    return text_card("\n".join(lines).rstrip(), "Agents")
