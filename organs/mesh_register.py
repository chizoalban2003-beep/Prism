"""Bundled organ: mesh_register — add a peer PRISM device to the local mesh."""
ORGAN_META = {
    "intent": "mesh_register",
    "description": "Register another PRISM device as a mesh peer so tasks can be forwarded to it",
    "version": "1.0",
    "capabilities": ["network"],
}

ORGAN_POLICY = {
    "risk_level": "medium",
    "requires_approval": True,
    "irreversible": False,
    "max_per_session": None,
}


def execute(intent: str, message: str, ctx: dict):
    from prism_mesh import get_mesh
    from prism_responses import text_card

    params = ctx.get("params") or {}
    name  = (params.get("name") or "").strip()
    host  = (params.get("host") or "").strip()
    port  = int(params.get("port") or 8742)
    token = (params.get("token") or "").strip()

    if not host:
        # Try to parse "register <name> at <host>:<port> token <tok>" from message
        import re
        m = re.search(r"\bat\s+([\w.\-]+)(?::(\d+))?", message or "")
        if m:
            host = m.group(1)
            port = int(m.group(2) or port)
        m2 = re.search(r"\btoken\s+([A-Za-z0-9._\-]+)", message or "")
        if m2 and not token:
            token = m2.group(1)
        m3 = re.search(r"\bregister\s+(\w[\w\-]*)", message or "", re.IGNORECASE)
        if m3 and not name:
            name = m3.group(1)

    if not host:
        return text_card(
            "I need a host to register a peer. Try: "
            "'register laptop at 192.168.1.42:8742 token <tok>'",
            "Mesh register",
        )

    mesh = get_mesh()
    peer = mesh.register_peer(name or host, host, port, token)
    caps = peer.capabilities or {}
    summary = caps.get("summary") or "no capabilities discovered yet"
    return text_card(
        f"Registered peer <strong>{peer.name}</strong> at "
        f"{peer.host}:{peer.port} (id {peer.peer_id}).\n\n{summary}",
        "Mesh peer registered",
    )
