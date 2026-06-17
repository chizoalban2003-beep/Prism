"""Bundled organ: mesh_orchestrate — forward a task to a registered peer device."""
ORGAN_META = {
    "intent": "mesh_orchestrate",
    "description": "Forward a chat message or task to another PRISM device on the mesh",
    "version": "1.0",
    "capabilities": ["network"],
}

ORGAN_POLICY = {
    "risk_level": "high",
    "requires_approval": True,
    "irreversible": False,
    "max_per_session": None,
}


def execute(intent: str, message: str, ctx: dict):
    from prism_mesh import get_mesh
    from prism_responses import text_card

    params = ctx.get("params") or {}
    peer_name = (params.get("peer") or params.get("device") or "").strip()
    task      = (params.get("task") or "").strip()
    fwd_msg   = (params.get("message") or "").strip()

    mesh = get_mesh()
    if not peer_name:
        # Try to parse "on <name>" or "via <name>" from the message
        import re
        m = re.search(r"\b(?:on|via|to)\s+(?:my\s+)?([\w\-]+)", message or "", re.IGNORECASE)
        if m:
            peer_name = m.group(1)

    if not peer_name:
        peers = mesh.list_peers()
        if not peers:
            return text_card(
                "No mesh peers registered yet. Add one with: "
                "'register laptop at 192.168.1.42:8742 token <tok>'.",
                "No peers",
            )
        names = ", ".join(p.name for p in peers)
        return text_card(
            f"Which device should I forward to? Registered peers: {names}.",
            "Pick a peer",
        )

    peer = mesh.find_peer_by_name(peer_name)
    if peer is None:
        return text_card(
            f"No peer named '{peer_name}'. Register one first.",
            "Unknown peer",
        )

    # Default: forward the original chat message to the peer's /chat
    payload = fwd_msg or message or ""
    if task:
        result = mesh.forward_task(peer.peer_id, task, params.get("task_params") or {})
        ok = bool(result.get("success"))
        body = result.get("output") or result.get("error") or ""
        title = f"{peer.name}: {task}"
        return text_card(
            f"{'✓' if ok else '✗'} {body[:1500]}",
            title,
        )

    result = mesh.forward_chat(peer.peer_id, payload)
    # Result is the peer's PrismCard.to_json() — surface the body inline
    title = result.get("title") or f"From {peer.name}"
    body = result.get("body") or ""
    if not body and result.get("error"):
        body = f"Forwarding failed: {result['error']}"
    return text_card(body or "(empty reply)", title)
