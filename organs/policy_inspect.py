"""Bundled organ: policy_inspect — show declared ORGAN_POLICY for all loaded organs."""
ORGAN_META = {
    "intent": "policy_inspect",
    "description": "Show the declared risk policy (risk_level, approval, irreversible) for every loaded organ",
    "version": "1.0",
    "capabilities": [],
}

ORGAN_POLICY = {
    "risk_level": "low",
    "requires_approval": False,
    "irreversible": False,
    "max_per_session": None,
}


def execute(intent: str, message: str, ctx: dict):
    from prism_responses import text_card

    loader = ctx.get("organ_loader")
    if loader is None:
        return text_card("organ_loader not available in context.", intent)

    intents = loader.list_organs() if hasattr(loader, "list_organs") else []
    if not intents:
        return text_card("No organs loaded.", intent)

    lines = ["Organ policy declarations:\n"]
    for name in intents:
        policy = loader.get_organ_policy(name) if hasattr(loader, "get_organ_policy") else {}
        if policy:
            risk        = policy.get("risk_level", "low")
            approval    = "yes" if policy.get("requires_approval") else "no"
            irreversible = "yes" if policy.get("irreversible") else "no"
            max_sess    = str(policy.get("max_per_session") or "∞")
            lines.append(
                f"  {name:30s}  risk={risk:<8}  approval={approval}"
                f"  irreversible={irreversible}  max/session={max_sess}"
            )
        else:
            lines.append(f"  {name:30s}  (no ORGAN_POLICY declared — legacy fallback)")

    return text_card("\n".join(lines), intent)
