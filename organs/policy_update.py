"""Bundled organ: policy_update — update a user-level allowance in the PolicyEngine."""
ORGAN_META = {
    "intent": "policy_update",
    "description": "Grant or revoke a user-level action allowance in the policy engine (e.g. allow email_send)",
    "version": "1.0",
}

ORGAN_POLICY = {
    "risk_level": "medium",
    "requires_approval": True,
    "irreversible": False,
    "max_per_session": 10,
}


def execute(intent: str, message: str, ctx: dict):
    import re

    from prism_responses import text_card

    policy = ctx.get("policy_engine")
    if policy is None:
        return text_card("PolicyEngine not available in context.", intent)

    # Parse: "allow email_send" / "deny calendar_write" / "revoke browser_task"
    m = re.search(r"\b(allow|grant|deny|revoke)\s+([\w_]+)", message, re.IGNORECASE)
    if not m:
        return text_card(
            "Could not parse the update. Use: 'allow <action>' or 'deny <action>'.", intent
        )

    verb, action = m.group(1).lower(), m.group(2).lower()
    allowed = verb in ("allow", "grant")

    try:
        if hasattr(policy, "set_allowance"):
            policy.set_allowance(action, allowed)
            status = "allowed" if allowed else "denied"
            return text_card(f"Policy updated: {action} → {status}.", intent)
        elif hasattr(policy, "allow") and allowed:
            policy.allow(action)
            return text_card(f"Policy updated: {action} → allowed.", intent)
        elif hasattr(policy, "deny") and not allowed:
            policy.deny(action)
            return text_card(f"Policy updated: {action} → denied.", intent)
        else:
            return text_card("PolicyEngine does not support runtime updates.", intent)
    except Exception as exc:
        return text_card(f"Policy update failed: {exc}", intent)
