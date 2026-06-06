"""
VEAX Control organ — bidirectional spectrum vector tuning.

Handles natural language intent, named presets, direct set/delta commands,
and state display.  Persists changes to ~/.prism/spectrum_state.json for
cross-session durability; updates the in-session singleton immediately so
the running chain sees the new gates on the very next step.

Example triggers:
  "show spectrum"  /  "current veax state"
  "use audit mode" /  "switch to scout"
  "set autonomy to 0.8"
  "increase verification, I need stricter proof today"
  "be more cautious"
  "reset to defaults"
"""
ORGAN_META = {
    "intent":      "veax_control",
    "description": (
        "read or update the VEAX spectrum vector "
        "(Verification / Evolution / Autonomy / Explanation)"
    ),
    "version":     "1.1",
    "capabilities": [],
}

ORGAN_POLICY = {
    "risk_level":        "low",
    "requires_approval": False,
    "irreversible":      False,
    "max_per_session":   None,
}

_SHOW_WORDS   = frozenset({"show", "current", "status", "what", "display", "read", "get", "check"})
_CHANGE_WORDS = frozenset({
    "set", "change", "update", "increase", "decrease", "use", "switch",
    "make", "more", "less", "higher", "lower", "turn", "apply", "to",
})


def execute(intent: str, message: str, ctx: dict):
    from prism_responses import text_card

    try:
        from prism_spectrum_middleware import (
            VEAX_PRESETS,
            SpectrumGates,
            get_current_gates,
            load_spectrum,
            nl_to_veax,
            render_gates,
            save_spectrum_state,
        )
    except ImportError as exc:
        return text_card(f"veax_control: import error — {exc}", intent)

    current = get_current_gates()
    if current is None:
        current, _ = load_spectrum()

    lower = message.lower().strip()

    # ── Fast path: pure read requests ─────────────────────────────────────────
    has_show   = any(w in lower for w in _SHOW_WORDS)
    has_change = any(w in lower for w in _CHANGE_WORDS)
    if (has_show and not has_change) or not lower:
        return text_card("Current VEAX Spectrum:\n\n" + render_gates(current), intent)

    # ── Fast path: named preset ────────────────────────────────────────────────
    for name, vals in VEAX_PRESETS.items():
        if name in lower:
            new_gates = SpectrumGates(**vals)
            save_spectrum_state(new_gates, preset=name)
            return text_card(
                f"Applied preset '{name}':\n\n" + render_gates(new_gates, preset=name),
                intent,
            )

    # ── Fast path: explicit reset ──────────────────────────────────────────────
    if "reset" in lower or "default" in lower:
        new_gates = SpectrumGates(**VEAX_PRESETS["balanced"])
        save_spectrum_state(new_gates, preset="balanced")
        return text_card(
            "Reset to balanced defaults:\n\n" + render_gates(new_gates, preset="balanced"),
            intent,
        )

    # ── LLM inference ─────────────────────────────────────────────────────────
    router = ctx.get("router")
    if router is None:
        return text_card(
            "No router available for NL parsing.\n\nCurrent:\n\n" + render_gates(current),
            intent,
        )

    parsed = nl_to_veax(message, current, router)
    action = parsed.get("action", "get")

    if action == "get":
        body = "Current VEAX Spectrum:\n\n" + render_gates(current)
        reasoning = parsed.get("reasoning", "")
        if reasoning:
            body += f"\n\n  ({reasoning})"
        return text_card(body, intent)

    if action == "preset":
        name = parsed.get("preset", "balanced")
        if name not in VEAX_PRESETS:
            name = "balanced"
        new_gates = SpectrumGates(**VEAX_PRESETS[name])
        save_spectrum_state(new_gates, preset=name)
        return text_card(
            f"Applied preset '{name}':\n\n" + render_gates(new_gates, preset=name),
            intent,
        )

    if action == "reset":
        new_gates = SpectrumGates(**VEAX_PRESETS["balanced"])
        save_spectrum_state(new_gates, preset="balanced")
        return text_card(
            "Reset to balanced defaults:\n\n" + render_gates(new_gates, preset="balanced"),
            intent,
        )

    # ── Apply set / delta ─────────────────────────────────────────────────────
    axes    = {"V": current.V, "E": current.E, "A": current.A, "X": current.X}
    changed: list[str] = []

    for k in ("V", "E", "A", "X"):
        if k not in parsed:
            continue
        raw = float(parsed[k])
        if action == "delta":
            new_val = max(0.0, min(1.0, axes[k] + raw))
            if abs(new_val - axes[k]) > 0.001:
                changed.append(f"  {k}: {axes[k]:.2f} → {new_val:.2f}  ({raw:+.2f})")
            axes[k] = new_val
        else:
            new_val = max(0.0, min(1.0, raw))
            if abs(new_val - axes[k]) > 0.001:
                changed.append(f"  {k}: {axes[k]:.2f} → {new_val:.2f}")
            axes[k] = new_val

    if not changed:
        reasoning = parsed.get("reasoning", "no axes affected")
        return text_card(
            f"No changes ({reasoning}).\n\nCurrent:\n\n" + render_gates(current),
            intent,
        )

    new_gates = SpectrumGates(**axes)
    save_spectrum_state(new_gates)
    reasoning  = parsed.get("reasoning", "")
    body  = "VEAX updated:\n" + "\n".join(changed)
    if reasoning:
        body += f"\n\nInterpreted as: {reasoning}"
    body += "\n\n" + render_gates(new_gates)
    return text_card(body, intent)
