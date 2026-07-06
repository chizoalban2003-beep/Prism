"""Bundled organ: computer_use — drive the desktop by looking at the screen.

The closed perception→action loop: screenshot → vision-capable LLM decides the
next click/type → actuate → repeat, bounded and approval-gated. This is the
autonomous-targeting capability on top of screenshot + input_control.

High-risk + approval-gated + irreversible: one approval authorises a whole
session, which is then bounded by a hard step limit and logs every actuation
to the PRISM inbox. Refuses to start unless display + input backend + a
vision-capable model are present.
"""
from __future__ import annotations

ORGAN_META = {
    "intent":      "computer_use",
    "description": "achieve a goal by looking at the screen and clicking/typing "
                   "— a bounded, approval-gated screenshot→vision→input loop",
    "version":     "1.0",
    "capabilities": ["system_ui", "input_synthesis", "screen_capture"],
    "inputs":  {"goal": "str"},
    "outputs": {"success": "bool", "steps": "int"},
}

ORGAN_POLICY = {
    "risk_level":        "high",
    "requires_approval": True,
    "irreversible":      True,
    "max_per_session":   None,
}

_MAX_STEPS = 8


def _parse_goal(message: str) -> str:
    import re
    m = (message or "").strip()
    g = re.sub(
        r"^(?:please\s+)?(?:use\s+the\s+computer\s+to|computer\s+use|"
        r"control\s+(?:my\s+)?(?:screen|desktop|computer)\s+to|"
        r"on\s+(?:my\s+)?screen)[,:]?\s*",
        "", m, flags=re.IGNORECASE).strip()
    return g or m


def _capture_screenshot() -> str | None:
    """Grab the primary monitor to a temp PNG via mss. None on failure."""
    try:
        import datetime
        from pathlib import Path

        import mss  # type: ignore[import]
        import mss.tools  # type: ignore[import]
        out = Path("~/.prism/screenshots").expanduser()
        out.mkdir(parents=True, exist_ok=True)
        p = out / f"cu_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        with mss.mss() as sct:
            img = sct.grab(sct.monitors[1])
            mss.tools.to_png(img.rgb, img.size, output=str(p))
        return str(p)
    except Exception:
        return None


def execute(intent: str, message: str, ctx: dict):
    from prism_responses import text_card

    goal = _parse_goal(message)
    if not goal:
        return text_card("What should I do on the screen?", "Computer use")

    try:
        from prism_computer_use import ComputerUse, actions_for, prerequisites
        from prism_device_agent import DeviceCapabilityScanner
        from prism_device_executor import run_argv
    except Exception as exc:
        return text_card(f"Computer-use unavailable: {exc}", "Computer use")

    caps = DeviceCapabilityScanner().scan()
    router = ctx.get("router")
    ready, missing = prerequisites(caps, router)
    if not ready:
        return text_card(
            "I can't drive the screen here yet — missing:\n"
            + "\n".join(f"  • {m}" for m in missing)
            + "\n\nComputer-use needs a local desktop, an input backend "
            "(ydotool/xdotool), a screenshot backend, and a vision-capable "
            "model.", "Computer use")

    backend = caps.best_tool("input_synth")

    def see(path, goal_, history):
        from prism_computer_use import _vision_prompt
        try:
            raw, _ = router.call(_vision_prompt(goal_, history), images=[path],
                                 json_mode=True, min_capability=2, max_tokens=300)
            return raw
        except Exception:
            return ""

    def act(step):
        seq = actions_for(backend, step)
        if not seq:
            return False
        for argv in seq:
            if not run_argv(argv, timeout=8).success:
                return False
        try:
            from prism_local_notify import deliver
            deliver("Computer-use step",
                    f"{step.action} {step.params}", source="computer_use")
        except Exception:
            pass
        return True

    driver = ComputerUse(capture=_capture_screenshot, see=see, act=act,
                         max_steps=_MAX_STEPS)
    result = driver.run(goal)

    lines = [f"Goal: {goal}",
             f"Outcome: {'✓ ' if result.success else '✗ '}{result.message}",
             f"Steps taken: {len(result.steps)}"]
    for i, s in enumerate(result.steps, 1):
        lines.append(f"  {i}. {s.action} {s.params}"
                     + (f" — {s.reason}" if s.reason else ""))
    card = text_card("\n".join(lines), "Computer use")
    card.card_data.update({"success": result.success,
                           "steps": len(result.steps), "goal": goal})
    return card
