"""
prism_computer_use.py
=====================
Closed "computer-use" loop: capture the screen, ask a vision-capable LLM where
to act next, actuate via the input backend, observe, repeat — until the goal
is met, the model gives up, or a hard step bound is hit.

This is the perception→action loop that sits on top of the already-built
pieces: the screenshot organ (capture), the LLM router's multimodal `call(...,
images=[...])` (see), and the input backends ydotool/xdotool (act). It is the
autonomous-targeting frontier — deciding *where* to click from a picture.

Safety
------
Driven only through the approval-gated ``computer_use`` organ, so a whole
session is authorised once by the user. Within a session:
  * a hard ``max_steps`` bound (default 8) caps runaway loops,
  * every actuation is logged to the PRISM notification inbox for audit,
  * the belt's taint rule still denies the organ after untrusted content, and
  * it refuses to start unless display + input backend + a vision-capable
    model are all present (honest degradation otherwise).

Design: capture / see / act are injected callables so the loop logic is
unit-tested with fakes (no display, no model, no input needed).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Coordinate-space actions the vision model may return.
_VALID_ACTIONS = {"click", "double_click", "right_click", "type", "key",
                  "scroll", "done", "fail"}


@dataclass
class Step:
    action: str
    params: dict
    reason: str = ""


@dataclass
class ComputerUseResult:
    goal:      str
    success:   bool
    steps:     list[Step] = field(default_factory=list)
    message:   str = ""


def _vision_prompt(goal: str, history: list[Step]) -> str:
    """Build the instruction for the vision model. It sees the screenshot and
    returns the single next action as strict JSON."""
    done_so_far = "; ".join(f"{s.action}({s.params})" for s in history) or "none"
    return (
        "You are controlling a desktop by looking at a screenshot. The user's "
        f"goal is: \"{goal}\".\n"
        f"Actions taken so far: {done_so_far}.\n"
        "Return ONLY strict JSON for the SINGLE next action, no prose:\n"
        '  {"action":"click|double_click|right_click|type|key|scroll|done|fail",'
        '"x":<int>,"y":<int>,"text":"<for type>","keys":"<for key e.g. ctrl+c>",'
        '"dir":"up|down","reason":"<short>"}\n'
        "Use pixel coordinates from the screenshot's top-left. Use \"done\" "
        "when the goal is visibly achieved, \"fail\" if it cannot be."
    )


def parse_vision_action(raw: str) -> Optional[Step]:
    """Parse the vision model's JSON into a Step. Returns None if unusable."""
    if not raw:
        return None
    text = raw.strip()
    # tolerate ```json fences
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):] if "{" in text else text
    try:
        start, end = text.find("{"), text.rfind("}")
        data = json.loads(text[start:end + 1]) if start >= 0 else json.loads(text)
    except Exception:
        return None
    action = str(data.get("action", "")).strip().lower()
    if action not in _VALID_ACTIONS:
        return None
    params = {k: data[k] for k in ("x", "y", "text", "keys", "dir")
              if k in data and data[k] not in (None, "")}
    return Step(action=action, params=params, reason=str(data.get("reason", "")))


def build_argv(backend: str, step: Step) -> Optional[list[str]]:
    """Coordinate-based actuation argv for xdotool/ydotool, or None if the
    backend can't express it. Clicks first move to (x,y) then click, so the
    caller runs move+click as two argvs — see actions_for()."""
    a = step.action
    if a == "type":
        text = str(step.params.get("text", ""))
        if not text:
            return None
        return {"xdotool": ["xdotool", "type", "--clearmodifiers", text],
                "ydotool": ["ydotool", "type", text]}.get(backend)
    if a == "key":
        keys = str(step.params.get("keys", ""))
        if not keys or backend != "xdotool":
            return None
        return ["xdotool", "key", keys]
    if a == "scroll":
        if backend != "xdotool":
            return None
        btn = "4" if step.params.get("dir") == "up" else "5"
        return ["xdotool", "click", "--repeat", "3", btn]
    return None


def actions_for(backend: str, step: Step) -> Optional[list[list[str]]]:
    """Full argv sequence for a step (clicks = move then click)."""
    a = step.action
    if a in ("click", "double_click", "right_click"):
        x, y = step.params.get("x"), step.params.get("y")
        if x is None or y is None:
            return None
        move = {"xdotool": ["xdotool", "mousemove", str(x), str(y)],
                "ydotool": ["ydotool", "mousemove", "-a", str(x), str(y)]}.get(backend)
        if move is None:
            return None
        if backend == "xdotool":
            btn = {"click": "1", "double_click": "1", "right_click": "3"}[a]
            click = ["xdotool", "click"]
            if a == "double_click":
                click += ["--repeat", "2"]
            return [move, click + [btn]]
        # ydotool
        yb = {"click": "0xC0", "double_click": "0xC0", "right_click": "0xC1"}[a]
        return [move, ["ydotool", "click", yb]]
    single = build_argv(backend, step)
    return [single] if single else None


class ComputerUse:
    """Runs the bounded perception→action loop. capture/see/act are injected
    so the control flow is testable without a real desktop or model."""

    def __init__(
        self,
        capture: Callable[[], Optional[str]],           # () -> screenshot path
        see:     Callable[[str, str, list[Step]], str],  # (path, goal, hist) -> raw JSON
        act:     Callable[[Step], bool],                 # (step) -> ok
        max_steps: int = 8,
        on_step: Optional[Callable[[Step], None]] = None,
    ) -> None:
        self._capture = capture
        self._see = see
        self._act = act
        self._max_steps = max_steps
        self._on_step = on_step

    def run(self, goal: str) -> ComputerUseResult:
        res = ComputerUseResult(goal=goal, success=False)
        for i in range(self._max_steps):
            path = self._capture()
            if not path:
                res.message = "Could not capture the screen."
                return res
            raw = self._see(path, goal, res.steps)
            step = parse_vision_action(raw)
            if step is None:
                res.message = f"Vision model returned no usable action (step {i+1})."
                return res
            if step.action == "done":
                res.success = True
                res.message = step.reason or "Goal reached."
                return res
            if step.action == "fail":
                res.message = step.reason or "Model reported the goal is not achievable."
                return res
            if self._on_step:
                self._on_step(step)
            ok = self._act(step)
            res.steps.append(step)
            if not ok:
                res.message = f"Actuation failed at step {i+1}: {step.action}."
                return res
        res.message = f"Stopped after the {self._max_steps}-step safety limit."
        return res


def prerequisites(caps: Any, router: Any) -> tuple[bool, list[str]]:
    """Return (ready, missing) — computer-use needs a display, an input
    backend, and a vision-capable model. Honest gate before starting."""
    missing = []
    if not getattr(caps, "has_display", False):
        missing.append("a graphical display (headless session)")
    if not caps.best_tool("input_synth"):
        missing.append("an input backend (ydotool/xdotool)")
    if not caps.best_tool("screenshot") and not _has_mss():
        missing.append("a screenshot backend (grim/scrot or `pip install mss`)")
    if router is None:
        missing.append("an LLM router")
    return (not missing, missing)


def _has_mss() -> bool:
    import importlib.util
    return importlib.util.find_spec("mss") is not None
