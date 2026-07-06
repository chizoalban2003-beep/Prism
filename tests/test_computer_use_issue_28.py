"""
tests/test_computer_use_issue_28.py
===================================
Closed computer-use loop: parse_vision_action, coordinate actuation
(actions_for), the bounded loop (done / fail / step-limit / capture-fail),
prerequisites gating, and the organ's honest degradation + high-risk policy.
All tested with fakes — no display, model, or input backend required.
"""
from __future__ import annotations

import importlib.util

import prism_computer_use as cu
from prism_intents import INTENTS
from prism_routing import route_intent


def _load():
    spec = importlib.util.spec_from_file_location(
        "computer_use", "organs/computer_use.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ORG = _load()


def _route(m):
    return route_intent(m, INTENTS, lambda _m: "")


class TestParseVisionAction:
    def test_valid(self):
        s = cu.parse_vision_action('{"action":"click","x":10,"y":20,"reason":"btn"}')
        assert s.action == "click" and s.params == {"x": 10, "y": 20}
        assert s.reason == "btn"

    def test_fenced_json(self):
        s = cu.parse_vision_action('```json\n{"action":"done","reason":"ok"}\n```')
        assert s.action == "done"

    def test_invalid_action_rejected(self):
        assert cu.parse_vision_action('{"action":"launch_missiles"}') is None

    def test_garbage_rejected(self):
        assert cu.parse_vision_action("not json at all") is None
        assert cu.parse_vision_action("") is None


class TestActuation:
    def test_click_is_move_then_click(self):
        seq = cu.actions_for("xdotool", cu.Step("click", {"x": 5, "y": 6}))
        assert seq == [["xdotool", "mousemove", "5", "6"],
                       ["xdotool", "click", "1"]]

    def test_double_and_right_click(self):
        dbl = cu.actions_for("xdotool", cu.Step("double_click", {"x": 1, "y": 2}))
        assert "--repeat" in dbl[1]
        rc = cu.actions_for("ydotool", cu.Step("right_click", {"x": 1, "y": 2}))
        assert rc[1] == ["ydotool", "click", "0xC1"]

    def test_type_and_key(self):
        assert cu.actions_for("xdotool", cu.Step("type", {"text": "hi"})) == [
            ["xdotool", "type", "--clearmodifiers", "hi"]]
        assert cu.actions_for("xdotool", cu.Step("key", {"keys": "ctrl+s"})) == [
            ["xdotool", "key", "ctrl+s"]]

    def test_click_without_coords_is_none(self):
        assert cu.actions_for("xdotool", cu.Step("click", {})) is None


class TestLoop:
    def _driver(self, sees, act=None, max_steps=8):
        it = iter(sees)
        acted = []

        def _act(step):
            acted.append(step.action)
            return True if act is None else act(step)
        d = cu.ComputerUse(capture=lambda: "/tmp/x.png",
                           see=lambda p, g, h: next(it),
                           act=_act, max_steps=max_steps)
        d._acted = acted
        return d

    def test_success_on_done(self):
        d = self._driver(['{"action":"click","x":1,"y":2}',
                          '{"action":"done","reason":"there"}'])
        r = d.run("open menu")
        assert r.success is True
        assert [s.action for s in r.steps] == ["click"]

    def test_fail_action_stops(self):
        d = self._driver(['{"action":"fail","reason":"no such button"}'])
        r = d.run("impossible")
        assert r.success is False and "no such button" in r.message

    def test_step_limit_bounds_runaway(self):
        d = self._driver(['{"action":"click","x":1,"y":1}'] * 20, max_steps=3)
        r = d.run("loop")
        assert r.success is False and "3-step" in r.message
        assert len(r.steps) == 3

    def test_capture_failure_is_honest(self):
        d = cu.ComputerUse(capture=lambda: None,
                           see=lambda p, g, h: "{}", act=lambda s: True)
        r = d.run("x")
        assert "capture" in r.message.lower()

    def test_unusable_vision_output_stops(self):
        d = self._driver(["complete garbage"])
        r = d.run("x")
        assert r.success is False and "no usable action" in r.message

    def test_actuation_failure_stops(self):
        d = self._driver(['{"action":"click","x":1,"y":2}'], act=lambda s: False)
        r = d.run("x")
        assert r.success is False and "Actuation failed" in r.message


class _Caps:
    def __init__(self, display=True, inp="xdotool", shot="grim"):
        self.has_display = display
        self._inp = inp
        self._shot = shot

    def best_tool(self, cat):
        return {"input_synth": self._inp, "screenshot": self._shot}.get(cat)


class TestPrerequisites:
    def test_all_present(self):
        ready, missing = cu.prerequisites(_Caps(), router=object())
        assert ready is True and missing == []

    def test_reports_each_missing(self):
        ready, missing = cu.prerequisites(
            _Caps(display=False, inp=None, shot=None), router=None)
        assert ready is False
        joined = " ".join(missing)
        assert "display" in joined and "input" in joined and "router" in joined


class TestOrgan:
    def test_high_risk_and_gated(self):
        assert ORG.ORGAN_POLICY["risk_level"] == "high"
        assert ORG.ORGAN_POLICY["requires_approval"] is True

    def test_degrades_honestly_without_backends(self, monkeypatch):
        import prism_device_agent as pda

        class Caps:
            has_display = False

            def best_tool(self, c):
                return None
        monkeypatch.setattr(pda.DeviceCapabilityScanner, "scan",
                            lambda self: Caps())
        card = ORG.execute("computer_use", "use the computer to open x",
                           {"router": object()})
        assert "missing" in card.body.lower()

    def test_parse_goal(self):
        assert ORG._parse_goal("use the computer to book a flight") == "book a flight"
        assert ORG._parse_goal("computer use: fill the form") == "fill the form"


class TestRouting:
    def test_computer_use_phrasings(self):
        for m in ("use the computer to book a flight",
                  "control my screen to open settings",
                  "computer use: fill the form"):
            assert _route(m) == "computer_use", m

    def test_does_not_steal_neighbours(self):
        assert _route("what can this computer do") != "computer_use"
        assert _route("type hello") == "input_control"
        assert _route("list my tasks") == "list_tasks"
