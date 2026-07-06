"""
tests/test_input_control_issue_28.py
====================================
input_control organ: synthesise keyboard/mouse input via ydotool/xdotool/wtype.
Pure _parse_action / _build_argv / _normalise_combo are unit-tested without a
live display; execute() is tested for honest degradation and, with a stubbed
run_argv + capability scan, for the happy path. Also asserts the organ is
high-risk + approval-gated and that its intent doesn't steal common phrasings.
"""
from __future__ import annotations

import importlib.util

from prism_intents import INTENTS
from prism_routing import route_intent


def _load():
    spec = importlib.util.spec_from_file_location(
        "input_control", "organs/input_control.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


IC = _load()


def _route(m):
    return route_intent(m, INTENTS, lambda _m: "")


class TestParse:
    def test_type_click_key_move_scroll(self):
        assert IC._parse_action("type hello world") == ("type", "hello world")
        assert IC._parse_action("left click") == ("click", "left")
        assert IC._parse_action("right click") == ("click", "right")
        assert IC._parse_action("double click") == ("click", "double")
        assert IC._parse_action("click") == ("click", "left")
        assert IC._parse_action("press ctrl+c") == ("key", "ctrl+c")
        assert IC._parse_action("press enter") == ("key", "Return")
        assert IC._parse_action("move the mouse to 400 300") == ("move", "400 300")
        assert IC._parse_action("scroll down") == ("scroll", "down 3")
        assert IC._parse_action("scroll up 5") == ("scroll", "up 5")

    def test_unknown_is_empty(self):
        assert IC._parse_action("do a barrel roll") == ("", "")

    def test_normalise_combo(self):
        assert IC._normalise_combo("ctrl+C") == "ctrl+c"
        assert IC._normalise_combo("ctrl+alt+del") == "ctrl+alt+Delete"
        assert IC._normalise_combo("esc") == "Escape"
        assert IC._normalise_combo("control + shift + t") == "ctrl+shift+t"


class TestArgv:
    def test_xdotool_full_support(self):
        assert IC._build_argv("xdotool", "type", "hi") == [
            "xdotool", "type", "--clearmodifiers", "hi"]
        assert IC._build_argv("xdotool", "key", "ctrl+c") == [
            "xdotool", "key", "ctrl+c"]
        assert IC._build_argv("xdotool", "click", "right") == [
            "xdotool", "click", "3"]
        assert IC._build_argv("xdotool", "click", "double") == [
            "xdotool", "click", "--repeat", "2", "1"]
        assert IC._build_argv("xdotool", "move", "400 300") == [
            "xdotool", "mousemove", "400", "300"]
        assert IC._build_argv("xdotool", "scroll", "up 3") == [
            "xdotool", "click", "--repeat", "3", "4"]

    def test_ydotool_partial(self):
        assert IC._build_argv("ydotool", "type", "hi") == ["ydotool", "type", "hi"]
        assert IC._build_argv("ydotool", "click", "left") == [
            "ydotool", "click", "0xC0"]
        # ydotool key needs raw keycodes — not expressed → None
        assert IC._build_argv("ydotool", "key", "ctrl+c") is None

    def test_wtype_type_only(self):
        assert IC._build_argv("wtype", "type", "hi") == ["wtype", "hi"]
        assert IC._build_argv("wtype", "click", "left") is None
        assert IC._build_argv("wtype", "move", "1 2") is None


class TestPolicy:
    def test_high_risk_and_approval_gated(self):
        assert IC.ORGAN_POLICY["risk_level"] == "high"
        assert IC.ORGAN_POLICY["requires_approval"] is True
        assert IC.ORGAN_POLICY["irreversible"] is True


class TestDegrade:
    def test_no_display(self, monkeypatch):
        import prism_device_agent as pda

        class FakeCaps:
            has_display = False
            session_type = ""

            def best_tool(self, c):
                return None
        monkeypatch.setattr(pda.DeviceCapabilityScanner, "scan",
                            lambda self: FakeCaps())
        card = IC.execute("input_control", "type hello", {})
        assert "no graphical" in card.body.lower() or "headless" in card.body.lower()

    def test_no_backend(self, monkeypatch):
        import prism_device_agent as pda

        class FakeCaps:
            has_display = True
            session_type = "wayland"

            def best_tool(self, c):
                return None
        monkeypatch.setattr(pda.DeviceCapabilityScanner, "scan",
                            lambda self: FakeCaps())
        card = IC.execute("input_control", "click", {})
        assert "install" in card.body.lower()

    def test_unrecognised_asks(self):
        card = IC.execute("input_control", "do something vague", {})
        assert "tell me" in card.body.lower() or "e.g." in card.body.lower()


class _FakeResult:
    def __init__(self, success=True, output="", error=""):
        self.success = success
        self.output = output
        self.error = error


class TestHappyPathStubbed:
    def _patch(self, monkeypatch, backend, result):
        import prism_device_agent as pda
        import prism_device_executor as pde

        class FakeCaps:
            has_display = True
            session_type = "x11"

            def best_tool(self, c):
                return backend
        monkeypatch.setattr(pda.DeviceCapabilityScanner, "scan",
                            lambda self: FakeCaps())
        monkeypatch.setattr(pde, "run_argv", lambda *a, **k: result)

    def test_type_success(self, monkeypatch):
        self._patch(monkeypatch, "xdotool", _FakeResult(success=True))
        card = IC.execute("input_control", "type hello world", {})
        assert card.card_data["action"] == "type"
        assert card.card_data["backend"] == "xdotool"

    def test_ydotool_socket_hint(self, monkeypatch):
        self._patch(monkeypatch, "ydotool",
                    _FakeResult(success=False, error="failed to connect socket"))
        card = IC.execute("input_control", "click", {})
        assert "ydotoold" in card.body.lower()


class TestRouting:
    def test_input_verbs_route(self):
        for m in ("type hello world", "left click", "press ctrl+c",
                  "move the mouse to 400 300", "scroll down", "click"):
            assert _route(m) == "input_control", m

    def test_does_not_steal_nouns_or_tasks(self):
        assert _route("what type of tea is best") != "input_control"
        assert _route("add task type up notes") == "add_task"
        assert _route("hit the gym") != "input_control"
        assert _route("list my tasks") == "list_tasks"
