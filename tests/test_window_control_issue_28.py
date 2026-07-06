"""
tests/test_window_control_issue_28.py
=====================================
window_control organ: pure argv-builder + list-parser + intent parsing are
tested directly; execute() is tested for honest degradation (no display / no
backend) and, with a stubbed subprocess + capability scan, for the happy path
on a machine that DOES have a backend.
"""
from __future__ import annotations

import importlib.util

from prism_intents import INTENTS
from prism_routing import route_intent


def _load():
    spec = importlib.util.spec_from_file_location(
        "window_control", "organs/window_control.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W = _load()


def _route(m):
    return route_intent(m, INTENTS, lambda _m: "")


class TestParse:
    def test_actions(self):
        assert W._parse_action("list windows") == ("list", "")
        assert W._parse_action("close the Firefox window") == ("close", "Firefox")
        assert W._parse_action("minimize") == ("minimize", "")
        assert W._parse_action("maximize the editor") == ("maximize", "editor")

    def test_focus_and_bring_to_front(self):
        assert W._parse_action("focus my terminal window") == ("focus", "terminal")
        assert W._parse_action("bring Chrome to the front") == ("focus", "Chrome")
        assert W._parse_action("bring up my terminal") == ("focus", "terminal")


class TestArgv:
    def test_wmctrl(self):
        assert W._build_argv("wmctrl", "list", "") == ["wmctrl", "-l"]
        assert W._build_argv("wmctrl", "focus", "FF") == ["wmctrl", "-a", "FF"]
        assert W._build_argv("wmctrl", "close", "FF") == ["wmctrl", "-c", "FF"]
        assert "maximized_vert" in " ".join(
            W._build_argv("wmctrl", "maximize", "E"))

    def test_xdotool_and_unsupported(self):
        assert W._build_argv("xdotool", "focus", "FF") == [
            "xdotool", "search", "--name", "FF", "windowactivate"]
        # xdotool cannot maximise directly
        assert W._build_argv("xdotool", "maximize", "E") is None
        # focus without a target is not expressible
        assert W._build_argv("wmctrl", "focus", "") is None


class TestListParse:
    def test_wmctrl_titles(self):
        out = ("0x03000007  0 host My Cool App\n"
               "0x0300000a  0 host Terminal")
        assert W._parse_window_list("wmctrl", out) == ["My Cool App", "Terminal"]

    def test_xdotool_ids(self):
        assert W._parse_window_list("xdotool", "12345\n67890") == ["12345", "67890"]


class TestDegrade:
    def test_no_display(self, monkeypatch):
        import prism_device_agent as pda

        class FakeCaps:
            has_display = False
            session_type = ""
            def best_tool(self, c): return None
        monkeypatch.setattr(pda.DeviceCapabilityScanner, "scan",
                            lambda self: FakeCaps())
        card = W.execute("window_control", "list windows", {})
        assert "headless" in card.body.lower() or "no graphical" in card.body.lower()

    def test_no_backend(self, monkeypatch):
        import prism_device_agent as pda

        class FakeCaps:
            has_display = True
            session_type = "wayland"
            def best_tool(self, c): return None
        monkeypatch.setattr(pda.DeviceCapabilityScanner, "scan",
                            lambda self: FakeCaps())
        card = W.execute("window_control", "focus Firefox window", {})
        assert "install" in card.body.lower()


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

    def test_list_returns_windows(self, monkeypatch):
        self._patch(monkeypatch, "wmctrl",
                    _FakeResult(output="0x01 0 host Alpha\n0x02 0 host Beta"))
        card = W.execute("window_control", "list windows", {})
        assert card.card_data["windows"] == ["Alpha", "Beta"]
        assert card.card_data["backend"] == "wmctrl"

    def test_focus_success(self, monkeypatch):
        self._patch(monkeypatch, "wmctrl", _FakeResult(success=True))
        card = W.execute("window_control", "focus Firefox window", {})
        assert "Firefox" in card.body and card.card_data["action"] == "focus"

    def test_focus_no_match_reports(self, monkeypatch):
        self._patch(monkeypatch, "wmctrl", _FakeResult(success=False, error=""))
        card = W.execute("window_control", "focus Nonexistent window", {})
        assert "couldn't" in card.body.lower() or "no match" in card.body.lower()


class TestRouting:
    def test_window_ops_route(self):
        for m in ("list windows", "close the Firefox window", "minimize",
                  "maximize the editor", "bring Chrome to the front",
                  "focus my terminal window"):
            assert _route(m) == "window_control", m

    def test_does_not_steal_tasks(self):
        assert _route("close my task buy milk") == "complete_task"
        assert _route("list my tasks") == "list_tasks"
