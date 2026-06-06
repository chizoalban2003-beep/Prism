"""
Tests for the VEAX bidirectional control interface.

Covers:
  - Named presets
  - State file persistence (save / load)
  - load_spectrum() layers in persisted state
  - render_gates() visual output
  - nl_to_veax() with mocked router
  - veax_control organ: show / preset / reset / set / delta / NL paths
  - PrismChain._sync_spectrum() picks up in-session updates
"""
from __future__ import annotations

import json

import pytest

from prism_spectrum_middleware import (
    VEAX_PRESETS,
    SpectrumGates,
    get_current_gates,
    load_spectrum,
    nl_to_veax,
    render_gates,
    save_spectrum_state,
    set_current_gates,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    """Redirect state file to tmp_path so tests never touch ~/.prism."""
    monkeypatch.setattr("prism_spectrum_middleware._STATE_PATH", tmp_path / "spectrum_state.json")
    set_current_gates(SpectrumGates(V=0.5, E=0.5, A=0.5, X=0.5))
    yield
    set_current_gates(SpectrumGates(V=0.5, E=0.5, A=0.5, X=0.5))


@pytest.fixture()
def balanced():
    return SpectrumGates(V=0.5, E=0.5, A=0.5, X=0.5)


# ── Named presets ─────────────────────────────────────────────────────────────

class TestVEAXPresets:
    def test_all_five_presets_exist(self):
        assert set(VEAX_PRESETS) == {"scout", "audit", "execution", "review", "balanced"}

    def test_audit_preset_values(self):
        p = VEAX_PRESETS["audit"]
        assert p["V"] == pytest.approx(0.9)
        assert p["E"] == pytest.approx(0.2)
        assert p["A"] == pytest.approx(0.3)
        assert p["X"] == pytest.approx(0.9)

    def test_scout_preset_values(self):
        p = VEAX_PRESETS["scout"]
        assert p["V"] == pytest.approx(0.3)
        assert p["E"] == pytest.approx(0.8)

    def test_execution_preset_low_X(self):
        assert VEAX_PRESETS["execution"]["X"] == pytest.approx(0.1)

    def test_all_preset_values_in_range(self):
        for name, vals in VEAX_PRESETS.items():
            for axis, v in vals.items():
                assert 0.0 <= v <= 1.0, f"{name}.{axis}={v} out of range"

    def test_preset_applied_to_gates(self):
        g = SpectrumGates(**VEAX_PRESETS["audit"])
        assert g.verification_threshold() == 5
        assert g.logicpolicy_verbosity() == "full"
        assert not g.requires_approval(False)   # A=0.3, just at boundary → False


# ── State file persistence ────────────────────────────────────────────────────

class TestStatePersistence:
    def test_save_creates_state_file(self, tmp_path):
        path = tmp_path / "spectrum_state.json"
        import prism_spectrum_middleware as psm
        psm._STATE_PATH = path
        gates = SpectrumGates(V=0.8, E=0.2, A=0.4, X=0.7)
        save_spectrum_state(gates)
        assert path.exists()

    def test_save_persists_values(self, tmp_path):
        import prism_spectrum_middleware as psm
        path = tmp_path / "s.json"
        psm._STATE_PATH = path
        gates = SpectrumGates(V=0.8, E=0.2, A=0.4, X=0.7)
        save_spectrum_state(gates)
        data = json.loads(path.read_text())
        assert data["V"] == pytest.approx(0.8)
        assert data["E"] == pytest.approx(0.2)

    def test_save_stores_preset_name(self, tmp_path):
        import prism_spectrum_middleware as psm
        psm._STATE_PATH = tmp_path / "s.json"
        save_spectrum_state(SpectrumGates(**VEAX_PRESETS["audit"]), preset="audit")
        data = json.loads((tmp_path / "s.json").read_text())
        assert data["preset"] == "audit"

    def test_save_updates_singleton(self):
        gates = SpectrumGates(V=0.9, E=0.1, A=0.9, X=0.1)
        save_spectrum_state(gates)
        assert get_current_gates() is gates

    def test_load_spectrum_reads_state_file(self, tmp_path, monkeypatch):
        state_file = tmp_path / "s.json"
        state_file.write_text('{"V": 0.9, "E": 0.1, "A": 0.8, "X": 0.7}')
        monkeypatch.setattr("prism_spectrum_middleware._STATE_PATH", state_file)
        gates, _ = load_spectrum()
        assert gates.V == pytest.approx(0.9)
        assert gates.E == pytest.approx(0.1)

    def test_load_spectrum_explicit_config_ignores_state_file(self, tmp_path, monkeypatch):
        state_file = tmp_path / "s.json"
        state_file.write_text('{"V": 0.9, "E": 0.1, "A": 0.8, "X": 0.7}')
        monkeypatch.setattr("prism_spectrum_middleware._STATE_PATH", state_file)
        gates, _ = load_spectrum(config={"spectrum": {"V": 0.3}})
        assert gates.V == pytest.approx(0.3)  # explicit config wins

    def test_missing_state_file_falls_through_to_defaults(self):
        # _STATE_PATH redirected to tmp_path that has no file
        gates, _ = load_spectrum()
        assert gates.V == pytest.approx(0.5)  # TOML or fallback default


# ── render_gates ──────────────────────────────────────────────────────────────

class TestRenderGates:
    def test_render_contains_all_axes(self, balanced):
        out = render_gates(balanced)
        for axis in ("V", "E", "A", "X"):
            assert axis in out

    def test_render_contains_block_chars(self, balanced):
        out = render_gates(balanced)
        assert "█" in out
        assert "░" in out

    def test_render_shows_value(self):
        g = SpectrumGates(V=0.8, E=0.2, A=0.5, X=0.5)
        out = render_gates(g)
        assert "0.80" in out
        assert "0.20" in out

    def test_render_shows_preset_name(self, balanced):
        out = render_gates(balanced, preset="audit")
        assert "audit" in out

    def test_render_no_preset_label_when_none(self, balanced):
        out = render_gates(balanced)
        assert "Preset:" not in out

    def test_render_high_V_shows_strict_proof(self):
        g = SpectrumGates(V=0.9, E=0.5, A=0.5, X=0.5)
        assert "strict proof" in render_gates(g)

    def test_render_low_V_shows_accept_all(self):
        g = SpectrumGates(V=0.1, E=0.5, A=0.5, X=0.5)
        assert "accept all" in render_gates(g)

    def test_render_full_bar_at_max(self):
        g = SpectrumGates(V=1.0, E=1.0, A=1.0, X=1.0)
        out = render_gates(g)
        assert "██████████" in out

    def test_render_empty_bar_at_zero(self):
        g = SpectrumGates(V=0.0, E=0.0, A=0.0, X=0.0)
        out = render_gates(g)
        assert "░░░░░░░░░░" in out


# ── nl_to_veax (mocked router) ────────────────────────────────────────────────

class MockRouter:
    def __init__(self, response: str):
        self._resp = response

    def call(self, prompt, system="", json_mode=False, max_tokens=300, min_capability=0):
        return self._resp, "mock"


class TestNLToVEAX:
    def test_set_action(self, balanced):
        router = MockRouter('{"action": "set", "A": 0.8, "reasoning": "user wants more autonomy"}')
        result = nl_to_veax("give it more autonomy", balanced, router)
        assert result["action"] == "set"
        assert result["A"] == pytest.approx(0.8)

    def test_delta_action(self, balanced):
        router = MockRouter('{"action": "delta", "V": 0.2, "reasoning": "increase verification"}')
        result = nl_to_veax("be more strict", balanced, router)
        assert result["action"] == "delta"
        assert result["V"] == pytest.approx(0.2)

    def test_preset_action(self, balanced):
        router = MockRouter('{"action": "preset", "preset": "audit", "reasoning": "audit mode"}')
        result = nl_to_veax("use audit mode", balanced, router)
        assert result["action"] == "preset"
        assert result["preset"] == "audit"

    def test_get_action_on_parse_error(self, balanced):
        router = MockRouter("not valid json {{{{")
        result = nl_to_veax("something", balanced, router)
        assert result["action"] == "get"

    def test_router_exception_returns_get(self, balanced):
        class BrokenRouter:
            def call(self, *a, **kw):
                raise RuntimeError("network error")
        result = nl_to_veax("test", balanced, BrokenRouter())
        assert result["action"] == "get"


# ── veax_control organ ────────────────────────────────────────────────────────

@pytest.fixture()
def organ():
    from organs.veax_control import execute
    return execute


@pytest.fixture()
def ctx_no_router():
    return {}


class TestOrganShow:
    def test_show_returns_current_state(self, organ, ctx_no_router):
        set_current_gates(SpectrumGates(V=0.7, E=0.3, A=0.5, X=0.8))
        card = organ("veax_control", "show spectrum", ctx_no_router)
        assert "0.70" in card.body
        assert "Verification" in card.body

    def test_empty_message_returns_state(self, organ, ctx_no_router):
        card = organ("veax_control", "", ctx_no_router)
        assert "VEAX" in card.body

    def test_status_message_returns_state(self, organ, ctx_no_router):
        card = organ("veax_control", "what is my current spectrum?", ctx_no_router)
        assert "Current VEAX Spectrum" in card.body

    def test_check_returns_state(self, organ, ctx_no_router):
        card = organ("veax_control", "check veax", ctx_no_router)
        assert "Verification" in card.body


class TestOrganPresets:
    def test_audit_preset_applied(self, organ, ctx_no_router):
        card = organ("veax_control", "use audit mode", ctx_no_router)
        assert "audit" in card.body.lower()
        assert get_current_gates().V == pytest.approx(0.9)

    def test_scout_preset_applied(self, organ, ctx_no_router):
        organ("veax_control", "switch to scout", ctx_no_router)
        assert get_current_gates().E == pytest.approx(0.8)

    def test_execution_preset_applied(self, organ, ctx_no_router):
        organ("veax_control", "execution mode", ctx_no_router)
        assert get_current_gates().X == pytest.approx(0.1)

    def test_preset_persisted(self, organ, ctx_no_router, tmp_path):
        import prism_spectrum_middleware as psm
        path = tmp_path / "s.json"
        psm._STATE_PATH = path
        organ("veax_control", "apply review mode", ctx_no_router)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["preset"] == "review"

    def test_review_preset_applied(self, organ, ctx_no_router):
        organ("veax_control", "use review preset", ctx_no_router)
        g = get_current_gates()
        assert g.V == pytest.approx(0.8)
        assert g.A == pytest.approx(0.2)


class TestOrganReset:
    def test_reset_returns_balanced(self, organ, ctx_no_router):
        set_current_gates(SpectrumGates(**VEAX_PRESETS["audit"]))
        card = organ("veax_control", "reset to defaults", ctx_no_router)
        assert "balanced" in card.body.lower() or "Reset" in card.body
        g = get_current_gates()
        assert g.V == pytest.approx(0.5)

    def test_default_keyword_resets(self, organ, ctx_no_router):
        organ("veax_control", "go back to default settings", ctx_no_router)
        assert get_current_gates().V == pytest.approx(0.5)


class TestOrganNLWithRouter:
    def _ctx(self, response: str) -> dict:
        return {"router": MockRouter(response)}

    def test_set_via_nl(self, organ):
        ctx = self._ctx('{"action": "set", "A": 0.9, "reasoning": "maximize autonomy"}')
        card = organ("veax_control", "I want full autonomy", ctx)
        assert get_current_gates().A == pytest.approx(0.9)
        assert "0.90" in card.body

    def test_delta_via_nl(self, organ):
        set_current_gates(SpectrumGates(V=0.5, E=0.5, A=0.5, X=0.5))
        ctx = self._ctx('{"action": "delta", "V": 0.2, "reasoning": "more strict"}')
        organ("veax_control", "be more careful with verification", ctx)
        assert get_current_gates().V == pytest.approx(0.7)

    def test_delta_clamps_at_1(self, organ):
        set_current_gates(SpectrumGates(V=0.9, E=0.5, A=0.5, X=0.5))
        ctx = self._ctx('{"action": "delta", "V": 0.5, "reasoning": "push V high"}')
        organ("veax_control", "even more verification", ctx)
        assert get_current_gates().V == pytest.approx(1.0)

    def test_delta_clamps_at_0(self, organ):
        set_current_gates(SpectrumGates(V=0.1, E=0.5, A=0.5, X=0.5))
        ctx = self._ctx('{"action": "delta", "V": -0.5, "reasoning": "push V low"}')
        organ("veax_control", "be very permissive", ctx)
        assert get_current_gates().V == pytest.approx(0.0)

    def test_no_change_returns_current(self, organ):
        ctx = self._ctx('{"action": "get", "reasoning": "already optimal"}')
        card = organ("veax_control", "is this good?", ctx)
        assert "Current VEAX Spectrum" in card.body

    def test_preset_via_router(self, organ):
        ctx = self._ctx('{"action": "preset", "preset": "scout", "reasoning": "explore mode"}')
        organ("veax_control", "I want to explore freely", ctx)
        assert get_current_gates().E == pytest.approx(0.8)

    def test_reasoning_shown_in_output(self, organ):
        ctx = self._ctx('{"action": "set", "X": 0.9, "reasoning": "user wants verbose traces"}')
        card = organ("veax_control", "give me detailed explanations", ctx)
        assert "verbose traces" in card.body or "Interpreted as" in card.body

    def test_parse_error_shows_current(self, organ):
        ctx = self._ctx("invalid json }{{{")
        card = organ("veax_control", "do something weird", ctx)
        assert "Current VEAX Spectrum" in card.body


# ── PrismChain._sync_spectrum ─────────────────────────────────────────────────

class TestChainSync:
    def test_sync_picks_up_singleton_change(self):
        from prism_chain import PrismChain
        chain = PrismChain()  # no router — minimal init
        original = chain._spectrum_gates
        new_gates = SpectrumGates(V=0.9, E=0.1, A=0.8, X=0.7)
        set_current_gates(new_gates)
        chain._sync_spectrum()
        assert chain._spectrum_gates is new_gates
        assert chain._spectrum_gates is not original

    def test_sync_no_op_when_unchanged(self):
        from prism_chain import PrismChain
        chain = PrismChain()
        set_current_gates(chain._spectrum_gates)
        chain._sync_spectrum()
        assert chain._spectrum_gates.V == pytest.approx(0.5)

    def test_sync_no_op_when_singleton_none(self):
        import prism_spectrum_middleware as psm
        from prism_chain import PrismChain
        chain = PrismChain()
        original = chain._spectrum_gates
        old = psm._current_gates
        psm._current_gates = None
        chain._sync_spectrum()
        assert chain._spectrum_gates is original
        psm._current_gates = old
