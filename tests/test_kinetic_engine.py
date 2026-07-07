"""
Tests for prism_kinetic_engine — compound personal signal engine.
"""
from __future__ import annotations

import time

from prism_kinetic_engine import (
    FACTOR_DOMAIN_MAP,
    ActionWindow,
    CrossDomainLink,
    DecisionLever,
    KineticEngine,
    PersonalSignal,
)

# ---------------------------------------------------------------------------
# PersonalSignal
# ---------------------------------------------------------------------------

class TestPersonalSignal:
    def _sig(self, raw=65.0, mu=60.0, sigma=10.0, impact=1.0, confidence=1.0):
        return PersonalSignal(
            domain="health", signal_type="hrv_drop",
            raw_value=raw, mu=mu, sigma=sigma,
            impact=impact, confidence=confidence,
        )

    def test_z_score_positive(self):
        sig = self._sig(raw=80, mu=60, sigma=10)
        assert abs(sig.z_score - 2.0) < 1e-9

    def test_z_score_negative(self):
        sig = self._sig(raw=40, mu=60, sigma=10)
        assert abs(sig.z_score - (-2.0)) < 1e-9

    def test_z_score_zero_sigma(self):
        sig = self._sig(sigma=0)
        assert sig.z_score == 0.0

    def test_expected_value(self):
        sig = self._sig(impact=0.8, confidence=0.9)
        assert abs(sig.expected_value - 0.72) < 1e-9

    def test_timestamp_set(self):
        sig = self._sig()
        assert sig.timestamp <= time.time()


# ---------------------------------------------------------------------------
# FACTOR_DOMAIN_MAP
# ---------------------------------------------------------------------------

class TestFactorDomainMap:
    def test_hrv_is_health(self):
        assert FACTOR_DOMAIN_MAP["hrv_recovery"] == "health"

    def test_circadian_is_energy(self):
        assert FACTOR_DOMAIN_MAP["circadian_energy"] == "energy"

    def test_system_busy_is_cognitive(self):
        assert FACTOR_DOMAIN_MAP["system_busy"] == "cognitive"

    def test_work_context_is_temporal(self):
        assert FACTOR_DOMAIN_MAP["work_context"] == "temporal"

    def test_screen_activity_is_social(self):
        assert FACTOR_DOMAIN_MAP["screen_activity"] == "social"


# ---------------------------------------------------------------------------
# CrossDomainLink
# ---------------------------------------------------------------------------

class TestCrossDomainLink:
    def test_lambda_effective_full_confidence(self):
        link = CrossDomainLink("health", "cognitive", lambda_base=0.4, confidence=1.0)
        assert abs(link.lambda_effective - 0.4) < 1e-9

    def test_lambda_effective_half_confidence(self):
        link = CrossDomainLink("health", "cognitive", lambda_base=0.4, confidence=0.5)
        assert abs(link.lambda_effective - 0.2) < 1e-9

    def test_lambda_effective_zero_confidence(self):
        link = CrossDomainLink("health", "cognitive", lambda_base=0.4, confidence=0.0)
        assert link.lambda_effective == 0.0


# ---------------------------------------------------------------------------
# DecisionLever defaults
# ---------------------------------------------------------------------------

class TestDecisionLever:
    def test_defaults(self):
        lv = DecisionLever("test", "Test", "desc")
        assert lv.net_torque == 0.0
        assert lv.torque_integral == 0.0
        assert lv.activated is False
        assert lv.activate_threshold == 3.0
        assert lv.deactivate_threshold == 1.5


# ---------------------------------------------------------------------------
# KineticEngine — factory and basic API
# ---------------------------------------------------------------------------

class TestKineticEngineFactory:
    def test_for_prism_creates_three_levers(self):
        eng = KineticEngine.for_prism()
        ids = {lv["lever_id"] for lv in eng.lever_status()}
        assert {"intervene_now", "defer_decision", "proactive_assist"} == ids

    def test_for_prism_has_six_links(self):
        eng = KineticEngine.for_prism()
        assert len(eng.link_status()) == 6

    def test_add_lever(self):
        eng = KineticEngine()
        eng.add_lever(DecisionLever("x", "X", "desc"))
        assert len(eng.lever_status()) == 1

    def test_add_link(self):
        eng = KineticEngine()
        eng.add_link(CrossDomainLink("health", "cognitive", 0.3))
        assert len(eng.link_status()) == 1


# ---------------------------------------------------------------------------
# KineticEngine.ingest — torque accumulation
# ---------------------------------------------------------------------------

class TestKineticEngineIngest:
    def _engine(self) -> KineticEngine:
        eng = KineticEngine(action_threshold=100.0)  # high threshold — no windows fire
        eng.add_lever(DecisionLever("L", "L", "desc",
                                    activate_threshold=1000.0,
                                    deactivate_threshold=500.0))
        eng.add_link(CrossDomainLink("health", "cognitive", lambda_base=1.0))
        return eng

    def test_ingest_increases_ema_torque(self):
        eng = self._engine()
        sig = PersonalSignal("health", "hrv", raw_value=90, mu=60, sigma=10)
        eng.ingest(sig)
        lv = eng.lever_status()[0]
        assert lv["ema_torque"] != 0.0

    def test_ingest_increases_integral(self):
        eng = self._engine()
        sig = PersonalSignal("health", "hrv", raw_value=90, mu=60, sigma=10)
        eng.ingest(sig)
        lv = eng.lever_status()[0]
        assert lv["torque_integral"] > 0.0

    def test_ingest_returns_list(self):
        eng = self._engine()
        sig = PersonalSignal("health", "hrv", raw_value=60, mu=60, sigma=10)
        windows = eng.ingest(sig)
        assert isinstance(windows, list)


# ---------------------------------------------------------------------------
# KineticEngine — crisis bypass
# ---------------------------------------------------------------------------

class TestKineticEngineCrisis:
    def test_black_swan_fires_window(self):
        eng = KineticEngine.for_prism()
        # Z = (raw - mu) / sigma = (90 - 0) / 10 = 9 ≥ 8
        sig = PersonalSignal(
            "health", "hrv_drop",
            raw_value=90, mu=0, sigma=10,
            impact=1.0, confidence=1.0,
        )
        windows = eng.ingest(sig)
        # At least one window should fire (crisis bypass)
        assert len(windows) > 0

    def test_black_swan_window_is_crisis(self):
        eng = KineticEngine.for_prism()
        sig = PersonalSignal(
            "health", "hrv_drop",
            raw_value=90, mu=0, sigma=10,
        )
        windows = eng.ingest(sig)
        assert any(w.is_crisis for w in windows)


# ---------------------------------------------------------------------------
# KineticEngine — active_windows and compound_phi_delta
# ---------------------------------------------------------------------------

class TestKineticEngineWindows:
    def test_active_windows_empty_initially(self):
        eng = KineticEngine.for_prism()
        assert eng.active_windows() == []

    def test_compound_phi_delta_zero_no_windows(self):
        eng = KineticEngine.for_prism()
        assert eng.compound_phi_delta() == 0.0

    def test_compound_phi_delta_positive_after_crisis(self):
        eng = KineticEngine.for_prism()
        sig = PersonalSignal("health", "hrv_drop", raw_value=90, mu=0, sigma=10)
        eng.ingest(sig)
        windows = eng.active_windows(max_age_seconds=3600)
        if windows:
            assert eng.compound_phi_delta() != 0.0

    def test_active_windows_max_age_filter(self):
        eng = KineticEngine.for_prism()
        sig = PersonalSignal("health", "hrv_drop", raw_value=90, mu=0, sigma=10)
        eng.ingest(sig)
        # Windows older than 0 seconds → empty
        assert eng.active_windows(max_age_seconds=0) == []


# ---------------------------------------------------------------------------
# KineticEngine — callback
# ---------------------------------------------------------------------------

class TestKineticEngineCallback:
    def test_on_action_callback_fires(self):
        fired = []
        eng = KineticEngine.for_prism()
        eng.on_action(lambda w: fired.append(w))
        sig = PersonalSignal("health", "hrv_drop", raw_value=90, mu=0, sigma=10)
        eng.ingest(sig)
        assert len(fired) > 0

    def test_callback_exception_does_not_propagate(self):
        eng = KineticEngine.for_prism()
        eng.on_action(lambda w: 1 / 0)  # intentional error
        sig = PersonalSignal("health", "hrv_drop", raw_value=90, mu=0, sigma=10)
        eng.ingest(sig)  # must not raise


# ---------------------------------------------------------------------------
# KineticEngine — link confidence update
# ---------------------------------------------------------------------------

class TestKineticEngineLinkConfidence:
    def test_update_link_confidence(self):
        eng = KineticEngine.for_prism()
        eng.update_link_confidence("health", "cognitive", 0.6)
        links = {(lk["source_domain"], lk["target_domain"]): lk["confidence"]
                 for lk in eng.link_status()}
        assert abs(links[("health", "cognitive")] - 0.6) < 1e-9

    def test_confidence_clamped_above_one(self):
        eng = KineticEngine.for_prism()
        eng.update_link_confidence("health", "cognitive", 1.5)
        links = {(lk["source_domain"], lk["target_domain"]): lk["confidence"]
                 for lk in eng.link_status()}
        assert links[("health", "cognitive")] == 1.0

    def test_confidence_clamped_below_zero(self):
        eng = KineticEngine.for_prism()
        eng.update_link_confidence("health", "cognitive", -0.5)
        links = {(lk["source_domain"], lk["target_domain"]): lk["confidence"]
                 for lk in eng.link_status()}
        assert links[("health", "cognitive")] == 0.0


# ---------------------------------------------------------------------------
# ActionWindow.to_proactive_message
# ---------------------------------------------------------------------------

class TestActionWindow:
    def test_user_message_is_human_not_telemetry(self):
        # #28-129: the user-facing message is plain language; the lever_id and
        # Z/ΔA maths live in debug_line() for logs, not in the notification.
        sig = PersonalSignal("health", "hrv_drop", raw_value=80, mu=60, sigma=10)
        win = ActionWindow(
            window_id="abc123", lever_id="intervene_now", source_signal=sig,
            v_potential=0.8, v_current=0.0, c_friction=0.1, delta_a=0.7,
        )
        msg = win.to_proactive_message()
        assert "intervene_now" not in msg          # no jargon to the user
        assert "Z=" not in msg and "ΔA" not in msg
        assert len(msg) > 20                        # a real sentence
        assert "intervene_now" in win.debug_line()  # id preserved for logs

    def test_crisis_message_signals_urgency(self):
        sig = PersonalSignal("health", "hrv_drop", raw_value=90, mu=0, sigma=10)
        win = ActionWindow(
            window_id="abc", lever_id="intervene_now", source_signal=sig,
            v_potential=1.0, v_current=0.0, c_friction=0.1, delta_a=0.9,
            is_crisis=True,
        )
        msg = win.to_proactive_message()
        assert "time-sensitive" in msg.lower()
