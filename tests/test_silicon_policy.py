"""Tests for SiliconResponsePolicy and ExecutionBudget."""
from prism_silicon_policy import _Q_ORDER, ExecutionBudget, SiliconResponsePolicy, get_policy


class TestExecutionBudget:
    def test_default_no_throttle(self):
        b = ExecutionBudget()
        assert b.capability_ceil == 3
        assert b.max_tokens == 1500
        assert not b.speculative
        assert not b.disable_evolution
        assert b.throttle_reason == ""

    def test_dataclass_fields(self):
        b = ExecutionBudget(capability_ceil=1, max_tokens=400, speculative=True,
                            disable_evolution=True, quantization_hint="int4",
                            throttle_reason="test")
        assert b.quantization_hint == "int4"
        assert b.throttle_reason == "test"


class TestSiliconResponsePolicy:
    def setup_method(self):
        self.policy = SiliconResponsePolicy()

    def test_healthy_state_no_throttle(self):
        b = self.policy._compute(delta_b=0.0, phase_name="STABLE", delta_h_override=0.0)
        assert b.throttle_reason == ""
        assert b.capability_ceil == 3
        assert b.max_tokens == 1500

    def test_moderate_pressure_reduces_tokens(self):
        b = self.policy._compute(delta_b=0.35, phase_name="STABLE", delta_h_override=0.0)
        assert b.max_tokens <= 1000
        assert b.capability_ceil == 3
        assert b.throttle_reason != ""

    def test_high_pressure_reduces_capability(self):
        b = self.policy._compute(delta_b=0.55, phase_name="STABLE", delta_h_override=0.0)
        assert b.capability_ceil == 2
        assert b.speculative is True

    def test_critical_pressure_disables_evolution(self):
        b = self.policy._compute(delta_b=0.75, phase_name="STABLE", delta_h_override=0.0)
        assert b.capability_ceil == 1
        assert b.disable_evolution is True
        assert b.max_tokens <= 400
        assert b.quantization_hint == "int4"

    def test_liquid_phase_tightens_regardless_of_debt(self):
        b = self.policy._compute(delta_b=0.0, phase_name="LIQUID", delta_h_override=0.0)
        assert b.capability_ceil == 1
        assert b.speculative is True

    def test_viscous_phase_reduces_tokens(self):
        b = self.policy._compute(delta_b=0.0, phase_name="VISCOUS", delta_h_override=0.0)
        assert b.max_tokens <= 900
        assert b.speculative is True

    def test_crystal_phase_no_override(self):
        b = self.policy._compute(delta_b=0.0, phase_name="CRYSTAL", delta_h_override=0.0)
        # CRYSTAL = healthy, no tightening
        assert b.capability_ceil == 3
        assert b.throttle_reason == ""

    def test_combined_pressure_uses_max(self):
        # delta_b=0.2 (below moderate), but delta_h=0.6 (high)
        b = self.policy._compute(delta_b=0.2, phase_name="STABLE", delta_h_override=0.6)
        assert b.capability_ceil == 2  # high pressure wins

    def test_phase_override_never_loosens_capability(self):
        # Critical debt sets capability_ceil=1, CRYSTAL should not loosen it
        b = self.policy._compute(delta_b=0.75, phase_name="CRYSTAL", delta_h_override=0.0)
        assert b.capability_ceil == 1

    def test_quantization_hint_tightens_toward_int4(self):
        b = self.policy._compute(delta_b=0.0, phase_name="LIQUID", delta_h_override=0.0)
        assert b.quantization_hint == "int4"

    def test_quantization_order_int4_most_aggressive(self):
        assert _Q_ORDER.index("int4") < _Q_ORDER.index("int8") < _Q_ORDER.index("fp16") < _Q_ORDER.index("fp32")

    def test_ttl_caches_budget(self):
        b1 = self.policy.current_budget(delta_b=0.0, phase_name="STABLE", delta_h=0.0)
        b2 = self.policy.current_budget(delta_b=0.9, phase_name="LIQUID", delta_h=0.9)
        # Should return cached b1 even though inputs changed drastically
        assert b1 is b2

    def test_ttl_expired_recomputes(self):
        self.policy._last_ts = 0.0  # force expiry
        b1 = self.policy.current_budget(delta_b=0.0, phase_name="STABLE", delta_h=0.0)
        self.policy._last_ts = 0.0  # force expiry again
        b2 = self.policy.current_budget(delta_b=0.8, phase_name="LIQUID", delta_h=0.8)
        assert b1 is not b2
        assert b2.capability_ceil < b1.capability_ceil

    def test_gpu_quantization_target_cpu_returns_none(self):
        # On this machine (no GPU), should return None
        b = ExecutionBudget(quantization_hint="int4")
        result = self.policy.gpu_quantization_target(b)
        assert result is None  # no GPU in test environment

    def test_thermal_throttle_ratio_no_crash(self):
        # Just verifies it runs without error on any Linux machine
        ratio = SiliconResponsePolicy._thermal_throttle_ratio()
        assert 0.0 <= ratio <= 1.0

    def test_extended_delta_h_override_passthrough(self):
        result = self.policy._extended_delta_h(0.42)
        assert result == 0.42


class TestPolicySingleton:
    def test_get_policy_returns_same_instance(self):
        p1 = get_policy()
        p2 = get_policy()
        assert p1 is p2


class TestBiologicalPressureMethod:
    def test_biological_pressure_zero_on_fresh_bridge(self):
        from prism_perception import BiometricVEAXBridge
        bridge = BiometricVEAXBridge()
        assert bridge.biological_pressure() == 0.0

    def test_biological_pressure_after_debt(self):
        from prism_perception import BiometricVEAXBridge
        bridge = BiometricVEAXBridge()
        bridge.dynamics.add_debt("V", 1.5)
        bridge.dynamics.add_debt("E", 1.0)
        pressure = bridge.biological_pressure()
        assert pressure > 0.0
        assert pressure <= 1.0
