"""
tests/test_lora_registry.py
Tests for Vector V: Dynamic Micro-LoRA Swapping / Task-Specialized Prompt Registry
"""
from unittest.mock import patch

import prism_lora_registry as _mod
from prism_lora_registry import (
    LoRAAdapter,
    LoRARegistry,
    get_registry,
)


def _fresh_registry() -> LoRARegistry:
    return LoRARegistry()


# ── Structural tests ──────────────────────────────────────────────────────────

def test_registry_has_all_adapters():
    reg = _fresh_registry()
    expected_ids = {
        "code-analyst", "factual-audit", "creative-scout",
        "fast-executor", "deep-analyst", "recovery-gentle",
    }
    assert set(reg._adapters.keys()) == expected_ids


# ── Selection logic ───────────────────────────────────────────────────────────

def test_select_recovery_on_critical_bio_debt():
    reg = _fresh_registry()
    adapter = reg.select(phase_name="STABLE", bio_debt=0.8)
    assert adapter.adapter_id == "recovery-gentle"


def test_select_fast_on_liquid_phase():
    reg = _fresh_registry()
    adapter = reg.select(phase_name="LIQUID", bio_debt=0.0)
    assert adapter.adapter_id == "fast-executor"


def test_select_factual_on_crystal_phase():
    reg = _fresh_registry()
    adapter = reg.select(phase_name="CRYSTAL", bio_debt=0.0)
    assert adapter.adapter_id == "factual-audit"


def test_select_by_task_hint_code():
    reg = _fresh_registry()
    adapter = reg.select(phase_name="STABLE", bio_debt=0.0, task_hint="code")
    assert adapter.adapter_id == "code-analyst"


def test_select_by_task_hint_creative():
    reg = _fresh_registry()
    adapter = reg.select(phase_name="STABLE", bio_debt=0.0, task_hint="creative")
    assert adapter.adapter_id == "creative-scout"


def test_select_default_deep_analyst():
    reg = _fresh_registry()
    adapter = reg.select(phase_name="STABLE", bio_debt=0.0, task_hint="")
    assert adapter.adapter_id == "deep-analyst"


# ── Prompt injection ──────────────────────────────────────────────────────────

def test_inject_system_prompt_prepends_template():
    reg = _fresh_registry()
    adapter = reg._adapters["code-analyst"]
    result = reg.inject_system_prompt("explain this bug", adapter)
    assert result.startswith(adapter.system_prompt)
    assert "explain this bug" in result
    # Should have separator
    assert "---" in result


def test_inject_empty_system_prompt_returns_prompt():
    reg = _fresh_registry()
    adapter = LoRAAdapter(
        adapter_id="empty-test",
        task_type="test",
        system_prompt="",
    )
    result = reg.inject_system_prompt("my prompt", adapter)
    assert result == "my prompt"


# ── GPU / weight loading ──────────────────────────────────────────────────────

def test_gpu_available_returns_bool():
    reg = _fresh_registry()
    result = reg.gpu_available()
    assert isinstance(result, bool)


def test_load_weights_cpu_fallback_returns_false():
    reg = _fresh_registry()
    # Force GPU unavailable
    with patch.object(reg, "gpu_available", return_value=False):
        adapter = reg._adapters["code-analyst"]
        result = reg.load_weights(adapter)
    assert result is False


# ── Singleton ─────────────────────────────────────────────────────────────────

def test_get_registry_singleton():
    # Reset module singleton first
    _mod._registry = None
    reg1 = get_registry()
    reg2 = get_registry()
    assert reg1 is reg2, "get_registry() should return the same singleton instance"


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_bio_debt_threshold_exactly_at_0_7():
    """bio_debt=0.7 is NOT above 0.7, so should not trigger recovery mode."""
    reg = _fresh_registry()
    adapter = reg.select(phase_name="STABLE", bio_debt=0.7)
    # 0.7 is not > 0.7, should fall through to default
    assert adapter.adapter_id == "deep-analyst"


def test_bio_debt_below_threshold_not_recovery():
    reg = _fresh_registry()
    adapter = reg.select(phase_name="STABLE", bio_debt=0.5)
    assert adapter.adapter_id != "recovery-gentle"
