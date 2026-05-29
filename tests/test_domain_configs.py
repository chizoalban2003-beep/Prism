from __future__ import annotations

from domain_configs import ALL_DOMAINS, MEDICAL, DomainDecisionModel


def test_all_domains_present():
    assert len(ALL_DOMAINS) == 6


def test_model_evaluate_returns_diagnosis():
    model = DomainDecisionModel(MEDICAL)
    diagnosis = model.evaluate(
        "Elderly (65+)",
        {"severity": 0.85, "vital_signs": 0.70, "deteriorating": 0.60},
    )

    assert diagnosis.primary_plank.name
    assert diagnosis.activations


def test_cross_profile_compare_all_profiles():
    model = DomainDecisionModel(MEDICAL)
    result = model.cross_profile_compare({"severity": 0.7})

    assert len(result) == len(MEDICAL.profiles)


def test_sensitivity_sweep_len():
    model = DomainDecisionModel(MEDICAL)
    sweep = model.sensitivity_sweep("Middle-aged", "severity", steps=5)

    assert len(sweep) == 5


def test_fulcrum_moves_with_factors():
    model = DomainDecisionModel(MEDICAL)
    low = model.evaluate("Middle-aged", {"severity": 0.0})
    high = model.evaluate("Middle-aged", {"severity": 1.0})

    assert high.fulcrum_position > low.fulcrum_position
