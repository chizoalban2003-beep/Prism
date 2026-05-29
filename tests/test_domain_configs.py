from __future__ import annotations

from decision_spectrum import OutcomeDiagnosis
from domain_configs import ALL_DOMAINS, MEDICAL, DomainDecisionModel


def test_all_domains():
    assert len(ALL_DOMAINS) == 6


def test_evaluate_returns_diagnosis():
    diagnosis = DomainDecisionModel(MEDICAL).evaluate("Elderly", {"severity": 0.8})
    assert isinstance(diagnosis, OutcomeDiagnosis)
