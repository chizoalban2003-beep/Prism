from __future__ import annotations

import json

from domain_configs import MEDICAL, DomainDecisionModel
from domain_validator import DomainValidator, LabeledDecision


def _best_case() -> LabeledDecision:
    model = DomainDecisionModel(MEDICAL)
    diagnosis = model.evaluate("Elderly (65+)", {"severity": 0.95, "vital_signs": 0.8})
    return LabeledDecision(
        case_id="1",
        domain="Medical",
        profile="Elderly (65+)",
        factor_values={"severity": 0.95, "vital_signs": 0.8},
        expert_choice=diagnosis.primary_plank.name,
    )


def test_validate_perfect_case():
    validator = DomainValidator("Medical")
    result = validator.validate([_best_case()])

    assert result.accuracy == 1.0


def test_validate_wrong_case():
    validator = DomainValidator("Medical")
    case = _best_case()
    wrong_choice = next(plank.name for plank in MEDICAL.planks if plank.name != case.expert_choice)
    case.expert_choice = wrong_choice

    result = validator.validate([case])

    assert result.accuracy == 0.0


def test_top3_accuracy_gte_accuracy():
    validator = DomainValidator("Medical")
    result = validator.validate([_best_case()])

    assert result.top3_accuracy >= result.accuracy


def test_load_json_parses_cases(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{
        "case_id": "1",
        "domain": "Medical",
        "profile": "Middle-aged",
        "factor_values": {"severity": 0.5},
        "expert_choice": "Urgent clinic",
    }]), encoding="utf-8")

    cases = DomainValidator("Medical").load_json(str(path))

    assert len(cases) == 1
    assert isinstance(cases[0], LabeledDecision)


def test_report_markdown_has_sections():
    validator = DomainValidator("Medical")
    result = validator.validate([_best_case()])
    report = validator.generate_report(result)

    assert "## Summary" in report
    assert "## By Profile" in report
