from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from decision_spectrum import OutcomeDiagnosis
from domain_configs import ALL_DOMAINS, DomainConfig, DomainDecisionModel

logger = logging.getLogger(__name__)


@dataclass
class LabeledDecision:
    """One expert-labeled decision for validation."""
    case_id: str
    domain: str
    profile: str
    factor_values: dict[str, float]
    expert_choice: str
    outcome: str = ""
    notes: str = ""


@dataclass
class DomainValidationResult:
    domain: str
    n_cases: int
    accuracy: float
    top3_accuracy: float
    avg_fulcrum_gap: float
    by_profile: dict[str, float]
    confusion: dict[str, dict]


class DomainValidator:
    """
    Validates a DomainConfig against expert-labeled decisions.

    Three data sources:
      CSV file: case_id,domain,profile,factor1,factor2,...,expert_choice
      JSON list: [{case_id,domain,profile,factor_values:{},expert_choice}]
      In-memory: list[LabeledDecision]
    """

    def __init__(self, domain: str = None):
        self.domain = domain
        config = ALL_DOMAINS.get(domain) if domain else None
        self.model = DomainDecisionModel(config) if config else None
        self._last_profile_counts: dict[str, int] = {}

    def load_csv(self, path: str) -> list[LabeledDecision]:
        """
        CSV columns: case_id, domain, profile, [factor_ids...], expert_choice
        Factor columns match DomainFactor.id values in the config.
        """
        rows: list[LabeledDecision] = []
        with Path(path).expanduser().open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                domain = row.get("domain") or self.domain or ""
                config = ALL_DOMAINS.get(domain)
                factor_ids = {f.id for f in config.factors} if config else set()
                factor_values: dict[str, float] = {}
                for key, value in row.items():
                    if key in factor_ids and value not in (None, ""):
                        factor_values[key] = float(value)
                rows.append(LabeledDecision(
                    case_id=row.get("case_id", ""),
                    domain=domain,
                    profile=row.get("profile", ""),
                    factor_values=factor_values,
                    expert_choice=row.get("expert_choice", ""),
                    outcome=row.get("outcome", ""),
                    notes=row.get("notes", ""),
                ))
        return rows

    def load_json(self, path: str) -> list[LabeledDecision]:
        """JSON: list of {case_id,domain,profile,factor_values,expert_choice}"""
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        return [
            LabeledDecision(
                case_id=item.get("case_id", ""),
                domain=item.get("domain") or self.domain or "",
                profile=item.get("profile", ""),
                factor_values=dict(item.get("factor_values", {})),
                expert_choice=item.get("expert_choice", ""),
                outcome=item.get("outcome", ""),
                notes=item.get("notes", ""),
            )
            for item in payload
        ]

    def validate(
        self,
        cases: list[LabeledDecision],
        domain: str = None,
    ) -> DomainValidationResult:
        """
        For each case: call DomainDecisionModel.evaluate() and compare
        top recommendation to expert_choice.
        accuracy    = fraction where activations[0].plank.name == expert_choice
        top3        = fraction where expert_choice in top 3 activations
        fulcrum_gap = mean(|model_fulcrum - expert_plank_position|)
        """
        resolved_domain = domain or self.domain
        if resolved_domain is None:
            if not cases:
                raise ValueError("domain is required when validating empty cases")
            resolved_domain = cases[0].domain
        config = ALL_DOMAINS.get(resolved_domain)
        if config is None:
            raise KeyError(f"Unknown domain: {resolved_domain}")
        model = DomainDecisionModel(config)

        relevant = [case for case in cases if case.domain == resolved_domain]
        confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        profile_hits: dict[str, int] = defaultdict(int)
        profile_totals: dict[str, int] = defaultdict(int)
        fulcrum_gaps: list[float] = []
        correct = 0
        top3 = 0

        plank_positions = {plank.name: plank.position for plank in config.planks}

        for case in relevant:
            diagnosis = model.evaluate(case.profile, case.factor_values)
            recommended = diagnosis.primary_plank.name
            top_choices = [activation.plank.name for activation in diagnosis.top(3)]
            expert_position = plank_positions.get(case.expert_choice, diagnosis.fulcrum_position)

            profile_totals[case.profile] += 1
            confusion[recommended][case.expert_choice] += 1
            fulcrum_gaps.append(abs(diagnosis.fulcrum_position - expert_position))

            if recommended == case.expert_choice:
                correct += 1
                profile_hits[case.profile] += 1
            if case.expert_choice in top_choices:
                top3 += 1

        total = len(relevant)
        self._last_profile_counts = dict(profile_totals)
        by_profile = {
            profile: (profile_hits[profile] / count if count else 0.0)
            for profile, count in profile_totals.items()
        }
        return DomainValidationResult(
            domain=resolved_domain,
            n_cases=total,
            accuracy=(correct / total if total else 0.0),
            top3_accuracy=(top3 / total if total else 0.0),
            avg_fulcrum_gap=(sum(fulcrum_gaps) / total if total else 0.0),
            by_profile=by_profile,
            confusion={rec: dict(experts) for rec, experts in confusion.items()},
        )

    def compare_domains(
        self,
        cases_by_domain: dict[str, list[LabeledDecision]],
    ) -> list[DomainValidationResult]:
        """Run validate() for each domain and return sorted by accuracy."""
        results = [
            DomainValidator(domain).validate(cases, domain=domain)
            for domain, cases in cases_by_domain.items()
        ]
        return sorted(results, key=lambda result: result.accuracy, reverse=True)

    def generate_report(
        self,
        result: DomainValidationResult,
        output_path: str = None,
        fmt: str = "markdown",
    ) -> str:
        """
        Markdown report sections:
          ## Summary  (accuracy, n_cases, domain)
          ## By Profile  (table: profile | n | accuracy)
          ## Confusion  (model recommendation vs expert choice)
          ## Recommendations  (which factor weights to adjust)
        """
        if fmt == "json":
            report = json.dumps(result.__dict__, indent=2, default=str)
        elif fmt == "html":
            rows = "".join(
                f"<tr><td>{profile}</td><td>{self._last_profile_counts.get(profile, 0)}</td><td>{acc:.2%}</td></tr>"
                for profile, acc in sorted(result.by_profile.items())
            )
            report = (
                f"<h2>Summary</h2><p>Domain: {result.domain} · Cases: {result.n_cases} · "
                f"Accuracy: {result.accuracy:.2%} · Top-3: {result.top3_accuracy:.2%}</p>"
                f"<h2>By Profile</h2><table><tr><th>Profile</th><th>N</th><th>Accuracy</th></tr>{rows}</table>"
            )
        else:
            lines = [
                "## Summary",
                f"- Domain: {result.domain}",
                f"- Cases: {result.n_cases}",
                f"- Accuracy: {result.accuracy:.2%}",
                f"- Top-3 Accuracy: {result.top3_accuracy:.2%}",
                f"- Avg Fulcrum Gap: {result.avg_fulcrum_gap:.3f}",
                "",
                "## By Profile",
                "| Profile | N | Accuracy |",
                "|---|---:|---:|",
            ]
            for profile, accuracy in sorted(result.by_profile.items()):
                lines.append(
                    f"| {profile} | {self._last_profile_counts.get(profile, 0)} | {accuracy:.2%} |"
                )

            lines.extend(["", "## Confusion"])
            if result.confusion:
                for model_choice, expert_counts in sorted(result.confusion.items()):
                    counts = ", ".join(
                        f"{expert}: {count}" for expert, count in sorted(expert_counts.items())
                    )
                    lines.append(f"- {model_choice} → {counts}")
            else:
                lines.append("- No cases.")

            lines.extend(["", "## Recommendations"])
            if result.accuracy >= 0.75:
                lines.append("- Current weights appear broadly aligned with expert labels.")
            elif result.top3_accuracy > result.accuracy:
                lines.append("- Review factor weights around the most-confused top recommendations.")
                lines.append("- Consider tightening bandwidth or increasing discriminative factor weights.")
            else:
                lines.append("- Revisit profile baselines and high-impact factor weights.")
            report = "\n".join(lines)

        if output_path:
            Path(output_path).expanduser().write_text(report, encoding="utf-8")
        return report
