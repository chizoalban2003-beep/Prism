from __future__ import annotations

import copy
from dataclasses import dataclass

from decision_spectrum import DecisionBeam, DecisionPlank, Factor, OutcomeDiagnosis, SpectrumFulcrum


@dataclass
class DomainPlank:
    name: str
    position: float
    payoff: float
    cost: float
    risk: float
    probability: float = 0.7


@dataclass
class DomainFactor:
    id: str
    label: str
    weight: float
    direction: float
    range: float
    description: str = ""


@dataclass
class DomainProfile:
    name: str
    fixed_fulcrum: float
    description: str = ""


@dataclass
class DomainConfig:
    name: str
    domain: str
    planks: list[DomainPlank]
    factors: list[DomainFactor]
    profiles: list[DomainProfile]
    bandwidth: float = 0.18
    version: str = "1.0"
    calibrated: bool = False


class DomainDecisionModel:
    def __init__(self, config: DomainConfig):
        if config is None:
            raise ValueError("DomainDecisionModel requires a DomainConfig")
        self.config = copy.deepcopy(config)

    def make_beam(
        self,
        profile_name: str,
        factor_values: dict[str, float] | None = None,
    ) -> DecisionBeam:
        profile = next(p for p in self.config.profiles if p.name == profile_name)
        beam = DecisionBeam(
            f"{self.config.domain}_{profile_name}",
            bandwidth=self.config.bandwidth,
            fulcrum=SpectrumFulcrum(),
        )
        for plank in self.config.planks:
            beam.add_plank(
                DecisionPlank(
                    plank.name,
                    plank.position,
                    plank.payoff,
                    plank.cost,
                    plank.risk,
                    plank.probability,
                )
            )

        values = factor_values or {}
        for factor in self.config.factors:
            value = max(0.0, min(1.0, float(values.get(factor.id, 0.5))))
            if factor.direction >= 0:
                target = min(1.0, profile.fixed_fulcrum + value * factor.range)
            else:
                target = max(0.0, profile.fixed_fulcrum - value * factor.range)
            beam.fulcrum.add_factor(
                Factor(
                    factor.id,
                    value,
                    factor.weight,
                    target,
                    factor.description or factor.label,
                )
            )

        beam.fulcrum.add_factor(
            Factor(
                "_base",
                1.0,
                2.0,
                profile.fixed_fulcrum,
                f"Profile: {profile.description or profile.name}",
            )
        )
        return beam

    def evaluate(
        self,
        profile_name: str,
        factor_values: dict[str, float] | None = None,
    ) -> OutcomeDiagnosis:
        return self.make_beam(profile_name, factor_values).evaluate()

    def sensitivity_sweep(
        self,
        profile_name: str,
        factor_id: str,
        steps: int = 5,
        factor_values: dict[str, float] | None = None,
    ) -> list[OutcomeDiagnosis]:
        beam = self.make_beam(profile_name, factor_values)
        return beam.sensitivity_sweep(factor_id, steps=steps)

    def cross_profile_compare(
        self,
        factor_values: dict[str, float] | None = None,
    ) -> list[dict]:
        results: list[dict] = []
        for profile in self.config.profiles:
            diagnosis = self.evaluate(profile.name, factor_values)
            results.append({
                "profile": profile.name,
                "fulcrum": diagnosis.fulcrum_position,
                "recommended": diagnosis.primary_plank.name,
                "confidence": diagnosis.activations[0].activation,
                "expected_net": diagnosis.expected_net,
            })
        return results


MEDICAL = DomainConfig(
    name="Medical",
    domain="Medical",
    bandwidth=0.16,
    planks=[
        DomainPlank("Self-care advice", 0.00, 15, 2, 5, 0.95),
        DomainPlank("Pharmacist consult", 0.14, 30, 3, 8, 0.92),
        DomainPlank("Primary care soon", 0.28, 55, 4, 12, 0.86),
        DomainPlank("Same-day GP", 0.43, 85, 5, 18, 0.80),
        DomainPlank("Urgent clinic", 0.57, 120, 6, 24, 0.74),
        DomainPlank("Emergency assessment", 0.71, 165, 8, 32, 0.66),
        DomainPlank("Emergency A&E now", 0.86, 240, 10, 40, 0.58),
        DomainPlank("ICU escalation", 1.00, 320, 12, 55, 0.48),
    ],
    factors=[
        DomainFactor("severity", "Severity", 3.8, +1, 0.70, "Clinical severity"),
        DomainFactor("vital_signs", "Vital signs instability", 3.0, +1, 0.60, "Vitals deterioration"),
        DomainFactor("deteriorating", "Deteriorating", 2.8, +1, 0.55, "Worsening trajectory"),
        DomainFactor("comorbidity", "Comorbidity burden", 1.6, +1, 0.25, "Underlying risk"),
        DomainFactor("support_at_home", "Support at home", 1.2, -1, 0.20, "Home support lowers urgency"),
        DomainFactor("diagnostic_uncertainty", "Diagnostic uncertainty", 1.4, +1, 0.20, "Uncertain picture"),
    ],
    profiles=[
        DomainProfile("Child", 0.42),
        DomainProfile("Young adult", 0.40),
        DomainProfile("Middle-aged", 0.48),
        DomainProfile("Elderly (65+)", 0.58),
        DomainProfile("Pregnant", 0.54),
        DomainProfile("Immunocompromised", 0.64),
    ],
)

FINANCIAL = DomainConfig(
    name="Financial",
    domain="Financial",
    bandwidth=0.18,
    planks=[
        DomainPlank("Hold cash", 0.00, 15, 2, 5, 0.97),
        DomainPlank("Short-term bonds", 0.16, 45, 3, 9, 0.92),
        DomainPlank("Balanced allocation", 0.32, 85, 5, 16, 0.86),
        DomainPlank("Core equity tilt", 0.48, 120, 6, 24, 0.80),
        DomainPlank("Growth overweight", 0.64, 165, 8, 35, 0.72),
        DomainPlank("Thematic concentration", 0.80, 210, 10, 48, 0.62),
        DomainPlank("Opportunistic high risk", 1.00, 270, 12, 62, 0.50),
    ],
    factors=[
        DomainFactor("risk_tolerance", "Risk tolerance", 3.2, +1, 0.65),
        DomainFactor("time_horizon", "Time horizon", 2.6, +1, 0.45),
        DomainFactor("liquidity_need", "Liquidity need", 2.5, -1, 0.50),
        DomainFactor("income_stability", "Income stability", 1.8, +1, 0.20),
        DomainFactor("market_stress", "Market stress", 1.9, -1, 0.35),
        DomainFactor("goal_urgency", "Goal urgency", 1.4, -1, 0.20),
    ],
    profiles=[
        DomainProfile("Capital preservation", 0.18),
        DomainProfile("Income seeker", 0.28),
        DomainProfile("Balanced investor", 0.45),
        DomainProfile("Growth investor", 0.62),
        DomainProfile("Aggressive trader", 0.78),
    ],
)

LEGAL = DomainConfig(
    name="Legal",
    domain="Legal",
    bandwidth=0.17,
    planks=[
        DomainPlank("Document and monitor", 0.00, 20, 2, 6, 0.95),
        DomainPlank("Internal review", 0.18, 55, 4, 12, 0.88),
        DomainPlank("Counsel consultation", 0.36, 90, 5, 20, 0.82),
        DomainPlank("Formal notice", 0.54, 135, 7, 28, 0.74),
        DomainPlank("Settlement posture", 0.72, 185, 9, 40, 0.66),
        DomainPlank("Immediate litigation", 1.00, 260, 12, 58, 0.52),
    ],
    factors=[
        DomainFactor("exposure", "Exposure", 3.3, +1, 0.65),
        DomainFactor("evidence_strength", "Evidence strength", 2.7, +1, 0.45),
        DomainFactor("time_pressure", "Time pressure", 1.9, +1, 0.30),
        DomainFactor("regulatory_risk", "Regulatory risk", 2.4, +1, 0.35),
        DomainFactor("settlement_leverage", "Settlement leverage", 1.8, -1, 0.20),
        DomainFactor("reputational_risk", "Reputational risk", 1.7, +1, 0.18),
    ],
    profiles=[
        DomainProfile("Employee-side", 0.44),
        DomainProfile("Employer-side", 0.40),
        DomainProfile("Commercial plaintiff", 0.56),
        DomainProfile("Commercial defendant", 0.46),
        DomainProfile("Regulatory response", 0.62),
    ],
)

HR = DomainConfig(
    name="HR",
    domain="HR",
    bandwidth=0.17,
    planks=[
        DomainPlank("Coach informally", 0.00, 18, 2, 5, 0.96),
        DomainPlank("Document concern", 0.20, 48, 3, 10, 0.90),
        DomainPlank("Performance plan", 0.40, 88, 5, 18, 0.82),
        DomainPlank("Formal warning", 0.60, 130, 6, 28, 0.74),
        DomainPlank("Escalate to HRBP", 0.80, 175, 8, 38, 0.66),
        DomainPlank("Immediate suspension", 1.00, 240, 10, 55, 0.54),
    ],
    factors=[
        DomainFactor("policy_breach", "Policy breach", 3.4, +1, 0.65),
        DomainFactor("performance_gap", "Performance gap", 2.6, +1, 0.45),
        DomainFactor("employee_history", "Employee history", 1.8, +1, 0.20),
        DomainFactor("manager_bias_risk", "Manager bias risk", 1.7, -1, 0.20),
        DomainFactor("business_impact", "Business impact", 2.0, +1, 0.30),
        DomainFactor("retention_value", "Retention value", 1.5, -1, 0.18),
    ],
    profiles=[
        DomainProfile("New hire", 0.28),
        DomainProfile("Individual contributor", 0.38),
        DomainProfile("People manager", 0.48),
        DomainProfile("Senior leader", 0.56),
        DomainProfile("High-potential employee", 0.34),
    ],
)

SUPPLY_CHAIN = DomainConfig(
    name="Supply Chain",
    domain="Supply Chain",
    bandwidth=0.19,
    planks=[
        DomainPlank("Monitor shipment", 0.00, 20, 2, 5, 0.96),
        DomainPlank("Re-sequence orders", 0.18, 58, 4, 12, 0.90),
        DomainPlank("Expedite primary supplier", 0.36, 98, 6, 20, 0.82),
        DomainPlank("Shift inventory", 0.54, 145, 8, 30, 0.74),
        DomainPlank("Dual-source activation", 0.72, 195, 10, 42, 0.64),
        DomainPlank("Emergency air freight", 1.00, 270, 14, 62, 0.48),
    ],
    factors=[
        DomainFactor("demand_spike", "Demand spike", 3.0, +1, 0.55),
        DomainFactor("supplier_risk", "Supplier risk", 3.1, +1, 0.60),
        DomainFactor("inventory_cover", "Inventory cover", 2.5, -1, 0.50),
        DomainFactor("transport_disruption", "Transport disruption", 2.1, +1, 0.32),
        DomainFactor("margin_sensitivity", "Margin sensitivity", 1.5, -1, 0.20),
        DomainFactor("customer_priority", "Customer priority", 1.8, +1, 0.22),
    ],
    profiles=[
        DomainProfile("Lean operations", 0.42),
        DomainProfile("Balanced network", 0.50),
        DomainProfile("Resilience-first", 0.62),
        DomainProfile("Premium service", 0.66),
    ],
)

CLIMATE = DomainConfig(
    name="Climate",
    domain="Climate",
    bandwidth=0.18,
    planks=[
        DomainPlank("Monitor and publish", 0.00, 18, 2, 5, 0.96),
        DomainPlank("Adaptation planning", 0.18, 52, 4, 10, 0.90),
        DomainPlank("Operational mitigation", 0.36, 92, 6, 18, 0.84),
        DomainPlank("Capital resilience project", 0.54, 140, 8, 28, 0.76),
        DomainPlank("Portfolio reallocation", 0.72, 190, 10, 40, 0.68),
        DomainPlank("Emergency climate response", 1.00, 260, 14, 58, 0.54),
    ],
    factors=[
        DomainFactor("hazard_severity", "Hazard severity", 3.4, +1, 0.65),
        DomainFactor("exposure", "Exposure", 2.8, +1, 0.45),
        DomainFactor("adaptive_capacity", "Adaptive capacity", 2.2, -1, 0.40),
        DomainFactor("timeline_urgency", "Timeline urgency", 1.9, +1, 0.28),
        DomainFactor("regulatory_pressure", "Regulatory pressure", 1.7, +1, 0.20),
        DomainFactor("community_sensitivity", "Community sensitivity", 1.6, +1, 0.18),
    ],
    profiles=[
        DomainProfile("Municipal planner", 0.40),
        DomainProfile("Corporate risk lead", 0.48),
        DomainProfile("Infrastructure operator", 0.58),
        DomainProfile("Emergency manager", 0.68),
    ],
)


ALL_DOMAINS: dict[str, DomainConfig] = {
    "Medical": MEDICAL,
    "Financial": FINANCIAL,
    "Legal": LEGAL,
    "HR": HR,
    "Supply Chain": SUPPLY_CHAIN,
    "Climate": CLIMATE,
}
