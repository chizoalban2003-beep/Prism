from __future__ import annotations

from dataclasses import dataclass

from decision_spectrum import DecisionBeam, DecisionPlank, Factor, OutcomeDiagnosis


@dataclass
class DomainPlank:
    name: str
    position: float
    payoff: float
    cost: float
    risk: float
    probability: float = 0.75


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
    bandwidth: float = 0.17


class DomainDecisionModel:
    _PROFILE_ALIASES = {
        "Medical": {
            "young adult": "Young_adult",
            "middle-aged": "Middle_aged",
            "middle aged": "Middle_aged",
            "elderly (65+)": "Elderly",
        },
    }

    def __init__(self, config: DomainConfig):
        self.config = config

    def _resolve_profile(self, profile_name: str | None) -> DomainProfile:
        if not profile_name:
            return self.config.profiles[0]
        aliases = self._PROFILE_ALIASES.get(self.config.domain, {})
        wanted = aliases.get(profile_name.lower(), profile_name)
        for profile in self.config.profiles:
            if profile.name == wanted:
                return profile
        for profile in self.config.profiles:
            if self._normalise(profile.name) == self._normalise(wanted):
                return profile
        return self.config.profiles[0]

    @staticmethod
    def _normalise(value: str) -> str:
        return ''.join(ch for ch in value.lower() if ch.isalnum())

    def make_beam(self, profile_name: str, factor_values: dict | None = None) -> DecisionBeam:
        profile = self._resolve_profile(profile_name)
        beam = DecisionBeam(f"{self.config.domain}_{profile.name}", bandwidth=self.config.bandwidth)
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
            target = (
                min(1.0, profile.fixed_fulcrum + value * factor.range)
                if factor.direction > 0
                else max(0.0, profile.fixed_fulcrum - value * factor.range)
            )
            beam.fulcrum.add_factor(Factor(factor.id, value, factor.weight, target, factor.description or factor.label))
        beam.fulcrum.add_factor(Factor("_base", 1.0, 2.0, profile.fixed_fulcrum))
        return beam

    def evaluate(self, profile_name: str, factor_values: dict | None = None) -> OutcomeDiagnosis:
        return self.make_beam(profile_name, factor_values).evaluate()

    def sensitivity_sweep(
        self,
        profile_name: str,
        factor_id: str,
        steps: int = 5,
        factor_values: dict | None = None,
    ) -> list[OutcomeDiagnosis]:
        return self.make_beam(profile_name, factor_values).sensitivity_sweep(factor_id, steps=steps)

    def cross_profile_compare(self, factor_values: dict | None = None) -> list[dict]:
        results = []
        for profile in self.config.profiles:
            diagnosis = self.evaluate(profile.name, factor_values)
            results.append(
                {
                    "profile": profile.name,
                    "recommended": diagnosis.primary_plank.name,
                    "fulcrum": round(diagnosis.fulcrum_position, 3),
                    "confidence": round(diagnosis.activations[0].activation, 3),
                }
            )
        return results


MEDICAL = DomainConfig(
    name="Medical",
    domain="Medical",
    bandwidth=0.16,
    planks=[
        DomainPlank("Monitor_home", 0.00, 10, 2, 0.96),
        DomainPlank("GP", 0.14, 30, 3, 0.90),
        DomainPlank("Urgent_GP", 0.28, 55, 4, 0.84),
        DomainPlank("Walk_in", 0.43, 70, 5, 0.78),
        DomainPlank("AE_4hr", 0.57, 90, 6, 0.72),
        DomainPlank("Emergency_AE", 0.72, 120, 8, 0.65),
        DomainPlank("999", 0.86, 180, 10, 0.90),
        DomainPlank("Critical", 1.00, 200, 12, 0.80),
    ],
    factors=[
        DomainFactor("severity", "severity", 3.5, +1, 0.85),
        DomainFactor("vital_signs", "vital_signs", 3.0, +1, 0.80),
        DomainFactor("deteriorating", "deteriorating", 2.5, +1, 0.75),
        DomainFactor("duration", "duration", 2.0, +1, 0.50),
        DomainFactor("age_risk", "age_risk", 1.8, +1, 0.40),
        DomainFactor("comorbidities", "comorbidities", 1.5, +1, 0.35),
        DomainFactor("response", "response", 1.2, -1, 0.30),
    ],
    profiles=[
        DomainProfile("Child", 0.55),
        DomainProfile("Young_adult", 0.35),
        DomainProfile("Middle_aged", 0.45),
        DomainProfile("Elderly", 0.55),
        DomainProfile("Chronic", 0.52),
        DomainProfile("Immunocompromised", 0.60),
    ],
)

FINANCIAL = DomainConfig(
    name="Financial",
    domain="Financial",
    bandwidth=0.17,
    planks=[
        DomainPlank("All_cash", 0.00, 20, 2, 0.99),
        DomainPlank("Gov_bonds", 0.14, 45, 3, 0.94),
        DomainPlank("Consv_balanced", 0.28, 70, 4, 0.87),
        DomainPlank("Mod_balanced", 0.43, 95, 5, 0.80),
        DomainPlank("Growth", 0.57, 125, 6, 0.72),
        DomainPlank("Equity_focus", 0.71, 155, 8, 0.64),
        DomainPlank("High_growth", 0.85, 185, 10, 0.55),
        DomainPlank("Concentrated", 1.00, 210, 12, 0.45),
    ],
    factors=[
        DomainFactor("time_horizon", "time_horizon", 3.0, +1, 0.75),
        DomainFactor("risk_tolerance", "risk_tolerance", 2.5, +1, 0.65),
        DomainFactor("income_stability", "income_stability", 2.0, +1, 0.50),
        DomainFactor("market_conditions", "market_conditions", 1.8, +1, 0.45),
        DomainFactor("liquidity_need", "liquidity_need", 2.0, -1, 0.55),
        DomainFactor("existing_wealth", "existing_wealth", 1.2, +1, 0.30),
        DomainFactor("years_to_goal", "years_to_goal", 2.0, +1, 0.60),
    ],
    profiles=[
        DomainProfile("Retiree", 0.18),
        DomainProfile("Pre_retirement", 0.38),
        DomainProfile("Mid_career", 0.55),
        DomainProfile("Young_pro", 0.68),
        DomainProfile("HNW", 0.58),
        DomainProfile("Institution", 0.50),
    ],
)

LEGAL = DomainConfig(
    name="Legal",
    domain="Legal",
    bandwidth=0.18,
    planks=[
        DomainPlank("Settle_now", 0.00, 40, 3, 0.95),
        DomainPlank("Negotiate", 0.17, 80, 5, 0.84),
        DomainPlank("Mediation", 0.33, 110, 7, 0.76),
        DomainPlank("Arbitration", 0.50, 140, 9, 0.66),
        DomainPlank("Part_litigate", 0.64, 175, 12, 0.55),
        DomainPlank("Full_litigate", 0.80, 220, 16, 0.44),
        DomainPlank("Appeal", 1.00, 300, 22, 0.28),
    ],
    factors=[
        DomainFactor("evidence_strength", "evidence_strength", 3.5, +1, 0.80),
        DomainFactor("precedent", "precedent", 2.5, +1, 0.65),
        DomainFactor("client_resources", "client_resources", 2.0, +1, 0.55),
        DomainFactor("opposing_strength", "opposing_strength", 1.8, -1, 0.45),
        DomainFactor("jurisdiction", "jurisdiction", 1.5, +1, 0.40),
        DomainFactor("time_pressure", "time_pressure", 1.2, -1, 0.35),
        DomainFactor("reputational", "reputational", 1.0, +1, 0.25),
    ],
    profiles=[
        DomainProfile("Individual", 0.35),
        DomainProfile("SME", 0.30),
        DomainProfile("Corp_claimant", 0.65),
        DomainProfile("Corp_defendant", 0.45),
        DomainProfile("Public_body", 0.55),
        DomainProfile("Class_action", 0.72),
    ],
)

HR = DomainConfig(
    name="HR",
    domain="HR",
    bandwidth=0.19,
    planks=[
        DomainPlank("Redeploy", 0.00, 30, 2, 0.88),
        DomainPlank("Internal_transfer", 0.14, 50, 3, 0.84),
        DomainPlank("Promote", 0.28, 70, 4, 0.76),
        DomainPlank("External_junior", 0.43, 90, 6, 0.68),
        DomainPlank("External_lateral", 0.57, 120, 8, 0.60),
        DomainPlank("Senior_hire", 0.71, 155, 11, 0.52),
        DomainPlank("Exec_search", 0.85, 200, 16, 0.42),
        DomainPlank("Acquihire", 1.00, 280, 22, 0.30),
    ],
    factors=[
        DomainFactor("skill_gap", "skill_gap", 3.0, +1, 0.75),
        DomainFactor("urgency", "urgency", 2.5, +1, 0.65),
        DomainFactor("internal_talent", "internal_talent", 3.0, -1, 0.70),
        DomainFactor("budget", "budget", 2.0, +1, 0.55),
        DomainFactor("market_supply", "market_supply", 1.8, -1, 0.45),
        DomainFactor("culture_risk", "culture_risk", 1.5, -1, 0.35),
    ],
    profiles=[
        DomainProfile("Startup", 0.58),
        DomainProfile("Scaleup", 0.52),
        DomainProfile("Enterprise", 0.40),
        DomainProfile("Public_sector", 0.32),
        DomainProfile("Prof_services", 0.55),
    ],
)

SUPPLY_CHAIN = DomainConfig(
    name="Supply Chain",
    domain="Supply Chain",
    bandwidth=0.18,
    planks=[
        DomainPlank("Hold_inventory", 0.00, 20, 2, 0.94),
        DomainPlank("Standard_reorder", 0.14, 50, 4, 0.88),
        DomainPlank("Accelerate", 0.28, 75, 6, 0.82),
        DomainPlank("Dual_source", 0.43, 100, 8, 0.74),
        DomainPlank("Safety_stock", 0.57, 130, 10, 0.68),
        DomainPlank("Spot_market", 0.71, 165, 14, 0.58),
        DomainPlank("Emergency", 0.85, 200, 18, 0.48),
        DomainPlank("Redesign", 1.00, 280, 25, 0.38),
    ],
    factors=[
        DomainFactor("demand_forecast", "demand_forecast", 2.5, +1, 0.65),
        DomainFactor("lead_time_risk", "lead_time_risk", 3.0, +1, 0.75),
        DomainFactor("stock_days", "stock_days", 3.5, -1, 0.80),
        DomainFactor("supplier_risk", "supplier_risk", 2.0, +1, 0.55),
        DomainFactor("cost_premium", "cost_premium", 1.5, -1, 0.40),
        DomainFactor("geopolitical", "geopolitical", 1.8, +1, 0.50),
    ],
    profiles=[
        DomainProfile("JIT_manufacturer", 0.38),
        DomainProfile("Retailer", 0.48),
        DomainProfile("Healthcare", 0.62),
        DomainProfile("Defence", 0.70),
        DomainProfile("Ecommerce", 0.55),
    ],
)

CLIMATE = DomainConfig(
    name="Climate",
    domain="Climate",
    bandwidth=0.20,
    planks=[
        DomainPlank("Voluntary", 0.00, 20, 3, 0.88),
        DomainPlank("Incremental", 0.14, 45, 5, 0.82),
        DomainPlank("Carbon_light", 0.28, 80, 8, 0.72),
        DomainPlank("Renewable_trans", 0.43, 120, 10, 0.64),
        DomainPlank("Accelerated", 0.57, 160, 14, 0.54),
        DomainPlank("Binding_targets", 0.71, 200, 18, 0.44),
        DomainPlank("Emergency", 0.85, 240, 22, 0.36),
        DomainPlank("Radical", 1.00, 280, 28, 0.28),
    ],
    factors=[
        DomainFactor("climate_urgency", "climate_urgency", 3.5, +1, 0.80),
        DomainFactor("economic_capacity", "economic_capacity", 2.0, +1, 0.55),
        DomainFactor("political_capital", "political_capital", 3.0, +1, 0.70),
        DomainFactor("public_support", "public_support", 2.5, +1, 0.65),
        DomainFactor("trade_exposure", "trade_exposure", 1.8, -1, 0.45),
        DomainFactor("energy_security", "energy_security", 1.5, -1, 0.40),
    ],
    profiles=[
        DomainProfile("Small_island", 0.75),
        DomainProfile("G7_economy", 0.55),
        DomainProfile("Emerging", 0.38),
        DomainProfile("Fossil_dependent", 0.25),
        DomainProfile("Tech_leader", 0.65),
    ],
)

ALL_DOMAINS = {
    "Medical": MEDICAL,
    "Financial": FINANCIAL,
    "Legal": LEGAL,
    "HR": HR,
    "Supply Chain": SUPPLY_CHAIN,
    "Climate": CLIMATE,
}
