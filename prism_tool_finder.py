from __future__ import annotations

import logging
from dataclasses import dataclass, field

from decision_spectrum import DecisionBeam, DecisionPlank, Factor

logger = logging.getLogger(__name__)


@dataclass
class ExecutionOption:
    """One possible way to execute a task when no direct tool exists."""

    name: str
    description: str
    execution_type: str
    url: str = ""
    phone: str = ""
    estimated_cost: float = 0.0
    time_to_execute: float = 0.0
    automation_level: str = "manual"
    available: bool = True
    source: str = ""


@dataclass
class ToolDiscoveryResult:
    task: str
    options: list[ExecutionOption]
    recommended: ExecutionOption
    confidence: float
    search_summary: str


EXECUTION_PATH_PLANKS = [
    ("Tell user — manual steps", 0.00, 20, 1, 2, 0.99),
    ("Show phone number — user calls", 0.12, 55, 2, 5, 0.97),
    ("Open their website — assisted", 0.26, 70, 5, 12, 0.82),
    ("Install their app — assisted", 0.38, 75, 10, 18, 0.76),
    ("Use delivery aggregator", 0.52, 85, 5, 10, 0.91),
    ("Social media message", 0.64, 55, 5, 28, 0.62),
    ("Suggest alternative provider", 0.76, 80, 4, 15, 0.88),
    ("Browser automation attempt", 0.88, 90, 18, 45, 0.52),
    ("Synthesise API integration", 1.00, 95, 25, 55, 0.45),
]


class ToolFinder:
    """
    Discovers how to execute a task when no direct integration exists.
    Uses web search + known aggregators + the decision engine to rank options.
    """

    AGGREGATORS = [
        "deliveroo.co.uk",
        "just-eat.co.uk",
        "ubereats.com",
        "thuisbezorgd.nl",
        "doordash.com",
        "grubhub.com",
    ]

    APP_STORES = {
        "android": "https://play.google.com/store/search?q={query}&c=apps",
        "ios": "https://apps.apple.com/search?term={query}",
    }

    _TYPE_POSITIONS = {
        "manual": 0.00,
        "phone": 0.12,
        "website": 0.26,
        "app_install": 0.38,
        "aggregator": 0.52,
        "social": 0.64,
        "alternative": 0.76,
        "api_synthesis": 1.00,
    }

    def __init__(self, collaborator=None, user_platform: str = "android"):
        self.collaborator = collaborator
        self.user_platform = user_platform
        self._cache: dict[str, ToolDiscoveryResult] = {}

    def find(
        self,
        task: str,
        provider_name: str,
        urgency: float = 0.50,
        cost_tolerance: float = 0.50,
        prefers_auto: float = 0.50,
        budget_left: float = 1.00,
    ) -> ToolDiscoveryResult:
        cache_key = f"{provider_name}:{task}:{urgency:.2f}:{cost_tolerance:.2f}:{prefers_auto:.2f}:{budget_left:.2f}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        options = self._discover_options(task, provider_name)
        recommended = self._rank_options(options, urgency, cost_tolerance, prefers_auto, budget_left)
        result = ToolDiscoveryResult(
            task=task,
            options=options,
            recommended=recommended,
            confidence=0.75 if len(options) > 2 else 0.50,
            search_summary=self._summarise(provider_name, options),
        )
        self._cache[cache_key] = result
        return result

    def _discover_options(self, task: str, provider_name: str) -> list[ExecutionOption]:
        del task
        options: list[ExecutionOption] = []
        name_slug = provider_name.lower().replace(" ", "+")

        options.append(
            ExecutionOption(
                name=f"Manual: tell user how to order from {provider_name}",
                description="PRISM explains the steps; user completes manually",
                execution_type="manual",
                automation_level="manual",
                available=True,
                source="builtin",
            )
        )

        store_url = self.APP_STORES.get(self.user_platform, "").format(query=name_slug)
        app_url = store_url
        if self.collaborator and hasattr(self.collaborator, "check_app_store"):
            try:
                app_result = self.collaborator.check_app_store(provider_name, self.user_platform)
                app_url = app_result.get("url") or store_url
            except Exception as exc:
                logger.debug("App store lookup failed for %s: %s", provider_name, exc)
        if store_url:
            options.append(
                ExecutionOption(
                    name=f"Install {provider_name} app",
                    description=f"Find and install from {self.user_platform.title()} store",
                    execution_type="app_install",
                    url=app_url,
                    time_to_execute=5.0,
                    automation_level="assisted",
                    available=True,
                    source="app_store",
                )
            )

        if self.collaborator:
            try:
                research = self.collaborator.research(
                    f"{provider_name} official website online ordering",
                    factor_names=["website_url", "has_online_ordering", "phone_number"],
                )
                website_url = str(research.findings.get("website_url", ""))
                if website_url:
                    options.append(
                        ExecutionOption(
                            name=f"Order via {provider_name} website",
                            description="Open their website in browser",
                            execution_type="website",
                            url=website_url,
                            time_to_execute=3.0,
                            automation_level="assisted",
                            available=True,
                            source="web_search",
                        )
                    )
                phone = None
                if hasattr(self.collaborator, "find_phone_number"):
                    phone = self.collaborator.find_phone_number(provider_name)
                phone = phone or str(research.findings.get("phone_number", "")) or ""
                if phone:
                    options.append(
                        ExecutionOption(
                            name=f"Call {provider_name} to order",
                            description="Phone order — PRISM dials, user orders",
                            execution_type="phone",
                            phone=phone,
                            time_to_execute=2.0,
                            automation_level="assisted",
                            available=True,
                            source="web_search",
                        )
                    )
            except Exception as exc:
                logger.debug("Research failed for %s: %s", provider_name, exc)

        aggregator_names = ["Deliveroo", "Just Eat", "Uber Eats"]
        presence: dict[str, bool] = {}
        if self.collaborator and hasattr(self.collaborator, "check_aggregator_presence"):
            try:
                presence = self.collaborator.check_aggregator_presence(provider_name, aggregator_names)
            except Exception as exc:
                logger.debug("Aggregator check failed for %s: %s", provider_name, exc)
        for aggregator in aggregator_names:
            options.append(
                ExecutionOption(
                    name=f"Order via {aggregator}",
                    description=f"Check if {provider_name} is on {aggregator}",
                    execution_type="aggregator",
                    estimated_cost=2.99,
                    time_to_execute=2.0,
                    automation_level="full",
                    available=presence.get(aggregator, True),
                    source="known_aggregator",
                )
            )

        options.append(
            ExecutionOption(
                name=f"Suggest alternative to {provider_name}",
                description="Find a similar provider PRISM already has tools for",
                execution_type="alternative",
                time_to_execute=1.0,
                automation_level="full",
                available=True,
                source="builtin",
            )
        )

        return options

    def _rank_options(
        self,
        options: list[ExecutionOption],
        urgency: float,
        cost_tolerance: float,
        prefers_auto: float,
        budget_left: float,
    ) -> ExecutionOption:
        if not options:
            return ExecutionOption("Manual", "No options found", "manual")

        beam = DecisionBeam("execution_path", bandwidth=0.20)
        for plank_name, pos, payoff, cost, risk, prob in EXECUTION_PATH_PLANKS:
            beam.add_plank(DecisionPlank(plank_name, pos, payoff, cost, risk, prob))

        beam.fulcrum.add_factor(Factor("urgency", urgency, 3.5, 0.30 + urgency * 0.55))
        beam.fulcrum.add_factor(Factor("cost_tolerance", cost_tolerance, 2.5, 0.25 + cost_tolerance * 0.50))
        beam.fulcrum.add_factor(Factor("prefers_auto", prefers_auto, 2.0, 0.20 + prefers_auto * 0.65))
        beam.fulcrum.add_factor(Factor("budget_left", budget_left, 3.0, max(0.05, budget_left * 0.60)))

        diagnosis = beam.evaluate()
        target_position = diagnosis.primary_plank.position

        def score(option: ExecutionOption) -> float:
            if not option.available:
                return float("-inf")
            position = self._TYPE_POSITIONS.get(option.execution_type, 0.0)
            closeness = 1.0 - abs(position - target_position)
            auto_bonus = {
                "manual": 0.0,
                "assisted": 0.12 * prefers_auto,
                "full": 0.20 * prefers_auto,
            }.get(option.automation_level, 0.0)
            urgency_bonus = max(0.0, (3.0 - option.time_to_execute)) * urgency * 0.20
            time_penalty = option.time_to_execute * max(0.0, 0.6 - urgency) * 0.04
            cost_penalty = option.estimated_cost * max(0.2, 1.0 - cost_tolerance) * (1.2 - min(budget_left, 1.0)) * 0.12
            free_bonus = 0.35 if budget_left <= 0.0 and option.estimated_cost <= 0.0 else 0.0
            manual_bonus = 0.25 if budget_left <= 0.0 and option.execution_type == "manual" else 0.0
            urgency_type_bonus = 0.0
            if urgency >= 0.8 and option.execution_type in {"phone", "aggregator"}:
                urgency_type_bonus = 0.30
            urgency_type_penalty = 0.0
            if urgency >= 0.8 and option.execution_type == "alternative":
                urgency_type_penalty = 0.45
            return (
                closeness
                + auto_bonus
                + urgency_bonus
                + free_bonus
                + manual_bonus
                + urgency_type_bonus
                - time_penalty
                - cost_penalty
                - urgency_type_penalty
            )

        return max(options, key=score)

    def _summarise(self, provider: str, options: list[ExecutionOption]) -> str:
        types = [option.execution_type for option in options]
        has_app = "app_install" in types
        has_web = "website" in types
        has_agg = "aggregator" in types
        parts = []
        if has_agg:
            parts.append("available on delivery apps")
        if has_web:
            parts.append("website ordering found")
        if has_app:
            parts.append("app in store")
        if not parts:
            parts.append("manual ordering only")
        return f"{provider}: {', '.join(parts)}. {len(options)} options found."
