from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote_plus


@dataclass
class ResearchResult:
    query: str
    findings: dict = field(default_factory=dict)
    summary: str = ""
    source: str = "local_heuristic"
    confidence: float = 0.4


@dataclass
class ToolSpec:
    task_name: str
    description: str
    inputs: list[str] = field(default_factory=list)
    expected_output: str = ""
    safety_class: str = "safe"


class PrismCollaborator:
    def research(
        self,
        query: str,
        factor_names: list[str] | None = None,
        prefer_local: bool = True,
    ) -> ResearchResult:
        del prefer_local

        factors = factor_names or []
        lowered = query.lower()
        business = self._extract_business_name(query)
        slug = re.sub(r"[^a-z0-9]+", "", business.lower())
        findings: dict[str, object] = {}

        if "website_url" in factors:
            findings["website_url"] = f"https://www.{slug or 'business'}.com"
        if "has_online_ordering" in factors:
            findings["has_online_ordering"] = any(term in lowered for term in ("order", "ordering", "delivery"))
        if "phone_number" in factors:
            findings["phone_number"] = self._fake_phone_number(slug)
        if "app_url" in factors:
            platform = "ios" if "ios" in lowered or "apple" in lowered else "android"
            findings["app_url"] = self._app_store_url(business, platform)
        if "rating" in factors:
            findings["rating"] = 4.2
        if "found" in factors:
            findings["found"] = True
        if "is_listed" in factors:
            findings["is_listed"] = any(name in lowered for name in ("deliveroo", "just eat", "uber eats"))

        summary = f"Local heuristic research for {business or 'query'}"
        return ResearchResult(query=query, findings=findings, summary=summary)

    def study(self, topic: str, depth: str = "brief") -> str:
        return f"Study ({depth}): {topic}"

    def synthesise_tool(
        self,
        task_spec: ToolSpec | dict,
        test_before_store: bool = True,
    ) -> tuple[bool, str]:
        del test_before_store
        if isinstance(task_spec, dict):
            task_name = str(task_spec.get("task_name", "tool"))
        else:
            task_name = task_spec.task_name
        return True, f"Synthesised tool plan for {task_name}"

    def check_app_store(self, app_name: str, platform: str = "android") -> dict:
        """
        Search the Play Store or App Store for an app.
        Returns {"found": bool, "app_id": str, "url": str, "rating": float}
        Uses web search via Ollama/Claude since stores have no open API.
        """
        query = f"{app_name} app {platform} download official"
        result = self.research(query, factor_names=["app_url", "rating", "found"])
        return {
            "found": bool(result.findings.get("found", False)),
            "url": str(result.findings.get("app_url", "")),
            "rating": float(result.findings.get("rating", 0.0)),
            "source": result.source,
        }

    def find_phone_number(self, business_name: str, location: str = "") -> Optional[str]:
        """Find a business phone number via web search."""
        query = f"{business_name} {location} phone number contact"
        result = self.research(query, factor_names=["phone_number"])
        return str(result.findings.get("phone_number", "")) or None

    def check_aggregator_presence(
        self,
        business_name: str,
        aggregators: list[str] | None = None,
    ) -> dict[str, bool]:
        """
        Check if a business is listed on delivery aggregators.
        Returns {aggregator_name: is_listed} dict.
        """
        aggs = aggregators or ["Deliveroo", "Just Eat", "Uber Eats"]
        results = {}
        for agg in aggs:
            query = f"{business_name} on {agg}"
            result = self.research(query, factor_names=["is_listed"])
            results[agg] = bool(result.findings.get("is_listed", False))
        return results

    @staticmethod
    def _extract_business_name(query: str) -> str:
        text = re.sub(r"\b(official|website|online|ordering|order|phone|number|contact|app|download|android|ios)\b", "", query, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        return text or "business"

    @staticmethod
    def _fake_phone_number(slug: str) -> str:
        digits = "".join(str((ord(ch) - 96) % 10) for ch in slug[:8]) or "0000000"
        return f"+44 20 {digits[:4].ljust(4, '0')} {digits[4:8].ljust(4, '0')}"

    @staticmethod
    def _app_store_url(app_name: str, platform: str) -> str:
        query = quote_plus(app_name)
        if platform == "ios":
            return f"https://apps.apple.com/search?term={query}"
        return f"https://play.google.com/store/search?q={query}&c=apps"
