"""
tests/test_organ_tool_schemas_issue_28.py
=========================================
organ_tool_schemas() exports enabled organs as OpenAI-function-calling
tool definitions — the enabling primitive for the LLM→policy→organ tool
loop (docs/rfc-agentic-loop.md). Schemas are advisory; dispatch_organ's
gates stay authoritative.
"""
from __future__ import annotations

from prism_organ_loader import OrganLoader


def _loader():
    return OrganLoader()  # constructor loads bundled + user organs


class TestToolSchemaExport:
    def test_every_enabled_organ_exports_a_valid_schema(self):
        loader = _loader()
        tools = loader.organ_tool_schemas()
        assert len(tools) == len(
            [d for d in loader.list_organ_details() if d["enabled"]])
        for t in tools:
            assert t["type"] == "function"
            fn = t["function"]
            assert fn["name"] and fn["description"]
            assert fn["parameters"]["required"] == ["message"]
            assert "message" in fn["parameters"]["properties"]

    def test_policy_facts_surface_in_description(self):
        loader = _loader()
        by_name = {t["function"]["name"]: t for t in loader.organ_tool_schemas()}
        shell = by_name.get("shell_run")
        assert shell is not None
        desc = shell["function"]["description"]
        assert "risk: critical" in desc
        assert "requires user approval" in desc

    def test_max_risk_filters_the_tool_belt(self):
        loader = _loader()
        low = loader.organ_tool_schemas(max_risk="low")
        allt = loader.organ_tool_schemas()
        assert 0 < len(low) < len(allt)
        names = {t["function"]["name"] for t in low}
        assert "shell_run" not in names

    def test_disabled_organs_are_excluded(self):
        loader = _loader()
        some = loader.list_organs()[0]
        loader.disable(some)
        names = {t["function"]["name"] for t in loader.organ_tool_schemas()}
        assert some not in names
