"""
tests/test_agent_fallbacks_issue_28.py
======================================
Two KDE-optional fallbacks (#28-106):

1. replan() hard-required the KDE module — the chat UI renders a
   Re-plan button on every daily-plan card, but in a KDE-less deploy the
   endpoint answered "Re-plan unavailable". It now routes the composed
   refinement message through the normal chat path (universal_plan →
   LLM planner), same as a typed "plan my day".
2. The identity card read only from KDE — one live install had 11k
   persona observations while chat claimed "no profile yet". With KDE
   absent, the card now renders the top persona traits; the onboarding
   pointer remains only for a truly empty persona.
"""
from __future__ import annotations

from prism_agent import PrismAgent


class TestReplanWithoutKde:
    def test_replan_falls_back_to_chat_path(self, offline_llm):
        agent = PrismAgent()
        agent._kde = None
        card = agent.replan(
            instructions="move deep work to the morning",
            tasks=[{"time": "07:00", "title": "Mobility"}],
        )
        assert card is not None
        assert card.title != "Re-plan unavailable"


class TestIdentityCardPersonaFallback:
    def test_persona_traits_render_when_kde_absent(self, offline_llm):
        agent = PrismAgent()
        agent._kde = None
        persona = getattr(agent, "_persona", None)
        if persona is None:
            import pytest
            pytest.skip("persona subsystem not initialised in this build")
        persona.update_trait("technical_depth", "high", 0.8, delta=3)
        persona.update_trait("communication_style", "concise", 0.6, delta=2)
        card = agent.chat("identity profile")
        assert card.card_type.value == "identity"
        labels = [d["label"] for d in card.card_data["domains"]]
        assert any("technical depth" in label for label in labels)
        assert "observations" in card.card_data["insight"]

    def test_empty_persona_still_points_at_ceremony(self, offline_llm, tmp_path):
        agent = PrismAgent()
        agent._kde = None
        agent._persona = None
        card = agent.chat("identity profile")
        assert "onboarding ceremony" in (card.body or "")


class TestStatusCard:
    def test_status_reports_llm_tasks_subsystems(self, offline_llm):
        # Old reply was "Connected. KDE: offline. KSA: active." — useless
        # to a user. Must answer from cached state only (no discovery).
        agent = PrismAgent()
        card = agent.chat("status")
        body = card.body or ""
        assert "LLM:" in body
        assert "Subsystems:" in body
