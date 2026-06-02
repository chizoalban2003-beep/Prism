from __future__ import annotations

from prism_llm_router import (
    PROVIDER_CATALOGUE,
    PROVIDER_COSTS,
    LLMOption,
    LLMRouter,
)


def test_provider_costs_has_expected_keys():
    """PROVIDER_COSTS must contain all major commercial providers plus ollama."""
    required_models = [
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-20250514",
        "gpt-4o-mini",
        "gpt-4o",
        "gemini-1.5-flash",
        "deepseek-chat",
        "mistral-small-latest",
        "llama-3.1-8b-instant",
        "ollama/any",
    ]
    for model in required_models:
        assert model in PROVIDER_COSTS, f"Missing cost entry for {model}"
        cost_in, cost_out = PROVIDER_COSTS[model]
        assert cost_in >= 0, f"Negative input cost for {model}"
        assert cost_out >= 0, f"Negative output cost for {model}"


def test_provider_catalogue_has_expected_providers():
    """PROVIDER_CATALOGUE must include all supported provider ids."""
    required_pids = {"anthropic", "openai", "google", "deepseek",
                     "mistral", "groq", "together", "ollama", "custom"}
    catalogue_pids = {entry[0] for entry in PROVIDER_CATALOGUE}
    assert required_pids == catalogue_pids


def test_llm_option_cost_fields():
    """LLMOption must expose cost tracking and budget helper methods."""
    opt = LLMOption(
        provider="openai",
        model="gpt-4o-mini",
        endpoint="https://api.openai.com",
        available=True,
        capability=3,
        cost_per_1k_in=0.00015,
        cost_per_1k_out=0.0006,
        monthly_budget=5.0,
        monthly_spent=1.0,
    )
    assert opt.estimated_cost(1000) > 0
    assert opt.budget_remaining() == 4.0
    assert opt.within_budget(1000) is True

    # Exhaust the budget
    opt.monthly_spent = 5.0
    assert opt.within_budget() is False


def test_llm_option_unlimited_budget():
    """monthly_budget == 0 means unlimited — within_budget always True."""
    opt = LLMOption(
        provider="ollama",
        model="mistral",
        endpoint="http://localhost:11434",
        available=True,
        capability=2,
        cost_per_1k_in=0.0,
        cost_per_1k_out=0.0,
        monthly_budget=0.0,    # unlimited
        monthly_spent=999.0,
    )
    assert opt.within_budget() is True
    assert opt.within_budget(100_000) is True
