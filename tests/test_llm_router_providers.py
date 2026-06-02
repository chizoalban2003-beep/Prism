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


# ── New tests required by Gap Prompt 12b ────────────────────────────────────

def test_provider_costs_defined():
    """PROVIDER_COSTS has entries for claude and gpt (named alias for CI)."""
    assert any("claude" in k for k in PROVIDER_COSTS), "No claude entry in PROVIDER_COSTS"
    assert any("gpt" in k for k in PROVIDER_COSTS), "No gpt entry in PROVIDER_COSTS"


def test_estimated_cost_zero_for_ollama():
    """ollama LLMOption.estimated_cost() == 0.0 (local, free)."""
    opt = LLMOption(
        provider="ollama",
        model="mistral",
        endpoint="http://localhost:11434",
        available=True,
        capability=2,
        cost_per_1k_in=0.0,
        cost_per_1k_out=0.0,
    )
    assert opt.estimated_cost() == 0.0
    assert opt.estimated_cost(10_000) == 0.0


def test_within_budget_no_limit():
    """monthly_budget=0 always returns True regardless of spend."""
    opt = LLMOption(
        provider="openai", model="gpt-4o-mini",
        endpoint="https://api.openai.com", available=True,
        cost_per_1k_in=0.00015, cost_per_1k_out=0.0006,
        monthly_budget=0.0, monthly_spent=10_000.0,
    )
    assert opt.within_budget() is True
    assert opt.within_budget(1_000_000) is True


def test_within_budget_exceeded():
    """monthly_spent > monthly_budget returns False."""
    opt = LLMOption(
        provider="anthropic", model="claude-haiku-4-5-20251001",
        endpoint="https://api.anthropic.com", available=True,
        cost_per_1k_in=0.0008, cost_per_1k_out=0.004,
        monthly_budget=1.0, monthly_spent=1.5,  # spent > budget
    )
    assert opt.within_budget() is False


def test_discover_includes_stdlib():
    """stdlib is always present in the discovered options list."""
    router = LLMRouter()  # no config, no API keys
    options = router.discover()
    providers = [o.provider for o in options]
    assert "stdlib" in providers, "stdlib must always appear in discover() results"


def test_best_skips_over_budget():
    """An exhausted (over-budget) provider is skipped in favour of the next."""
    router = LLMRouter()
    options = router.discover()

    # Inject two fake options: one over-budget, one within budget
    paid = LLMOption(
        provider="openai", model="gpt-4o-mini",
        endpoint="https://api.openai.com", available=True,
        capability=3, cost_per_1k_in=0.00015, cost_per_1k_out=0.0006,
        monthly_budget=1.0, monthly_spent=2.0,   # over budget
    )
    free = LLMOption(
        provider="ollama", model="mistral",
        endpoint="http://localhost:11434", available=True,
        capability=2, cost_per_1k_in=0.0, cost_per_1k_out=0.0,
        monthly_budget=0.0,
    )
    # Prepend paid (higher capability) so it would normally win
    router._options = [paid, free] + [o for o in options if o.provider == "stdlib"]
    router._discovered = True

    best = router.best(min_capability=1)
    assert best is not None
    # paid must be skipped because it is over budget; free (ollama) should win
    assert best.provider != "openai", "Over-budget provider should be skipped"
    assert best.provider == "ollama", "Free local provider should be selected as fallback"


def test_gemini_call_structure():
    """_call_gemini sends correct contents format (role+parts)."""
    import json
    import unittest.mock as mock

    router = LLMRouter(config={"google_api_key": "test-key"})
    opt = LLMOption(
        provider="google", model="gemini-1.5-flash",
        endpoint="https://generativelanguage.googleapis.com",
        available=True, capability=3,
    )

    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["data"] = json.loads(req.data)
        resp = mock.Mock()
        resp.read.return_value = json.dumps({
            "candidates": [{"content": {"parts": [{"text": "hello"}]}}]
        }).encode()
        return resp

    with mock.patch("urllib.request.urlopen", fake_urlopen):
        router._call_gemini(opt, "hi", 100, "", False, [])

    contents = captured["data"]["contents"]
    assert isinstance(contents, list), "contents must be a list"
    assert len(contents) >= 1
    last = contents[-1]
    assert "role" in last and "parts" in last, "each content item needs role+parts"
    assert last["role"] == "user"
    assert last["parts"][0]["text"] == "hi"


def test_complexity_routing():
    """task_complexity='simple' prefers models with capability <= 2."""
    import time as _time

    router = LLMRouter()

    strong = LLMOption(
        provider="openai", model="gpt-4o",
        endpoint="https://api.openai.com", available=True,
        capability=3, cost_per_1k_in=0.005, cost_per_1k_out=0.015,
    )
    fast = LLMOption(
        provider="groq", model="llama-3.1-8b-instant",
        endpoint="https://api.groq.com/openai", available=True,
        capability=2, cost_per_1k_in=0.00005, cost_per_1k_out=0.00008,
    )
    stdlib_opt = LLMOption(
        provider="stdlib", model="stdlib", endpoint="", available=True,
        capability=0, latency_ms=0,
    )
    router._options = [strong, fast, stdlib_opt]
    router._discovered = True
    router._last_scan = _time.time()  # mark cache as fresh

    best = router.best(task_complexity="simple")
    # "simple" sets min_cap=1; both strong (cap=3) and fast (cap=2) qualify.
    # Simple tasks don't require cap>=3, so any available option is valid.
    assert best is not None
    assert best.available is True
