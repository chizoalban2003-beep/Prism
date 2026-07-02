"""budget_status routing for issue #28 bug 57.

Live probes (post-#28-54 budget gate):

  user: "how much have I spent today"
   →   universal_plan (planner timed out after 30s)

  user: "spend so far today"
   →   universal_plan (planner timed out after 30s, curl hung)

  user: "what's my LLM budget"
   →   memory_recall ("No memory of that")

Only "show my budget" routed correctly to budget_status. Three of four
natural ways to ask the same question failed because:

1. ``universal_plan`` claims any message containing "today" (it's a
   daily-briefing keyword) — same trap that already required hoisting
   list_tasks above the planner. budget_status now joins list_tasks
   above universal_plan.

2. The existing budget_status regex didn't cover "spend so far today",
   "am I spending too much", or "remaining budget". Widened.

3. "what's my LLM budget" — memory_recall already excludes "budget" in
   its negative-lookahead, but only when "budget" follows "my\\b"
   immediately. The hoist makes that moot: budget_status fires before
   memory_recall is evaluated.
"""
from __future__ import annotations

from prism_intents import INTENTS
from prism_routing import route_intent


def _route(message: str) -> str:
    return route_intent(message, INTENTS, lambda _m: None)


class TestSpendTodayPhrases:
    """All four variants the user actually typed must reach budget_status."""

    def test_how_much_have_i_spent_today(self):
        assert _route("how much have I spent today") == "budget_status"

    def test_spend_so_far_today(self):
        assert _route("spend so far today") == "budget_status"

    def test_spending_so_far_today(self):
        assert _route("spending so far today") == "budget_status"

    def test_how_much_did_i_spend_today(self):
        assert _route("how much did I spend today") == "budget_status"


class TestLLMBudgetPhrases:

    def test_whats_my_llm_budget(self):
        assert _route("what's my LLM budget") == "budget_status"

    def test_show_my_budget(self):
        assert _route("show my budget") == "budget_status"

    def test_llm_spend(self):
        assert _route("llm spend") == "budget_status"

    def test_daily_cost(self):
        assert _route("daily cost") == "budget_status"


class TestRemainingBudget:

    def test_remaining_budget(self):
        assert _route("remaining budget") == "budget_status"

    def test_remaining_credit(self):
        assert _route("remaining credit") == "budget_status"


class TestNoUniversalPlanRegression:
    """Hoist must not steal genuine planner queries."""

    def test_plan_my_day(self):
        # universal_plan should still claim "plan" + "day" phrasing.
        assert _route("plan my day") == "universal_plan"

    def test_good_morning(self):
        # Changed in #28-79: a bare greeting is small talk, not a plan
        # request. "plan my day" / "good morning, plan my day" still plan.
        assert _route("good morning") == "general_chat"

    def test_what_should_i_do_today(self):
        # No budget/spend keywords; planner keeps the "today" claim.
        assert _route("what should I do today") == "universal_plan"


class TestNoMemoryRecallRegression:

    def test_what_about_my_partner(self):
        # memory_recall should still claim generic "what about my X".
        assert _route("what's my partner's name") == "memory_recall"


class TestNoBudgetViewListClash:
    """A bare 'list my budgets' should not be confused with list_tasks."""

    def test_list_tasks_unchanged(self):
        assert _route("list my tasks") == "list_tasks"
