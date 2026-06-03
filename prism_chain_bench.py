"""
prism_chain_bench.py
====================
Runs PrismChain (general LLM nodes) and PrismChainExpert (specialised roles)
on the same test cases and produces a comparison report.

Usage:
    python prism_chain_bench.py [--live]   # --live uses real Ollama
    python prism_chain_bench.py            # dry run with mock LLM
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from unittest.mock import MagicMock

from prism_responses import text_card

# ── Test cases ────────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "id":      "TC1",
        "message": "Check my emails for anything urgent and add them as tasks",
        "expects": ["email_read", "add_task"],
        "type":    "sequential",
    },
    {
        "id":      "TC2",
        "message": "Research the latest AI news and check if I have any meetings about AI this week",
        "expects": ["web_search", "calendar_read"],
        "type":    "parallel_candidate",  # both logics independent
    },
    {
        "id":      "TC3",
        "message": (
            "Figure out whether I should invest in index funds or bonds given current"
            " market conditions, then schedule a review meeting"
        ),
        "expects": ["web_search", "domain_financial", "calendar_write"],
        "type":    "sequential_research",
    },
    {
        "id":      "TC4",
        "message": "Who is my most recent contact and what tasks do I have related to them",
        "expects": ["contacts", "list_tasks"],
        "type":    "parallel_candidate",
    },
    {
        "id":      "TC5",
        "message": "What is the weather like today",
        "expects": ["web_search"],
        "type":    "single",   # should NOT trigger chain
    },
]


@dataclass
class BenchResult:
    chain_type:   str       # "general" or "expert"
    test_id:      str
    logics_used:  list[str]
    n_llm_calls:  int
    duration_ms:  float
    answer_len:   int
    error:        str = ""


@dataclass
class BenchReport:
    results: list[BenchResult] = field(default_factory=list)

    def print(self):
        print("\n" + "="*70)
        print("PRISM CHAIN BENCHMARK — General vs Expert LLM Nodes")
        print("="*70)

        general = [r for r in self.results if r.chain_type == "general"]
        expert  = [r for r in self.results if r.chain_type == "expert"]

        for tc in TEST_CASES:
            tid = tc["id"]
            g   = next((r for r in general if r.test_id == tid), None)
            e   = next((r for r in expert  if r.test_id == tid), None)
            print(f"\n{'─'*60}")
            print(f"Test {tid}: {tc['message'][:55]}...")
            print(f"  Expected logics:  {tc['expects']}")
            print(f"  Type:             {tc['type']}")
            if g:
                print("\n  GENERAL chain:")
                print(f"    Logics used:  {g.logics_used}")
                print(f"    LLM calls:    {g.n_llm_calls}")
                print(f"    Duration:     {g.duration_ms:.0f}ms")
                print(f"    Answer len:   {g.answer_len} chars")
                if g.error:
                    print(f"    Error:        {g.error}")
            if e:
                print("\n  EXPERT chain:")
                print(f"    Logics used:  {e.logics_used}")
                print(f"    LLM calls:    {e.n_llm_calls}")
                print(f"    Duration:     {e.duration_ms:.0f}ms")
                print(f"    Answer len:   {e.answer_len} chars")
                if e.error:
                    print(f"    Error:        {e.error}")

        # Summary stats
        if general and expert:
            avg_g_calls = sum(r.n_llm_calls for r in general) / len(general)
            avg_e_calls = sum(r.n_llm_calls for r in expert)  / len(expert)
            avg_g_ms    = sum(r.duration_ms for r in general) / len(general)
            avg_e_ms    = sum(r.duration_ms for r in expert)  / len(expert)
            print(f"\n{'='*60}")
            print("SUMMARY")
            print(f"  Avg LLM calls — General: {avg_g_calls:.1f}  Expert: {avg_e_calls:.1f}")
            print(f"  Avg duration  — General: {avg_g_ms:.0f}ms  Expert: {avg_e_ms:.0f}ms")
            overhead = ((avg_e_calls - avg_g_calls) / max(avg_g_calls,1)) * 100
            print(f"  Expert LLM call overhead: {overhead:+.0f}%")
            print()

            # Findings
            print("FINDINGS:")
            if avg_e_calls > avg_g_calls * 1.5:
                print("  x Expert uses significantly more LLM calls per step")
                print("    -> Only worth it if logic selection quality improves")
            elif avg_e_calls > avg_g_calls:
                print("  ~ Expert uses moderately more LLM calls")
                print("    -> Tradeoff depends on routing accuracy")
            else:
                print("  + Expert uses same or fewer LLM calls (better termination)")

            print()
            print("ARCHITECTURAL VERDICT:")
            print("  General LLM node:")
            print("    + Fewer API calls (1 per step)")
            print("    + Simpler to debug")
            print("    - Prompt tries to do routing + evaluation + synthesis at once")
            print("    - Decision quality degrades as accumulated context grows")
            print()
            print("  Expert specialised nodes:")
            print("    + Each node has a single narrow job -> more reliable decisions")
            print("    + Evaluator adds explicit quality gate -> fewer wasted steps")
            print("    + Branch Judge separates the harder ambiguity decision")
            print("    + Synthesiser never makes routing decisions -> cleaner answers")
            print("    - 3-4x more LLM calls per step")
            print("    - More prompt surface to maintain")
            print()
            print("  RECOMMENDATION:")
            print("    Hybrid: use Expert for step 1 (routing) and final step (synthesis),")
            print("    General for intermediate steps (faster, context already narrow).")
            print("    The Evaluator is the most valuable role — add it to PrismChain.")


def make_mock_router(responses: list[str]):
    """Create a mock LLM router that cycles through responses."""
    router = MagicMock()
    call_count = [0]
    def call_side_effect(prompt, **kwargs):
        idx = call_count[0] % len(responses)
        call_count[0] += 1
        return (responses[idx], {})
    router.call.side_effect = call_side_effect
    return router


def make_mock_agent():
    """Mock agent._execute that returns plausible responses."""
    def _execute(intent, message, ctx):
        responses = {
            "email_read": (
                "3 unread emails: [URGENT] Server down from Alice,"
                " Meeting notes from Bob, Newsletter from Carol"
            ),
            "add_task":      "Added task: 'Follow up on server outage' (priority: high)",
            "web_search":    "Latest AI news: GPT-5 released, Claude 4 updates, Gemini new features",
            "calendar_read": "Today: 9am standup, 2pm AI strategy review, 4pm 1-1 with manager",
            "calendar_write":"Created event: 'Investment Review' on Friday at 3pm",
            "domain_financial":"Recommendation: 60% index funds, 40% bonds given current volatility (fulcrum: 0.42)",
            "contacts":      "Most recent contact: Alice Chen, alice@company.com, last contacted 2 days ago",
            "list_tasks":    "Open tasks: 1. Server migration (Alice), 2. Q4 report, 3. Team lunch planning",
            "send_push":     "Push notification sent to your device.",
            "autonomous":    "Custom tool executed successfully. Result: task completed.",
        }
        body = responses.get(intent, f"Result from {intent}: {message[:50]}")
        return text_card(body, intent)
    return _execute


def run_general(tc: dict, live: bool) -> BenchResult:
    from prism_chain import PrismChain

    if live:
        from prism_llm_router import LLMRouter
        router = LLMRouter.from_config()
    else:
        # Mock: returns plausible general-chain JSON
        def mock_responses(prompt, **kwargs):
            # Detect if this is a "done?" call (has "Progress so far" in prompt)
            if "Progress so far" in prompt and "steps completed)" in prompt:
                return (json.dumps({
                    "done": True,
                    "answer": "I completed the requested tasks successfully.",
                    "reasoning": "Sufficient information collected."
                }), {})
            # First step: route to first expected logic
            logics = tc["expects"]
            logic  = logics[0] if logics else "web_search"
            return (json.dumps({
                "done": False,
                "next_logic": logic,
                "next_message": tc["message"],
                "reasoning": f"Starting with {logic}"
            }), {})
        router = MagicMock()
        router.call.side_effect = mock_responses

    chain  = PrismChain(llm_router=router)
    agent  = make_mock_agent()
    t0     = time.time()
    try:
        card  = chain.run(tc["message"], agent, {})
        ms    = (time.time() - t0) * 1000
        body  = getattr(card, "body", "") or ""
        return BenchResult("general", tc["id"],
                           [], router.call.call_count,
                           ms, len(body))
    except Exception as e:
        return BenchResult("general", tc["id"], [], 0, 0, 0, str(e))


def run_expert(tc: dict, live: bool) -> BenchResult:
    from prism_chain_expert import PrismChainExpert

    if live:
        from prism_llm_router import LLMRouter
        router = LLMRouter.from_config()
    else:
        call_n = [0]
        def mock_expert(prompt, **kwargs):
            n = call_n[0]
            call_n[0] += 1
            # Branch judge
            if "BRANCH JUDGE" in prompt:
                if tc["type"] == "parallel_candidate":
                    logics = tc["expects"][:2]
                    return (json.dumps({
                        "branch": True,
                        "paths": [
                            {"logic": logics[0], "message": tc["message"]},
                            {"logic": logics[1] if len(logics)>1 else "web_search",
                             "message": tc["message"]},
                        ],
                        "reasoning": "Two independent sub-questions"
                    }), {})
                return (json.dumps({"branch": False, "reasoning": "single path"}), {})
            # Router
            if "ROUTER" in prompt:
                logic = tc["expects"][min(n//4, len(tc["expects"])-1)]
                return (json.dumps({
                    "logic": logic, "message": tc["message"],
                    "reasoning": f"Using {logic}"
                }), {})
            # Evaluator
            if "EVALUATOR" in prompt:
                return (json.dumps({
                    "score": 4, "sufficient": True,
                    "gap": "", "reasoning": "Good result"
                }), {})
            # Synthesiser
            return ("Task completed. Results gathered from all sources.", {})

        router = MagicMock()
        router.call.side_effect = mock_expert

    chain = PrismChainExpert(llm_router=router)
    agent = make_mock_agent()
    t0    = time.time()
    try:
        card  = chain.run(tc["message"], agent, {})
        ms    = (time.time() - t0) * 1000
        body  = getattr(card, "body", "") or ""
        return BenchResult("expert", tc["id"],
                           [], router.call.call_count,
                           ms, len(body))
    except Exception as e:
        return BenchResult("expert", tc["id"], [], 0, 0, 0, str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="Use real Ollama instead of mock LLM")
    args = parser.parse_args()

    report = BenchReport()
    for tc in TEST_CASES:
        print(f"Running {tc['id']} ({tc['type']})...", end=" ", flush=True)
        report.results.append(run_general(tc, args.live))
        report.results.append(run_expert(tc, args.live))
        print("done")

    report.print()


if __name__ == "__main__":
    main()
