# RFC: LLM → policy → [organ + organ-policy] as tools → LLM → policy → …

Status: research note (#28-108). The enabling primitive
(`OrganLoader.organ_tool_schemas()`) is implemented; the loop itself is not.

## The question

What if PRISM's architecture were just a policied tool loop —

```
message → LLM (organs presented as tools)
        → policy gate on the proposed call
        → organ executes under its own ORGAN_POLICY
        → result returns to the LLM
        → LLM → policy → organ → … until a final answer
```

— instead of the current routing pipeline?

## What the current pipeline actually is

Measured flow for one chat turn (see `prism_chat_tiers.py`,
`prism_routing.py`, `prism_organ_dispatch.py`):

```
message
 → tier dispatcher: orchestrator? expert chain? chain? composer?   (heuristic gates)
 → _route(): ~200-pattern regex table → LLM classifier fallback (1 call, 24 tokens)
 → _execute(intent) → handlers or dispatch_organ():
       L1 ConstitutionGuard (capability rules)
       L2 ORGAN_POLICY (approval gate, session ceiling, per-organ rate limit)
       L3 BudManager (capability-scoped ctx, token lifecycle)
 → one card
```

Two observations matter:

1. **The organs are already tools.** Uniform contract
   (`execute(intent, message, ctx)`), self-describing `ORGAN_META`,
   machine-readable `ORGAN_POLICY`, and a three-layer gate at the single
   dispatch chokepoint. This is a tool-calling architecture that predates
   its own tool-calling loop.
2. **The tiers above the regex table are hand-rolled agentic loops.**
   Orchestrator, expert chain, chain, composer — each approximates "let
   the model decompose and sequence steps" with bespoke code from the
   pre-function-calling era.

## Assessment of the pure loop

### What it fixes

- **The regex-gap bug class disappears.** Every routing bug shipped this
  cycle was the table being wrong: "unmatched" hit `predict_match` via
  the substring "match"; "remove task X" fell into `list_tasks` because
  no completion intent existed (#28-104); "git status" hit `\bstatus\b`
  (#28-96). A model reading 52 tool schemas doesn't make these mistakes.
- **Multi-step composition comes free.** "Check tomorrow's weather and
  if it's raining move my run indoors and remind me" is one loop with
  three tool calls; today it needs the composer to recognise it, or it
  degrades to one wrong intent.
- **Policy placement improves.** Today policy judges one organ call in
  isolation. In the loop, the gate sees the whole trajectory ("the model
  read an email, and now proposes shell_run — deny") — strictly more
  context than L1/L2 get now.
- **Four tiers collapse into one mechanism** — long-term, the
  orchestrator/chains/composer become prompt + loop, deleting a lot of
  bespoke code.

### What it costs (measured on this deployment)

- **Latency and money.** Today most daily-driver turns are ZERO LLM
  calls (reminder set: 0.2s, weather: 0.1s, tasks: 0.1s — measured).
  DeepSeek round-trips run 1.4–5s. A 3-hop loop with 52 tool schemas
  (~2–4k tokens) in context is 5–15s and ~5–10k tokens per message —
  for messages the regex table answers today for free. We just spent
  #28-101 deleting ONE redundant 1.1k-token call per message (45% of
  spend); the pure loop multiplies exactly that cost class.
- **Offline death.** PRISM's core promise is local-first. Reminders,
  notes, weather, tasks, status all work with no LLM at all. A pure
  loop is a brick without a provider — and this install spent two days
  in exactly that state after a key rotation.
- **Determinism.** ~4,000 tests pass because routing is a pure function.
  LLM routing makes every routing regression probabilistic or mock-heavy.
- **Prompt injection becomes the main threat model.** Organs return
  untrusted content (web, email). A model that picks the NEXT tool after
  reading untrusted content is the classic injection→exfiltration chain.
  Today's single-hop design structurally caps this; the loop must handle
  it with taint rules (see below), not vibes.

## Recommendation: hybrid — keep the fast path, make the fallback agentic

```
message
 → regex table (free, instant, offline, deterministic)   ← ~90% of turns
 → HIT  → handler/organ through the existing L1/L2/L3 gate (unchanged)
 → MISS → bounded tool loop (max 3 hops):
            LLM + organ_tool_schemas(max_risk=…)
            each proposed call → dispatch_organ() — SAME gate, unchanged
            requires_approval → approval card, loop state parked in
            _pending_approval, resumed on approve
            tool results → back to the LLM → final synthesis card
```

Where today's fallback single-shot-classifies into one intent label
(24 tokens of output, then hope), the loop reasons with the actual tool
belt — multi-step where needed — while everything the regex table
already handles stays instant, free, and offline-capable.

### Loop guardrails (non-negotiable)

1. **The gate is authoritative, the schema is advisory.** Proposed calls
   go through `dispatch_organ()` unchanged — constitution, approval,
   ceilings, rate limits, bud scoping all apply. The LLM never executes
   anything; it only proposes.
2. **Taint rule.** After any tool result from an untrusted source
   (email/web/document content), the remaining hops get a reduced belt:
   `organ_tool_schemas(max_risk="low")` and no outbound organs
   (email_send, phone_call, shell_run). This is mechanical, not judged
   by the model.
3. **Hop and budget caps.** Max 3 hops; each hop debits prism_budget;
   ceiling reached → degrade to the single-shot classifier.
4. **No LLM → old behaviour.** Loop unavailable offline → fall back to
   today's classifier-or-general_chat path. Offline capability is
   unchanged by construction.

### Migration steps

1. ✅ `OrganLoader.organ_tool_schemas()` — organs as OpenAI-format tool
   definitions, policy facts in the description, `max_risk` belt filter
   (#28-108, tested).
2. ✅ `LLMRouter.call_tools(messages, tools)` — structured tool-call
   round-trip for openai_compat / ollama / claude backends, provider
   budget ceilings applied per hop, calls ledgered as `tool_loop`
   (#28-109).
3. ✅ `prism_tool_loop.py` — the bounded loop, wired as the shadow
   rollout of step 4: PrismAgent runs it exactly where routing lands on
   `general_chat`. Policy is split deliberately: the **user** owns the
   `[tool_loop]` config (enabled / max_hops / max_risk / deny /
   allow_only); the **Prism's self-preservation** is mechanical and
   non-configurable — dispatch_organ's L1/L2/L3 gates unchanged, taint
   rule (untrusted content → low-risk belt, outbound organs denied),
   critical organs excluded from the default belt, offline → old path
   by construction. A `requires_approval` organ surfaces the approval
   card as the turn's outcome; the existing approve flow executes it on
   consent (full mid-loop resume is future work) (#28-109).
4. ✅ Shadow rollout — subsumed by 3 (loop runs only on the
   `general_chat` shrug; strictly additive).
5. ✅ Tier folding (#28-111). When a chain or composer trigger fires,
   the dispatcher now tries the tool loop FIRST with the larger
   ``max_hops_multistep`` budget (default 5) — decomposing and
   sequencing steps is the loop's native job. The legacy tier still
   runs whenever the loop declines (offline, disabled, empty belt), so
   LLM-less behaviour is bit-identical, and
   ``[tool_loop].fold_tiers = false`` restores the legacy order
   outright. The chain/composer code stays in the tree as that
   fallback; deleting it outright is a later decision once tool_loop
   ledger rows show the folded path handling the traffic.

   **Orchestrator evaluation — keep it.** ``should_orchestrate``
   claims conditional multi-domain task graphs with horizon pauses:
   nodes that wait DAYS on a trigger condition, persist across daemon
   restarts (TaskGraph + HorizonGoal storage), and resume on approval.
   A bounded synchronous loop cannot express "pause this branch until
   the physio replies next week" — that requires durable graph state,
   which the orchestrator owns. The loop and the orchestrator are not
   competitors: the loop covers minutes-scale multi-step requests, the
   orchestrator covers cross-session ones. Revisit only if the loop
   ever gains checkpoint/resume persistence. Expert chain (tier 0.5)
   keeps precedence for research-heavy requests for the same reason
   the orchestrator does: its evaluate-and-retry machinery is not yet
   expressible in the loop.

## Answer in one line

Pure `llm→policy→organ→llm→…` is the right *shape* for the fallback but
the wrong *default* for a local-first assistant on paid/slow inference —
keep the deterministic fast path, replace the single-shot classifier
with the policied loop, and let the tiers collapse into it gradually.
