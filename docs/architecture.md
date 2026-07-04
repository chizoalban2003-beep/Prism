# PRISM Architecture Reference

## Overview

PRISM is a local-first personal AI assistant that runs entirely on the user's machine with no cloud
dependencies at runtime. All inference is performed by a locally hosted Ollama instance (with a
Claude API fallback when explicitly configured), all state is stored in SQLite databases under
`~/.prism/`, and every subsystem — from the HTTP server to the organ bus to the background
reflection daemon — communicates over in-process Python calls or localhost sockets. No data leaves
the device unless the user explicitly triggers an integration (email send, calendar write, etc.).

---

## Module Namespaces

| Prefix | Role |
|--------|------|
| `prism_*` | Core agent runtime: memory, chain reasoning, planning, soul, organs, perception, policy, LLM routing, task queue, daemon. |
| `kde_*` | HTTP server + CLI interface: ASGI app (`prism_asgi.py`), config, profiles, and the `kde` / `prism` CLI entry points. |
| `ksa_*` | Knowledge / Skills Agent subsystem: skill registry, optimizer, executor, router, and the `ksa` CLI entry point. |

Supporting modules with no prefix (`artifact_store`, `digital_identity`, `identity_bus`,
`decision_spectrum`, `sport_*`, `moment_*`, `vision_analyzer`, …) provide domain-specific
capabilities used by the three primary namespaces above.

---

## Layer Diagram

```
┌─────────────────────────────────────────────────┐
│  Layer 1 — Interfaces                           │
│  kde_cli · prism_asgi · prism_chat · ksa_cli         │
├─────────────────────────────────────────────────┤
│  Layer 2 — Agent                                │
│  prism_agent · prism_chat · prism_daemon        │
├─────────────────────────────────────────────────┤
│  Layer 3 — Reasoning                            │
│  prism_chain · prism_chain_expert               │
│  prism_planner · prism_llm_router               │
├─────────────────────────────────────────────────┤
│  Layer 4 — Identity                             │
│  prism_soul · prism_policy · prism_instructions │
│  digital_identity · identity_bus · artifact_store│
├─────────────────────────────────────────────────┤
│  Layer 5 — Organs                               │
│  prism_organ_bus · prism_organ_loader           │
│  organs/* (currency_convert, weather_check, …)  │
├─────────────────────────────────────────────────┤
│  Layer 6 — Subsystems                           │
│  prism_memory · prism_perception                │
│  prism_proactive · prism_reflection             │
│  prism_outcome_tracker · prism_horizon          │
│  prism_task_queue · ksa_* · kde_*               │
├─────────────────────────────────────────────────┤
│  Layer 7 — Persistence                          │
│  SQLite ~/.prism/*.db  ·  prism_config.toml     │
└─────────────────────────────────────────────────┘
```

---

## Data Flows

### User message → response

```
User input (HTTP POST /chat  or  CLI stdin)
  │
  ▼
prism_asgi / kde_cli
  │  calls
  ▼
prism_agent.chat(message)
  │  builds context from prism_memory, prism_soul, prism_instructions
  │  applies prism_policy (CEO→Manager delegation check)
  ▼
prism_chain.run(prompt, context)
  │  routes via prism_llm_router (Claude → Ollama → stdlib fallback)
  │  may invoke prism_chain_expert for structured reasoning
  ▼
Organ dispatch (prism_organ_bus)
  │  prism_organ_loader selects matching organs
  │  organs execute (weather_check, finance_summary, …)
  ▼
Response assembled → returned to caller
  │
  └─► Soul / memory feedback loop
        prism_memory.store(exchange)
        prism_soul.update(signal)          ← identity shaped by interaction
        prism_outcome_tracker.record()     ← outcome fed back to soul + horizon
        prism_horizon.check_now()          ← long-range goal progress evaluated
```

---

## Persistence

All runtime state lives under `~/.prism/`. Each file is a standard SQLite database unless noted.

| File | Contents |
|------|----------|
| `memory.db` | Conversation history + semantic search index (BM25 fallback). |
| `soul.db` | Queryable soul state — beliefs, values, running identity model. |
| `chains.db` | Chain reasoning run history and cached results. |
| `chains_expert.db` | Expert-mode chain runs (structured multi-step reasoning). |
| `prism.db` | Top-level agent state: session metadata, context snapshots. |
| `tasks.db` | Background task queue — status, progress, results. |
| `outcomes.db` | Outcome tracker records — action → measured result pairs. |
| `policy.db` | Policy rules and delegation grants (CEO→Manager model). |
| `policy_audit.db` | Immutable audit log of every policy decision. |
| `identity.db` | Digital identity profiles and cross-session continuity. |
| `identity_bus.db` | Identity bus signal log (event sourcing for identity layer). |
| `artifacts.db` | Artifact store — files, outputs, and generated assets. |
| `bus.db` | General identity bus event store. |
| `organ_bus.db` | OrganBus signal replay buffer (recent organ activations). |
| `horizon.db` | Long-range goal state — milestones, deadlines, progress. |
| `tools.db` | Tool registry used by the executor agent. |
| `executions.db` | Execution history from prism_executor_agent. |

---

## Background Workers

`prism_daemon` starts the following threads on launch. All are daemon threads (auto-killed when the
main process exits) and use a shared `_SHUTDOWN` event for clean stop.

| Worker name | Module function | Default interval | Purpose |
|-------------|-----------------|-----------------|---------|
| `bus-flush` | `_bus_flush_worker` | 60 s | Flushes LOW-priority OrganBus signals in batches. |
| `horizon` | `_horizon_worker` | 300 s (5 min) | Evaluates horizon goal triggers. |
| `health` | `_health_worker` | 120 s | Logs a health summary line (chain count, soul state). |
| `reflection` | `_reflection_worker` | 604 800 s (7 days) | Runs the weekly reflection cycle: patterns, belief proposals, stale goal detection. |
| `outcome-feed` | `_outcome_feed_worker` | 3 600 s (1 hr) | Feeds outcome deltas into soul and horizon planner. |

---

## Adding an Organ

Organs are the unit of capability in PRISM — lightweight callables that execute one focused action
(lookup, transform, IO) and return structured data to the chain.

### Step 1 — Create the organ module

```
organs/my_organ.py
```

The file must define the `ORGAN_POLICY` dict and an `execute` function (see step 2 and 3).

### Step 2 — Declare `ORGAN_POLICY`

```python
ORGAN_POLICY = {
    "name": "my_organ",
    "description": "One sentence: what this organ does.",
    "triggers": ["keyword1", "keyword2"],   # words in user intent that activate this organ
    "requires": [],                          # capability flags (e.g. "network", "filesystem")
    "output_schema": {"result": "str"},      # keys the chain can reference
}
```

### Step 3 — Implement `execute(ctx)`

```python
def execute(ctx: dict) -> dict:
    """
    ctx keys guaranteed present:
      ctx["message"]   – raw user message
      ctx["intent"]    – parsed intent string
      ctx["agent"]     – reference to PrismAgent (for memory / soul access)
    Returns a dict matching output_schema.
    """
    # ... your logic here ...
    return {"result": "value"}
```

### Step 4 — Register in `prism_organ_loader.py`

Open `prism_organ_loader.py` and add your organ to the `ORGAN_REGISTRY` list:

```python
from organs.my_organ import ORGAN_POLICY, execute as my_organ_execute

ORGAN_REGISTRY = [
    # ... existing entries ...
    {"policy": ORGAN_POLICY, "fn": my_organ_execute},
]
```

### Step 5 — Write tests

Create `tests/test_my_organ.py`:

```python
from organs.my_organ import execute

def test_my_organ_basic():
    ctx = {"message": "test", "intent": "keyword1", "agent": None}
    result = execute(ctx)
    assert "result" in result
    assert isinstance(result["result"], str)
```

Run with `pytest -x -q tests/test_my_organ.py`.
