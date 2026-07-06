# Research note: is PRISM a self-extending, policied command centre yet?

Status: capability assessment (#28-114), verified live on a real install
on 2026-07-06. Companion to docs/rfc-agentic-loop.md.

## The question

Can PRISM, through the policied agentic loop, (a) handle and *learn to
handle* tasks — controlling the user's hardware, making calls, sending
email under the user's permission/notification/budget/policy/planning —
and (b) act as a command centre where the user runs their desktop,
building connections between apps to share info, data, and resources?

## (a) Task handling + learning — YES, and the loop is closed. Verified.

Four layers, all live:

1. **Acting under policy.** The tool loop proposes; `dispatch_organ`
   disposes — L1 constitution, L2 approval cards + rate ceilings, L3
   capability-scoped execution, provider budget per hop, taint rule
   after untrusted content. `email_send`, `phone_call` (Twilio),
   `telegram_send`, `smart_home_control`, `shell_run`, `file_write`,
   `system_power` all exist as organs and all stop at approval cards.
   Verified live this cycle: multi-step weather→note request executed
   through the loop with every call gated.
2. **Learning to act (the closed loop).** When no organ matches,
   the autonomous path synthesises one (PrismCollaborator → AST safety
   visitor → SSRF guard → quarantine → promote). This is not
   theoretical: **this install has 9 self-synthesised organs**
   (generate_haiku, synth_compute_sha256_string, synth_encrypt_message,
   cross_reference, …) — and all 9 are in the tool-loop belt via
   organ_tool_schemas(). Synthesis grows the belt; the loop can call
   what PRISM taught itself yesterday. That is "learning to handle
   tasks" in the literal sense.
3. **Learning how to behave.** Persona crystalliser (11k observations
   on this install), calibration feedback ("too aggressive"), LoRA
   denial→DPO ingestion, and 8 standing instructions in
   instructions.db — plain-language rules taught once, enforced on
   every matching turn.
4. **Planning across time.** Planner (minutes), orchestrator task
   graphs + horizon goals (days/weeks, survive restarts), proactive
   triggers + reminders (fire with the tab closed).

Per-user divergence is structural: the belt (which organs exist),
the policies (constitution + [tool_loop] config), the persona weights,
and the standing instructions are all local state — two Prisms
genuinely stop being the same software.

## (b) Command centre / app-mesh — the spine exists, three gaps.

**Exists today:** MCP client (stdio + Streamable HTTP) — any MCP server
(filesystem, browser, Slack, GitHub, databases…) becomes callable
tooling through the same chat surface and the same gates; organ I/O
schemas + `composable_with()` for wiring producer→consumer; federation
mesh + mobile sync across devices; IDE extension; PWA; device
executor/scanner for local apps and files.

**Gap 1 — MCP is dark on this install** — CLOSED (#28-116). MCP
tools now join the tool-loop belt through the same gates as organs.
`prism_tool_loop._belt()` returns provider-safe function names
(`mcp.demo.echo` → `mcp_demo_echo`) and a name→intent map for dispatch;
any `mcp.*` tool is treated as both untrusted-source and outbound for
taint purposes. Verified live: the loop engaged with 61 tools incl. 2
MCP, called `mcp.demo.local_time` through the gate, then correctly
tainted the belt (→34 low-risk, outbound + MCP denied). Enable via
`[mcp] enabled=true` + `[[mcp.servers]]` in the config overlay.

**Gap 2 — no persistent pipes** — CLOSED (#28-116). `prism_pipelines`
adds a `PipelineStore` (`~/.prism/pipelines.db`) and NL grammar
"save pipeline <name>: <steps> [every N unit]". Four intents
(`pipeline_save/run/list/delete`) are recognised by a regex-only
*priority route* that fires BEFORE the chain/composer fold — a saved
instruction routinely contains "and"/"then" and task keywords that
would otherwise run the steps instead of saving them. Running a
pipeline replays its instruction through the policied tool loop
(`agent.run_pipeline`); scheduled pipelines fire from the daemon's
`pipeline` worker via the proactive delivery path. Verified live:
full save/list/run/delete lifecycle + scheduled ("every 15m") parse.

**Gap 3 — desktop control is shallow** — addressed via Gap 1. Deep
per-app automation (draft in the mail client, manipulate a
spreadsheet, window management) arrives as per-app MCP servers on the
now-live belt rather than PRISM-native code — the same gates, taint
rules, and approval flow apply. No separate PRISM subsystem is
warranted; the extension point is the MCP config.

## Honest constraints

- Telephony/email need credentials (Twilio, app password) — the setup
  cards exist, the accounts are the user's job.
- Synthesis is deliberately narrow: stdlib-only, AST-gated,
  quarantined until promoted. It learns *small* tools, not whole
  integrations — by design; big integrations should arrive as MCP
  servers or organ packs, not synthesised code.
- Every outbound/actuating capability is approval-gated; a command
  centre that never surprises you is the point, not a limitation.
