# Changelog

All notable changes to PRISM are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
from v1.0 onward; pre-1.0 releases may break compatibility on minor
version bumps.

## [Unreleased]

## [0.2.4] — 2026-06-24

Tiny routing hotfix surfaced during the v0.2.3 live-test session. With
the new reasoning pre-filter in place, "explain database deadlocks" was
still misrouting to `smart_home` because the offending regex matched at
the *first-pass* intent table, never reaching the LLM classifier where
the pre-filter sits. Root cause: the `smart_home` intent's
`lock|unlock` alternation had no word boundaries, so any substring
containing `lock` — `deadlocks`, `blocking`, `roadblock`, … — matched.

### Fixed
- **`smart_home` lock/unlock now word-boundaried.** `lock|unlock` →
  `\b(?:un)?lock\b`. `lock the front door` / `unlock my phone` still
  route to `smart_home`; sentences merely containing `lock` as a
  substring fall through. Same class of bug as issue #26 bug 2, but at
  the regex layer where the LLMClassifier pre-filter can't help.

### Added
- `TestSmartHomeWordBoundary` in `tests/test_routing_issue_26.py`
  pinning the four positive/negative cases discovered during testing.

## [0.2.3] — 2026-06-24

Router-quality hotfix driven by live-test feedback on v0.2.2 (issue #26).
Three bugs all surfaced from the same DeepSeek-backed chat session: PRISM
mis-routed plain personal-fact questions to Wikipedia, the LLM classifier
latched onto ambient topic keywords ("deadlocks" → devices_list), and
"remember that my X is Y" was stored as a standing rule with the
imperative still attached instead of an indexable memory entry.

### Fixed
- **Personal-fact recall now reaches `PrismMemory`.** Added a
  `memory_recall` intent positioned after the specific `my_*` routes and
  before the wikipedia/web-search catch-alls, so "what is my favourite
  colour?" hits the local store instead of being redirected to an
  encyclopaedia. Negative lookahead excludes routes that already own
  "my X" (`my profile`, `my calendar`, `my email`, `my budget`, …) so
  none of them lose their existing behaviour.
- **`PrismInstructions.parse_fact` extracts `(key, value)` from
  "remember that my X is Y"** and the chat prelude now routes those
  assertions into `PrismMemory` (source=`"fact"`) instead of dropping
  them into the standing-rule store. The verbatim-imperative storage
  bug ("Stored: remember that my favourite colour is blue") is gone —
  facts come back as plain `"My favourite colour is blue."` from
  recall.
- **Imperative prefixes stripped from stored rules.** `parse_from_chat`
  now peels `remember:`, `remember that `, `from now on, `, `make sure `,
  `don't forget `, `note:`, `rule:` (plus an optional leading `please `)
  off the saved text. `always` / `never` / `whenever` / `every time` are
  preserved because they carry the rule's quantifier — stripping them
  would invert meaning.
- **LLM intent classifier pre-filters reasoning questions.** Messages
  framed as `explain X`, `why does X`, `how does X work`, `name three`,
  `tell me about X`, `compare X`, `define X` short-circuit to
  `general_chat` before the classifier prompt runs. v0.2.2's classifier
  blindly trusted its output, so an explanation like "explain database
  deadlocks" got routed to `devices_list` because the word "deadlocks"
  overlapped a tool-intent keyword. The pre-filter is regex-only and
  costs no LLM call.

### Added
- `tests/test_routing_issue_26.py` — covers the new `memory_recall`
  intent (positive cases plus negative checks that `my_profile` /
  `my_growth` / `calendar_read` / `email_read` / `show_policies` still
  win), the reasoning-question pre-filter, and an end-to-end test that
  forces a misfiring classifier and confirms reasoning-frame messages
  fall through to `general_chat` regardless of what the LLM returns.
- `tests/test_instructions.py` extended with prefix-strip,
  `parse_fact`, and "facts don't land in the rule store" tests.

## [0.2.2] — 2026-06-24

Second packaging+UX hotfix surfaced during live install testing of v0.2.1.
v0.2.1 closed the root-module gap but `organs/` was still missing from
the wheel — 40+ bundled organs (including the `gdrive_search` /
`notion_query` / `dropbox_fetch` document-store organs and the new
`agents_inventory`) never reached pip installs. Also caught a stale
config-path default in `LLMRouter.from_config` that silently stranded
user config at the daemon level, and a first-run UX issue where the
default federation auth posture logged an ERROR-level boot message on
single-node setups (the dominant use case).

### Fixed
- **`organs/` package now ships in the wheel.** `pyproject.toml` declares
  `packages = ["organs"]` and the directory carries an `__init__.py`. The
  organ loader still uses file-based imports (unchanged), so the empty
  `__init__` is purely a setuptools handshake.
- **`LLMRouter.from_config` default path corrected** from
  `~/.prism/config.toml` to `~/.prism/prism_config.toml` — matching what
  `prism_agent_bootstrap.load_toml_config` actually reads. The two
  defaults had silently disagreed since the bootstrap fix in v0.2.0, so
  the daemon-level router (used by `GET /agents`) saw an empty config
  even when the user-config file was correctly populated.
- **Daemon reuses `PrismAgent._router` instead of constructing a parallel
  one.** `_state["llm_router"]` and `agent._router` now agree on what's
  configured, so `/agents` no longer under-reports providers.

### Changed
- **Federation auth default flipped to permissive** for single-node /
  local-first runs (the dominant use case — server binds to 127.0.0.1
  anyway). Multi-node deployments must opt in to strict mode via
  `[federation].require_auth = true` and a token. The boot-time
  advisory drops from ERROR to INFO when permissive. Existing strict
  deployments are unaffected as long as they had `require_auth = true`
  explicitly set (the example config now documents this as the
  multi-node escape hatch).

### Added
- `tests/test_packaging.py` extended:
  - asserts `organs` is declared in `[tool.setuptools].packages`
  - asserts every `organs/*.py` exposes `ORGAN_META` and `execute()`
    (so a packaging miss can't hide behind silent loader-skips)
  - asserts `LLMRouter.from_config` default path matches the bootstrap
    loader's

## [0.2.1] — 2026-06-24

Packaging hotfix. v0.2.0's wheel was missing 25 root-level modules from
`pyproject.toml`'s `py-modules` list — including ones the daemon imports
at boot (`prism_agent_bootstrap`, `prism_mesh`, `prism_mcp`,
`prism_identity_learning`, `prism_perception_cluster`,
`prism_chat_subsystems`, `prism_routes_mesh`, `prism_routes_mcp`) and the
brand-new `prism_agent_registry` / `prism_routes_agents` that v0.2.0
shipped. Editable installs and run-from-clone setups were unaffected, but
`pip install prism-platform @ git+…@v0.2.0 && prism` died with
`ModuleNotFoundError: prism_agent_bootstrap`. CI never caught it because
the test suite runs against the working tree, not the built wheel.

### Fixed
- All 25 missing modules added to `py-modules` in `pyproject.toml`.
  Pure-pip installs of the daemon now boot.

### Added
- `tests/test_packaging.py` — static guard that diffs `*.py` in the repo
  root against `py-modules` and fails the suite if any root-level module
  isn't packaged. Catches this exact regression class going forward.

## [0.2.0] — 2026-06-23

A substantial minor release. PRISM grew an MCP client, a capability-aware
device mesh, three document-workspace organs (Drive / Notion / Dropbox),
a unified read-only registry that aggregates every agent surface (LLM /
organ / MCP / mesh), a CEO/manager governance bridge layer, and a
near-total init-path refactor — all on the same L1 → L2 → L3 gate.

### Added
- **Unified agent registry.** `prism_agent_registry.inventory()` pulls
  from `LLMRouter.status_summary()`, `OrganLoader._organs`,
  `MCPManager.status()`/`list_tools()`, and `PrismMesh.list_peers()` and
  normalises every entry to `{kind, name, status, capabilities[], …}`.
  Exposed via `GET /agents?capability=…` (`prism_routes_agents.py`) and
  via the `agents_inventory` chat-routable organ. The aggregator is
  read-only — it answers *what exists*, not *what to use*.
- **Document store integrations.** Three new `internet_read` organs —
  `gdrive_search` (Drive v3 files API, OAuth2 token), `notion_query`
  (Notion v1 search API, integration token), `dropbox_fetch` (Dropbox
  `files/search_v2`, app token). Token resolution falls back to
  env vars (`GDRIVE_TOKEN`, `NOTION_TOKEN`, `DROPBOX_TOKEN`); missing
  tokens render a setup card instead of raising.

### Changed
- **Cross-component wiring moved inline.** `PrismAgent._wire_backpatches`
  consolidation method removed. Each cross-cluster back-patch
  (`chain → soul`, `chain → persona`, `chain → organ_loader`,
  `horizon → chain`, `outcome_tracker ↔ crystalliser`, `outcome_tracker
  → kinetic`, `ml_assembler → tracker`, `orchestrator → persona`) now
  lives inside the factory closure that constructs the second-to-arrive
  component. -45 LOC in `prism_agent.py`; dependency graph readable
  from the construction site.
- **Config resolution aligns with docs.** `load_toml_config` now tries
  `~/.prism/prism_config.toml` first (the location documented in
  QUICKSTART and the architecture diagram) and falls back to the path
  passed in. `prism_config.example.toml` updated to recommend the
  user-config location — survives reinstalls.

### Fixed
- **First-run logs are deterministic.** The daemon now emits one of
  three explicit lines at startup — *"Identity ceremony complete"*,
  *"No soul seed found — running with default identity"*, or
  *"Identity seed loaded"* — so a non-developer no longer sees the
  ambiguous "No soul seed found" and assumes PRISM is broken.
- **LLM reachability check at startup.** Probes the configured Ollama
  host (4 s timeout) and surfaces *reachable*, *API-key fallback in
  use*, or *unreachable-and-no-key with a pointer to
  `python3 prism_daemon.py --setup-llm`*. The daemon previously
  silently degraded to no-LLM mode if Ollama wasn't installed.

### Chat-path & agent bootstrap refactoring (2026-06-22 → 06-23)

`PrismAgent.__init__` and `_execute` were both growing past comfortable.
The bulk of construction and chat-path branching is now in dedicated
factories so the nucleus reads top-to-bottom and so testing can target
one cluster at a time. No user-facing API change.

#### Changed
- `prism_agent_bootstrap.py` extracted: TOML loading, LLM config build,
  and a `safe_init()` wrapper that fail-softs any single init site.
- `prism_identity_learning.py`, `prism_perception_cluster.py`,
  `prism_chat_subsystems.py` extracted: three factories that build the
  identity/learning, perception/proactive/kinetic, and
  chain/composer/loader/expert clusters with their cross-wires inline.
- `prism_chat_context.py`, `prism_chat_graph_bridge.py`,
  `prism_chat_tiers.py`, `prism_routing.py`, `prism_unknown_handler.py`,
  `prism_organ_dispatch.py` extracted: chat-prelude, WAL graph bridge,
  Tier 0–3 dispatcher, intent routing, managerial-PA synthesis
  fallback, and the L1→L2→L3 organ execution gate.
- `prism_goal_intents.py`, `prism_pa_intents.py`,
  `prism_info_intents.py` extracted: themed intent handler groupings
  pulled out of `_execute`.
- Ten remaining init sites swept onto `safe_init`; dead phase H
  removed.

### Model Context Protocol (MCP) client (2026-06-20)

PRISM is now an MCP client. Configured MCP servers expose their tools,
resources, and prompts through the same chat surface as native organs.

#### Added
- `prism_mcp.py` — `MCPManager` orchestrates handshake, caches
  `tools/list`, dispatches `tools/call`. Supports both **stdio** and
  **Streamable HTTP** transports.
- `prism_routes_mcp.py` — `GET /mcp/status`, `GET /mcp/servers`,
  `GET /mcp/tools`, `POST /mcp/connect`, `POST /mcp/call`.
- MCP tools routable directly from chat; `mcp_arguments` pass through
  the L1/L2/L3 gate exactly like a native organ.
- Resources + prompts supported, not just tools.

### CEO/manager governance bridges (2026-06-18 → 06-19)

Thirteen bridges over two days aligning the codebase with the
CEO/manager mental model: the only surfaces the user touches are
**permissions, instructions, policy, budget, and plug-ins**.

#### Added
- **Budget primitive.** `prism_budget.py` enforces daily/monthly USD
  ceilings on LLM spend with soft warning bands; free-provider bypass.
  Routable via the `budget_status` intent and the `[budget]` config
  section.
- **Persona policy export.** What the manager learned about the user
  is now inspectable as policy rather than opaque embeddings.
- **DAG composition planner.** `prism_organ_planner.py` reads
  `ORGAN_META.inputs/outputs` and `composable_with()` to wire chains
  automatically — the foundation for PowerBI-style arrows between
  organs and buds.
- **Typed organ I/O schemas.** `ORGAN_META.inputs/outputs` declared by
  every shipped organ (optional, backward-compatible).
- **Portable Organ Packs.** `prism_organ_pack.py` exports bundles of
  organs as hash-verified JSON for sharing; imports run the same AST
  safety validation as on-the-fly synthesis.
- **Auto-pick organs.** `_llm_classify` injects `loader.known_intents()`
  so a freshly synthesised organ is callable next turn; the LLM picks
  from the live loader, not a frozen list.
- **Routable synthesised organs.** Same mechanism for synthesis output.
- **Mechanical-scope capability gates.** Foundation for chaining
  Twilio/smart-home organs with software organs.
- **`frontend_mutate` capability gate.** `PrismCard.body` is rendered
  unescaped, so a new capability is detected via HTML/JS signal scan,
  listed as critical, and added to `never_synthesize_capabilities`.

### Security hardening (2026-06-18 → 06-22)

#### Added
- Federation auth defaults to **strict** (fail-safe). Was opt-in
  via `PRISM_FEDERATION_REQUIRE_AUTH=1`.
- Federation **peer pinning** + KSAgent daemon wiring.
- **Synthesis quarantine**: newly synthesised organs land in a
  quarantine area until policy approves.
- **Forbidden intent-name patterns blocked at synthesis** — `system_*`,
  `agent_*`, and reserved router sentinels can no longer be claimed by
  an LLM-generated organ.
- Constitution **`never_log` privacy** enforced at the routing layer.
- Home Assistant token now sourceable via env var; no longer requires
  plaintext in `prism_config.toml`.

#### Changed
- BudManager ctx tightened to **least privilege** — only keys declared
  by the organ's capability manifest are visible during execution.

### Memory durability bridge (2026-06-20)

Conversation memory is now written through the WAL graph
(`prism_chat_graph_bridge.py`). Every chat turn becomes a graph node
with an `answered_by` edge, so recall is durable across crashes via
the same WAL replay path used by the rest of the memory system.

### M12 — SIAM-aligned learning + routing wave (2026-06-17)

Four directions that close feedback loops PRISM was missing: plan
execution telemetry feeds the horizon planner, user denials now both
guard the runtime gate AND train the personalised LoRA, and the device
mesh routes by capability instead of by explicit peer name. The wave is
deliberately aligned with the SIAM blueprint pillars (Semantic Manifest,
Guardrails, Semantic Daemon).

#### Added
- **M12a — mesh capability-aware auto-routing.** `prism_mesh.py` grows
  `score_peer_for_task` and `find_capable_peer`. The `mesh_orchestrate`
  organ now auto-routes when the user omits a peer name and one peer
  uniquely matches the task's capability hints (browser, ffmpeg, git,
  image, compress, package_manager…); ties surface a "pick a peer"
  prompt instead of silently guessing. Forwards also enforce `MAX_HOPS
  = 2` via a `_hop` counter, so chained A→B→C→… loops self-terminate.
- **M12b — denial → standing-rule extraction.** `PrismInstructions.
  classify_denial` detects "never/always/from now on/whenever/…"
  markers in the textarea reason on an approval card. `PrismAgent.
  record_denial` now dual-writes: the existing task-scoped retry guard
  plus a broad-trigger standing rule keyed by TRIGGER_MAP category. The
  `/device/approve` confirmation note adapts — "Saved as a rule for all
  email requests" instead of the generic "Noted: I'll remember this."
- **M12c — LoRA denial → DPO pair ingestion.** `PrismLoraTrainer.
  _collect_dpo_pairs` now reads both `OutcomeRecord.correction` and the
  PrismInstructions DB. Standing rules, task-scoped denials, and
  `always` rules each get a tailored prompt/chosen/rejected shape so
  the user's "no" actively trains the LoRA, not just gates the runtime.
- **M12d — plan execution telemetry.** New `prism_plan_telemetry.py`
  module persists every `DailyPlan` (request, primary focus, rationale,
  step list) to `~/.prism/plan_telemetry.db` with per-step status and
  outcome record cross-links. `PrismAgent.replan` reads the previous
  plan's telemetry summary, prepends it to the KDE prompt, and marks
  the prior plan as superseded by the new one. New routes: `GET
  /plan/latest`, `GET /plan/{plan_id}`, `POST /plan/{plan_id}/step/
  {step_index}`.

#### Fixed
- `PrismLoraTrainer._collect_dpo_pairs` called `tracker.recent(limit=
  500)` against an `n`-keyworded signature — silently emitted zero
  pairs from outcome corrections. Now passes `n=500`.

#### Notes
- Wave is covered by 36 new tests (15 mesh, 8 instructions/denial,
  4 agent-level replan wiring, 8 LoRA DPO, 5 routes). CI green on
  3.11 + 3.12.
- The SIAM↔PRISM alignment report from the previous turn is the
  rationale behind the d → b → c → a sequence. Escrow ledger card,
  which the report flagged as a missing pillar, is deferred to a
  later milestone.

## [0.1.3] — 2026-06-16

Patch release. A second smoke-test on the v0.1.2 daemon (boot → probe
every endpoint with a real bearer token, not a `MagicMock`) surfaced
five more orphan routes the v0.1.1 audit had missed. Same root cause:
the test fixture pre-populates every attribute the route handler
might touch (`agent._hub`, `agent._assistant`, `agent._profile`,
`agent.morning_briefing`, …), so the tests pass even though the
attributes don't exist on real `PrismAgent`. `curl` finds the bug in
under a second.

### Removed
- `GET /plan` in `prism_routes_agent.py` — called missing
  `agent.morning_briefing()`. `POST /plan` is the working alternative
  (it uses `PrismPlanner` directly, not an agent method) and remains.
- `POST /rate` in `prism_routes_core.py` — called missing
  `agent._assistant.rate_day(agent._profile.name, …)`. Neither
  `_assistant` nor `_profile` exists on `PrismAgent`.
- `POST /session` in `prism_routes_core.py` — called missing
  `agent.log_session()`.
- `GET /devices` in `prism_routes_core.py` — called missing
  `agent._hub.list_devices()`. `device_hub.py` exists as a module but
  is not wired into `PrismAgent.__init__`.
- `POST /device/sync` in `prism_routes_core.py` — called missing
  `agent.sync_devices()`.

### Notes
- Pattern is the same `MagicMock`-hides-bug failure as v0.1.1; the
  fix is the same (delete the orphan route). Restoring the features
  would require also wiring `DeviceHub`, an `_assistant` object, and
  user `_profile` into `PrismAgent.__init__` — out of scope for a
  patch.
- Smoke-test discipline: after the v0.1.1 audit I shipped v0.1.2
  (test-suite fix only) without re-running the daemon-probe step,
  reasoning the functional code hadn't changed. Five 500s later, the
  lesson lands: *every* release gets a real-client smoke test, not
  just feature ones.

## [0.1.2] — 2026-06-16

Patch release. After v0.1.1 a full pytest run took 7h06m and 9 tests
failed once Ollama was installed on the dev box: `LLMRouter.discover()`
found tinyllama and used it, but tinyllama responses take >30 s — past
the `pytest-timeout` default. Two root causes, addressed separately.

### Fixed
- `test_vision_chain.py::test_images_none_is_default` (and its sister)
  tried to disable discovery via `_options=[]; _discovered=True` but
  forgot `_last_scan`. `discover()`'s cache check is
  `time.time() - _last_scan < 60`, and the sentinel `0.0` always
  evaluates false → re-discovery ran and hit Ollama anyway. Fix: also
  set `_last_scan = time.time()`.
- `test_llm_router.py::test_call_tuple` and
  `test_llm_router_history.py::test_call_accepts_history` were
  checking router return *shape* (tuple of two strings) with no
  mocking. They passed pre-Ollama only because connection-refused
  returned in 2 s and fell through to stdlib. Same `_last_scan` fix
  applied — they now assert the shape deterministically without a
  real LLM in the loop.

### Changed
- `pyproject.toml` adds `addopts = "-m 'not slow'"` under
  `[tool.pytest.ini_options]` and registers the `slow` marker. Six
  genuine integration tests of `PrismCollaborator`, `PrismAgent.chat`,
  and the nucleus topology that *do* exercise the LLM call path
  end-to-end are now marked `@pytest.mark.slow` and excluded from the
  default run. Opt in with `pytest -m slow` when an LLM is up and
  you have patience.

### Result
- Full suite: **2822 passed, 2 skipped, 6 deselected in 3m09s**
  (was 9 failed, 2819 passed, 2 skipped in 7h06m).

## [0.1.1] — 2026-06-16

Patch release. End-to-end smoke test on a fresh host (boot → HTTP →
auth → Ollama/tinyllama chat round-trip) surfaced a set of orphan
routes whose target agent methods were never implemented; the unit
tests passed because the test stub was a `MagicMock`. Removed so a
first user with `curl` doesn't hit confusing 500s.

### Removed
- `GET /reflect`, `GET /history`, `GET /artifacts`, `POST /artifacts/rate`
  in `prism_routes_agent.py` — all five called methods that don't exist
  on `PrismAgent` (`reflect`, `_assistant`, `recent_artifacts`,
  `rate_artifact`).
- `GET /identity`, `GET /identity/domains`, `POST /identity/observe`,
  `POST /identity/reset` in `prism_routes_agent.py` — same pattern.
  The canonical identity snapshot lives at `GET /identity/dashboard`,
  the HTML page at `GET /identity/ui`, the cross-device export at
  `GET /federation/identity`.
- `POST /ask` in `prism_routes_core.py` — called missing `agent.ask`.
  Use `POST /chat` for the working conversational entry point.

### Fixed
- README API table and capability matrix referenced the removed `/reflect`
  and `/identity` paths; updated to point at `/identity/dashboard`,
  `/identity/ui`, `/identity/onboard`, and `/reflection`.

## [0.1.0] — 2026-06-16

First public release. Local-first personal-AI daemon: physics-based
decision engine, organ topology with three-layer security, federated mesh
sync, and a Jarvis-class identity model that crystallises from your
actual decisions.

### Added
- Nucleus–Organ topology with `ConstitutionGuard` (L1), `ORGAN_POLICY`
  per-organ gate (L2), and `BudManager` ephemeral scoped agents (L3).
- 35 bundled organs + LLM-synthesised user organs with AST safety check.
- Federated mesh sync with Lamport vector clock, HMAC-SHA256 replay
  protection, and bearer-token auth.
- FastAPI/ASGI HTTP+WS surface on `127.0.0.1:8742` with bearer-token
  middleware and per-host token-bucket rate limiting on streaming
  endpoints.
- Crystallisation engine: identity domains, soul beliefs, persona traits,
  outcome tracking, reflection.
- `prism-tray` (system tray), `kde` (kinetic dashboard CLI), `ksa`
  (kinetic-sport agent CLI) entry points.
- `Dockerfile` + `docker-compose.yml` for containerised deploys.

### Security
- AST safety visitor blocks `eval`/`exec`/`compile`/`__import__`/`open` as
  bare-name references too (defeats `e = eval; e('1+1')` rebinds).
- Sandbox-escape attrs `__mro__`, `__subclasses__`, `__bases__`,
  `__globals__`, `__class__` and filesystem op `rmdir` are now blocked.
- Marker sanitiser in `PrismChain._sanitize_for_prompt` is now
  case-insensitive and tolerates whitespace inside delimiters.
- `/_health` exempted from bearer auth so orchestrators can probe.
- Shared SSRF guard (`prism_ssrf.is_safe_external_url`) resolves DNS
  hosts and refuses any address that resolves into loopback, private,
  link-local, multicast, or reserved space; applied to the browser
  agent, federation push, and `/federation/announce`.
- `prism_autonomous._install_requirements` now uses an explicit PyPI
  allow-list — the LLM cannot pick arbitrary package names.
- `prism_collaborator.synthesise_tool` and
  `prism_executor_agent._save_code` run synthesised code through the AST
  safety check before persistence and execution.
- `organs/file_write` switched from deny-list to allow-list of user-data
  roots; refuses dotfile names and `.service` / `.desktop` payloads.
- `prism_device_agent.open_app` / `install_package` validate input
  against a strict shape and reject path-traversal / scheme tricks.

[Unreleased]: https://github.com/chizoalban2003-beep/Prism/compare/v0.2.4...HEAD
[0.2.4]: https://github.com/chizoalban2003-beep/Prism/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/chizoalban2003-beep/Prism/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/chizoalban2003-beep/Prism/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/chizoalban2003-beep/Prism/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/chizoalban2003-beep/Prism/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/chizoalban2003-beep/Prism/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/chizoalban2003-beep/Prism/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/chizoalban2003-beep/Prism/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/chizoalban2003-beep/Prism/releases/tag/v0.1.0
