# Changelog

All notable changes to PRISM are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
from v1.0 onward; pre-1.0 releases may break compatibility on minor
version bumps.

## [Unreleased]

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

[Unreleased]: https://github.com/chizoalban2003-beep/Prism/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/chizoalban2003-beep/Prism/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/chizoalban2003-beep/Prism/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/chizoalban2003-beep/Prism/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/chizoalban2003-beep/Prism/releases/tag/v0.1.0
