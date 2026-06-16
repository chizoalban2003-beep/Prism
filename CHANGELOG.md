# Changelog

All notable changes to PRISM are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
from v1.0 onward; pre-1.0 releases may break compatibility on minor
version bumps.

## [Unreleased]

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

[Unreleased]: https://github.com/chizoalban2003-beep/Prism/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/chizoalban2003-beep/Prism/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/chizoalban2003-beep/Prism/releases/tag/v0.1.0
