<p align="center">
  <img src="docs/logo.svg" width="420" alt="PRISM — Decision Intelligence"/>
</p>

<h1 align="center">PRISM — Decision Intelligence</h1>
<p align="center"><strong>Crystallised into you.</strong></p>

<p align="center">
  A local-first platform that decides, explains, and executes — for any user, in any domain.<br>
  Not a chatbot. Not a rules engine. Not an LLM wrapper.<br>
  A physics-inspired decision model that learns from your outcomes,<br>
  runs on your hardware, and belongs entirely to you.
</p>

<p align="center">
  <a href="https://github.com/chizoalban2003-beep/Prism/actions"><img src="https://github.com/chizoalban2003-beep/Prism/actions/workflows/ci.yml/badge.svg" alt="CI status"/></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python"/>
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="MIT"/>
  <img src="https://img.shields.io/badge/cloud-none-orange" alt="no cloud"/>
  <img src="https://img.shields.io/badge/runs-locally-orange" alt="local"/>
</p>

---

## What PRISM is

PRISM is a local personal AI assistant that decides, plans, and acts for any user across any domain — all on your own hardware. It combines three things no existing tool does simultaneously:

**A physics-based decision engine** that produces interpretable, personalised recommendations with named causes — not black-box predictions. Every decision is a Gaussian activation over a spectrum of options, weighted by user-specific factors.

**A full execution layer** that carries out approved actions, finds tools when they don't exist, learns new integrations on demand, and follows standing instructions you teach it once in plain language.

**A continuous learning identity** that crystallises from your actual decisions over time — becoming more accurate for you specifically, not for a population average. Feedback is as simple as "that was too aggressive" or "good call".

---

## Architecture

```
User input (chat / voice / CLI / REST API)
         │
         ▼
  ┌─────────────────────────────────────────────┐
  │  PrismAgent  (prism_agent.py)               │
  │  Four-tier routing · standing instructions  │
  │  Chat history · memory injection            │
  └──────────┬──────────────────────────────────┘
             │
    ┌────────▼────────────────────────────────────────────────────────┐
    │  ChainOrchestrator — Tier 0  (prism_orchestrator.py)            │
    │  "Prefrontal cortex" — decomposes multi-step tasks into a       │
    │  TaskGraph DAG; executes nodes in dependency order (serial or   │
    │  parallel); pauses cross-session via HorizonGoal; synthesises   │
    │  final answer.  Five profiles: reactive · analytical ·          │
    │  verification · creative · negotiation.                         │
    └────────┬────────────────────────────────────────────────────────┘
             │ delegates nodes to ↓
    ┌────────▼────────────────────────────────────────────────────────┐
    │  Four-Tier Reasoning Cascade                                     │
    │                                                                  │
    │  Tier 1 — Expert Chain (prism_chain_expert.py)                  │
    │    Router LLM → Logic+Policy → Evaluator LLM (1-5 score)        │
    │    → Branch Judge → Synthesiser LLM  [research queries]         │
    │                                                                  │
    │  Tier 2 — General Chain (prism_chain.py)                        │
    │    LLM₁ → Logic+Policy → Evaluator gate → LLM₂ → ... → LLMₙ   │
    │    Adaptive: plan emerges from real intermediate results         │
    │    Evaluator early-exits when result quality score ≥ 4/5        │
    │    Branches: up to 3 parallel logics when genuinely ambiguous   │
    │    Memory recall: top-5 relevant past entries prepended          │
    │                                                                  │
    │  Tier 3 — Static Composer (prism_composer.py)                   │
    │    LLM decomposes upfront → DAG → sequential/parallel execute   │
    │    [multi-step requests with clear dependencies]                 │
    │                                                                  │
    │  Tier 4 — Single Intent (prism_agent._execute)                  │
    │    Regex route → one logic module  [simple one-shot requests]   │
    └──────────┬──────────────────────────────────────────────────────┘
               │
    ┌──────────▼──────────┐       ┌─────────────────┐
    │  KDEAgent           │       │  KSAgent         │
    │  Sport + Domain     │       │  Developer tasks │
    └──────────┬──────────┘       └────────┬────────┘
               │                           │
    ┌──────────▼───────────────────────────▼────────┐
    │        Decision Engine                         │
    │  decision_spectrum.py                          │
    │  p = Σ(w·v·t)/Σ(w·v)  ← fulcrum             │
    │  activation = Gaussian kernel over options    │
    │  AdaptiveFulcrum.observe() ← online learn     │
    └────────────────────────────────────────────────┘

Personal Assistant Layer (all local):
  ┌──────────┐ ┌──────────┐ ┌──────────────┐ ┌────────────┐
  │  Email   │ │ Calendar │ │ Web Search   │ │  Contacts  │
  │ IMAP/SMTP│ │ CalDAV / │ │ Brave/DDG    │ │  SQLite    │
  │          │ │  Google  │ │              │ │  + Google  │
  └──────────┘ └──────────┘ └──────────────┘ └────────────┘
  ┌──────────┐ ┌──────────┐ ┌──────────────┐ ┌────────────┐
  │  Tasks   │ │ SmartHome│ │ Push (ntfy)  │ │  Browser   │
  │ Todoist/ │ │ Home Asst│ │  free, local │ │ Playwright │
  │ GitHub/  │ │  REST    │ │              │ │            │
  │ Linear / │ └──────────┘ └──────────────┘ └────────────┘
  │  Local   │
  └──────────┘

Background loop:
  PrismProactive  →  triggers (calendar, budget, recovery, wearable, calibration)
  PrismMemory     →  short/long-term memory (SQLite + TF-IDF)
  PrismPerception →  context (time, biometrics, system state)
  PrismVoice      →  STT input (Whisper local / SpeechRecognition)
  TaskQueue       →  async background tasks with live progress
  PrismCalibration→  conversational feedback → model adjustment
```

---

## Capabilities

### Decision Engine
- Physics-inspired Gaussian kernel decision model (`decision_spectrum.py`)
- Named factors, interpretable outputs — no black box
- Online learning via `AdaptiveFulcrum.observe()` — no retraining
- Conversational calibration: "that was too aggressive" adjusts factor weights

### Sports Intelligence
- Match prediction, injury risk, performance, transfer value
- Real-time moment analysis (1v1, shot, cross, penalty, drive, etc.)
- Duel network from match events — attacker vs defender win rates
- StatsBomb open-data pipeline; validated on 10 La Liga seasons
- Sports: Football, Basketball, Tennis, Rugby, Boxing, MMA, Wrestling, Cricket

### Domain Decision Framework
- Medical triage · Financial portfolio · Legal strategy
- HR hiring · Supply chain · Climate policy
- Same engine — different configuration, zero code changes

### Personal Assistant
| Capability | Module | Status |
|---|---|---|
| Chat interface | `prism_chat.py`, `prism_agent.py` | Working |
| Email read/send | `prism_email.py`, `organs/email_send.py` | Working (needs config) |
| Calendar read/write (CalDAV/iCal/Google) | `prism_calendar.py`, `organs/calendar_write.py` | Working (needs config) |
| Phone calls + SMS (Twilio) | `organs/phone_call.py` | Working — `pip install twilio`, add `[twilio]` to config |
| Web search | `prism_search.py` | Working (DDG free; Brave/Serp optional) |
| Push notifications | `prism_push.py` | Working (ntfy.sh, free) |
| Contacts | `prism_contacts.py` | Working (local + Google optional; auto-resolved in email/call) |
| Tasks | `prism_tasks.py` | Working (local + Todoist/GitHub/Linear) |
| Smart home | `prism_smart_home.py` | Working (Home Assistant) |
| Browser automation | `prism_browser_agent.py` | Working (needs playwright) |
| Device tasks | `prism_device_agent.py` | Working |
| Memory | `prism_memory.py` | Working — recalled at chain start; SQLite + semantic search |
| Standing instructions | `prism_instructions.py` | Working |
| Proactive triggers | `prism_proactive.py` | Working |
| Scheduled reminders | `prism_proactive.py` | Working |
| Wearable sync trigger | `prism_proactive.py` | Working |
| TTS | `prism_tts.py` | Working (espeak fallback) |
| Voice input (Whisper) | `prism_voice.py` | Working — Whisper local / faster-whisper / SpeechRecognition; `pip install openai-whisper` |
| LLM routing (Ollama) | `prism_llm_router.py` | Working |
| LLM routing (Claude API) | `prism_llm_router.py` | Working (needs API key) |
| Multi-user | `prism_agent.py` | Working (`[user].name` in config) |
| Unknown-tool PA fallback | `prism_agent.py` | Working (discovers + plans integrations) |
| Autonomous tool synthesis | `prism_autonomous.py` | Working — synthesises, installs, sandboxes (AST+subprocess), caches |
| Multi-step task orchestration | `prism_orchestrator.py` | Working — DAG decomposition, parallel execution, 5 chain profiles |
| Cross-session goals | `prism_horizon.py` | Working — persists HorizonGoals across restarts; resumes on session start |
| Approval gate | `prism_agent.py` | Working — `requires_approval` organs block until explicit confirmation |
| Adaptive reasoning chain | `prism_chain.py` | Working — alternating LLM→Logic+Policy→Evaluator spine, branches |
| Expert reasoning chain | `prism_chain_expert.py` | Working — Router/Evaluator/BranchJudge/Synthesiser specialised nodes |
| Evaluator quality gate | `prism_chain.py` | Working — per-step 1-5 score, early exit when sufficient |
| Logic composition (DAG) | `prism_composer.py` | Working — LLM decomposes task → parallel/sequential DAG |
| Outcome learning | `prism_outcome_tracker.py` | Working — Bayesian belief updates on done/abandoned/corrected outcomes |
| Crystallised user persona | `prism_persona.py` | Working — behavioural profile grows from every interaction |
| Continuous crystallisation | `prism_crystalliser.py` | Working — heuristic (every turn) + LLM deep analysis (hourly daemon) |
| Living narrative | `prism_narrative.py` | Working — weekly/monthly synthesis stored to memory; `my narrative` to read |

---

## Living User Model

PRISM crystallises to each user over time through three interlocking systems:

### PrismPersona — the crystallised self

A behavioural profile that grows alongside `PrismSoul` (which stores values and beliefs). Persona stores **how you operate** — patterns inferred from watching you work:

```
[Alice — crystallised profile]
Style: direct and technical · concise responses preferred
Active hours: 9am–6pm · peak: Tue/Wed mornings
Patterns: defers strategic decisions after 7pm · approves reversible changes readily
Preferences: 30min default meetings · prefers data over prose summaries
Confidence: 47 observations · 8 patterns · 12 traits
```

This compact description is injected into **every LLM call** — chain, orchestrator, and expert chain — so responses are calibrated to the specific user from the first word.

### PrismCrystalliser — the extraction engine

Runs in two modes with no manual input needed:

- **After every message** (zero LLM cost): heuristics extract message length → response preference, vocabulary → technical depth, approval/cancel → risk tolerance, time of day → active hours histogram
- **Hourly daemon tick**: sends the last 20 conversation turns + outcome stats + calibration events to the LLM; parses structured JSON; updates traits and patterns
- **Weekly**: full 7-day recrystallisation pass

Corrections deepen learning: when you say "no, not like that — I meant X", that correction is immediately extracted as an explicit preference.

### PrismNarrative — the living story

Chat commands:
- `my profile` — full crystallised profile: persona + soul beliefs + current snapshot
- `my narrative` — weekly synthesis: what happened, what shifted, what PRISM learned
- `what have you learned about me` — growth report: trait confidence gains, pattern counts, outcome trends

Weekly narratives are stored to `PrismMemory` as `source="narrative"` — they become semantically searchable, so future sessions can recall "three weeks ago PRISM noted you prefer X."

The three systems feed each other: outcomes update beliefs (soul) → beliefs shape decisions → decisions create patterns (persona) → patterns inform every future response.

---

## Autonomous Execution

PRISM is a managerial PA with full autonomy. When asked to do something it has no built-in tool for, instead of returning instructions to the user it:

1. **Synthesises a Python tool on demand** — the LLM writes a self-contained `execute(task, params) -> str` module
2. **AST safety check** — a strict `_SafetyVisitor` (Python AST walker) rejects any code calling `eval`, `exec`, `os.system`, `shutil.rmtree`, `socket.connect`, `open(..., "w"`, or dangerous imports — no string-pattern bypass possible
3. **Subprocess isolation** — synthesised code runs in a separate process (30s timeout) via a temp runner script, never in-process
4. **Installs pip dependencies** — any required packages are installed automatically with `pip install --quiet`
5. **Caches the tool** — stored as JSON in `~/.prism/tools/` (SHA256 key + fuzzy name match); reused for identical or similar future tasks
6. **Push-notifies on completion** — if push is configured, you get a notification on your phone when done

### Approval gate

Any organ or autonomous action with `requires_approval: True` in its `ORGAN_POLICY` is **blocked at execution** — PRISM stores the pending call and returns a confirmation prompt before taking any action:

```
You: send email to alice@example.com about tomorrow's meeting
PRISM: email_send requires approval before executing.
       Action: send email to alice@example.com about tomorrow's meeting
       Say yes or approve to confirm, or cancel to abort.
You: yes
PRISM: Sent to alice@example.com — "Tomorrow's meeting"
```

This applies to: `email_send`, `phone_call`, `calendar_write`, autonomous tasks, and any custom organ with `requires_approval: True`. Approvals expire after 5 minutes. `cancel` drops the pending action with no side effects.

### Viewing accumulated tools

Say **"what tools have you learned?"** (or "tool list", "acquired tools", "new capabilities") to see everything PRISM has synthesised:

```
You: what tools have you learned?
PRISM: Learned tools (3)
• weather_lookup — fetches current weather via Open-Meteo (used 4×)
• currency_convert — converts currencies using exchangerate.host (used 2×)
• hacker_news — fetches top Hacker News stories via the public API (used 1×)
```

### Tool cache location

All synthesised tools are stored in `~/.prism/tools/` as JSON files containing the tool name, description, synthesised code, requirements, use count, and last result. They persist across sessions — PRISM accumulates capability over time without re-synthesising.

### Safety blocklist

The following patterns are **always blocked** regardless of LLM output:

| Pattern | Reason |
|---|---|
| `os.system(` | Shell injection |
| `eval(` / `exec(` | Arbitrary code execution |
| `shutil.rmtree` | Recursive deletion |
| `os.remove(` | File deletion |
| `socket.connect` | Raw socket access |
| `.chmod(` | Permission changes |
| `open(..., "w"` | Unguarded file write |

---

## Reasoning Chains

For complex requests PRISM uses an alternating chain architecture instead of a single LLM call:

```
Message: "research async Python patterns and add a task to refactor my code"

Step 1 — LLM node:  decides to call web_search
Step 1 — Logic:     web_search("async Python patterns")
Step 1 — Policy:    no action flags
Step 1 — Evaluator: score 4/5 — sufficient, early exit
         Synthesiser LLM composes final answer from accumulated results
```

**Four-tier cascade** selects the right architecture per request:

| Tier | Module | When used |
|---|---|---|
| 0 — Orchestrator | `prism_orchestrator.py` | Cross-domain multi-step, conditional, cross-session ("if hotel confirms, book flight") |
| 1 — Expert | `prism_chain_expert.py` | "research", "analyse", "decide", "compare", "evaluate" |
| 2 — General | `prism_chain.py` | Multi-goal, conditional, "and then", "after that" |
| 3 — Composer | `prism_composer.py` | Multiple steps with clear "and" / "then" dependency |
| 4 — Single | `prism_agent._execute()` | Simple one-shot requests |

**Evaluator quality gate** runs after every logic step in the general chain. If the Evaluator scores the result ≥ 4/5 (`sufficient=True`), the chain exits early and a Synthesiser LLM composes the final answer — reducing wasted steps without the full +200% Expert overhead.

**Branching**: when genuinely ambiguous, the LLM spawns up to 3 parallel logic executions. Results are merged before the next LLM node, turning the spine into a tree.

**Hybrid chain intelligence** (production-adopted theory experiments):

| Component | Module | What it does |
|---|---|---|
| `InterceptorPolicy` | `prism_chain_theory.py` | 8-rule deterministic rerouter — fires with zero LLM calls when errors, delivery failures, or permission denials are detected |
| `SoftLogic` | `prism_chain_theory.py` | In-node LLM softener for noisy logics (`web_search`, `email_read`, `device_task`, `browser_task`) — compresses raw output to 3 key facts before the next Router call |
| `SubChainLogic` | `prism_chain_theory.py` | Research intent runs a 3-step mini-chain internally (`web_search → parse_result → cross_reference → Synthesiser`) — the outer chain sees one clean result string |

The `research` intent is registered in the logic registry and handled directly in `prism_agent._execute()`, ensuring SubChainLogic is invoked whether the request arrives via Tier 0 Expert, Tier 1 General chain, or Tier 3 direct execution.

View recent chains: say `show chain history` or call `GET /chain/recent`.

---

## Voice input setup

PRISM supports local speech-to-text via `prism_voice.py`. Three backends in priority order:

| Backend | Install | Notes |
|---|---|---|
| **openai-whisper** | `pip install openai-whisper` | Local, fully offline, best accuracy |
| **faster-whisper** | `pip install faster-whisper` | Local, ~4× faster, lighter model |
| **SpeechRecognition** | `pip install SpeechRecognition` | Requires internet (Google free tier) |

```bash
# Recommended — fully local, no cloud:
pip install openai-whisper sounddevice

# Then optionally configure in prism_config.toml:
```

```toml
[voice]
enabled     = true
model       = "base"    # tiny | base | small | medium | large
language    = "en"      # ISO 639-1; leave blank for auto-detect
sample_rate = 16000
```

**Chat commands:**
- `voice status` — check which backend is active
- `voice on` / `voice off` — enable/disable
- `transcribe /path/to/audio.wav` — transcribe a file

**REST API:** `POST /voice/transcribe` with `{"path": "/tmp/clip.wav"}` or raw audio bytes.

Falls back gracefully when no backend is installed — PRISM remains fully functional via text.

---

## Claude API key setup

PRISM uses Ollama (local) by default and falls back to Claude API when Ollama is unavailable or returns empty:

1. Get an API key from [console.anthropic.com](https://console.anthropic.com)
2. Add to `prism_config.toml`:

```toml
[llm]
claude_api_key = "sk-ant-..."
```

Or set `ANTHROPIC_API_KEY` environment variable.

---

## Phone calls and SMS (Twilio)

PRISM can make outbound voice calls and send SMS via [Twilio](https://twilio.com):

1. Create a free Twilio account and get a phone number at [console.twilio.com](https://console.twilio.com)
2. Install the library: `pip install twilio` (or `pip install ".[full]"`)
3. Add credentials to `prism_config.toml`:

```toml
[twilio]
account_sid = "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
auth_token  = "your_auth_token_here"
from_number = "+14155552671"   # your Twilio number in E.164 format
```

Or set environment variables `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM`.

**Usage:**
- `call +447700900000 and say "your meeting starts in 5 minutes"`
- `text +447700900000 say "running 10 minutes late"`
- `call Alice` — resolves from your contacts automatically

**Approval gate:** Phone calls are `irreversible` and `requires_approval`. PRISM will always ask for confirmation before dialling.

---

## Linear task integration

PRISM supports [Linear](https://linear.app) as a task provider via GraphQL API:

1. Get your API key from Linear → Settings → API → Personal API Keys
2. Add to `prism_config.toml`:

```toml
[tasks]
provider       = "auto"
linear_api_key = "lin_api_..."
```

When `linear_api_key` is set and no Todoist/GitHub tokens are configured, tasks automatically route to Linear. Say:

- `add task: Fix the login bug`
- `show my tasks`

---

## Scheduled reminders

PRISM supports natural language reminder scheduling:

- `remind me in 30 minutes to call Alice`
- `remind me at 3pm to check the oven`
- `don't let me forget to submit the report by 5pm`

Reminders fire via the proactive loop (polling every 60 seconds by default) and can send push notifications if `[push].topic` is configured.

---

## Google Calendar OAuth

PRISM supports Google Calendar via OAuth2 access token:

1. Set up a Google Cloud project and enable Calendar API at [developers.google.com/calendar](https://developers.google.com/calendar)
2. Obtain an OAuth2 access token (use `google-auth` library or the OAuth playground)
3. Add to `prism_config.toml`:

```toml
[calendar]
provider     = "google"
google_token = "ya29...."
```

PRISM reads/writes your primary calendar. Token refresh is **automatic** — when the access token expires, PRISM reads `google_creds.json`, calls `oauth2.googleapis.com/token` with the stored `refresh_token`, and writes the updated `access_token` and `expiry` back to disk. The file must contain:

```json
{
  "access_token":  "ya29.…",
  "refresh_token": "1//…",
  "client_id":     "….apps.googleusercontent.com",
  "client_secret": "…",
  "expiry":        "2025-01-01T00:00:00Z"
}
```

Point PRISM at the file via config:

```toml
[calendar]
provider      = "google"
google_creds  = "~/.prism/google_creds.json"
```

---

## Multi-user support

PRISM scopes the active user from `[user].name` in `prism_config.toml`:

```toml
[user]
name = "Alice"
```

Policies, calibration history, and standing instructions use this name as the user key. To support multiple users on the same machine, run separate instances with separate config files.

---

## Quick start

```bash
git clone https://github.com/chizoalban2003-beep/Prism.git
cd Prism
pip install -r requirements.txt
# Optional: pip install playwright && playwright install chromium
# Optional: install Ollama from https://ollama.ai, then: ollama pull mistral
```

If you want installed CLI entry points:

```bash
pip install .
prism --help
kde --help
ksa --help
```

Run `python3 prism_daemon.py --ceremony` on first boot to create your soul seed

### Chat interface

```bash
python kde_cli.py server --port 8742
```

Open **http://localhost:8742** — the PRISM chat interface. Type any request in plain language:

- `plan my day`
- `check my emails`
- `what's on my calendar today`
- `add task: finish the report by Friday`
- `search the web for Python async tutorials`
- `send me a push notification about my meeting`
- `that was too aggressive` — calibrates the model

### Developer agent (KSA)

```bash
python ksa_cli.py run "quietly scan my project folder in the background"
python ksa_cli.py status
python ksa_cli.py history file_index_stealth
```

### Sports platform (KDE)

```bash
python kde_cli.py morning
python kde_cli.py ask "predict Manchester City vs Arsenal"
python kde_cli.py ask "assess my squad injury risk"
python kde_cli.py reflect
```

---

## Configuration (`prism_config.toml`)

The repository ships a ready-to-edit `prism_config.toml`. All sections are optional — PRISM works without any configuration and degrades gracefully when integrations are missing.

```toml
[user]
role  = "universal"        # developer | athlete | coach | analyst | universal
name  = "PRISM User"
sport = "Football"
team  = ""

[agent]
db_path      = "~/.prism/prism.db"
ollama_model = "mistral"            # remove to disable LLM routing
ollama_host  = "http://localhost:11434"

# Email (IMAP + SMTP) — optional
[email]
provider  = "gmail"                 # "gmail" | "imap"
address   = "you@gmail.com"
password  = ""                      # Gmail: App Password from myaccount.google.com
imap_host = "imap.gmail.com"
smtp_host = "smtp.gmail.com"

# Calendar (CalDAV or iCal URL) — optional
[calendar]
provider  = "ical_url"              # "ical_url" | "caldav"
ical_url  = "webcal://..."          # paste your calendar URL

# Web search — optional (DDG works without any key)
[search]
provider      = "auto"              # "brave" | "serp" | "ddg" | "auto"
brave_api_key = ""                  # api.search.brave.com/app/keys

# Push notifications via ntfy.sh — optional, free
[push]
topic = "prism-yourname-2024"       # any unique topic name
server = "https://ntfy.sh"

# Contacts — optional
[contacts]
google_token = ""                   # Google People API OAuth token

# Tasks — optional (local SQLite always works)
[tasks]
provider      = "auto"              # "todoist" | "github" | "local" | "auto"
todoist_token = ""
github_token  = ""
github_repo   = "owner/repo"

# Smart home (Home Assistant) — optional
[smarthome]
ha_url   = "http://homeassistant.local:8123"
ha_token = ""                       # Long-lived access token from HA profile

# Phone calls + SMS via Twilio — optional
[twilio]
account_sid = ""                    # ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
auth_token  = ""                    # from console.twilio.com
from_number = ""                    # your Twilio number, e.g. +14155552671

[[devices]]
name       = "Apple Watch"
type       = "apple_watch"
watch_path = "~/Downloads/apple_health_export"
```

---

## REST API

Start the local server (binds to 127.0.0.1 only):

```bash
python kde_cli.py server --port 8742
```

### Chat & General

| Method | Route | Description |
|---|---|---|
| GET | `/` or `/chat` | PRISM chat web UI |
| GET | `/status` | Agent status, Ollama availability |
| POST | `/chat` | `{"message":"..."}` → PrismCard JSON |
| POST | `/plan` | `{"task":"...", "context":{}}` → plan card |
| GET | `/reflect` | Learned state for current user |

### Sports & Prediction

| Method | Route | Description |
|---|---|---|
| GET | `/predict/match?home=X&away=Y&sport=football` | Match prediction |
| GET | `/predict/injury?name=X&recovery=0.7&load=0.5` | Injury risk |
| GET | `/predict/performance?name=X&form=0.6` | Performance prediction |
| GET | `/predict/transfer?name=X&age=24&performance=0.6` | Transfer value |
| GET | `/predict/brief?home=X&away=Y` | Full pre-match brief |
| GET | `/moment/analyze?sport=Football&moment_type=1v1_keeper&player=X` | Moment analysis |
| POST | `/moment/calibrate` | Record outcome, trigger learning |
| POST | `/moment/live_frame` | Feed live tracking frame |
| GET | `/moment/history?player=X` | Player moment history |
| GET | `/duel/network` | Full duel network |
| GET | `/duel/player?player=X` | Player attack profile |

### Domain Decisions

| Method | Route | Description |
|---|---|---|
| GET | `/domain/list` | All available domains |
| GET | `/domain/evaluate?domain=Medical&severity=0.8` | Evaluate a case |
| POST | `/domain/validate` | Validate against expert labels |
| GET | `/domain/sensitivity?domain=X&profile=Y&factor=Z` | Factor sweep |

### Personal Assistant

| Method | Route | Description |
|---|---|---|
| GET | `/email/status` | Email configured? |
| GET | `/email/inbox?n=20` | Fetch inbox |
| POST | `/email/send` | `{"to":"...","subject":"...","body":"..."}` |
| GET | `/calendar/status` | Calendar configured? |
| GET | `/calendar/today` | Today's events |
| GET | `/instructions` | List standing instructions |
| POST | `/instructions` | Add `{"text":"...","trigger":"always"}` |
| GET | `/discovery/services` | All discovered service integrations |
| POST | `/discovery/build` | Build integration `{"service_id":"..."}` |
| GET | `/search?q=query` | Web search (Brave/DDG) |
| GET | `/push/status` | Push notification status |
| GET | `/smarthome/status` | Smart home status |
| POST | `/smarthome` | `{"action":"turn_on","entity_id":"..."}` |
| POST | `/voice/transcribe` | `{"path":"/tmp/clip.wav"}` or raw audio bytes → transcript |
| GET | `/voice/status` | STT backend, model, enabled flag |
| GET | `/chain/recent?n=5` | Recent general chain runs with avg eval score |
| GET | `/chain/expert/recent?n=5` | Recent expert chain runs |
| GET | `/horizon/goals` | List horizon goals (`?status=watching\|triggered\|paused\|completed\|abandoned`) |
| GET | `/horizon/status` | Planner summary with counts per status |
| POST | `/horizon/goal` | `{"intent":"…","trigger_condition":"…","completion_condition":"…","expires_in_days":30}` |
| POST | `/horizon/goal/<id>/complete` | Mark goal completed `{"notes":"…"}` |
| POST | `/horizon/goal/<id>/abandon` | Abandon goal `{"reason":"…"}` |
| POST | `/horizon/goal/<id>/context` | Deposit facts into accumulated context `{key: value, …}` |
| GET | `/organs` | List loaded organ intents and descriptions |
| GET | `/organ_bus/history` | Recent organ bus call history |
| GET | `/organ_bus/subscribers` | Active organ bus subscribers |

### Memory & Perception

| Method | Route | Description |
|---|---|---|
| GET | `/memory/search?q=query&n=5` | Search long-term memory |
| POST | `/memory/ingest` | Add `{"content":"...","source":"note"}` |
| GET | `/perception/status` | Active perception channels |
| POST | `/perception/ingest` | Inject biometric data |
| GET | `/proactive` | Pending proactive events |

### Background Tasks

| Method | Route | Description |
|---|---|---|
| GET | `/tasks?n=10` | Recent background tasks |
| GET | `/tasks/<id>` | Single task progress |

---

## How the learning loop works

1. User sends a message → `PrismAgent` routes the intent
2. If a decision is produced (sport moment, domain, plan), it is saved as `_last_decision`
3. User gives feedback: "that was too aggressive" / "good call"
4. `PrismCalibration.detect()` classifies the direction
5. `PrismCalibration.process()` adjusts the factor weight via `AdaptiveFulcrum.observe()`
6. The adjustment is persisted to `~/.prism/calibration.db` — survives restarts
7. Future decisions for the same domain use the updated weights

Proactive calibration prompts fire every 3 days if no feedback has been given.

---

## Organ system

### ORGAN_POLICY — per-organ risk declarations

Every bundled organ in `organs/` may declare an `ORGAN_POLICY` dict at module level:

```python
ORGAN_POLICY = {
    "risk_level":        "low",   # "low" | "medium" | "high"
    "requires_approval": False,   # prompt user before executing?
    "irreversible":      False,   # cannot be undone?
    "max_per_session":   None,    # integer cap, or None for unlimited
}
```

`prism_organ_loader.py` reads this dict when loading each organ and makes it available via `get_organ_policy(intent)`.  `prism_chain.py` enforces the policy (blocks high-risk actions, writes to the audit log, enforces session caps).

### Bundled policy organs

| Organ intent | Module | Purpose |
|---|---|---|
| `task_reminder` | `organs/task_reminder.py` | Show overdue/due-today tasks; add new reminders with optional due date |
| `policy_audit` | `organs/policy_audit.py` | Query the SQLite audit log (`~/.prism/prism_audit.db`) for recent policy decisions |
| `policy_inspect` | `organs/policy_inspect.py` | Dump the declared `ORGAN_POLICY` for every currently loaded organ |
| `policy_update` | `organs/policy_update.py` | Update a loaded organ's live policy at runtime (risk_level, approval flag, session cap) |

### OrganBus

`OrganBus` (in `prism_organ_bus.py`) is an LLM-mediated publish/subscribe bus that lets PRISM's internal engines communicate without knowing each other's data schemas.  An engine emits an `OrganSignal`; the bus uses an LLM to translate the signal payload into each subscriber's vocabulary before delivery.

```python
bus = OrganBus(llm=my_llm)
bus.subscribe("policy", policy_engine.handle)
bus.publish(OrganSignal(source="physics", signal_type="injury_risk_elevated",
                        payload={"risk": 0.78, "muscle_group": "hamstring"}))
```

REST endpoints: `GET /organ_bus/history`, `GET /organ_bus/subscribers`.

### PrismSoul

`PrismSoul` (in `prism_soul.py`) is the persistent identity layer.  It maintains a *belief graph* — observed vs stated values for attributes like stress, focus, and energy — and injects a compact identity context string into every LLM prompt so responses stay consistent with the user's current state.

Key concepts:
- **Belief graph** — stores `(attribute, stated_value, observed_value)` triples in SQLite
- **Lenses** — user-defined value filters (e.g. "I prioritise recovery over training load")
- **Delta signal** — fires when stated and observed values diverge significantly

### Identity ceremony (`prism_identity_ceremony.py`)

Run on first boot via `python3 prism_daemon.py --ceremony`.  Guides the user through a 7-question onboarding flow (LLM-facilitated or heuristic fallback) and seeds the `PrismSoul` belief graph with initial values.

### prism_daemon

`prism_daemon.py` is the background process that keeps PRISM alive between chat sessions.  It runs on a configurable tick (default 60 s) and on each tick:

1. Flushes pending `OrganBus` signals
2. Evaluates `HorizonGoal` conditions (fires task queue entries when conditions are met)
3. Checks proactive triggers (calendar, reminders, calibration prompts)

Systemd-compatible — exits cleanly on SIGTERM.  Run with `--daemon` to detach, `--ceremony` to trigger identity onboarding.

---

## Project structure

```
PRISM/
├── Core engine
│   ├── decision_spectrum.py    DecisionBeam, Factor, AdaptiveFulcrum
│   ├── ksa_lever.py            ThreeBarSystem — original physics layer
│   └── ksa_registry.py         SnapshotRegistry — versioned SQLite store
│
├── KSA — developer agent
│   ├── ksa_agent.py            KSAgent — task routing + execution
│   ├── ksa_executor.py         FileIndex, Search, Shell executors
│   ├── ksa_jarvis.py           Jarvis — artifact memory + learning
│   ├── ksa_router.py           MasterFulcrum intent router
│   ├── ksa_fixes.py            LiveWeightInjector, GroundTruthOptimizer
│   ├── ksa_cli.py              CLI entry point
│   └── ksa_config.py           Config loader
│
├── KDE platform
│   ├── kde_agent.py            KDEAgent — unified sports + domain agent
│   ├── kde_server.py           Local REST API (stdlib http.server)
│   ├── prism_pwa.py            PWA mobile companion — installable app at /mobile
│   ├── kde_dashboard.py        HTML reports + terminal dashboard
│   ├── kde_cli.py              CLI entry point
│   ├── kde_config.py           Config loader
│   ├── kde_profiles.py         Profile catalogue and role defaults
│   └── kde_ui.py               SPA served at localhost:8742
│
├── Sport intelligence
│   ├── sport_spectrum.py       SportConfig, DuelModel, ALL_SPORTS
│   ├── sports_pro.py           SportsProAssistant, DailyPlanner
│   ├── daily_workflow.py       Morning briefing, session log, evening review
│   ├── prediction_engine.py    Match, injury, performance, transfer predictions
│   ├── duel_analyzer.py        1v1 duel network from match events
│   ├── moment_analyzer.py      Real-time moment analysis, ALL_MOMENT_CONFIGS
│   ├── moment_configs_ext.py   Extended sport moment configs
│   ├── moment_pipeline.py      StatsBomb batch + live tracking pipeline
│   ├── moment_validator.py     Season-scale accuracy validation
│   └── sport_data.py           StatsBomb open-data connector
│
├── Device integration
│   ├── device_hub.py           GoPro, Apple Health, Garmin, Whoop, Oura
│   ├── media_processor.py      Video/image pipeline (ffmpeg + Pillow)
│   └── vision_analyzer.py      Local vision AI via Ollama LLaVA
│
├── PRISM chat + identity
│   ├── prism_agent.py          Unified PRISM orchestration layer (four-tier routing)
│   ├── prism_chat.py           Local chat interface and UI payloads
│   ├── prism_responses.py      Response formatting helpers
│   ├── prism_perception.py     Perceptual context engine — time, location, device state
│   ├── prism_memory.py         Short- and long-term memory store
│   ├── prism_planner.py        Goal decomposition and multi-step planning
│   ├── prism_llm_router.py     LLM routing (Ollama / Claude API / OpenAI-compat)
│   ├── prism_tts.py            Text-to-speech output layer
│   ├── prism_voice.py          Speech-to-text input (Whisper / faster-whisper / SR)
│   ├── prism_proactive.py      Proactive trigger evaluation and scheduling
│   ├── prism_smart_home.py     Smart-home device command layer
│   ├── prism_task_queue.py     Async task queue for background execution
│   ├── prism_calibration.py    Conversational feedback → model adjustment
│   ├── digital_identity.py     User identity state and profile signals
│   ├── identity_bus.py         Cross-module identity event bus
│   └── artifact_store.py       Artifact collection with identity tagging
│
├── Orchestration & reasoning chains
│   ├── prism_orchestrator.py   ChainOrchestrator — TaskGraph DAG, 5 profiles, cross-session pause
│   ├── prism_chain.py          General alternating LLM→Logic+Policy→Evaluator chain
│   ├── prism_chain_expert.py   Expert chain — Router/Evaluator/BranchJudge/Synthesiser
│   ├── prism_chain_bench.py    Benchmark: general vs expert, mock + live modes
│   └── prism_composer.py       Static DAG composer for multi-step requests
│
├── Autonomous execution
│   ├── prism_autonomous.py     Tool synthesis (AST safety + subprocess sandbox + cache)
│   ├── prism_horizon.py        Cross-session long-horizon goal persistence (SQLite)
│   ├── prism_outcome_tracker.py Bayesian belief updates from task outcomes
│   ├── prism_organ_bus.py          LLM-mediated pub/sub bus between PRISM logic engines
│   ├── prism_organ_bus_experiment.py  Experimental organ bus extensions
│   └── organs/                 Bundled organ modules
│       ├── currency_convert.py     Currency conversion via live exchange rates
│       ├── weather_check.py        Current weather for any city
│       ├── finance_summary.py      Local CSV/JSON ledger summariser
│       ├── document_read.py        Local document (markdown/txt) reader
│       ├── meeting_brief.py        Pre-meeting brief from calendar details
│       ├── health_summary.py       Health metrics summariser (steps, sleep, HRV)
│       ├── task_reminder.py        Surface overdue/due-today tasks; add new reminders
│       ├── email_send.py           Send email — LLM-parsed, contact-resolved, approval-gated
│       ├── calendar_write.py       Create calendar events or find free slots
│       ├── phone_call.py           Outbound voice call or SMS via Twilio
│       ├── policy_audit.py         Query the policy audit log (SQLite)
│       ├── policy_inspect.py       Dump ORGAN_POLICY for every loaded organ
│       └── policy_update.py        Update a live organ's policy at runtime
│
├── Personal assistant
│   ├── prism_email.py          IMAP/SMTP email reader and sender
│   ├── prism_calendar.py       Calendar event management (CalDAV + iCal + Google)
│   ├── prism_search.py         Web search (Brave / SerpAPI / DuckDuckGo)
│   ├── prism_push.py           Push notifications via ntfy.sh
│   ├── prism_contacts.py       Contact management (local SQLite + Google)
│   ├── prism_tasks.py          Task management (local + Todoist + GitHub)
│   ├── prism_browser_agent.py  Headless web navigation and scraping
│   ├── prism_device_agent.py   On-device task execution (files, shell, apps)
│   ├── prism_device_executor.py Safe subprocess and file-system executor
│   ├── prism_device_resolver.py App and tool resolver for installed software
│   └── prism_device_scanner.py Installed-app and capability scanner
│
├── Execution intelligence
│   ├── prism_policy.py         Resource allocation + policy engine
│   ├── prism_tool_finder.py    Alternative execution path discovery
│   ├── prism_collaborator.py   Claude/Ollama research + tool synthesis
│   ├── prism_executor_agent.py Agentic execution with tool registry + sandboxing
│   ├── prism_instructions.py   Standing instructions — rules taught once, applied always
│   └── prism_service_discovery.py Universal handler for unknown services
│
├── Sport task executors
│   ├── sport_executor.py       Video analysis, highlight reel, reports
│   └── sport_tasks.py          Training plan, scouting, nutrition, social
│
├── Domain framework
│   ├── domain_configs.py       Medical · Financial · Legal · HR · Supply Chain · Climate
│   └── domain_validator.py     Expert-label accuracy validation
│
└── tests/                      1131 pytest tests — all passing
```

---

## Validated sports domains

| Sport | Configured moments |
|---|---|
| Football | 1v1 keeper · winger cross · penalty |
| Basketball | Drive to basket · isolation · pick-roll · post-up · fast break |
| Tennis | Serve (deuce) · serve (ad) · baseline rally · net approach |
| Rugby Union | Ball carrier contact · breakdown · lineout |
| Boxing | In range · counter |
| MMA | Clinch · ground top position |
| Wrestling | Takedown attempt |
| Cricket | Batting delivery |

**Validation**: 2,732 shot moments analysed against 10 La Liga seasons (2004–2018, StatsBomb open data). 100% model–player action agreement.

---

## Running the tests

```bash
python -m pytest tests/ -q
# 1282+ tests pass in ~115 seconds

# With coverage report:
python -m pytest tests/ -q --cov=. --cov-report=term-missing:skip-covered
```

---

## Extending PRISM

### Adding a new sport moment config

```python
# In moment_configs_ext.py or a new file:
from moment_analyzer import MomentSportConfig, MomentOption

MY_SPORT_CONFIG = MomentSportConfig(
    sport="Volleyball",
    moment_type="spike",
    options=[
        MomentOption("cross_court", position=0.2, ev=0.7),
        MomentOption("line",        position=0.8, ev=0.6),
    ],
    bandwidth=0.3,
)
# Register in ALL_MOMENT_CONFIGS and it's live in the API.
```

### Adding a new domain

```python
# In domain_configs.py:
from domain_configs import DomainConfig, DomainPlank, DomainFactor, DomainProfile

MY_DOMAIN = DomainConfig(
    domain="Cybersecurity",
    planks=[
        DomainPlank("immediate_patch", position=0.1, description="Patch now"),
        DomainPlank("monitor_watchlist", position=0.5, description="Monitor"),
        DomainPlank("defer", position=0.9, description="Low risk, defer"),
    ],
    factors=[
        DomainFactor("severity", weight=1.0),
        DomainFactor("exposure", weight=0.8),
    ],
    profiles=[
        DomainProfile("Production System", fixed_fulcrum=0.2),
        DomainProfile("Dev Environment",   fixed_fulcrum=0.6),
    ],
)
ALL_DOMAINS["Cybersecurity"] = MY_DOMAIN
```

### Adding a new executor (KSA)

```python
from ksa_executor import BaseExecutor

class MyToolExecutor(BaseExecutor):
    def execute(self, task: str, params: dict) -> dict:
        # implement
        return {"output": "done", "success": True}

agent.register("my_tool", ["my", "tool", "keywords"],
               MyToolExecutor(), description="My custom tool")
```

---

## What is still missing for a full local personal assistant

All major gaps from the initial build have been bridged. The table below reflects the current state:

| Capability | Status | Notes |
|---|---|---|
| Voice input (Whisper) | **Working** | `prism_voice.py` — local Whisper; `pip install openai-whisper` |
| LLM fallback (Claude API) | **Working** | `prism_llm_router.py` — auto-falls back when Ollama unavailable |
| Google Calendar OAuth | **Working** | Set `[calendar] google_token` in config |
| Contact auto-extraction | **Working** | LLM extracts contacts from memory entries when Ollama available |
| Linear task integration | **Working** | GraphQL API via `[tasks] linear_api_key` |
| Scheduled reminders | **Working** | "remind me in 30 mins" → `PrismProactive.schedule_in()` |
| Multi-user support | **Working** | Scoped by `[user].name` in config; run separate instances for isolation |
| Adaptive reasoning chains | **Working** | LLM↔Logic+Policy alternating spine with Evaluator quality gate |
| Autonomous tool synthesis | **Working** | AST safety + subprocess sandbox + pip auto-install + cache |
| iOS / Android companion | **Working (PWA)** | `prism_pwa.py` — installable PWA at `/mobile`; push via ntfy.sh; no app store needed |
| Token refresh for Google OAuth | **Working** | Auto-refresh via `google_creds.json` — stores `access_token`, `refresh_token`, `client_id`, `client_secret`, `expiry` |
| Horizon goals | `prism_horizon.py` | **Working** — cross-session goal watching; say "watch for X when Y" in chat |
| Organ registry | `prism_organ_loader.py` | **Working** — synthesised tools persist; say "what organs do you have" |
| Identity layer | `prism_soul.py` | Working — belief graph, user-defined lenses, stated vs observed delta, LLM context injection |
| Identity ceremony | `prism_identity_ceremony.py` | Working — 7-question LLM-facilitated onboarding, heuristic fallback |
| Continuous daemon | `prism_daemon.py` | Working — systemd-compatible, OrganBus flush, horizon evaluation, --ceremony flag |

---

## Docker

```bash
docker build -t prism .
docker run -p 8742:8742 prism
```

---

## Dependencies

```
Python 3.11+
psutil          resource monitoring
Pillow          image processing
pytest          testing
ruff            linting
```

Optional:
```
ffmpeg          video processing (brew/apt install ffmpeg)
Ollama + mistral  LLM routing (https://ollama.ai)
Ollama + llava    vision analysis (ollama pull llava)
playwright        browser automation (pip install playwright && playwright install chromium)
```

No numpy · no torch · no langchain · no openai required. All decision mathematics is pure Python arithmetic.

---

## License

MIT
