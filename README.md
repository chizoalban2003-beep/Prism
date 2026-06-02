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
User input (chat / CLI / REST API)
         │
         ▼
  ┌─────────────────────────────────────────────┐
  │  PrismAgent  (prism_agent.py)               │
  │  Intent routing · standing instructions     │
  │  Chat history · memory injection            │
  └──────────┬──────────────────────────────────┘
             │
    ┌────────▼────────┐       ┌─────────────────┐
    │  KDEAgent       │       │  KSAgent         │
    │  Sport + Domain │       │  Developer tasks │
    └────────┬────────┘       └────────┬────────┘
             │                         │
    ┌────────▼─────────────────────────▼────────┐
    │        Decision Engine                     │
    │  decision_spectrum.py                      │
    │  p = Σ(w·v·t)/Σ(w·v)  ← fulcrum          │
    │  activation = Gaussian kernel over options │
    │  AdaptiveFulcrum.observe() ← online learn  │
    └────────────────────────────────────────────┘

Personal Assistant Layer (all local):
  ┌──────────┐ ┌──────────┐ ┌──────────────┐ ┌────────────┐
  │  Email   │ │ Calendar │ │ Web Search   │ │  Contacts  │
  │ IMAP/SMTP│ │ CalDAV   │ │ Brave/DDG    │ │  SQLite    │
  └──────────┘ └──────────┘ └──────────────┘ └────────────┘
  ┌──────────┐ ┌──────────┐ ┌──────────────┐ ┌────────────┐
  │  Tasks   │ │ SmartHome│ │ Push (ntfy)  │ │  Browser   │
  │ Todoist/ │ │ Home Asst│ │  free, local │ │ Playwright │
  │ GitHub/  │ │  REST    │ │              │ │            │
  │  Local   │ └──────────┘ └──────────────┘ └────────────┘
  └──────────┘

Background loop:
  PrismProactive  →  triggers (calendar, budget, recovery, calibration)
  PrismMemory     →  short/long-term memory store (SQLite + TF-IDF)
  PrismPerception →  context (time, biometrics, system state)
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
| Email read/send | `prism_email.py` | Working (needs config) |
| Calendar read/write (CalDAV/iCal) | `prism_calendar.py` | Working (needs config) |
| Calendar read/write (Google) | `prism_calendar.py` | Working (needs OAuth token) |
| Web search | `prism_search.py` | Working (DDG free; Brave/Serp optional) |
| Push notifications | `prism_push.py` | Working (ntfy.sh, free) |
| Contacts | `prism_contacts.py` | Working (local + Google optional) |
| Tasks | `prism_tasks.py` | Working (local + Todoist/GitHub/Linear) |
| Smart home | `prism_smart_home.py` | Working (Home Assistant) |
| Browser automation | `prism_browser_agent.py` | Working (needs playwright) |
| Device tasks | `prism_device_agent.py` | Working |
| Memory | `prism_memory.py` | Working (SQLite + TF-IDF) |
| Standing instructions | `prism_instructions.py` | Working |
| Proactive triggers | `prism_proactive.py` | Working |
| Scheduled reminders | `prism_proactive.py` | Working |
| Wearable sync trigger | `prism_proactive.py` | Working |
| TTS | `prism_tts.py` | Working (espeak fallback) |
| Voice input (Whisper) | `prism_perception.py` | Working (opt-in; needs pyaudio + openai-whisper) |
| LLM routing (Ollama) | `prism_llm_router.py` | Working |
| LLM routing (Claude API) | `prism_llm_router.py` | Working (needs API key) |
| Multi-user | `prism_agent.py` | Working (`[user].name` in config) |
| Unknown-tool PA fallback | `prism_agent.py` | Working (discovers + plans integrations) |
| Autonomous tool synthesis | `prism_autonomous.py` | Working — synthesises, installs, executes, caches |

---

## Autonomous Execution

PRISM is a managerial PA with full autonomy. When asked to do something it has no built-in tool for, instead of returning instructions to the user it:

1. **Synthesises a Python tool on demand** — the LLM writes a self-contained `execute(task, params) -> str` module
2. **Safety-checks the code** — a strict pattern blocklist rejects any code containing `eval`, `exec`, `os.system`, `shutil.rmtree`, `socket.connect`, or unguarded file writes
3. **Installs pip dependencies** — any required packages are installed automatically with `pip install --quiet`
4. **Executes and returns the result** — the module is dynamically loaded and run in-process
5. **Caches the tool** — synthesised tools are stored as JSON in `~/.prism/tools/` and reused for identical or similar future tasks
6. **Push-notifies on completion** — if push is configured, you get a notification on your phone when done

### Approval gate

Actions that may affect external systems (send emails, make purchases, delete data) are held for one-time approval before executing:

```
You: order 10 roses from florist.com
PRISM: I can do this, but it involves external purchase which may
       affect external systems.
       Say 'yes, go ahead' to authorise, or 'cancel' to stop.
You: yes, go ahead
PRISM: Approved. Executing autonomously. Task ID: `a3f8b2c1`
       I'll notify you when done.
```

Approval is remembered per-conversation — once approved, PRISM acts. `cancel` or `no` drops the pending task with no side effects.

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

## Voice input setup

PRISM supports local speech-to-text via OpenAI Whisper (no cloud):

```bash
pip install openai-whisper pyaudio
```

Then in `prism_config.toml`:

```toml
[agent]
enable_voice = true
```

Say "Hey Prism, ..." — PRISM transcribes locally and routes the command. Falls back gracefully if pyaudio or whisper are not installed.

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

PRISM reads/writes your primary calendar. Token refresh is not automatic — refresh via your OAuth flow and update the config.

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
│   ├── prism_agent.py          Unified PRISM orchestration layer
│   ├── prism_chat.py           Local chat interface and UI payloads
│   ├── prism_responses.py      Response formatting helpers
│   ├── prism_perception.py     Perceptual context engine — time, location, device state
│   ├── prism_memory.py         Short- and long-term memory store
│   ├── prism_planner.py        Goal decomposition and multi-step planning
│   ├── prism_llm_router.py     Local LLM routing via Ollama
│   ├── prism_tts.py            Text-to-speech output layer
│   ├── prism_proactive.py      Proactive trigger evaluation and scheduling
│   ├── prism_smart_home.py     Smart-home device command layer
│   ├── prism_task_queue.py     Async task queue for background execution
│   ├── prism_calibration.py    Conversational feedback → model adjustment
│   ├── digital_identity.py     User identity state and profile signals
│   ├── identity_bus.py         Cross-module identity event bus
│   └── artifact_store.py       Artifact collection with identity tagging
│
├── Personal assistant
│   ├── prism_email.py          IMAP/SMTP email reader and sender
│   ├── prism_calendar.py       Calendar event management (CalDAV + iCal)
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
└── tests/                      769 pytest tests — all passing
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
# 769 tests pass in ~80 seconds
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

| Gap | Status | Notes |
|---|---|---|
| Voice input (Whisper) | Not wired | `prism_tts.py` exists; Whisper import/pipeline missing from `prism_agent.py`. Install `openai-whisper` and wire `PrismPerception` voice channel. |
| LLM (without Ollama) | Partial | Works with Ollama; Claude API key field exists but not fully exercised |
| Google Calendar OAuth | Not implemented | CalDAV + iCal URL work; full Google OAuth flow needs `google-auth` library |
| Contact auto-extraction | Partial | Manual add + Google sync work; LLM extraction from memory needs Ollama |
| Linear task integration | Stub | API key field exists in config; `_resolve_provider()` returns "linear" but no API calls implemented |
| Notification scheduling | Partial | Push works on-demand; cron-style "remind me at 3pm" not wired to a scheduler |
| Multi-user support | Not implemented | All state is single-user; `_user` defaults to `"default"` |
| iOS / Android companion | Not implemented | Push via ntfy.sh works; native app would enable bidirectional |

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
