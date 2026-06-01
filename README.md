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
  <img src="https://github.com/chizoalban2003-beep/Prism/actions/workflows/ci.yml/badge.svg" alt="CI"/>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python"/>
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="MIT"/>
  <img src="https://img.shields.io/badge/cloud-none-orange" alt="no cloud"/>
  <img src="https://img.shields.io/badge/runs-locally-orange" alt="local"/>
</p>

<p align="center">
  <strong>Domains:</strong>
  sport &nbsp;·&nbsp; medical &nbsp;·&nbsp; legal &nbsp;·&nbsp; financial &nbsp;·&nbsp;
  HR &nbsp;·&nbsp; supply chain &nbsp;·&nbsp; climate &nbsp;·&nbsp; developer tools
</p>

---

## What it is

PRISM is a local personal AI assistant that decides, plans, and acts
for any user across any domain — all on your own hardware.

It combines three things no existing tool does simultaneously:

**A physics-based decision engine** that produces interpretable,
personalised recommendations with named causes — not black-box predictions.

**A full execution layer** that carries out approved actions, finds tools
when they don't exist, and learns new integrations on demand.

**A continuous learning identity** that crystallises from your actual
decisions over time — becoming more accurate for you specifically,
not for a population average.

It works for developers, athletes, coaches, clinicians, analysts,
business owners, and anyone who makes consequential decisions regularly.
The same engine that advises a striker on shot selection advises a doctor
on triage urgency and a founder on go-to-market strategy — because all
three are bounded decisions under uncertainty with named contextual factors.

---

## Who it is for

| Role | What PRISM does |
|---|---|
| **Developer** | Routes tasks, manages system resources, learns your workflow via KSA |
| **Professional athlete** | Plans daily training load, manages recovery, analyzes session footage |
| **Coach** | Tactical preparation, squad load management, opposition scouting |
| **Sports analyst** | Duel network analysis, moment prediction, StatsBomb data pipeline |
| **Performance director** | Squad risk overview, transfer value estimation, season forecasting |
| **Enterprise (Medical / Financial / Legal)** | Domain-agnostic decision support with full audit trail |

---

## The engine

Every decision passes through the same three-step process regardless of domain:

```
Fixed fulcrum      = who the agent IS (profile, style, role)
Movable factors    = what the statistics say RIGHT NOW
Gaussian kernel    = probability distribution over options

p = Σ(w·v·t) / Σ(w·v)                 ← weighted centroid (fulcrum)
act_i = exp(-½((pos-p)/bw)²) / Σ(exp) ← normalised Gaussian activation
```

The engine learns online via `AdaptiveFulcrum.observe()` — no retraining,
no gradient descent, no cloud call. Factor weights drift toward the
configurations that produced the best real outcomes.

---

## Quick start

```bash
git clone https://github.com/chizoalban2003-beep/Prism.git
cd Prism
pip install -r requirements.txt
# Optional: ffmpeg (video), Ollama (local LLM fallback)
```

If you want installed CLI entry points instead of calling the Python files directly:

```bash
pip install .
prism --help
kde --help
ksa --help
```

### As a developer agent (KSA)

```bash
python ksa_cli.py run "quietly scan my project folder in the background"
python ksa_cli.py status
python ksa_cli.py history file_index_stealth
```

```python
from ksa_agent import KSAgent
from ksa_executor import FileIndexExecutor

agent = KSAgent(db_path="~/.ksa/state.db", auto_optimise=True)
agent.register("file_index_stealth", ["index","scan","files"],
               FileIndexExecutor(), description="Background file indexing")
outcome = agent.run("quietly scan my project folder")
```

### As a sports platform (KDE)

```bash
python kde_cli.py morning                         # daily briefing
python kde_cli.py ask "analyse my session footage"
python kde_cli.py ask "predict Manchester City vs Arsenal"
python kde_cli.py ask "assess my squad injury risk"
python kde_cli.py reflect                         # show what the agent learned
```

```python
from kde_agent import KDEAgent
from sports_pro import Role

agent = KDEAgent.setup(name="Marcus", role=Role.ATHLETE,
                       sport="Football", team="City FC")

# Morning planning from wearable data
brief = agent.morning_briefing(hrv_ms=58, sleep_hrs=6.8, soreness=3, energy=3)

# Match prediction
pred = agent.ask("predict next match vs Arsenal")

# Real-time moment analysis (1v1 keeper, defenders closing in 2.5s)
result = agent.ask("analyse moment: striker vs keeper, 2 defenders 2.5s away")
```

### As a domain decision platform

```python
from domain_configs import ALL_DOMAINS, DomainDecisionModel

# Medical triage
model   = DomainDecisionModel(ALL_DOMAINS["Medical"])
verdict = model.evaluate("Elderly (65+)", {
    "severity": 0.85, "vital_signs": 0.70, "deteriorating": 0.60
})
print(verdict.primary_plank.name)    # "Emergency A&E now"
print(verdict.risk_adjusted_return)  # urgency score

# Financial portfolio
fin_model = DomainDecisionModel(ALL_DOMAINS["Financial"])
portfolio = fin_model.evaluate("Young professional", {
    "time_horizon": 0.85, "risk_tolerance": 0.72, "market_conditions": 0.55
})
print(portfolio.primary_plank.name)  # "Equity focused"
```

---

## Configuration

PRISM/KDE detects your role from the first config file it finds in this order:
`--config` flag · `$KDE_CONFIG` env · `~/.kde/config.toml` · `~/.kde/kde.toml` · `./prism_config.toml` · `./kde_config.toml`

The repository includes `prism_config.toml` as a ready-to-edit sample.

```toml
[user]
role  = "athlete"          # developer | athlete | coach | analyst | universal
name  = "Marcus"
sport = "Football"
team  = "City FC"

[agent]
db_path      = "~/.kde/kde.db"
media_dir    = "~/.kde/media"
auto_watch   = true
ollama_model = "mistral"   # remove to disable LLM routing
ollama_host  = "http://localhost:11434"

[[devices]]
name       = "Apple Watch"
type       = "apple_watch"
watch_path = "~/Downloads/apple_health_export"

[[devices]]
name       = "GoPro Hero 12"
type       = "gopro"
watch_path = "~/GoPro/DCIM"
api_url    = "http://10.5.5.9:8080"
```

---

## REST API

Start the local server (binds to 127.0.0.1 only — never exposed externally):

```bash
python kde_cli.py server --port 8742
```

Key endpoints:

| Method | Route | Description |
|---|---|---|
| GET | `/status` | Agent status and loaded modules |
| GET | `/plan?date=…` | Today's daily plan |
| POST | `/ask` | Natural-language task |
| GET | `/predict/match?home=X&away=Y&…` | Match prediction |
| GET | `/predict/injury?name=X&recovery=0.7&…` | Injury risk |
| GET | `/moment/analyze?sport=Football&…` | Real-time moment analysis |
| POST | `/moment/calibrate` | Record outcome, trigger learning |
| POST | `/moment/live_frame` | Feed live tracking frame |
| GET | `/domain/evaluate?domain=Medical&…` | Domain decision |
| POST | `/domain/validate` | Validate against expert labels |
| GET | `/duel/network?match_id=…` | Match duel network |
| GET | `/reflect` | Learned state for current user |

---

## Validated domains

**Sports** (validated against 10 La Liga seasons, StatsBomb open data):

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
| *Any sport* | Add a `MomentSportConfig` — no engine changes needed |

**Enterprise domains** (same engine, different configuration):
Medical triage · Financial portfolio · Legal case strategy ·
HR hiring · Supply chain · Climate policy

**Daily life** (personal assistant mode):
Task planning · Email · Calendar · Device tasks · Smart home · Web navigation

---

## Project structure

```
PRISM/
│
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
│   ├── digital_identity.py     User identity state and profile signals
│   ├── identity_bus.py         Cross-module identity event bus
│   └── artifact_store.py       Artifact collection with identity tagging
│
├── Execution intelligence
│   ├── prism_policy.py         Resource allocation + policy engine (CEO→Manager)
│   ├── prism_tool_finder.py    Alternative execution path discovery
│   ├── prism_collaborator.py   Claude/Ollama research + tool synthesis
│   └── prism_executor_agent.py Agentic execution with tool registry + sandboxing
│
├── Sport task executors
│   ├── sport_executor.py       Video analysis, highlight reel, reports
│   └── sport_tasks.py          Training plan, scouting, nutrition, social
│
├── Domain framework
│   ├── domain_configs.py       Medical · Financial · Legal · HR · Supply Chain · Climate
│   └── domain_validator.py     Expert-label accuracy validation
│
├── Optimisation
│   └── ksa_optimizer.py        Snapshot and executor optimisation helpers
│
└── tests/                      Pytest suite covering the shipped modules
```

---

## How it flows

1. A user calls `prism`, `kde`, or `ksa` CLI (or the REST API, or Python API directly)
2. The relevant agent (`prism_agent.py`, `kde_agent.py`, or `ksa_agent.py`) receives natural-language input
3. Intent routing (`ksa_router.py` / KDE routing) maps the input to a registered executor or sport/domain module
4. The appropriate executor or domain model calls into the Gaussian kernel engine (`decision_spectrum.py`)
5. Results are returned as an inspectable decision distribution with named causes — nothing is a black box
6. The policy engine checks the recommended action against the user's
   resource allocation — budget, spending limits, preferred providers,
   blacklist, time window
7. If approved: the tool finder identifies how to execute (built-in tool,
   app, website, aggregator, phone, or synthesised integration)
8. The executor agent runs the action and logs it with a full audit trail
9. The artifact store saves the result, tagged to the current identity version
10. Outcomes are fed back via `AdaptiveFulcrum.observe()` to shift factor
    weights — the system gets more accurate for this specific user over time

---

## Validation — La Liga 10 seasons (2004–2018)

| Metric | Result |
|---|---|
| Shot moments analysed (in-box) | 2,732 |
| Model–player action agreement | 100% |
| Duel events extracted | 15,758 |
| Seasons covered | 10 |
| Data source | StatsBomb Open Data |

Next milestone: 200 expert-labeled decisions → measured accuracy on optimal/suboptimal binary classification (target: >60%).

---

## Docker

Run the full platform in one command — no local Python setup needed:

```
docker build -t prism .
docker run -p 8742:8742 prism
```

Then open http://localhost:8742 for the PRISM chat interface.

---

## Dependencies

```
Python 3.11+
psutil          resource monitoring
Pillow          image processing
```

Optional:
```
ffmpeg          video processing (brew/apt install ffmpeg)
Ollama + mistral  natural language routing (https://ollama.ai)
Ollama + llava    vision analysis (ollama pull llava)
```

No numpy · no torch · no langchain · no openai · no cloud services.
All decision mathematics is pure Python arithmetic.

---

## Running the tests

```bash
python -m pytest tests/ -q
# current suite passes locally in ~20 seconds
```

---

## The honest ceiling

This system matches situations to known patterns. It does not reason about novel problems outside its configured decision space. Its sports predictions need calibration by domain experts before professionals would trust them for high-stakes decisions. Complex non-linear factor interactions (high age AND high comorbidity) are not fully modelled yet — that requires the next engineering step of cross-factor interaction terms.

What it does reliably: produce interpretable, adaptive, locally-run decision distributions that get measurably better with use.

---

## License

MIT
