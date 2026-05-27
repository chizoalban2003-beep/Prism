# KSA — Kinetic State Agent

A local-first, hardware-native AI agent that uses a **physics simulation metaphor** (levers, fulcrums, torque) to make routing and resource-allocation decisions — without a neural network.

---

## How it works

Every decision passes through a cascade of three mechanical levers:

```
Input Prompt
     │
     ▼
MasterFulcrum (Router)
     │  keyword scoring → alias → LLM fallback → bootstrap
     ▼
ThreeBarSystem (Physics Engine)
  Lever 0 — Input Lever   (raw weighted inputs)
  Lever 1 — Logic Lever   (constraint bias via fulcrum offset)
  Lever 2 — Balancer Bar  (safety monitor; can override cascade)
     │
     ▼
EquilibriumResult (tilt=LEFT/RIGHT/BALANCED, confidence 0–1)
     │
     ▼
ExecutorRegistry → primary / secondary / safe action
     │
     ▼
SnapshotRegistry (SQLite) + KineticOptimizer (hill-climb)
```

After each successful run the optimizer perturbs the lever geometry with Gaussian noise and saves an improved snapshot if confidence increases. This creates a self-tuning feedback loop with no gradient descent.

---

## Project structure

```
KSA/
├── ksa_lever.py      3-bar lever physics engine + snapshot serialisation
├── ksa_registry.py   SQLite-backed snapshot registry + version control
├── ksa_router.py     MasterFulcrum — intent router (keyword / alias / LLM)
├── ksa_executor.py   Hardware execution layer (file index, search, shell)
├── ksa_optimizer.py  Kinetic optimizer — gradient-free hill-climbing
├── ksa_agent.py      Top-level orchestrator wiring all layers
├── ksa_cli.py        Command-line interface entry point
├── ksa_config.py     Config loader (TOML / JSON, with sensible defaults)
├── requirements.txt  Python dependencies
└── tests/
    ├── test_lever.py
    ├── test_registry.py
    ├── test_router.py
    ├── test_executor.py
    └── test_optimizer.py
```

---

## Installation

```bash
pip install -r requirements.txt
```

> **Python 3.9+** required.  
> TOML config files additionally require Python 3.11+ (stdlib `tomllib`) or the `tomli` back-port (`pip install tomli`).

---

## Quick start

### Run from the CLI

```bash
# Route a natural-language prompt through the full pipeline
python ksa_cli.py run "quietly scan my project folder in the background"

# Show all registered tasks and snapshot versions
python ksa_cli.py status

# View version history for a task
python ksa_cli.py history file_index_stealth

# Roll back a task to the previous snapshot
python ksa_cli.py rollback file_index_stealth

# Promote a specific version to current
python ksa_cli.py promote file_index_stealth 3

# Remove old snapshots, keeping the 5 most recent
python ksa_cli.py prune file_index_stealth --keep 5

# Delete all snapshots for a task
python ksa_cli.py delete file_index_stealth

# Load a ThreeBarSystem state from a JSON snapshot file
python ksa_cli.py snapshot load my_snapshot.json --task-name my_task
```

### Global flags

| Flag | Description |
|---|---|
| `--db PATH` | SQLite database path (default: `~/.ksa/state.db`) |
| `--config PATH` | TOML or JSON config file path |
| `--dry-run` | Run without touching the filesystem or shell |
| `--verbose` / `-v` | Enable DEBUG logging |
| `--quiet` / `-q` | Suppress all output except errors |

### Use as a library

```python
from ksa_agent import KSAgent
from ksa_executor import FileIndexExecutor, LocalSearchExecutor, ShellExecutor

agent = KSAgent(db_path="~/.ksa/state.db", dry_run=True, auto_optimise=True)

agent.register(
    task_name   = "file_index_stealth",
    keywords    = ["index", "scan", "files", "folder", "background"],
    executor    = FileIndexExecutor(),
    aliases     = ["index"],
    description = "Background file indexing",
)

outcome = agent.run("quietly scan my project folder")
print(outcome)
```

---

## Configuration

KSA looks for a config file in this order:

1. Explicit `--config` flag
2. `$KSA_CONFIG` environment variable
3. `~/.ksa/config.toml`
4. `~/.ksa/config.json`
5. `./ksa_config.toml`
6. `./ksa_config.json`

Falls back to built-in defaults if none is found.

### Example `ksa_config.toml`

```toml
db_path       = "~/.ksa/state.db"
working_dir   = "."
dry_run       = false
auto_optimise = true
confidence_floor = 0.25

# Optional: Ollama LLM fallback resolver
# ollama_model = "mistral"
# ollama_host  = "http://localhost:11434"

[[tasks]]
task_name   = "file_index_stealth"
keywords    = ["index", "scan", "files"]
aliases     = ["index"]
description = "Background file indexing"
executor    = "FileIndexExecutor"
```

---

## Built-in executors

| Executor | `task_name` | `primary` | `secondary` | `safe` |
|---|---|---|---|---|
| `FileIndexExecutor` | `file_index_stealth` | `nice -n 19 find … -type f` | `find . -maxdepth 1 -type f` | no-op |
| `LocalSearchExecutor` | `local_search` | `grep -rl QUERY DIR` | `find . -name '*QUERY*'` | return cached index |
| `ShellExecutor` | `shell_generic` | run command as-is | `timeout 5 COMMAND` | no-op |

The tilt direction routes execution:

- `LEFT` → `primary`
- `RIGHT` → `secondary`
- `BALANCED` or `override_active` → `safe`

---

## Snapshot registry

All lever configurations are stored as versioned snapshots in SQLite.  
Each successful run can produce an improved snapshot via the `KineticOptimizer`.

```
registry.save(task_name, system)           → version int
registry.load(task_name)                   → ThreeBarSystem  (current best)
registry.promote(task_name, version)       → mark a version as current
registry.rollback(task_name)               → revert to previous version
registry.best_version(task_name)           → version with best avg score
registry.prune(task_name, keep=5)          → trim old versions
registry.delete_task(task_name)            → remove all versions
```

---

## Running the tests

```bash
pytest tests/ -v
```

All 100 tests should pass in < 1 second.

---

## Design principles

| Principle | Implementation |
|---|---|
| **Transparency** | Every decision maps to an inspectable `(W, F, L)` vector stored in SQLite |
| **Low footprint** | Decision core is pure math — no LLM on the hot path |
| **Self-optimisation** | Hill-climbing optimizer updates lever geometry after each run |
| **Surgical LLM use** | Ollama called only when keyword confidence < floor (default 25%) |
| **Safety** | Lever 2 (Balancer) can override the cascade if instability is detected |
