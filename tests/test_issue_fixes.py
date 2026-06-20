"""Regression tests for issues found during the live infrastructure shakedown."""
from __future__ import annotations

import textwrap

from ksa_agent import KSAgent
from ksa_executor import LocalSearchExecutor
from ksa_registry import SnapshotRegistry
from prism_organ_planner import compose, execute_plan

# ── ksa: SnapshotRegistry first-run on a fresh (non-existent) home ──────────────

def test_snapshot_registry_creates_missing_parent(tmp_path):
    """`ksa status` on a clean home used to crash with 'unable to open database
    file' because the parent dir didn't exist. The registry must create it."""
    db = tmp_path / "does" / "not" / "exist" / "ksa_state.db"
    reg = SnapshotRegistry(str(db))          # must not raise
    assert db.parent.exists()
    assert reg.list_tasks() == []


# ── ksa: KSAgent.run threads the prompt into the executor payload ───────────────

def test_ksagent_run_threads_query_into_payload(tmp_path):
    """LocalSearchExecutor used to always get an empty payload and return
    'No query provided in ctx.payload[query]'. run() must populate payload."""
    agent = KSAgent(db_path=str(tmp_path / "ksa.db"), working_dir=str(tmp_path),
                    auto_optimise=False, dry_run=True)
    agent.register(
        task_name="local_search",
        keywords=["search", "find", "locate", "grep"],
        executor=LocalSearchExecutor(),
        aliases=["search", "find"],
    )
    outcome = agent.run("find budget files")
    assert outcome.return_code == 0
    assert "No query provided" not in (outcome.stderr or "")


# ── organ planner: empty plan vs cycle are distinct messages ────────────────────

class _StubLoader:
    def __init__(self, schemas):
        self._s = schemas

    def get(self, intent):
        return self._s.get(intent, {}).get("fn")

    def get_organ_schema(self, intent):
        e = self._s.get(intent, {})
        return {"inputs": e.get("inputs", {}), "outputs": e.get("outputs", {})}


def test_execute_plan_empty_message():
    loader = _StubLoader({})
    plan = compose(loader, [])               # no valid intents → empty plan
    result = execute_plan(loader, plan)
    assert result["executed"] == 0
    assert "empty plan" in result["errors"]["_plan"]
    assert "cycle" not in result["errors"]["_plan"]


def test_execute_plan_cycle_message():
    ident = lambda i, m, c: {"v": 1}         # noqa: E731
    loader = _StubLoader({
        "a": {"inputs": {"v": "int"}, "outputs": {"v": "int"}, "fn": ident},
        "b": {"inputs": {"v": "int"}, "outputs": {"v": "int"}, "fn": ident},
    })
    plan = compose(loader, ["a", "b"])       # mutual arrows → cycle
    result = execute_plan(loader, plan)
    assert result["executed"] == 0
    assert "cycle" in result["errors"]["_plan"]


# ── autonomous: AST safety is single-sourced from the organ loader (strict) ─────

def test_autonomous_safety_blocks_file_writes():
    from prism_autonomous import _is_safe_code
    code = textwrap.dedent("""
        from pathlib import Path
        def execute(task, params):
            Path('/tmp/x').write_text('data')
            return 'done'
    """)
    safe, reason = _is_safe_code(code)
    assert not safe
    assert "write_text" in reason


def test_autonomous_safety_allows_str_replace():
    # str.replace() is safe and was previously over-blocked by the duplicate
    # checker; the single-sourced loader allows it.
    from prism_autonomous import _is_safe_code
    code = "def execute(task, params):\n    return task.replace('a', 'b')"
    safe, _ = _is_safe_code(code)
    assert safe


def test_autonomous_safety_still_blocks_eval_and_os():
    from prism_autonomous import _is_safe_code
    assert not _is_safe_code("def execute(t, p):\n    return eval(t)")[0]
    assert not _is_safe_code("import os\ndef execute(t, p):\n    os.system('ls')")[0]


# ── daemon wires the previously-unreachable subsystems into ASGI state ──────────

def test_build_asgi_state_wires_subsystems():
    """causality / multi-user / mobile routes used to 503 in a live daemon
    because _build_asgi_state never injected their dependencies."""
    import prism_daemon
    from prism_agent import PrismAgent
    agent = PrismAgent()
    state = prism_daemon._build_asgi_state(agent)
    for key in ("causal_reasoner", "user_registry", "household_bus", "mobile_sync"):
        assert state.get(key) is not None, f"{key} not wired into ASGI state"
