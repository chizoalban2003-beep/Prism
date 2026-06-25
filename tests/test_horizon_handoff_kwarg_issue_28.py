"""HorizonPlanner _execute_step kwarg fix for issue #28 bug 43.

Live test: every Horizon goal in /tasks landed in ``failed`` status with::

    HorizonPlanner._hand_off.<locals>._execute_step() got an unexpected
    keyword argument 'prompt'

prism_task_queue.py:98 invokes ``fn(**params)`` (kwargs unpacked), but
the closure was defined as ``_execute_step(params: dict)`` — a single
positional dict. The interface mismatch had silently failed every
horizon execution since the planner shipped on 2026-06-02.

Fix: redefine the closure to accept ``prompt`` (and any future
TaskQueue params) as kwargs, falling back to the closure-captured
description.
"""
from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestHandOffSignature:
    """Spot-check the closure signature without spinning a full planner."""

    def test_handoff_source_uses_kwarg(self):
        # The literal failure mode was: `def _execute_step(params: dict)`.
        # Make sure the closure signature accepts `prompt=` so that
        # TaskQueue's `fn(**params)` call shape doesn't blow up.
        source = (_PROJECT_ROOT / "prism_horizon.py").read_text()
        # Find the inner def line.
        idx = source.find("def _execute_step(")
        assert idx >= 0, "Expected `_execute_step` closure in HorizonPlanner._hand_off"
        sig_end = source.find(")", idx)
        signature = source[idx:sig_end + 1]
        assert "prompt" in signature, (
            f"signature {signature!r} must accept `prompt=` kwarg"
        )
        assert "params: dict" not in signature, (
            "signature still uses positional `params: dict` — TaskQueue "
            "calls fn(**params), which produces unexpected-keyword errors"
        )


class TestTaskQueueContract:
    """Pin the contract on the queue side too, so future refactors don't
    silently re-break the interface."""

    def test_taskqueue_unpacks_params(self):
        # Read prism_task_queue.py and confirm it still uses fn(**params).
        # If somebody changes the queue to pass params positionally, this
        # test catches it so the closure signature can be updated in lockstep.
        src = (_PROJECT_ROOT / "prism_task_queue.py").read_text()
        assert "fn(**params)" in src, (
            "TaskQueue must invoke steps as `fn(**params)`; if this changes, "
            "update HorizonPlanner._execute_step signature too."
        )


class TestHandOffClosureExecutes:
    """End-to-end: build a HorizonPlanner with stub deps and confirm the
    closure executes without raising the kwarg error."""

    def _make_planner(self, tmp_path):
        # Lazy import; the test relies on the planner module being importable.
        from prism_horizon import HorizonPlanner

        class _StubLLM:
            def call(self, prompt, **_):
                return f"executed:{prompt[:30]}", "stub"

        class _StubQueue:
            captured: list = []

            def submit(self, *, title, steps, on_complete=None):
                # Replay the real queue's invocation pattern.
                for step in steps:
                    fn = step["fn"]
                    params = step.get("params", {})
                    result = fn(**params) if params else fn()
                    self.captured.append({"title": step["title"], "result": result})
                return "stub-task"

        queue = _StubQueue()
        planner = HorizonPlanner(
            llm_router=_StubLLM(),
            task_queue=queue,
            db_path=str(tmp_path / "horizon.db"),
        )
        return planner, queue

    def test_closure_runs_under_real_queue_shape(self, tmp_path):
        planner, queue = self._make_planner(tmp_path)
        goal_id = planner.add(
            intent="probe goal",
            trigger_condition="never fires; we test the hand-off path",
        )
        goal = planner._load_goal(goal_id)
        # Drive the hand-off directly — bypass the watcher.
        planner._hand_off(goal, resume=False)
        # If the kwarg bug were still present, _StubQueue's `fn(**params)`
        # would have raised TypeError before reaching `captured`.
        assert queue.captured, "expected the closure to execute via the queue"
        first = queue.captured[0]
        assert "Execute:" in first["title"]
        # Sanity: the LLM stub got the prompt.
        assert "executed:" in first["result"]
