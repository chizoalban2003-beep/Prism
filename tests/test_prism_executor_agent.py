from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from prism_executor_agent import ExecutionPlan, ExecutorRecord, PrismExecutorAgent, ToolRegistry


def test_plan_exposes_executor_metadata_without_approval(tmp_path):
    registry = ToolRegistry(db_path=str(tmp_path / "tools.db"))
    registry.register(
        ExecutorRecord(
            "weather_lookup",
            "check the weather",
            handler=lambda context: {"success": True, "status": "executed", "context": context},
            task_name="weather_check",
            source="builtin",
            safety_class="read_only",
            success_count=4,
            failure_count=1,
        )
    )
    agent = PrismExecutorAgent(registry=registry, db_path=str(tmp_path / "executions.db"))

    plan = agent.plan("weather check", context={"estimated_cost": 0})

    assert isinstance(plan, ExecutionPlan)
    assert plan.executor_found is True
    assert plan.executor_name == "weather_check"
    assert plan.executor_source == "builtin"
    assert plan.approval_needed is False
    assert plan.estimated_cost == "free"
    assert plan.confidence == 0.8


def test_execute_synthesises_registers_and_runs_executor(tmp_path):
    class _Collaborator:
        def synthesise_tool(self, spec):
            assert spec.task_name == "order_lunch"
            code = """
import json
import sys

if __name__ == "__main__":
    payload = json.loads(sys.argv[1])
    print(json.dumps({"success": True, "status": "executed", "result": payload["item"]}))
"""
            return True, code

    agent = PrismExecutorAgent(
        registry=ToolRegistry(db_path=str(tmp_path / "tools.db")),
        collaborator=_Collaborator(),
        on_approval=lambda plan: True,
        db_path=str(tmp_path / "executions.db"),
    )

    plan = agent.plan("order lunch", context={"category": "food", "estimated_cost": 12.5})
    result = agent.execute(plan, {"item": "ramen", "category": "food", "estimated_cost": 12.5})

    assert plan.executor_found is True
    assert plan.executor_source == "learned"
    assert result.success is True
    assert result.status == "executed"
    assert result.executor_used == "order_lunch"
    assert result.output["result"] == "ramen"

    learned = agent.registry.find("order lunch", top_n=1)
    assert learned
    assert learned[0].source == "learned"
    assert Path(learned[0].code_path).exists()

    with sqlite3.connect(tmp_path / "executions.db") as connection:
        row = connection.execute("SELECT task, executor, success, status, output_json FROM log").fetchone()
    assert row is not None
    assert row[0] == "order lunch"
    assert row[1] == "order_lunch"
    assert row[2] == 1
    assert row[3] == "executed"
    assert json.loads(row[4])["result"] == "ramen"
