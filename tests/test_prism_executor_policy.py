from __future__ import annotations

from prism_executor_agent import ExecutorRecord, PrismExecutorAgent
from prism_policy import PolicyEngine, ResourceAllocation


def test_reject_blocks(tmp_path):
    engine = PolicyEngine(db_path=str(tmp_path / "policy.db"))
    engine.set_allocation(
        "alice",
        "food",
        ResourceAllocation(name="food", blacklisted=["Bad Shop"], per_action_limit=10.0),
    )
    called = {"value": False}

    def handler(plan, context):
        called["value"] = True
        return {"success": True}

    agent = PrismExecutorAgent(policy_engine=engine)
    agent.registry.register(ExecutorRecord("order_food", "order food", handler))
    plan = agent.plan("order food", context={})

    result = agent.execute(
        plan,
        {"user": "alice", "category": "food", "provider": "Bad Shop", "estimated_cost": 5.0},
    )

    assert result.success is False
    assert result.status == "policy_rejected"
    assert called["value"] is False


def test_escalate_denied(tmp_path):
    engine = PolicyEngine(db_path=str(tmp_path / "policy.db"))
    engine.set_allocation(
        "alice",
        "transport",
        ResourceAllocation(name="transport", per_action_limit=5.0, auto_approve_below=2.0),
    )
    called = {"value": False}

    def handler(plan, context):
        called["value"] = True
        return {"success": True}

    agent = PrismExecutorAgent(policy_engine=engine, on_approval=lambda plan: False)
    agent.registry.register(ExecutorRecord("book_transport", "book transport", handler))
    plan = agent.plan("book transport", context={})

    result = agent.execute(
        plan,
        {"user": "alice", "category": "transport", "provider": "Taxi Co", "estimated_cost": 20.0},
    )

    assert result.success is False
    assert result.status == "escalated_denied"
    assert called["value"] is False


def test_approve_proceeds(tmp_path):
    engine = PolicyEngine(db_path=str(tmp_path / "policy.db"))
    engine.set_allocation(
        "alice",
        "food",
        ResourceAllocation(name="food", auto_approve_below=10.0, per_action_limit=15.0),
    )
    called = {"value": False}

    def handler(plan, context):
        called["value"] = True
        return {"success": True, "status": "executed", "message": "done"}

    agent = PrismExecutorAgent(policy_engine=engine)
    agent.registry.register(ExecutorRecord("order_food", "order food", handler))
    plan = agent.plan("order food", context={})

    result = agent.execute(
        plan,
        {"user": "alice", "category": "food", "provider": "Good Shop", "estimated_cost": 4.0},
    )

    assert result.success is True
    assert result.status == "executed"
    assert called["value"] is True
