from __future__ import annotations

from prism_policy import PolicyEngine, ResourceAllocation


def test_approve_below_limit(tmp_path):
    engine = PolicyEngine(db_path=str(tmp_path / "policy.db"))
    engine.set_allocation(
        "alice",
        "food",
        ResourceAllocation(name="food", auto_approve_below=5.0, per_action_limit=10.0),
    )

    verdict, reason = engine.evaluate("alice", "food", "Tesco", 2.0)

    assert verdict == PolicyEngine.Verdict.APPROVE
    assert "Auto-approved" in reason


def test_escalate_above_limit(tmp_path):
    engine = PolicyEngine(db_path=str(tmp_path / "policy.db"))
    engine.set_allocation(
        "alice",
        "transport",
        ResourceAllocation(name="transport", auto_approve_below=5.0, per_action_limit=10.0),
    )

    verdict, reason = engine.evaluate("alice", "transport", "Uber", 20.0)

    assert verdict == PolicyEngine.Verdict.ESCALATE
    assert "per-action limit" in reason


def test_reject_blacklisted(tmp_path):
    engine = PolicyEngine(db_path=str(tmp_path / "policy.db"))
    engine.set_allocation(
        "alice",
        "food",
        ResourceAllocation(name="food", blacklisted=["Uber Eats"], per_action_limit=20.0),
    )

    verdict, reason = engine.evaluate("alice", "food", "Uber Eats", 4.0)

    assert verdict == PolicyEngine.Verdict.REJECT
    assert "blacklist" in reason


def test_parse_budget_update(tmp_path):
    engine = PolicyEngine(db_path=str(tmp_path / "policy.db"))

    message = engine.parse_policy_update("set my transport budget to £20", "alice")
    allocation = engine.get_policy("alice").allocations["transport"]

    assert message == "✓ Transport monthly budget set to £20.00"
    assert allocation.monthly_limit == 20.0


def test_parse_never_use(tmp_path):
    engine = PolicyEngine(db_path=str(tmp_path / "policy.db"))

    message = engine.parse_policy_update("never use Uber Eats", "alice")
    allocation = engine.get_policy("alice").allocations["default"]

    assert message == "✓ Added 'Uber Eats' to your blacklist"
    assert "Uber Eats" in allocation.blacklisted


def test_parse_auto_approve(tmp_path):
    engine = PolicyEngine(db_path=str(tmp_path / "policy.db"))

    message = engine.parse_policy_update("auto-approve food under £8", "alice")
    allocation = engine.get_policy("alice").allocations["food"]

    assert message == "✓ Food orders under £8.00 will auto-approve"
    assert allocation.auto_approve_below == 8.0
