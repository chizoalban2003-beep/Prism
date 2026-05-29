from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ResourceAllocation:
    """
    What the user has committed to the system for autonomous execution.
    Like a department budget with operating rules.
    """

    name: str
    currency: str = "GBP"
    total_budget: float = 0.0
    per_action_limit: float = 5.0
    monthly_limit: float = 0.0
    auto_approve_below: float = 2.0
    preferred_providers: list[str] = field(default_factory=list)
    blacklisted: list[str] = field(default_factory=list)
    time_window: str = "any"
    notifications: str = "financial"
    notes: str = ""


@dataclass
class PolicySet:
    """
    Complete set of resource allocations and operating policies for one user.
    Persisted in SQLite. User edits via chat.
    """

    user_name: str
    allocations: dict[str, ResourceAllocation] = field(default_factory=dict)
    global_limit: float = 500.0
    escalate_at: float = 0.85
    created_at: float = field(default_factory=time.time)
    version: int = 1


class PolicyEngine:
    """
    Enforces the user's ResourceAllocation policy against proposed actions.
    """

    class Verdict:
        APPROVE = "approve"
        ESCALATE = "escalate"
        REJECT = "reject"

    def __init__(self, db_path: str = "~/.prism/policy.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS policies(
                    user TEXT PRIMARY KEY,
                    policy_json TEXT NOT NULL,
                    updated REAL NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS spend_log(
                    id TEXT PRIMARY KEY,
                    user TEXT NOT NULL,
                    category TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    amount REAL NOT NULL,
                    approved INTEGER NOT NULL,
                    ts REAL NOT NULL
                )
                """
            )

    def set_allocation(self, user: str, name: str, alloc: ResourceAllocation) -> None:
        """Set or update one resource allocation for a user."""
        policy = self.get_policy(user)
        policy.allocations[name] = alloc
        self._save(user, policy)

    def get_policy(self, user: str) -> PolicySet:
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT policy_json FROM policies WHERE user=?",
                (user,),
            ).fetchone()

        if not row:
            return PolicySet(user_name=user)

        data = json.loads(row[0])
        policy = PolicySet(
            user_name=user,
            global_limit=data.get("global_limit", 500.0),
            escalate_at=data.get("escalate_at", 0.85),
            created_at=data.get("created_at", time.time()),
            version=data.get("version", 1),
        )
        for name, allocation_data in data.get("allocations", {}).items():
            policy.allocations[name] = ResourceAllocation(**allocation_data)
        return policy

    def _save(self, user: str, policy: PolicySet) -> None:
        data = {
            "allocations": {name: vars(alloc) for name, alloc in policy.allocations.items()},
            "global_limit": policy.global_limit,
            "escalate_at": policy.escalate_at,
            "created_at": policy.created_at,
            "version": policy.version,
        }
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO policies VALUES(?,?,?)",
                (user, json.dumps(data), time.time()),
            )

    def evaluate(
        self,
        user: str,
        category: str,
        provider: str,
        estimated_cost: float,
        action: str = "",
    ) -> tuple[str, str]:
        """
        Returns (verdict, reason).
        """
        del action

        policy = self.get_policy(user)
        allocation = policy.allocations.get(category) or policy.allocations.get("default")

        if allocation is None:
            return (
                self.Verdict.ESCALATE,
                f"No policy set for '{category}'. Set a budget to enable autonomous execution.",
            )

        provider_name = provider.strip()

        if any(blocked.lower() in provider_name.lower() for blocked in allocation.blacklisted):
            return self.Verdict.REJECT, f"{provider_name} is on your blacklist for {category}."

        if not self._in_window(allocation.time_window):
            return (
                self.Verdict.ESCALATE,
                f"Outside allowed time window ({allocation.time_window}) for {category}.",
            )

        spent = self._monthly_spend(user, category)
        limit = allocation.monthly_limit if allocation.monthly_limit > 0 else allocation.total_budget
        if limit > 0 and spent + estimated_cost > limit * policy.escalate_at:
            remaining = max(limit - spent, 0.0)
            return (
                self.Verdict.ESCALATE,
                f"{category} budget at {spent / limit:.0%}. £{remaining:.2f} remaining this month. Approval needed.",
            )

        if estimated_cost > allocation.per_action_limit:
            return (
                self.Verdict.ESCALATE,
                f"£{estimated_cost:.2f} exceeds your per-action limit of £{allocation.per_action_limit:.2f} for {category}.",
            )

        if estimated_cost <= allocation.auto_approve_below:
            self._log_spend(user, category, provider_name, estimated_cost, True)
            return (
                self.Verdict.APPROVE,
                f"Auto-approved: £{estimated_cost:.2f} for {provider_name} (under £{allocation.auto_approve_below:.2f} limit).",
            )

        return (
            self.Verdict.APPROVE,
            f"Within policy: £{estimated_cost:.2f} for {category} via {provider_name}.",
        )

    def preferred_providers(self, user: str, category: str) -> list[str]:
        """Return preferred providers for a category, if set."""
        policy = self.get_policy(user)
        allocation = policy.allocations.get(category)
        return list(allocation.preferred_providers) if allocation else []

    def _in_window(self, window: str) -> bool:
        """Check if current time is within the policy window."""
        if window == "any":
            return True

        from datetime import datetime

        now = datetime.now()
        if window == "weekdays":
            return now.weekday() < 5

        if ":" in window and "-" in window:
            start_text, end_text = [part.strip() for part in window.split("-", maxsplit=1)]
            try:
                start = datetime.strptime(start_text, "%H:%M").time()
                end = datetime.strptime(end_text, "%H:%M").time()
            except ValueError as exc:
                raise ValueError(
                    f"Invalid time window '{window}'. Expected HH:MM-HH:MM format."
                ) from exc
            current = now.time()
            if start <= end:
                return start <= current <= end
            return current >= start or current <= end

        return True

    def _monthly_spend(self, user: str, category: str) -> float:
        cutoff = time.time() - (30 * 86400)
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT COALESCE(SUM(amount), 0)
                FROM spend_log
                WHERE user=? AND category=? AND ts>? AND approved=1
                """,
                (user, category, cutoff),
            ).fetchone()
        return float(row[0]) if row else 0.0

    def spend_summary(self, user: str, category: str, days: int = 30) -> dict:
        cutoff = time.time() - (max(days, 1) * 86400)
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT COALESCE(SUM(amount), 0), COUNT(*)
                FROM spend_log
                WHERE user=? AND category=? AND ts>? AND approved=1
                """,
                (user, category, cutoff),
            ).fetchone()
        total = float(row[0]) if row else 0.0
        count = int(row[1]) if row else 0
        return {
            "user": user,
            "category": category,
            "days": max(days, 1),
            "approved_spend": total,
            "approved_actions": count,
        }

    def _log_spend(
        self,
        user: str,
        category: str,
        provider: str,
        amount: float,
        approved: bool,
    ) -> None:
        import uuid

        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "INSERT INTO spend_log VALUES(?,?,?,?,?,?,?)",
                (
                    str(uuid.uuid4()),
                    user,
                    category,
                    provider,
                    amount,
                    int(approved),
                    time.time(),
                ),
            )

    def parse_policy_update(self, message: str, user: str) -> Optional[str]:
        """
        Parse policy updates from natural language chat messages.
        Returns a confirmation string or None if not recognised.
        """
        budget_match = re.search(
            r"set\s+(?:my\s+)?(\w+)\s+(?:monthly\s+)?(?:budget|limit)\s+to\s+£?(\d+\.?\d*)",
            message,
            flags=re.IGNORECASE,
        )
        if budget_match:
            category = budget_match.group(1).lower()
            amount = float(budget_match.group(2))
            policy = self.get_policy(user)
            allocation = policy.allocations.get(category, ResourceAllocation(name=category))
            allocation.monthly_limit = amount
            policy.allocations[category] = allocation
            self._save(user, policy)
            return f"✓ {category.title()} monthly budget set to £{amount:.2f}"

        blacklist_match = re.search(
            r"never\s+(?:use|order from|buy from)\s+(.+)",
            message,
            flags=re.IGNORECASE,
        )
        if blacklist_match:
            provider = blacklist_match.group(1).strip()
            policy = self.get_policy(user)
            allocation = policy.allocations.get("default", ResourceAllocation(name="default"))
            if provider.lower() not in {item.lower() for item in allocation.blacklisted}:
                allocation.blacklisted.append(provider)
            policy.allocations["default"] = allocation
            self._save(user, policy)
            return f"✓ Added '{provider}' to your blacklist"

        auto_approve_match = re.search(
            r"auto.?approve\s+(\w+)\s+(?:orders?\s+)?(?:under|below)\s+£?(\d+\.?\d*)",
            message,
            flags=re.IGNORECASE,
        )
        if auto_approve_match:
            category = auto_approve_match.group(1).lower()
            amount = float(auto_approve_match.group(2))
            policy = self.get_policy(user)
            allocation = policy.allocations.get(category, ResourceAllocation(name=category))
            allocation.auto_approve_below = amount
            policy.allocations[category] = allocation
            self._save(user, policy)
            return f"✓ {category.title()} orders under £{amount:.2f} will auto-approve"

        return None
