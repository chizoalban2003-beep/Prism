from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

from prism_policy import PolicyEngine
from prism_tool_finder import ToolDiscoveryResult, ToolFinder


@dataclass
class ExecutorRecord:
    executor_id: str
    task_description: str
    handler: Callable
    success_count: int = 0
    failure_count: int = 0


@dataclass
class ExecutionPlan:
    task: str
    description: str = ""
    executors: list[ExecutorRecord] = field(default_factory=list)
    selected_executor: Optional[ExecutorRecord] = None
    diagnosis: object = None
    executor_source: str = ""
    discovery_result: Optional[ToolDiscoveryResult] = None


@dataclass
class ExecutionResult:
    success: bool
    output: dict
    status: str
    elapsed_ms: float
    message: str = ""


class ToolRegistry:
    def __init__(self) -> None:
        self._records: dict[str, ExecutorRecord] = {}

    def find(self, task_description: str, top_n: int = 3) -> list[ExecutorRecord]:
        lowered = task_description.lower()
        ranked = sorted(
            self._records.values(),
            key=lambda record: self._score(record, lowered),
            reverse=True,
        )
        return [record for record in ranked if self._score(record, lowered) > 0][:top_n]

    def register(self, record: ExecutorRecord) -> None:
        self._records[record.executor_id] = record

    def update_stats(self, executor_id: str, success: bool) -> None:
        record = self._records.get(executor_id)
        if record is None:
            return
        if success:
            record.success_count += 1
        else:
            record.failure_count += 1

    @staticmethod
    def _score(record: ExecutorRecord, lowered_task: str) -> int:
        words = {word for word in lowered_task.split() if word}
        haystack = f"{record.executor_id} {record.task_description}".lower()
        overlap = sum(1 for word in words if word in haystack)
        if lowered_task and lowered_task in haystack:
            overlap += 5
        return overlap


class PrismExecutorAgent:
    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        policy_engine: Optional[PolicyEngine] = None,
        tool_finder: Optional[ToolFinder] = None,
        on_approval: Optional[Callable[[ExecutionPlan], bool]] = None,
    ) -> None:
        self.registry = registry or ToolRegistry()
        self.policy_engine = policy_engine
        self.tool_finder = tool_finder
        self.on_approval = on_approval or (lambda plan: False)

    def plan(
        self,
        chosen_option: str,
        diagnosis=None,
        context: Optional[dict] = None,
    ) -> ExecutionPlan:
        context = context or {}
        executors = self.registry.find(chosen_option, top_n=3)
        plan = ExecutionPlan(
            task=chosen_option,
            description=f"Execute {chosen_option}",
            executors=executors,
            selected_executor=executors[0] if executors else None,
            diagnosis=diagnosis,
            executor_source=f"registry:{executors[0].executor_id}" if executors else "",
        )

        if not executors and self.tool_finder:
            discovery = self.tool_finder.find(
                task=chosen_option,
                provider_name=context.get("provider", chosen_option),
                urgency=context.get("urgency", 0.5),
                cost_tolerance=context.get("cost_tolerance", 0.5),
                prefers_auto=context.get("prefers_auto", 0.5),
                budget_left=context.get("budget_left", 1.0),
            )
            plan.discovery_result = discovery
            plan.executor_source = f"discovered:{discovery.recommended.execution_type}"
            plan.description = discovery.search_summary

        return plan

    def execute(self, plan: ExecutionPlan, context: Optional[dict] = None) -> ExecutionResult:
        context = context or {}
        started = time.perf_counter()
        user = context.get("user")

        if self.policy_engine and user:
            verdict, reason = self.policy_engine.evaluate(
                user=user,
                category=context.get("category", "general"),
                provider=context.get("provider", plan.task),
                estimated_cost=context.get("estimated_cost", 0.0),
            )
            if verdict == PolicyEngine.Verdict.REJECT:
                return ExecutionResult(False, {}, "policy_rejected", 0.0, reason)
            if verdict == PolicyEngine.Verdict.ESCALATE:
                approved = self.on_approval(plan)
                if not approved:
                    return ExecutionResult(False, {}, "escalated_denied", 0.0, reason)

        if plan.selected_executor is not None:
            result = self._invoke_executor(plan.selected_executor, plan, context)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            final = self._normalise_result(result, elapsed_ms)
            self.registry.update_stats(plan.selected_executor.executor_id, final.success)
            return final

        if plan.discovery_result is not None:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            option = plan.discovery_result.recommended
            output = {
                "task": plan.task,
                "recommended": asdict(option),
                "options": [asdict(item) for item in plan.discovery_result.options],
                "summary": plan.discovery_result.search_summary,
            }
            return ExecutionResult(True, output, f"discovered_{option.execution_type}", elapsed_ms, option.description)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return ExecutionResult(False, {}, "no_executor", elapsed_ms, "No executor or fallback path found.")

    @staticmethod
    def _invoke_executor(record: ExecutorRecord, plan: ExecutionPlan, context: dict):
        try:
            return record.handler(plan, context)
        except TypeError:
            return record.handler(context)

    @staticmethod
    def _normalise_result(result, elapsed_ms: float) -> ExecutionResult:
        if isinstance(result, ExecutionResult):
            result.elapsed_ms = elapsed_ms
            return result
        if isinstance(result, tuple) and len(result) == 2:
            success, payload = result
            output = payload if isinstance(payload, dict) else {"result": payload}
            return ExecutionResult(bool(success), output, "executed", elapsed_ms)
        if isinstance(result, dict):
            success = bool(result.get("success", True))
            status = str(result.get("status", "executed"))
            message = str(result.get("message", ""))
            return ExecutionResult(success, result, status, elapsed_ms, message)
        return ExecutionResult(True, {"result": result}, "executed", elapsed_ms)
