from __future__ import annotations

import json
import logging
import sqlite3
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from prism_policy import PolicyEngine
from prism_tool_finder import ToolDiscoveryResult, ToolFinder

logger = logging.getLogger(__name__)


SAFETY_LEVELS = {
    "read_only": 0,
    "write": 1,
    "financial": 2,
    "communication": 2,
    "system": 3,
}


@dataclass
class ExecutorRecord:
    executor_id: str
    task_description: str
    handler: Optional[Callable] = None
    success_count: int = 0
    failure_count: int = 0
    task_name: str = ""
    description: str = ""
    safety_class: str = "read_only"
    source: str = "builtin"
    code_path: str = ""
    success_rate: float = 0.0
    n_executions: int = 0
    last_used: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.task_name:
            self.task_name = self.executor_id
        if not self.description:
            self.description = self.task_description
        if not self.tags:
            text = f"{self.executor_id} {self.task_name} {self.task_description}".lower()
            self.tags = sorted({token for token in text.replace("_", " ").split() if token})
        computed_n = self.success_count + self.failure_count
        if self.n_executions <= 0 and computed_n > 0:
            self.n_executions = computed_n
        if self.n_executions > 0 and self.success_rate == 0.0:
            self.success_rate = self.success_count / self.n_executions


@dataclass
class ExecutionPlan:
    task: str
    description: str = ""
    executors: list[ExecutorRecord] = field(default_factory=list)
    selected_executor: Optional[ExecutorRecord] = None
    diagnosis: object = None
    executor_source: str = ""
    discovery_result: Optional[ToolDiscoveryResult] = None
    chosen_option: str = ""
    executor_found: bool = False
    executor_name: str = "none"
    approval_needed: bool = False
    estimated_cost: str = "unknown"
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if not self.chosen_option:
            self.chosen_option = self.task
        if self.selected_executor is None and self.executors:
            self.selected_executor = self.executors[0]
        if self.selected_executor is not None:
            self.executor_found = True
            self.executor_name = self.selected_executor.task_name
            if not self.executor_source:
                self.executor_source = self.selected_executor.source
            if self.confidence == 0.0:
                self.confidence = self.selected_executor.success_rate


@dataclass
class ExecutionResult:
    success: bool
    output: dict
    status: str
    elapsed_ms: float
    message: str = ""
    executor_used: str = ""
    error: str = ""


class ToolRegistry:
    BUILTIN_TAGS = {
        "uber": ["transport", "taxi", "ride"],
        "bus": ["transport", "bus", "oyster", "tfl", "pass"],
        "grocery": ["shopping", "food", "delivery", "supermarket"],
        "calendar": ["schedule", "reminder", "appointment"],
        "search": ["lookup", "find", "search", "check", "research"],
        "weather": ["weather", "forecast", "rain", "temperature"],
    }

    def __init__(self, db_path: str = "~/.prism/tools.db") -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, ExecutorRecord] = {}
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS executors(
                    id TEXT PRIMARY KEY,
                    task_name TEXT NOT NULL,
                    task_description TEXT NOT NULL,
                    safety_class TEXT NOT NULL,
                    source TEXT NOT NULL,
                    code_path TEXT NOT NULL,
                    success_rate REAL NOT NULL,
                    n_exec INTEGER NOT NULL,
                    last_used REAL NOT NULL,
                    tags_json TEXT NOT NULL
                )
                """
            )

    def _db_records(self) -> list[ExecutorRecord]:
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT
                    id, task_name, task_description, safety_class, source,
                    code_path, success_rate, n_exec, last_used, tags_json
                FROM executors
                """
            ).fetchall()

        records: list[ExecutorRecord] = []
        for row in rows:
            existing = self._records.get(row[0])
            records.append(
                ExecutorRecord(
                    executor_id=row[0],
                    task_description=row[2],
                    handler=existing.handler if existing else None,
                    task_name=row[1],
                    description=row[2],
                    safety_class=row[3],
                    source=row[4],
                    code_path=row[5],
                    success_rate=float(row[6]),
                    n_executions=int(row[7]),
                    last_used=float(row[8]),
                    tags=json.loads(row[9] or "[]"),
                )
            )
        return records

    def find(self, task_description: str, top_n: int = 3) -> list[ExecutorRecord]:
        lowered = task_description.lower().strip()
        candidates: dict[str, ExecutorRecord] = {record.executor_id: record for record in self._db_records()}
        candidates.update(self._records)

        scored: list[tuple[int, float, ExecutorRecord]] = []
        for record in candidates.values():
            score = self._score(record, lowered)
            if score > 0:
                scored.append((score, record.success_rate, record))

        scored.sort(key=lambda item: (item[0], item[1], item[2].last_used), reverse=True)
        return [record for _, _, record in scored[:top_n]]

    def register(self, record: ExecutorRecord) -> None:
        record.last_used = time.time()
        self._records[record.executor_id] = record
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO executors VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record.executor_id,
                    record.task_name,
                    record.task_description,
                    record.safety_class,
                    record.source,
                    record.code_path,
                    record.success_rate,
                    record.n_executions,
                    record.last_used,
                    json.dumps(record.tags),
                ),
            )

    def update_stats(self, executor_id: str, success: bool) -> None:
        record = self._resolve(executor_id)
        if record is None:
            return
        if success:
            record.success_count += 1
        else:
            record.failure_count += 1
        record.n_executions = record.success_count + record.failure_count
        if record.n_executions > 0:
            record.success_rate = record.success_count / record.n_executions
        record.last_used = time.time()
        self.register(record)

    def _resolve(self, executor_key: str) -> Optional[ExecutorRecord]:
        if executor_key in self._records:
            return self._records[executor_key]
        for record in self._records.values():
            if executor_key in {record.task_name, record.executor_id}:
                return record
        for record in self._db_records():
            if executor_key in {record.executor_id, record.task_name}:
                self._records.setdefault(record.executor_id, record)
                return self._records[record.executor_id]
        return None

    @classmethod
    def _score(cls, record: ExecutorRecord, lowered_task: str) -> int:
        if not lowered_task:
            return 0
        words = {word for word in lowered_task.replace("_", " ").split() if word}
        haystack = (
            f"{record.executor_id} {record.task_name} {record.task_description} "
            f"{record.description} {' '.join(record.tags)}"
        ).lower()
        overlap = sum(1 for word in words if word in haystack)
        if lowered_task in haystack:
            overlap += 5
        for keyword, builtin_tags in cls.BUILTIN_TAGS.items():
            if keyword in haystack:
                overlap += sum(1 for tag in builtin_tags if tag in lowered_task)
        return overlap


class PrismExecutorAgent:
    """Plans and executes tasks, escalating policy-gated work via `on_approval` when needed."""

    SPENDING_LIMITS = {
        "transport": 15.0,
        "grocery": 80.0,
        "food": 30.0,
        "default": 5.0,
    }

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        policy_engine: Optional[PolicyEngine] = None,
        tool_finder: Optional[ToolFinder] = None,
        collaborator=None,
        on_approval: Optional[Callable[[ExecutionPlan], bool]] = None,
        spending_limits: Optional[dict] = None,
        db_path: str = "~/.prism/executions.db",
        tool_registry: Optional[ToolRegistry] = None,
        autonomous=None,
    ) -> None:
        self.registry = tool_registry or registry or ToolRegistry()
        self.policy_engine = policy_engine
        self.tool_finder = tool_finder
        self.collaborator = collaborator or getattr(tool_finder, "collaborator", None)
        self.on_approval = on_approval or self._deny_approval
        self.spending_limits = spending_limits or dict(self.SPENDING_LIMITS)
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._autonomous = autonomous
        self._init_log_db()

    def plan(
        self,
        chosen_option: str,
        diagnosis=None,
        context: Optional[dict] = None,
    ) -> ExecutionPlan:
        context = context or {}
        executors = self.registry.find(chosen_option, top_n=3)
        selected = executors[0] if executors else None
        plan = ExecutionPlan(
            task=chosen_option,
            description=f"Execute {chosen_option}",
            executors=executors,
            selected_executor=selected,
            diagnosis=diagnosis,
            executor_source=(selected.source if selected else ""),
            chosen_option=chosen_option,
            executor_found=selected is not None,
            executor_name=(selected.task_name if selected else "none"),
            approval_needed=self._approval_needed_for(selected, context),
            estimated_cost=self._format_cost(context.get("estimated_cost", "unknown")),
            confidence=(selected.success_rate if selected else 0.0),
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
            plan.description = discovery.search_summary
            plan.estimated_cost = self._format_cost(
                context.get("estimated_cost", discovery.recommended.estimated_cost)
            )

        if not executors and self._can_synthesise():
            plan.executor_source = "will_synthesise"
            plan.approval_needed = True
        elif not executors and plan.discovery_result is not None:
            plan.executor_source = f"discovered:{plan.discovery_result.recommended.execution_type}"

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
                estimated_cost=self._numeric_cost(context.get("estimated_cost", 0.0)),
            )
            if verdict == PolicyEngine.Verdict.REJECT:
                return ExecutionResult(False, {}, "policy_rejected", 0.0, reason, error=reason)
            if verdict == PolicyEngine.Verdict.ESCALATE:
                approved = self.on_approval(plan)
                if not approved:
                    return ExecutionResult(False, {}, "escalated_denied", 0.0, reason, error=reason)
        elif plan.approval_needed:
            approved = self.on_approval(plan)
            if not approved:
                return ExecutionResult(False, {}, "approval_denied", 0.0, "Not approved by user", error="Not approved by user")

        if plan.selected_executor is not None:
            result = self._invoke_executor(plan.selected_executor, plan, context)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            final = self._normalise_result(result, elapsed_ms, executor_used=plan.executor_name)
            self.registry.update_stats(plan.selected_executor.executor_id, final.success)
            self._log(plan, final)
            return final

        if self._can_synthesise():
            synthesised = self._synthesise_executor(plan, context, started)
            self._log(plan, synthesised)
            return synthesised

        if plan.discovery_result is not None:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            option = plan.discovery_result.recommended
            output = {
                "task": plan.task,
                "recommended": asdict(option),
                "options": [asdict(item) for item in plan.discovery_result.options],
                "summary": plan.discovery_result.search_summary,
            }
            final = ExecutionResult(
                True,
                output,
                f"discovered_{option.execution_type}",
                elapsed_ms,
                option.description,
                executor_used=option.name,
            )
            self._log(plan, final)
            return final

        # Autonomous engine fallback: synthesise and run a tool on the fly
        if hasattr(self, '_autonomous') and self._autonomous:
            autonomous_result = self._autonomous.execute_sync(
                plan.task, context or {})
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            final = ExecutionResult(
                True, {"result": autonomous_result},
                "autonomous", elapsed_ms,
                autonomous_result[:200],
                executor_used="autonomous",
            )
            self._log(plan, final)
            return final

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        final = ExecutionResult(False, {}, "no_executor", elapsed_ms, "No executor or fallback path found.", error="No executor or fallback path found.")
        self._log(plan, final)
        return final

    def _invoke_executor(self, record: ExecutorRecord, plan: ExecutionPlan, context: dict):
        if callable(record.handler):
            try:
                return record.handler(plan, context)
            except TypeError:
                return record.handler(context)
        if record.code_path:
            return self._run_file_executor(record.code_path, context)
        return {"success": False, "status": "no_executor", "error": f"Executor '{record.executor_id}' is not runnable."}

    def _run_file_executor(self, code_path: str, context: dict) -> dict:
        path = Path(code_path)
        if not path.exists():
            return {"success": False, "status": "missing_executor", "error": f"Executor file not found: {code_path}"}
        try:
            result = subprocess.run(
                ["python3", str(path), json.dumps(context)],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception as exc:
            return {"success": False, "status": "executor_error", "error": str(exc)}

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if result.returncode != 0:
            return {
                "success": False,
                "status": "executor_error",
                "error": (stderr or stdout or f"Executor exited {result.returncode}")[:500],
            }
        if not stdout:
            return {"success": False, "status": "executor_error", "error": (stderr or "Executor produced no output")[:500]}
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            last_line = stdout.splitlines()[-1]
            try:
                payload = json.loads(last_line)
            except json.JSONDecodeError:
                return {"success": False, "status": "executor_error", "error": stdout[:500]}
        return payload if isinstance(payload, dict) else {"success": True, "result": payload}

    def _run_executor(self, executor_name: str, context: dict) -> dict:
        record = self.registry._resolve(executor_name)
        if record is None:
            return {"success": False, "status": "missing_executor", "error": f"Executor file not found: {executor_name}"}
        return self._invoke_executor(record, ExecutionPlan(task=executor_name), context)

    def _synthesise_executor(self, plan: ExecutionPlan, context: dict, started: float) -> ExecutionResult:
        from prism_collaborator import ToolSpec

        spec = ToolSpec(
            task_name=plan.task.replace(" ", "_").lower(),
            description=f"Execute: {plan.task}",
            inputs={key: type(value).__name__ for key, value in (context or {}).items()},
            expected_output={"success": "bool", "result": "str"},
            safety_class=context.get("safety_class", self._default_safety_class(context)),
            approval_required=True,
        )
        ok, code = self.collaborator.synthesise_tool(spec)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not ok:
            return ExecutionResult(False, {}, "synthesis_failed", elapsed_ms, code, executor_used="synthesis_failed", error=code)

        record = ExecutorRecord(
            executor_id=str(uuid.uuid4()),
            task_description=plan.task,
            task_name=spec.task_name,
            description=plan.task,
            safety_class=spec.safety_class,
            source="learned",
            code_path=self._save_code(code, spec.task_name),
            tags=sorted(set(plan.task.lower().replace("_", " ").split())),
        )
        self.registry.register(record)
        plan.executors = [record]
        plan.selected_executor = record
        plan.executor_found = True
        plan.executor_name = record.task_name
        plan.executor_source = "learned"

        result = self._invoke_executor(record, plan, context)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        final = self._normalise_result(result, elapsed_ms, executor_used=record.task_name)
        self.registry.update_stats(record.executor_id, final.success)
        return final

    @staticmethod
    def _normalise_result(result, elapsed_ms: float, executor_used: str = "") -> ExecutionResult:
        if isinstance(result, ExecutionResult):
            result.elapsed_ms = elapsed_ms
            if executor_used and not result.executor_used:
                result.executor_used = executor_used
            if not result.error and not result.success and result.message:
                result.error = result.message
            return result
        if isinstance(result, tuple) and len(result) == 2:
            success, payload = result
            output = payload if isinstance(payload, dict) else {"result": payload}
            return ExecutionResult(bool(success), output, "executed", elapsed_ms, executor_used=executor_used)
        if isinstance(result, dict):
            success = bool(result.get("success", True))
            status = str(result.get("status", "executed" if success else "failed"))
            message = str(result.get("message", result.get("error", "")))
            error = str(result.get("error", "")) if not success else ""
            return ExecutionResult(success, result, status, elapsed_ms, message, executor_used=executor_used, error=error)
        return ExecutionResult(True, {"result": result}, "executed", elapsed_ms, executor_used=executor_used)

    def _save_code(self, code: str, name: str) -> str:
        path = self.db_path.parent / "learned_tools" / f"{name}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code, encoding="utf-8")
        return str(path)

    def _init_log_db(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS log(
                    id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    executor TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    elapsed REAL NOT NULL,
                    ts REAL NOT NULL,
                    output_json TEXT NOT NULL
                )
                """
            )

    def _log(self, plan: ExecutionPlan, result: ExecutionResult) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO log VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    str(uuid.uuid4()),
                    plan.task,
                    result.executor_used or plan.executor_name or "none",
                    int(result.success),
                    result.status,
                    result.elapsed_ms,
                    time.time(),
                    json.dumps(result.output),
                ),
            )

    def _approval_needed_for(self, executor: Optional[ExecutorRecord], context: dict) -> bool:
        if context.get("approval_required") is True:
            return True
        if executor is None:
            return self._can_synthesise()
        if SAFETY_LEVELS.get(executor.safety_class, 2) >= 2:
            return True
        category = context.get("category", "default")
        limit = self.spending_limits.get(category, self.spending_limits.get("default", 0.0))
        estimated_cost = self._numeric_cost(context.get("estimated_cost", 0.0))
        return estimated_cost > limit if limit > 0 else False

    def _default_safety_class(self, context: dict) -> str:
        if context.get("category") in {"transport", "food", "grocery"}:
            return "financial"
        if context.get("message") or context.get("recipient"):
            return "communication"
        return "write"

    def _can_synthesise(self) -> bool:
        return bool(self.collaborator and hasattr(self.collaborator, "synthesise_tool"))

    @staticmethod
    def _numeric_cost(value) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.replace("£", "").replace("$", "").strip()
            try:
                return float(cleaned)
            except ValueError:
                return 0.0
        return 0.0

    @staticmethod
    def _format_cost(value) -> str:
        if isinstance(value, (int, float)):
            return "free" if float(value) <= 0 else f"£{float(value):.2f}"
        return str(value)

    @staticmethod
    def _deny_approval(plan: ExecutionPlan) -> bool:
        del plan
        return False
