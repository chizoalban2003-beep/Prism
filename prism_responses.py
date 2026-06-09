from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from prism_device_agent import DeviceTaskResult
    from prism_planner import PlanOfAction


class CardType(str, Enum):
    TEXT = "text"
    PLAN = "plan"
    PREDICTION = "prediction"
    MOMENT = "moment"
    RISK = "risk"
    SQUAD = "squad"
    DOMAIN = "domain"
    IDENTITY = "identity"
    ARTIFACTS = "artifacts"
    ERROR = "error"
    THINKING = "thinking"
    APPROVAL = "approval"


@dataclass
class PrismCard:
    card_type: CardType
    title: str
    body: str
    card_data: dict
    actions: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "type": self.card_type.value,
            "title": self.title,
            "body": self.body,
            "data": self.card_data,
            "actions": self.actions,
        }


def text_card(body: str, title: str = "") -> PrismCard:
    return PrismCard(CardType.TEXT, title, body, {})


def approval_card(task: str, reason: str, params: Optional[dict] = None) -> PrismCard:
    """
    Card shown when a device task requires user confirmation before executing.
    The chat UI renders two buttons: Approve and Deny.
    card_data contains everything needed to re-execute on approval.
    """
    import uuid as _uuid
    task_id = str(_uuid.uuid4())[:8]
    return PrismCard(
        card_type = CardType.APPROVAL,
        title     = "Approval required",
        body      = reason or f"Allow PRISM to: {task}",
        card_data = {
            "task_id": task_id,
            "task":    task,
            "params":  params or {},
            "reason":  reason,
        },
        actions = ["Approve", "Deny"],
    )



def plan_card(plan) -> PrismCard:
    data = {
        "primary_focus": getattr(plan, "primary_focus", ""),
        "activation": round(float(getattr(plan, "activation", 0.0)), 3),
        "warnings": list(getattr(plan, "warnings", []) or []),
        "tasks": [
            {
                "time": task.time_slot,
                "category": task.category,
                "title": task.title,
                "duration": task.duration_min,
            }
            for task in (getattr(plan, "tasks", []) or [])
        ],
    }
    return PrismCard(CardType.PLAN, "Daily Plan", "", data)


def prediction_card(pred) -> PrismCard:
    data = {
        "home": getattr(pred, "home_team", ""),
        "away": getattr(pred, "away_team", ""),
        "p_home": round(float(getattr(pred, "p_home_win", 0.0)), 3),
        "p_draw": round(float(getattr(pred, "p_draw", 0.0)), 3),
        "p_away": round(float(getattr(pred, "p_away_win", 0.0)), 3),
        "predicted": getattr(pred, "prediction", ""),
        "confidence": round(float(getattr(pred, "confidence", 0.0)), 3),
        "key_factors": [
            (name, round(float(contribution), 3), direction)
            for name, contribution, direction in (getattr(pred, "key_factors", []) or [])[:3]
        ],
    }
    return PrismCard(CardType.PREDICTION, "Match Prediction", "", data)


def risk_card(risk) -> PrismCard:
    data = {
        "athlete": getattr(risk, "athlete_name", ""),
        "risk_level": getattr(risk, "risk_level", ""),
        "prediction": getattr(risk, "prediction", ""),
        "confidence": round(float(getattr(risk, "confidence", 0.0)), 3),
        "recommendations": list(getattr(risk, "recommendations", []) or []),
    }
    return PrismCard(CardType.RISK, "Injury Risk", "", data)


def squad_card(risks: list) -> PrismCard:
    data = {
        "players": [
            {
                "name": getattr(risk, "athlete_name", risk.get("athlete_name") if isinstance(risk, dict) else ""),
                "risk_level": getattr(risk, "risk_level", risk.get("risk_level") if isinstance(risk, dict) else ""),
                "confidence": round(
                    float(getattr(risk, "confidence", risk.get("confidence") if isinstance(risk, dict) else 0.0)),
                    3,
                ),
            }
            for risk in (risks or [])
        ]
    }
    return PrismCard(CardType.SQUAD, "Squad Risk Overview", "", data)


def domain_card(domain_name: str, diag) -> PrismCard:
    data = {
        "domain": domain_name,
        "recommended": diag.primary_plank.name,
        "confidence": round(diag.activations[0].activation, 3),
        "fulcrum": round(diag.fulcrum_position, 3),
        "options": [
            {
                "name": activation.plank.name,
                "activation": round(activation.activation, 3),
                "position": activation.plank.position,
            }
            for activation in diag.activations
        ],
    }
    return PrismCard(CardType.DOMAIN, f"{domain_name} Recommendation", "", data)


def moment_card(result) -> PrismCard:
    data = {
        "sport": result.moment.sport,
        "moment_type": result.moment.moment_type,
        "recommended": result.recommended,
        "activation": round(result.activations[0][1] if result.activations else 0, 3),
        "xg": result.xg_contextual,
        "time_pressure": round(result.time_pressure, 3),
        "options": [
            {"name": name, "activation": round(activation, 3), "ev": round(ev, 1)}
            for name, activation, ev in result.activations
        ],
    }
    return PrismCard(CardType.MOMENT, "Moment Analysis", "", data)


def identity_card(identity_data: dict) -> PrismCard:
    return PrismCard(CardType.IDENTITY, "Your Decision Profile", "", identity_data)


def device_result_card(result: "DeviceTaskResult", task: str) -> PrismCard:
    """Card for device task execution results."""
    if result.needs_approval:
        params = {}
        try:
            import json as _j
            params = _j.loads(result.undo_command or "{}").get("params", {})
        except Exception:
            pass
        return approval_card(task, result.output, params)
    data = {
        "success":        result.success,
        "tool_used":      result.tool_used,
        "output":         result.output[:1000],
        "files_created":  result.files_created,
        "files_modified": result.files_modified,
        "elapsed_ms":     round(result.elapsed_ms, 1),
        "error":          result.error,
        "undo_available": bool(result.undo_command),
        "undo_command":   result.undo_command,
    }
    body = (
        f"✓ Done in {result.elapsed_ms:.0f}ms using {result.tool_used}"
        if result.success
        else f"✗ {result.error[:200]}"
    )
    actions = []
    if result.undo_command:
        actions.append("Undo this action")
    if not result.success and result.error:
        actions.append("Try a different approach")
    return PrismCard(
        card_type = CardType.TEXT,
        title     = f"Device task: {task[:50]}",
        body      = body,
        card_data = data,
        actions   = actions,
    )


def plan_of_action_card(plan: "PlanOfAction") -> PrismCard:
    """
    Renders a PlanOfAction as a PRISM chat card.

    card_data structure:
    {
      "task": str,
      "domain": str,
      "timeline": str,
      "fulcrum": float,
      "context_summary": str,
      "strategies": [
        {
          "name": str,
          "rank": int,
          "activation": float,        # 0-1 confidence
          "expected_value": float,
          "risk_score": float,
          "why": str,
          "steps": [{"order":int,"action":str,"timeline":str},...],
          "resources": [str,...],
          "outcome": str,
          "risks": [str,...],
          "has_full_plan": bool       # False for alternatives without plans
        }
      ]
    }
    """
    strategies = []
    for i, s in enumerate(plan.all_strategies):
        strategies.append({
            "name":           s.name,
            "rank":           i + 1,
            "activation":     round(s.activation, 3),
            "expected_value": round(s.expected_value, 1),
            "risk_score":     s.risk_score,
            "why":            s.why_recommended,
            "steps":          [{"order": st.order, "action": st.action,
                                 "timeline": st.timeline} for st in s.steps],
            "resources":      s.resources,
            "outcome":        s.expected_outcome,
            "risks":          s.risks,
            "has_full_plan":  len(s.steps) > 0,
        })
    return PrismCard(
        card_type = CardType.PLAN,
        title     = f"Plan of action — {plan.domain}",
        body      = plan.context_summary,
        card_data = {
            "task":            plan.task,
            "domain":          plan.domain,
            "timeline":        plan.timeline,
            "fulcrum":         round(plan.fulcrum_position, 3),
            "context_summary": plan.context_summary,
            "strategies":      strategies,
        },
        actions = [
            f"Full plan for {plan.all_strategies[1].name}" if len(plan.all_strategies) > 1 else "",
            "Explain the ranking",
            "Execute optimal strategy",
        ]
    )


def policy_view_card(data: dict) -> PrismCard:
    return PrismCard(CardType.TEXT, "Your operating policies",
        f"Global limit: {data.get('global_limit','—')} · "
        f"Escalate at: {data.get('escalate_at','—')}",
        data, actions=["Set a budget","Reset all policies"])


def task_list_card(tasks: list) -> PrismCard:
    items = [{"id":t.task_id,"title":t.title,"status":t.status
              if isinstance(t.status,str) else t.status.value,
              "progress":t.progress,"current_step":t.current_step,
              "error":t.error} for t in tasks]
    running = sum(1 for t in tasks
                  if (t.status if isinstance(t.status,str)
                      else t.status.value) == "running")
    return PrismCard(CardType.TEXT,"Task queue",
        f"{running} running · {len(tasks)} recent",
        {"tasks":items},
        actions=["Cancel running task"] if running else [])


def task_progress_card(progress) -> PrismCard:
    status = (progress.status if isinstance(progress.status,str)
              else progress.status.value)
    pct    = int(progress.progress * 100)
    body   = (f"{progress.current_step}" if status == "running"
              else "Completed" if status == "completed"
              else f"Failed: {progress.error[:100]}" if status == "failed"
              else status.title())
    return PrismCard(CardType.TEXT,
        f"{progress.title} — {pct}%", body,
        {"task_id":progress.task_id,"status":status,
         "progress":progress.progress,
         "steps_done":progress.steps_done,
         "steps_total":progress.steps_total,
         "result":progress.result},
        actions=["Cancel"] if status == "running" else [])
