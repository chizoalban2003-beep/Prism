from __future__ import annotations

import json
import logging
import re
import urllib.request
from typing import Optional

from prism_llm_router import LLMRouter
from prism_task_queue import TaskQueue
from domain_configs import ALL_DOMAINS, DomainDecisionModel
from prism_device_agent import PrismDeviceAgent, DeviceTaskResult
from prism_planner import PrismPlanner, PlanOfAction
from prism_perception import PrismPerception, ContextState
from prism_responses import (
    PrismCard,
    domain_card,
    identity_card,
    moment_card,
    plan_card,
    prediction_card,
    risk_card,
    squad_card,
    text_card,
)

logger = logging.getLogger(__name__)


class PrismAgent:
    """
    Unified PRISM agent. Routes natural language to sub-agents.
    Returns PrismCard for every request. Never raises.
    """

    INTENTS = [
        (r"plan|morning|daily|today|schedule", "plan"),
        (r"how (?:do|can|should) i|plan (?:for|to)|strategy for|help me (?:with|plan)|"
         r"what(?:'s| is) the best way|i want to|i need to|my goal is", "universal_plan"),
        (r"predict|match|fixture|vs|versus", "predict_match"),
        (r"injury|risk|squad|fitness|medical|available", "squad_risk"),
        (r"moment|1v1|keeper|shot|attack", "moment"),
        (r"session|footage|video|analyse.*play", "session"),
        (r"transfer|market|value|worth", "transfer"),
        (r"triage|chest|pain|fever|symptom|patient", "domain_medical"),
        (r"portfolio|invest|allocation|bonds|equity", "domain_financial"),
        (r"legal|case|litigat|settle|arbitrat", "domain_legal"),
        (r"hire|hiring|recruit|talent|headcount", "domain_hr"),
        (r"supply|procurement|inventory|stock", "domain_supply"),
        (r"climate|carbon|emission|energy\.policy", "domain_climate"),
        (r"identity|profile|who\.am|digital\.dna|crystal", "identity"),
        (r"artifact|history|past\.decision|what\.have\.i", "artifacts"),
        (r"status|connected|device|sync", "status"),
        (r"index|scan\.files|search\.code|grep|find\.file", "ksa_task"),
        (r"resize|convert|compress|rename|move|copy|delete|create file|"
         r"find file|search (?:in|for)|read file|list files|"
         r"run (?:command|script)|execute|open (?:app|file)|"
         r"install (?:package|app)|git (?:commit|push|pull|status)|"
         r"what(?:'s| is) (?:on|in) my|show me (?:my )?files", "device_task"),
        (r"show (?:my )?polic|what(?:'s| are) my (?:budget|polic|limit)|"
         r"current (?:polic|budget|limit)", "show_policies"),
        (r"set (?:my )?(\w+) (?:budget|limit)|auto.?approv|never use|"
         r"require approval|reset (?:all )?polic", "update_policy"),
        (r"(?:running|active|pending|recent) tasks?|task (?:status|progress)|"
         r"what(?:'s| is) (?:running|happening)", "task_status"),
        (r"help|what\.can|commands|options", "help"),
    ]

    def __init__(
        self,
        kde_agent=None,
        ksa_agent=None,
        ollama_host: str = "http://localhost:11434",
        text_model: str = "mistral",
        claude_api_key: str = None,
    ):
        self._kde = kde_agent
        self._ksa = ksa_agent
        self._ollama_host = ollama_host.rstrip('/')
        self._text_model = text_model
        self._claude_key = claude_api_key
        self._router = LLMRouter.from_config()
        self._queue  = TaskQueue()
        self._planner = PrismPlanner(
            ollama_host    = ollama_host,
            ollama_model   = text_model,
            claude_api_key = claude_api_key,
        )
        self._device = PrismDeviceAgent.setup(
            policy_engine = getattr(self, '_policy', None),
            on_approval   = self._request_approval,
            collaborator  = getattr(self, '_collaborator', None),
            user          = getattr(self, '_user', 'default'),
        )
        try:
            cfg = {}
            self._perception = PrismPerception.setup(
                enable_voice     = cfg.get("enable_voice", False),
                enable_screen    = cfg.get("enable_screen", False),
                enable_typing    = cfg.get("enable_typing", True),
                enable_system    = cfg.get("enable_system", True),
                enable_biometric = cfg.get("enable_biometric", True),
                on_voice_command = self.chat,
            )
            self._perception.start()
        except Exception:
            self._perception = None

    @classmethod
    def setup(
        cls,
        name: str,
        sport: str = "Football",
        team: str = "",
        db_path: str = "~/.prism/prism.db",
    ) -> "PrismAgent":
        try:
            from kde_agent import KDEAgent
            from sports_pro import Role

            kde = KDEAgent.setup(
                name=name,
                role=Role.UNIVERSAL,
                sport=sport,
                team=team,
                config=type(
                    'C',
                    (),
                    {
                        'db_path': db_path,
                        'media_dir': '~/.prism/media',
                        'ollama_model': 'mistral',
                        'ollama_host': 'http://localhost:11434',
                        'auto_watch': False,
                    },
                )(),
            )
        except Exception:
            kde = None
        try:
            from ksa_agent import KSAgent

            ksa = KSAgent(db_path=db_path.replace('prism.db', 'ksa.db'))
        except Exception:
            ksa = None
        return cls(kde_agent=kde, ksa_agent=ksa)

    def chat(self, message: str, context: dict | None = None) -> PrismCard:
        try:
            intent = self._route(message or "")
            if self._perception:
                percept_state = self._perception.current_context()
                if not context:
                    context = {}
                context["perception"] = percept_state.to_factor_updates()
                context["perception_summary"] = percept_state.summary
            return self._execute(intent, message or "", context or {})
        except Exception as exc:
            logging.exception("PrismAgent.chat error")
            return text_card(f"Something went wrong: {exc}", "Error")

    def _request_approval(self, task: str, reason: str) -> bool:
        """
        Default approval handler — returns False (deny) when no UI is connected.
        The chat UI overrides this to show an approval card to the user.
        Store the pending approval and return False; the chat loop will
        receive the approval response and retry execution.
        """
        self._pending_approval = {"task": task, "reason": reason}
        return False

    def _route(self, message: str) -> str:
        lowered = message.lower()
        for pattern, intent in self.INTENTS:
            if re.search(pattern, lowered):
                return intent
        return self._llm_classify(message) or "help"

    def _llm_classify(self, message: str) -> Optional[str]:
        try:
            labels = [intent for _, intent in self.INTENTS]
            prompt = (
                f"Classify this message into exactly one of: {labels}\n"
                f"Message: {message}\n"
                "Reply with ONLY the label, nothing else."
            )
            payload = json.dumps({"model": self._text_model, "prompt": prompt, "stream": False}).encode()
            request = urllib.request.Request(
                f"{self._ollama_host}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                result = json.loads(response.read()).get("response", "").strip().lower()
            return result if result in labels else None
        except Exception:
            return None

    def _execute(self, intent: str, message: str, ctx: dict) -> PrismCard:
        if intent == "device_task":
            from prism_responses import device_result_card
            result = self._device.execute(message, params=ctx.get("params", {}))
            return device_result_card(result, message)

        if intent == "universal_plan":
            from prism_responses import plan_of_action_card
            perception_factors = ctx.get("perception", {})
            user_context = {**perception_factors, **(ctx.get("user_factors", {}))}
            plan = self._planner.plan(
                task_description = message,
                user_context     = user_context,
                n_plans          = 4,
            )
            return plan_of_action_card(plan)

        if intent.startswith("domain_"):
            domain_key = {
                "domain_medical": "Medical",
                "domain_financial": "Financial",
                "domain_legal": "Legal",
                "domain_hr": "HR",
                "domain_supply": "Supply Chain",
                "domain_climate": "Climate",
            }.get(intent, "Medical")
            config = ALL_DOMAINS.get(domain_key)
            if config is None:
                return text_card(f"Domain '{domain_key}' not configured.")
            profile = ctx.get("profile") or config.profiles[min(2, len(config.profiles) - 1)].name
            perception_factors = ctx.get("perception", {})
            user_context = {**perception_factors, **(ctx.get("user_factors", {}))}
            factors = {factor.id: float(user_context.get(factor.id, ctx.get(factor.id, 0.5))) for factor in config.factors}
            diagnosis = DomainDecisionModel(config).evaluate(profile, factors)
            return domain_card(domain_key, diagnosis)

        if self._kde:
            try:
                result = self._kde.ask(message)
                output = getattr(result, 'output', result)
                try:
                    from sports_pro import DailyPlan
                    from prediction_engine import MatchPrediction, InjuryRiskPrediction
                except Exception:
                    DailyPlan = MatchPrediction = InjuryRiskPrediction = None
                if DailyPlan and isinstance(output, DailyPlan):
                    return plan_card(output)
                if MatchPrediction and isinstance(output, MatchPrediction):
                    return prediction_card(output)
                if InjuryRiskPrediction and isinstance(output, InjuryRiskPrediction):
                    return risk_card(output)
                if isinstance(output, list) and output and hasattr(output[0], 'risk_level'):
                    return squad_card(output)
                if hasattr(output, 'recommended') and hasattr(output, 'activations') and hasattr(output, 'moment'):
                    return moment_card(output)
                if isinstance(output, str):
                    return text_card(output)
                return text_card(str(output))
            except Exception as exc:
                logger.debug("KDE ask failed: %s", exc)

        if intent == "ksa_task" and self._ksa:
            try:
                return text_card(str(self._ksa.run(message)))
            except Exception:
                pass

        if intent == "identity":
            identity_data = {}
            if self._kde and hasattr(self._kde, 'identity'):
                try:
                    identity_data = self._kde.identity() or {}
                except Exception:
                    identity_data = {}
            elif self._kde and hasattr(self._kde, 'reflect'):
                try:
                    identity_data = self._kde.reflect() or {}
                except Exception:
                    identity_data = {}
            return identity_card(identity_data)

        if intent == "artifacts":
            return text_card("Artifacts are available via the /artifacts endpoint.", "Artifacts")
        if intent == "help":
            return text_card(
                "I can help with: plan my day · match prediction · squad risk · moment analysis · "
                "session footage · medical triage · financial portfolio · legal strategy · "
                "identity profile · developer tasks (scan files, search code).",
                "PRISM — What I can do",
            )
        if intent == "status":
            return text_card(
                f"Connected. KDE: {'active' if self._kde else 'offline'}. "
                f"KSA: {'active' if self._ksa else 'offline'}.",
                "Status",
            )
        if intent == "show_policies":
            from prism_responses import policy_view_card
            if hasattr(self, '_policy') and self._policy:
                data = self._policy.show_policies(self._user)
            else:
                data = {"allocations": {}, "note": "No policies set yet. "
                        "Try: 'set my food budget to £80'"}
            return policy_view_card(data)

        if intent == "update_policy":
            if hasattr(self, '_policy') and self._policy:
                result = self._policy.parse_policy_update(message, self._user)
                if result:
                    return text_card(result, "Policy updated")
            return text_card("Policy engine not configured.", "Policy")

        if intent == "task_status":
            from prism_responses import task_list_card
            tasks = self._queue.list_recent(5)
            return task_list_card(tasks)

        return text_card("I'm not sure how to help with that. Try: 'help'")
