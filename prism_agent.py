from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from domain_configs import ALL_DOMAINS, DomainDecisionModel
from kde_agent import KDEAgent, KDEConfig
from ksa_agent import KSAgent
from prism_responses import (
    CardType,
    PrismCard,
    domain_card,
    identity_card,
    moment_card,
    plan_card,
    prediction_card,
    risk_card,
    text_card,
)
from sports_pro import Role

logger = logging.getLogger(__name__)


class PrismAgent:
    """
    Unified PRISM agent wrapping KDEAgent + KSAgent.
    """

    INTENTS = [
        ("plan my day|morning|daily plan|today", "plan"),
        ("predict|match prediction|fixture", "predict_match"),
        ("injury|risk|squad|fitness", "injury_risk"),
        ("moment|1v1|keeper|analyse moment", "moment"),
        ("footage|video|session|analyse", "session"),
        ("transfer|value|market", "transfer"),
        ("scouting|opponent|scout", "scouting"),
        ("triage|medical|patient|symptoms", "domain_medical"),
        ("portfolio|invest|allocation|financial", "domain_financial"),
        ("legal|case|litigat|settle", "domain_legal"),
        ("hire|hiring|talent|recruitment", "domain_hr"),
        ("supply|procurement|inventory", "domain_supply"),
        ("my profile|identity|who am i|digital dna|crystal", "identity"),
        ("artifacts|history|what have i", "artifacts"),
        ("index|scan|file|search|grep", "ksa_task"),
        ("run|execute|shell|command", "ksa_task"),
        ("status|connected|devices", "status"),
        ("help|what can you", "help"),
    ]

    def __init__(
        self,
        kde_agent: Optional[KDEAgent] = None,
        ksa_agent: Optional[KSAgent] = None,
        ollama_host: str = "http://localhost:11434",
        text_model: str = "mistral",
    ):
        self.kde_agent = kde_agent
        self.ksa_agent = ksa_agent
        self.ollama_host = ollama_host.rstrip("/")
        self.text_model = text_model
        self._domain_models = {
            name: DomainDecisionModel(config)
            for name, config in ALL_DOMAINS.items()
        }

    @classmethod
    def setup(
        cls,
        name: str,
        role: Role,
        sport: str = "Football",
        team: str = "",
        db_path: str = "~/.prism/prism.db",
    ) -> "PrismAgent":
        prism_db = str(Path(db_path).expanduser())
        kde_cfg = KDEConfig(
            db_path=prism_db,
            media_dir=str(Path("~/.prism/media").expanduser()),
            ollama_host="http://localhost:11434",
            ollama_model="llava",
            text_model="mistral",
        )
        kde_agent = KDEAgent.setup(name=name, role=role, sport=sport, team=team, config=kde_cfg)
        ksa_agent = KSAgent(
            db_path=prism_db,
            working_dir=".",
            ollama_model=None,
            auto_optimise=False,
            dry_run=True,
        )
        return cls(kde_agent=kde_agent, ksa_agent=ksa_agent, ollama_host=kde_cfg.ollama_host, text_model=kde_cfg.text_model)

    def chat(self, message: str, context: dict = None) -> PrismCard:
        context = context or {}
        try:
            message = (message or "").strip()
            if not message:
                return self._bootstrap_card()
            intent = self._route(message)
            return self._execute_intent(intent, message, context)
        except Exception as exc:
            logger.exception("PRISM chat failed")
            return PrismCard(CardType.ERROR, "PRISM error", str(exc), {}, actions=["Try again", "Show help"])

    def _route(self, message: str) -> str:
        text = (message or "").lower()
        for pattern, intent in self.INTENTS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return intent
        llm_intent = self._classify_with_ollama(message)
        return llm_intent or "bootstrap"

    def _classify_with_ollama(self, message: str) -> Optional[str]:
        payload = {
            "model": self.text_model,
            "stream": False,
            "prompt": (
                "Return one PRISM intent label only: plan, predict_match, injury_risk, "
                "moment, session, transfer, scouting, domain_medical, domain_financial, "
                "domain_legal, domain_hr, domain_supply, identity, artifacts, ksa_task, "
                "status, help, bootstrap.\nMessage: " + message
            ),
        }
        try:
            req = urllib.request.Request(
                f"{self.ollama_host}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            response = str(data.get("response", "")).strip().splitlines()[0].strip().lower()
            valid = {intent for _, intent in self.INTENTS} | {"bootstrap"}
            return response if response in valid else None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, OSError):
            return None

    def _execute_intent(self, intent: str, message: str, context: dict) -> PrismCard:
        if intent == "plan":
            if self.kde_agent and hasattr(self.kde_agent, "morning_briefing"):
                brief = self.kde_agent.morning_briefing()
                return plan_card(getattr(brief, "plan", brief))
            return text_card("No sports planner is connected yet.", "Daily plan")

        if intent == "predict_match":
            output = self._ask_kde(message)
            if hasattr(output, "p_home_win") or hasattr(output, "p_draw"):
                return prediction_card(output)
            return text_card(self._stringify(output), "Match prediction")

        if intent == "injury_risk":
            output = self._ask_kde(message)
            if hasattr(output, "risk_level"):
                return risk_card(output)
            return text_card(self._stringify(output), "Risk assessment")

        if intent == "moment":
            output = self._ask_kde(message)
            if hasattr(output, "recommended") and hasattr(output, "activations"):
                return moment_card(output)
            return text_card(self._stringify(output), "Moment analysis")

        if intent.startswith("domain_"):
            domain_name = {
                "domain_medical": "Medical",
                "domain_financial": "Financial",
                "domain_legal": "Legal",
                "domain_hr": "HR",
                "domain_supply": "Supply Chain",
            }.get(intent, "Medical")
            diagnosis = self._evaluate_domain(domain_name, message, context)
            return domain_card(domain_name, diagnosis)

        if intent == "identity":
            return identity_card(self.reflect())

        if intent == "artifacts":
            tasks = []
            if self.ksa_agent is not None and hasattr(self.ksa_agent, "registry"):
                try:
                    tasks = list(self.ksa_agent.registry.list_tasks())
                except Exception:
                    tasks = []
            return PrismCard(
                CardType.ARTIFACTS,
                "Artifacts",
                f"{len(tasks)} artifact record(s) available.",
                {"artifacts": tasks},
                actions=["Show recent work", "Search artifacts"],
            )

        if intent == "ksa_task":
            if self.ksa_agent and hasattr(self.ksa_agent, "run"):
                try:
                    outcome = self.ksa_agent.run(message)
                    body = getattr(outcome, "stdout", "") or getattr(outcome, "stderr", "") or str(outcome)
                    return text_card(body.strip() or str(outcome), "Developer task")
                except Exception as exc:
                    return text_card(f"Developer task unavailable: {exc}", "Developer task")
            return text_card("Developer task routing is not configured yet.", "Developer task")

        if intent == "status":
            return text_card(json.dumps(self.status(), default=str, indent=2), "Status")

        if intent == "help":
            return self._help_card()

        if intent in {"session", "transfer", "scouting"}:
            output = self._ask_kde(message)
            return text_card(self._stringify(output), "PRISM")

        return self._bootstrap_card()

    def _ask_kde(self, message: str):
        if self.kde_agent and hasattr(self.kde_agent, "ask"):
            result = self.kde_agent.ask(message)
            return getattr(result, "output", result)
        return "KDE agent unavailable"

    def _evaluate_domain(self, domain_name: str, message: str, context: dict):
        model = self._domain_models[domain_name]
        config = ALL_DOMAINS[domain_name]
        profile = context.get("profile") or config.profiles[0].name
        factors = {factor.id: 0.5 for factor in config.factors}
        lowered = message.lower()
        for factor in config.factors:
            if factor.id in lowered or factor.label.lower() in lowered:
                factors[factor.id] = 0.8
        if "urgent" in lowered or "emergency" in lowered or "severe" in lowered:
            for key in ("severity", "vital_signs", "deteriorating"):
                if key in factors:
                    factors[key] = 0.85
        return model.evaluate(profile, factors)

    def _help_card(self) -> PrismCard:
        return text_card(
            "Try: plan my day, predict Arsenal vs City, assess injury risk, analyse a moment, triage chest pain, or scan my files.",
            "PRISM help",
        )

    def _bootstrap_card(self) -> PrismCard:
        return PrismCard(
            CardType.TEXT,
            "Welcome to PRISM",
            "PRISM routes developer, sport, and domain questions through one local chat interface.",
            {},
            actions=[
                "Plan my day",
                "Predict the next match",
                "Assess injury risk",
                "Triage a patient",
            ],
        )

    def _stringify(self, value) -> str:
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, default=str, indent=2)
        except TypeError:
            return str(value)

    def status(self) -> dict:
        kde_status = {}
        if self.kde_agent and hasattr(self.kde_agent, "status"):
            try:
                kde_status = dict(self.kde_agent.status())
            except Exception:
                kde_status = {"kde": "unavailable"}
        ksa_status = {}
        if self.ksa_agent and hasattr(self.ksa_agent, "status"):
            try:
                ksa_status = dict(self.ksa_agent.status())
            except Exception:
                ksa_status = {"ksa": "unavailable"}
        return {"kde": kde_status, "ksa": ksa_status}

    def reflect(self) -> dict:
        if self.kde_agent and hasattr(self.kde_agent, "reflect"):
            try:
                data = self.kde_agent.reflect()
                return data if isinstance(data, dict) else {"value": data}
            except Exception as exc:
                return {"error": str(exc)}
        return {"profile": "Unknown", "fixed_fulcrum": 0.5, "total_ratings": 0, "total_plans": 0}
