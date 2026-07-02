"""
prism_planner.py
================
PRISM Universal Task Planner

Accepts any natural language task or problem description.
Uses an LLM to understand the task and extract its structure.
Uses the decision engine to rank all strategies.
Uses the LLM again to generate a concrete action plan per strategy.
Returns a complete PlanOfAction — the full strategy landscape
from optimal to least optimal, each with actionable steps.

The LLM understands. The physics engine ranks. Together they plan.

Flow:
  User describes any problem in plain language
        ↓
  LLM extracts TaskProfile (strategies, factors, constraints)
        ↓
  DecisionBeam ranks all strategies against user context
        ↓
  LLM generates concrete ActionPlan per strategy
        ↓
  PlanOfAction returned — full landscape, most to least optimal
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

from prism_llm_router import parse_llm_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ActionStep:
    """One concrete step within a strategy's execution plan."""
    order:       int
    action:      str          # what to do
    timeline:    str          # "day 1-3", "week 2", "by end of month"
    resource:    str          # what you need to do it
    outcome:     str          # what success looks like


@dataclass
class StrategyPlan:
    """One strategy with its full action plan and decision engine score."""
    name:             str
    position:         float      # on the 0-1 risk spectrum
    activation:       float      # probability mass from decision engine
    expected_value:   float      # payoff × probability
    risk_score:       float      # 0-100
    steps:            list[ActionStep]
    timeline:         str        # total duration
    resources:        list[str]  # what you need overall
    expected_outcome: str        # what success looks like
    risks:            list[str]  # what could go wrong
    why_recommended:  str        # plain English: why this ranks here


@dataclass
class PlanOfAction:
    """
    Complete output of the universal planner.
    Contains all strategies ranked from most to least optimal.
    """
    task:               str
    domain:             str
    entity:             str           # who is doing this
    timeline:           str
    fulcrum_position:   float         # where the context sits on the spectrum
    recommended:        StrategyPlan  # top-ranked strategy
    all_strategies:     list[StrategyPlan]   # ranked best → least optimal
    context_summary:    str           # plain English context interpretation
    generated_at:       float = field(default_factory=time.time)

    def top(self, n: int = 3) -> list[StrategyPlan]:
        return self.all_strategies[:n]

    def to_chat_response(self) -> str:
        """Format for the PRISM chat interface."""
        lines = [
            f"**Task:** {self.task}",
            f"**Context:** {self.context_summary}",
            "",
            f"**Optimal strategy → {self.recommended.name}** "
            f"({self.recommended.activation:.0%} confidence)",
            "",
        ]
        for i, s in enumerate(self.all_strategies[:4]):
            label = "★ Optimal" if i == 0 else f"Alt {i}"
            lines.append(
                f"{label}: **{s.name}** ({s.activation:.0%}) — {s.why_recommended}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are a task analysis assistant. A user has described a problem or task.
Extract its structure as a JSON object for a decision planning system.

TASK DESCRIPTION:
{task_description}

USER CONTEXT (what we know about this user):
{user_context}

Return ONLY valid JSON with this exact structure:
{{
  "domain": "one of: business, career, fitness, finance, learning, health, creative, technical, personal, social, other",
  "entity": "who is doing this (e.g. solo founder, team of 5, individual, student)",
  "timeline": "realistic total timeline (e.g. 30 days, 3 months, 1 year)",
  "strategies": [
    {{
      "name": "strategy name (max 6 words)",
      "position": 0.0 to 1.0,
      "payoff": 10 to 200,
      "cost": 1 to 50,
      "risk": 1 to 100,
      "probability": 0.1 to 0.95
    }}
  ],
  "factors": [
    {{
      "id": "snake_case_id",
      "label": "human readable label",
      "value": 0.0 to 1.0,
      "weight": 1.0 to 4.0,
      "direction": 1 or -1
    }}
  ],
  "context_summary": "one sentence describing the user's situation and what drives the recommendation"
}}

Rules:
- strategies: 6-9 options covering the full range from safest/cheapest to most aggressive/expensive.
  Position 0.0 = safest/most conservative. Position 1.0 = highest risk/most aggressive.
  Order strategies by position ascending.
- factors: 4-7 contextual factors. Extract from user context and task description.
  direction +1 means high value pushes toward aggressive strategies.
  direction -1 means high value pushes toward conservative strategies.
  value: 0.0 to 1.0 representing the user's current state on that factor.
- Do not include any text outside the JSON object.
- Respond with valid JSON only."""


PLAN_GENERATION_PROMPT = """You are a practical action planning assistant.
A decision system has ranked strategies for the following task.
Generate a concrete action plan for the strategy provided.

TASK: {task}
STRATEGY: {strategy_name}
RANK: {rank} of {total} strategies ({activation:.0%} confidence score)
TIMELINE: {timeline}
ENTITY: {entity}

Return ONLY valid JSON:
{{
  "steps": [
    {{
      "order": 1,
      "action": "specific concrete action",
      "timeline": "when to do this",
      "resource": "what you need",
      "outcome": "what success looks like"
    }}
  ],
  "resources": ["resource1", "resource2", "resource3"],
  "expected_outcome": "what success looks like overall",
  "risks": ["risk1", "risk2"],
  "why_recommended": "one sentence explaining why this strategy ranks here given the context"
}}

Rules:
- steps: 4-7 concrete steps. Be specific, not generic.
- timeline: use the overall timeline ({timeline}) to anchor step timings.
- why_recommended: reference the actual context (budget, time, skills) not generic advice.
- No text outside the JSON object."""


# ---------------------------------------------------------------------------
# Core planner
# ---------------------------------------------------------------------------

class PrismPlanner:
    """
    Universal task planner. Give it any problem in plain language.
    It returns a complete ranked plan of action.

    Example:
        planner = PrismPlanner.setup()
        plan = planner.plan(
            "I want to run a marathon in 6 months. I currently jog 2km once a week.",
            user_context={"fitness_level": 0.2, "time_per_week_hrs": 0.4}
        )
        print(plan.recommended.name)           # "Structured training plan"
        print(plan.recommended.steps[0].action) # "Week 1: run 3×2km easy pace"
    """

    def __init__(
        self,
        ollama_host:  str = "http://localhost:11434",
        ollama_model: str = "mistral",
        claude_api_key: Optional[str] = None,
        prefer_claude:  bool = True,
        request_timeout: float = 30.0,
        claude_model: str = "claude-opus-4-8",
        llm_router: Optional[Any] = None,
    ):
        self.ollama_host    = ollama_host
        self.ollama_model   = ollama_model
        self.claude_api_key = claude_api_key
        self.claude_model   = claude_model
        self.prefer_claude  = prefer_claude and bool(claude_api_key)
        # When the agent hands us its LLMRouter, planning follows whatever
        # provider the user picked in /settings/llm (Ollama / Claude /
        # OpenAI-compatible) — including hot config changes — instead of
        # this module's own boot-time Claude/Ollama wiring. The direct
        # paths below remain as fallback for standalone use.
        self._llm_router    = llm_router
        # Interactive chat path uses this planner too — 120s was killing
        # the UX for greetings like "good morning" (which legitimately
        # routes here) when the local Ollama model was slow. 30s is
        # generous for a 3B model and bearable as worst-case chat wait.
        # Override via the kwarg for batch/offline planning.
        self.request_timeout = request_timeout

    @classmethod
    def setup(cls, **kwargs) -> PrismPlanner:
        """One-line setup."""
        return cls(**kwargs)

    # ── Main entry point ─────────────────────────────────────────────────

    def plan(
        self,
        task_description: str,
        user_context: Optional[dict] = None,
        n_plans:          int = 4,      # how many strategies to generate full plans for
        identity_profile: Optional[dict] = None,  # from CrystallisationEngine if available
    ) -> PlanOfAction:
        """
        Given any task description, return a complete ranked plan of action.

        Parameters:
            task_description: plain language problem or goal
            user_context:     dict of factor_id → float (0-1) values known about the user
            n_plans:          number of strategies to generate full action plans for
            identity_profile: user's crystallised identity if available
        """
        # Enrich context with identity profile if available
        context = dict(user_context or {})
        if identity_profile:
            for domain, profile in identity_profile.get("domains", {}).items():
                context.setdefault(f"identity_{domain}", profile.get("value", 0.5))

        # Step 1: LLM extracts task structure
        logger.info("Extracting task structure for: %s", task_description[:60])
        task_profile = self._extract_task_profile(task_description, context)
        if task_profile is None:
            # Degrade before giving up: small local models routinely time
            # out or emit unparseable JSON on the big extraction prompt,
            # but can still answer a tiny plain-text prompt. A quick plan
            # beats an error card.
            simple = self._simple_plan(task_description)
            if simple is not None:
                return simple
            return self._fallback_plan(task_description)

        # Step 2: Decision engine ranks strategies
        logger.info("Ranking %d strategies", len(task_profile["strategies"]))
        ranked = self._rank_strategies(task_profile, context)

        # Step 3: LLM generates action plans for top N strategies
        logger.info("Generating action plans for top %d strategies", n_plans)
        strategy_plans = []
        total = len(ranked)
        for i, (activation, plank_data) in enumerate(ranked[:n_plans]):
            plan = self._generate_action_plan(
                task        = task_description,
                strategy    = plank_data,
                rank        = i + 1,
                total       = total,
                activation  = activation,
                timeline    = task_profile.get("timeline", "90 days"),
                entity      = task_profile.get("entity", "user"),
            )
            strategy_plans.append(plan)

        # Add remaining strategies without full plans
        for _i, (activation, plank_data) in enumerate(ranked[n_plans:]):
            strategy_plans.append(StrategyPlan(
                name             = plank_data["name"],
                position         = plank_data["position"],
                activation       = activation,
                expected_value   = plank_data["payoff"] * plank_data["probability"],
                risk_score       = plank_data["risk"],
                steps            = [],
                timeline         = task_profile.get("timeline", ""),
                resources        = [],
                expected_outcome = "",
                risks            = [],
                why_recommended  = "Alternative option — full plan available on request.",
            ))

        # Recompute fulcrum for reporting
        beam = self._build_beam(task_profile, context)
        diag = beam.evaluate()

        return PlanOfAction(
            task             = task_description,
            domain           = task_profile.get("domain", "general"),
            entity           = task_profile.get("entity", "user"),
            timeline         = task_profile.get("timeline", ""),
            fulcrum_position = diag.fulcrum_position,
            recommended      = strategy_plans[0],
            all_strategies   = strategy_plans,
            context_summary  = task_profile.get("context_summary", ""),
        )

    # ── Strategy ranking ─────────────────────────────────────────────────

    def _build_beam(self, task_profile: dict, context: dict):
        from decision_spectrum import DecisionBeam, DecisionPlank, Factor
        beam = DecisionBeam("task_planner", bandwidth=0.18)
        for s in task_profile.get("strategies", []):
            beam.add_plank(DecisionPlank(
                s["name"], s["position"], s["payoff"],
                s["cost"], s["risk"], s["probability"]
            ))
        base = 0.45
        for f in task_profile.get("factors", []):
            val = context.get(f["id"], f.get("value", 0.5))
            if f["direction"] > 0:
                target = min(1.0, base + val * f.get("range", 0.50))
            else:
                target = max(0.0, base - val * f.get("range", 0.50))
            beam.fulcrum.add_factor(Factor(
                f["id"], val, f.get("weight", 2.0), target,
                f.get("label", f["id"])
            ))
        return beam

    def _rank_strategies(
        self, task_profile: dict, context: dict
    ) -> list[tuple[float, dict]]:
        """Returns list of (activation, strategy_dict) sorted by activation desc."""
        beam = self._build_beam(task_profile, context)
        diag = beam.evaluate()
        result = []
        for act in diag.activations:
            strat = next(
                (s for s in task_profile["strategies"] if s["name"] == act.plank.name),
                None
            )
            if strat:
                result.append((act.activation, strat))
        return result

    # ── LLM calls ────────────────────────────────────────────────────────

    def _extract_task_profile(
        self, task: str, context: dict
    ) -> Optional[dict]:
        prompt = EXTRACTION_PROMPT.format(
            task_description = task,
            user_context     = json.dumps(context, indent=2) if context else "No additional context provided."
        )
        raw = self._call_llm(prompt)
        return parse_llm_json(raw)

    def _generate_action_plan(
        self,
        task:       str,
        strategy:   dict,
        rank:       int,
        total:      int,
        activation: float,
        timeline:   str,
        entity:     str,
    ) -> StrategyPlan:
        prompt = PLAN_GENERATION_PROMPT.format(
            task             = task,
            strategy_name    = strategy["name"],
            rank             = rank,
            total            = total,
            activation       = activation,
            timeline         = timeline,
            entity           = entity,
        )
        raw  = self._call_llm(prompt)
        data = parse_llm_json(raw) or {}
        steps = [
            ActionStep(
                order    = s.get("order", i+1),
                action   = s.get("action", ""),
                timeline = s.get("timeline", ""),
                resource = s.get("resource", ""),
                outcome  = s.get("outcome", ""),
            )
            for i, s in enumerate(data.get("steps", []))
        ]
        return StrategyPlan(
            name             = strategy["name"],
            position         = strategy["position"],
            activation       = activation,
            expected_value   = strategy["payoff"] * strategy["probability"],
            risk_score       = strategy["risk"],
            steps            = steps,
            timeline         = timeline,
            resources        = data.get("resources", []),
            expected_outcome = data.get("expected_outcome", ""),
            risks            = data.get("risks", []),
            why_recommended  = data.get("why_recommended", ""),
        )

    def _call_llm(self, prompt: str) -> str:
        if self._llm_router is not None:
            try:
                text, _model = self._llm_router.call(
                    prompt, min_capability=2, max_tokens=1500)
                if text:
                    return text
            except Exception as exc:
                logger.warning("Planner via LLMRouter failed: %s", exc)
        if self.prefer_claude and self.claude_api_key:
            return self._call_claude(prompt)
        return self._call_ollama(prompt)

    def _call_claude(self, prompt: str) -> str:
        payload = json.dumps({
            "model": self.claude_model,
            "max_tokens": 1500,
            "messages": [{"role": "user", "content": prompt}]
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data    = payload,
            headers = {
                "Content-Type":      "application/json",
                "anthropic-version": "2023-06-01",
                "x-api-key":         self.claude_api_key,
            },
            method = "POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                data = json.loads(resp.read())
            return data["content"][0]["text"]
        except Exception as e:
            logger.warning("Claude call failed: %s", e)
            return self._call_ollama(prompt)

    def _call_ollama(self, prompt: str, *,
                     num_predict: Optional[int] = None,
                     timeout: Optional[float] = None) -> str:
        body: dict = {
            "model":  self.ollama_model,
            "prompt": prompt,
            "stream": False,
        }
        if num_predict:
            body["options"] = {"num_predict": num_predict}
        payload = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.ollama_host}/api/generate",
            data    = payload,
            headers = {"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(
                    req, timeout=timeout or self.request_timeout) as resp:
                return json.loads(resp.read()).get("response", "")
        except urllib.error.HTTPError as e:
            logger.warning("Ollama call failed: HTTP %s — model '%s' may not exist",
                           e.code, self.ollama_model)
            self._last_ollama_error = f"HTTP {e.code} (model '{self.ollama_model}' not found?)"
            return ""
        except Exception as e:
            etype = type(e).__name__
            if "timed out" in str(e).lower() or "timeout" in etype.lower():
                self._last_ollama_error = (
                    f"model '{self.ollama_model}' timed out after {self.request_timeout:g}s — "
                    "too slow for structured planning"
                )
            else:
                self._last_ollama_error = f"{etype}: {e}"
            logger.warning("Ollama call failed: %s", e)
            return ""

    SIMPLE_PLAN_PROMPT = (
        "Make a short practical plan for: {task}\n\n"
        "Answer in EXACTLY this format and nothing else:\n"
        "GOAL: <the goal in one short line>\n"
        "1. <first step, under 15 words>\n"
        "2. <second step>\n"
        "3. <third step>\n"
        "You may add steps 4 and 5 if genuinely needed."
    )

    def _simple_plan(self, task: str) -> Optional[PlanOfAction]:
        """Single-shot degraded planning for weak/slow local models.

        Plain numbered lines instead of nested JSON, a capped response
        length, and a bounded timeout — a model that fails the full
        extraction prompt can usually still manage this. Returns None
        when even this fails, letting _fallback_plan explain honestly.
        """
        raw = self._call_ollama(
            self.SIMPLE_PLAN_PROMPT.format(task=task[:300]),
            num_predict=160,
            timeout=self.request_timeout,
        )
        if not raw:
            return None
        goal = ""
        steps_txt: list[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.upper().startswith("GOAL:"):
                goal = line[5:].strip()
                continue
            m = re.match(r"^(\d+)[.)]\s*(.+)$", line)
            if m:
                steps_txt.append(m.group(2).strip())
        if not steps_txt:
            return None
        steps = [
            ActionStep(order=i + 1, action=s, timeline="", resource="",
                       outcome="")
            for i, s in enumerate(steps_txt[:6])
        ]
        strat = StrategyPlan(
            name             = goal or "Quick plan",
            position         = 0.5,
            activation       = 1.0,
            expected_value   = 0,
            risk_score       = 0,
            steps            = steps,
            timeline         = "",
            resources        = [],
            expected_outcome = goal,
            risks            = [],
            why_recommended  = (
                "Quick single-shot plan — your local model is too small for "
                "full multi-strategy planning. For ranked alternatives try "
                "`ollama pull llama3.2:3b`."
            ),
        )
        return PlanOfAction(
            task=task, domain="general", entity="user", timeline="",
            fulcrum_position=0.5, recommended=strat, all_strategies=[strat],
            context_summary=(
                "Quick plan (simple mode — structured planning needs a "
                "stronger local model)."
            ),
        )

    def _fallback_plan(self, task: str) -> PlanOfAction:
        """Return a minimal plan when LLM is unavailable."""
        reason = getattr(self, "_last_ollama_error", "no local LLM reachable")
        if "timed out" in reason or "too slow" in reason:
            hint = (
                f"Your local model ({self.ollama_model}) is too slow for full planning. "
                "Try a larger Ollama model: `ollama pull llama3.2:3b` or `ollama pull qwen2.5:3b`."
            )
        elif "not found" in reason or "HTTP 404" in reason:
            hint = (
                f"Model '{self.ollama_model}' is not installed. "
                f"Run: `ollama pull {self.ollama_model}` (or change [agent].text_model in prism_config.toml)."
            )
        else:
            hint = "Start Ollama with: `ollama serve` — or set [llm].claude_api_key in prism_config.toml."
        stub = StrategyPlan(
            name="Manual planning required", position=0.5, activation=1.0,
            expected_value=0, risk_score=0, steps=[], timeline="",
            resources=[], expected_outcome="",
            risks=[f"Planner LLM failed: {reason}"],
            why_recommended=hint,
        )
        return PlanOfAction(
            task=task, domain="unknown", entity="user",
            timeline="", fulcrum_position=0.5,
            recommended=stub, all_strategies=[stub],
            context_summary=f"Planner LLM unavailable — {reason}. {hint}"
        )
