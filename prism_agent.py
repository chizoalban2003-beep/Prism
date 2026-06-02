from __future__ import annotations

import json
import logging
import re
import urllib.request
from pathlib import Path
from typing import Optional

from prism_llm_router import LLMRouter
from prism_task_queue import TaskQueue
from domain_configs import ALL_DOMAINS, DomainDecisionModel
from prism_device_agent import PrismDeviceAgent
from prism_planner import PrismPlanner
from prism_perception import PrismPerception
from prism_memory import PrismMemory
from prism_tts import PrismTTS
from prism_proactive import PrismProactive, build_default_triggers
from prism_smart_home import PrismSmartHome
from prism_email    import PrismEmail
from prism_calendar import PrismCalendar
from prism_browser_agent import PrismBrowserAgent
from prism_instructions import PrismInstructions
from prism_service_discovery import PrismServiceDiscovery
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

from prism_search import PrismSearch
from prism_push   import PrismPush
from prism_contacts import PrismContacts
from prism_tasks    import PrismTasks
from prism_calibration import PrismCalibration
from prism_autonomous import PrismAutonomous

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
        (r"(?:read|check|show|any|my) (?:new )?(?:emails?|inbox|messages?)|"
         r"unread|what(?:'s| came) in",                        "email_read"),
        (r"(?:send|reply|write|draft) (?:an? )?email|"
         r"email (?:to|them|him|her)",                          "email_send"),
        (r"(?:what(?:'s| is) on my|check my|show) (?:calendar|schedule|agenda)|"
         r"(?:any|my) (?:meetings?|appointments?|events?) (?:today|tomorrow|this week)",
                                                                "calendar_read"),
        (r"(?:schedule|book|create|add) (?:a )?(?:meeting|event|appointment)|"
         r"(?:find|when(?:'s| is) the next) (?:free|available) (?:slot|time)",
                                                                "calendar_write"),
        (r"(?:go to|open|browse|visit|search (?:the )?web|find (?:on|online)|"
         r"look up|book|reserve|fill (?:in|out)|check (?:the )?(?:price|availability)|"
         r"what(?:'s| is) (?:on|the) website)",  "browser_task"),
        (r"show (?:my )?(?:instructions?|rules?|standing orders?)|"
         r"what (?:have you )?(?:remember|know) about my preferences",
         "show_instructions"),
        (r"(?:forget|remove|delete) (?:that |the )?(?:instruction|rule)|"
         r"stop (?:always|never)", "remove_instruction"),
        (r"(?:use|connect|integrate|set up|configure|add) (?:with )?(?!my )(?!the )"
         r"(?:[A-Z][a-z]+(?:\s[A-Z][a-z]+)*|[a-z]+\.[a-z]+)|"
         r"(?:can you|how do i) (?:use|access|connect to) ", "discover_service"),
        (r"search (?:the web|online|internet|for)|"
         r"look up|find (?:out|info|information)|"
         r"what(?:'s| is) (?:the )?(?:latest|current|today)|"
         r"research|who is|where is|when (?:did|does|is)",
         "web_search"),
        (r"(?:send|push) (?:me )?(?:a )?(?:notification|alert|reminder)|"
         r"notify me|ping me|alert me",
         "send_push"),
        (r"(?:find|search|look up|who is|contact|call|email) (?:my )?(?:contact|person|colleague|client|friend)",
         "contacts"),
        (r"(?:add|create|make|new) (?:a )?(?:task|todo|reminder|ticket|issue)|"
         r"(?:i need to|i have to|remember to|don't forget)",
         "add_task"),
        (r"(?:my )?(?:tasks?|todos?|to-do|to do|what(?:'s| is) (?:on my )?list|"
         r"pending|backlog|open issues?)",
         "list_tasks"),
        (r"(?:that was|you were|that(?:'s| is)) (?:wrong|right|too|not|off|correct|"
         r"perfect|bad|good)|(?:i (?:disagree|agree|wouldn't|would)|"
         r"too (?:aggressive|cautious|risky|safe|bold|timid)|"
         r"next time (?:consider|weight|prioritise)|"
         r"(?:more|less) (?:important|weight|focus))", "calibrate"),
        (r"(?:how am i|calibration|what have you learned|"
         r"how (?:accurate|well) (?:are you|is prism)|"
         r"show (?:my )?feedback history)", "calibration_summary"),
        (r"remind me|set (?:a )?reminder|alert me (?:in|at|when)|"
         r"don't let me forget|in (\d+) (?:minute|hour|day)|"
         r"at (\d+(?::\d+)?(?:am|pm)?)", "reminder"),
        (r"^(?:yes[,.]?|yeah[,.]?|go ahead|approved?|confirm|do it|proceed)[\s!.]*$",
         "approve_pending"),
        (r"^(?:no[,.]?|cancel|stop|don't|abort|never mind)[\s!.]*$",
         "cancel_pending"),
        (r"(?:what tools|learned tools|acquired tools|"
         r"what can you now do|new capabilities|tool list)",
         "list_tools"),
        (r"help|what\.can|commands|options", "help"),
        (r"turn (?:on|off)|set (?:the )?(?:lights?|thermostat|temp)|"
         r"lock|unlock|what(?:'s| is) (?:on|off)|smart home|home assistant",
         "smart_home"),
        # NOTE: broad email catch-all — maps to email_read to avoid duplication
        # with the more specific email_read/email_send intents above.
        (r"(?:check|read|show|open|fetch|get|list).*(?:email|inbox|mail)|"
         r"(?:email|mail).*(?:unread|new|recent)|send.*(?:email|mail)|"
         r"draft.*(?:email|reply)|reply.*email|email.*summary",
         "email_read"),
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

        # Load config early so all subsequent setup can use it
        self._config = {}
        self._user = "default"
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ImportError:
                tomllib = None  # type: ignore[assignment]
        if tomllib:
            _config_path = Path(__file__).parent / "prism_config.toml"
            try:
                with open(_config_path, "rb") as f:
                    self._config = tomllib.load(f)
            except Exception:
                pass
        self._user = self._config.get("user", {}).get("name", "default")

        # Build LLMRouter with claude_api_key from config or constructor arg
        _llm_cfg = self._config.get("llm", {})
        _claude_key_cfg = (_llm_cfg.get("claude_api_key", "")
                           or self._claude_key or "")
        self._router = LLMRouter.from_config(
            claude_api_key=_claude_key_cfg) if _claude_key_cfg else LLMRouter.from_config()

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
            self._memory = PrismMemory(ollama_host=ollama_host)
        except Exception as e:
            logger.warning("PrismMemory not available: %s", e)
            self._memory = None
        self._tts = PrismTTS.setup()
        # Config is loaded above — construct with real config directly
        self._smarthome = PrismSmartHome.from_config(self._config)
        self._email    = PrismEmail.from_config(self._config)
        self._calendar = PrismCalendar.from_config(self._config)
        self._browser  = PrismBrowserAgent.setup(
            llm_router = getattr(self, '_router', None),
            headless   = True,
        )
        self._instructions = PrismInstructions()
        self._discovery    = PrismServiceDiscovery(
            collaborator  = getattr(self, '_collaborator', None),
            tool_registry = getattr(
                getattr(self, '_device', None), '_registry', None),
        )
        self._chat_history: list[dict] = []
        try:
            cfg = self._config.get("agent", {}) if hasattr(self, '_config') and self._config else {}
            self._perception = PrismPerception.setup(
                enable_voice     = cfg.get("enable_voice", False),
                enable_screen    = cfg.get("enable_screen", False),
                enable_typing    = cfg.get("enable_typing", True),
                enable_system    = cfg.get("enable_system", True),
                enable_biometric = cfg.get("enable_biometric", True),
                on_voice_command = self.chat,
            )
            self._perception.start()
        except Exception as e:
            logger.warning("PrismPerception not available: %s", e)
            self._perception = None
        try:
            self._proactive = PrismProactive(
                on_event=self._handle_proactive_event)
            triggers = build_default_triggers(
                perception    = getattr(self, '_perception', None),
                policy_engine = getattr(self, '_policy', None),
                task_queue    = getattr(self, '_queue', None),
            )
            for t in triggers:
                self._proactive.register(t)
            self._proactive.start()
        except Exception as e:
            logger.warning("PrismProactive not available: %s", e)
            self._proactive = None

        self._search = PrismSearch.from_config(self._config)
        self._push   = PrismPush.from_config(self._config)
        if self._proactive:
            self._proactive._push = self._push
        self._contacts     = PrismContacts.from_config(self._config)
        self._task_mgr     = PrismTasks.from_config(self._config)
        self._calibration  = PrismCalibration()
        self._last_decision: dict = {}
        self._autonomous = PrismAutonomous(
            llm_router   = self._router,
            device_agent = self._device,
            policy_engine= getattr(self, '_policy', None),
            push         = self._push,
            task_queue   = self._queue,
        )

        # Re-construct email/calendar/smarthome with the real config now that
        # prism_config.toml has been loaded.  The initial construction above
        # used {} so these modules default to "unconfigured"; this pass wires
        # any credentials the user has provided in prism_config.toml.
        if self._config:
            self._smarthome = PrismSmartHome.from_config(self._config)
            self._email     = PrismEmail.from_config(self._config)
            self._calendar  = PrismCalendar.from_config(self._config)

    def _handle_proactive_event(self, event) -> None:
        """Store proactive notification for chat UI polling."""
        if hasattr(self, '_proactive_buffer'):
            self._proactive_buffer.append(event)
        else:
            self._proactive_buffer = [event]

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
        context = context or {}
        try:
            # 1. Check for standing instruction (store if detected)
            stored_instruction = self._instructions.parse_from_chat(message or "")
            if stored_instruction:
                return text_card(
                    f"✓ Remembered: {stored_instruction.text}",
                    "Instruction stored")

            # 2. Inject relevant instructions into context
            instructions_str = self._instructions.to_context_string(message or "")
            if instructions_str:
                context["standing_instructions"] = instructions_str

            # 3. Inject conversation history
            context["history"] = self._chat_history[-10:]

            # 4. Add to history
            self._chat_history.append({"role": "user", "content": message or ""})
            if len(self._chat_history) > 20:
                self._chat_history = self._chat_history[-20:]

            # 5. Perception context
            if self._perception:
                percept_state = self._perception.current_context()
                context["perception"] = percept_state.to_factor_updates()
                context["perception_summary"] = percept_state.summary

            # 6. Memory context
            if self._memory and message:
                try:
                    mem_results = self._memory.search(message, top_n=3)
                    if mem_results:
                        context["memory_context"] = [
                            {"title": r.entry.title, "excerpt": r.excerpt,
                             "source": r.entry.source, "score": round(r.score, 3)}
                            for r in mem_results
                        ]
                    self._memory.ingest_conversation("user", message)
                except Exception:
                    pass

            # 7. Route intent and execute
            intent = self._route(message or "")
            card = self._execute(intent, message or "", context)

            # 8. Store response in history
            if hasattr(card, 'body') and card.body:
                self._chat_history.append(
                    {"role": "assistant", "content": card.body[:500]})

            # 9. Memory ingestion for response
            if self._memory and card.body:
                try:
                    self._memory.ingest_conversation("assistant", card.body)
                except Exception:
                    pass

            if self._tts:
                self._tts.speak(card.body or "")
            return card
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
            _policy = getattr(self, '_policy', None)
            _user   = getattr(self, '_user', 'default')
            if _policy:
                data = _policy.show_policies(_user)
            else:
                data = {"allocations": {}, "note": "No policies set yet. "
                        "Try: 'set my food budget to £80'"}
            return policy_view_card(data)

        if intent == "update_policy":
            _policy = getattr(self, '_policy', None)
            _user   = getattr(self, '_user', 'default')
            if _policy:
                result = _policy.parse_policy_update(message, _user)
                if result:
                    return text_card(result, "Policy updated")
            return text_card("Policy engine not configured.", "Policy")

        if intent == "task_status":
            from prism_responses import task_list_card
            tasks = self._queue.list_recent(5)
            return task_list_card(tasks)

        if intent == "smart_home":
            if not self._smarthome.configured:
                return text_card(
                    "Smart home not configured. "
                    "Add ha_url and ha_token to prism_config.toml.",
                    "Smart Home")
            msg_lower = message.lower()
            entity = None
            for word in message.split():
                found = self._smarthome.find_entity(word)
                if found:
                    entity = found
                    break
            if "turn on" in msg_lower and entity:
                ok = self._smarthome.turn_on(entity.entity_id)
                return text_card(
                    f"{'Done' if ok else 'Failed'}: {entity.friendly_name} on",
                    "Smart Home")
            if "turn off" in msg_lower and entity:
                ok = self._smarthome.turn_off(entity.entity_id)
                return text_card(
                    f"{'Done' if ok else 'Failed'}: {entity.friendly_name} off",
                    "Smart Home")
            summary = self._smarthome.status_summary()
            return text_card(
                f"{summary['on_count']} devices on · "
                f"{summary['total_entities']} total · "
                f"domains: {', '.join(summary.get('domains', [])[:5])}",
                "Smart Home Status")

        if intent == "email_read":
            if not self._email.configured:
                return text_card("Email not configured. "
                                 "Add email settings to prism_config.toml.", "Email")
            messages = self._email.fetch_unread(n=10)
            summary  = self._email.summarise_inbox(
                messages, llm_router=getattr(self, '_router', None))
            return text_card(summary, f"Inbox — {len(messages)} unread")

        if intent == "email_send":
            if not self._email.configured:
                return text_card("Email not configured.", "Email")
            router = getattr(self, '_router', None)
            if router:
                prompt = (f"Extract email details from: '{message}'\n"
                          f"Return JSON: {{\"to\":\"...\",\"subject\":\"...\","
                          f"\"body\":\"...\"}}")
                raw, _ = router.call(prompt, min_capability=1, max_tokens=300,
                                      json_mode=True)
                try:
                    import json as _j
                    data = _j.loads(raw.strip().lstrip("```json").rstrip("```"))
                    ok   = self._email.send(data["to"], data["subject"], data["body"])
                    return text_card(
                        f"{'Sent' if ok else 'Failed to send'} to {data.get('to','')}",
                        "Email")
                except Exception:
                    pass
            return text_card("Could not parse email details. "
                             "Try: 'send email to name@example.com about...'", "Email")

        if intent == "calendar_read":
            if not self._calendar.configured:
                return text_card("Calendar not configured. "
                                 "Add calendar settings to prism_config.toml.", "Calendar")
            today    = self._calendar.today()
            next_ev  = self._calendar.next_event()
            if not today:
                msg = "Nothing scheduled today."
            else:
                msg = "\n".join(str(e) for e in today)
            if next_ev and next_ev.starts_in_mins <= 30:
                msg = f"⚠ {next_ev.title} starts in {next_ev.starts_in_mins} minutes\n\n" + msg
            return text_card(msg, f"Today — {len(today)} events")

        if intent == "calendar_write":
            if not self._calendar.configured:
                return text_card("Calendar not configured.", "Calendar")
            router = getattr(self, '_router', None)
            if "free slot" in message.lower() or "available" in message.lower():
                slot = self._calendar.find_free_slot()
                if slot:
                    return text_card(
                        f"Next free slot: {slot.strftime('%a %d %b at %H:%M')}",
                        "Calendar")
                return text_card("No free slots found in the next 48 hours.", "Calendar")
            parsed = self._calendar.parse_event_from_text(message, router)
            if parsed and parsed.get("start_iso"):
                from datetime import datetime as _dt
                start = _dt.fromisoformat(parsed["start_iso"])
                event = self._calendar.create_event(
                    title        = parsed.get("title", "New Event"),
                    start        = start,
                    duration_mins= parsed.get("duration_mins", 60),
                    location     = parsed.get("location", ""),
                    attendees    = parsed.get("attendees", []),
                )
                if event:
                    return text_card(f"Created: {event}", "Calendar")
            return text_card("Could not parse event details. "
                             "Try: 'schedule a meeting with X on Friday at 2pm'", "Calendar")

        if intent == "browser_task":
            if not self._browser.available:
                return text_card(
                    "Browser agent not available. "
                    "Install with: pip install playwright && playwright install chromium",
                    "Browser")
            queue = getattr(self, '_queue', None)
            if queue:
                def run_browser():
                    return self._browser.execute(message)
                task_id = queue.submit_single(f"Browser: {message[:40]}", run_browser)
                return text_card(
                    f"Browser task started. I'll let you know when done.\n"
                    f"Task ID: {task_id}",
                    "Browser Task")
            else:
                result = self._browser.execute(message)
                body   = result.extracted[:500] if result.success else result.error
                return text_card(body, "Browser Result")

        if intent == "show_instructions":
            instrs = self._instructions.all_active()
            if not instrs:
                return text_card("No standing instructions set. "
                                 "Tell me to 'always...' or 'never...' "
                                 "to set one.", "Standing Instructions")
            lines = "\n".join(f"• [{i.trigger}] {i.text}" for i in instrs)
            return text_card(lines, f"Your instructions ({len(instrs)})")

        if intent == "remove_instruction":
            instrs = self._instructions.all_active()
            if instrs:
                for instr in reversed(instrs):
                    if any(w in message.lower()
                           for w in instr.text.lower().split()[:3]):
                        self._instructions.remove(instr.instr_id)
                        return text_card(f"Removed: {instr.text}",
                                         "Instruction removed")
            return text_card("Couldn't find a matching instruction to remove.",
                             "Instructions")

        if intent == "discover_service":
            router = getattr(self, '_router', None)
            if router:
                name_prompt = (f"Extract the service/app/platform name from: "
                               f"'{message}'. Return ONLY the name, nothing else.")
                service_name, _ = router.call(name_prompt, min_capability=1,
                                              max_tokens=20)
                service_name = service_name.strip().strip('"\'')
            else:
                import re as _re
                words = _re.findall(r'[A-Z][a-zA-Z]+', message)
                service_name = words[0] if words else "unknown service"

            if not service_name:
                service_name = "unknown service"

            if self._discovery.is_known(service_name):
                existing = self._discovery.get(service_name)
                if existing and existing.configured:
                    return text_card(
                        f"I already have {service_name} connected "
                        f"via {existing.access_method}. "
                        f"What would you like to do with it?",
                        f"{service_name} — already integrated")

            service, questions = self._discovery.discover(
                service_name = service_name,
                user_intent  = message,
                constraints  = ctx.get("user_constraints", {}),
            )
            steps_text = "\n".join(f"{i+1}. {s}"
                                   for i, s in enumerate(service.setup_steps))
            q_text     = "\n".join(f"• {q}" for q in questions[:2])
            body = (
                f"I've researched **{service_name}** — {service.description}\n\n"
                f"Best integration method: **{service.access_method}**\n\n"
                f"To set this up:\n{steps_text}"
                + (f"\n\nI also need a few answers:\n{q_text}" if q_text else "")
            )
            return text_card(body, f"Connecting: {service_name}")

        if intent == "web_search":
            results = self._search.search(message, n=5)
            if not results:
                answer = self._search.quick_answer(message)
                if answer:
                    return text_card(answer, "Search result")
                return text_card("No results found.", "Search")
            router = getattr(self, '_router', None)
            if router and results:
                context_str = "\n".join(
                    f"{r.title}: {r.snippet}" for r in results[:4])
                prompt  = (f"Answer this query using the search results below.\n"
                           f"Query: {message}\nResults:\n{context_str}\n"
                           f"Give a concise factual answer in 2-3 sentences.")
                answer, _ = router.call(
                    prompt, min_capability=1, max_tokens=300,
                    conversation_history=self._chat_history[-4:])
                body = answer or "\n".join(
                    f"• {r.title}  {r.url}" for r in results[:4])
            else:
                body = "\n".join(
                    f"• {r.title}\n  {r.snippet}\n  {r.url}"
                    for r in results[:4])
            return text_card(body, f"Search · {self._search.status_summary()['provider']}")

        if intent == "send_push":
            if not self._push.configured:
                return text_card(
                    "Push not configured. Add topic to prism_config.toml [push]. "
                    "Get the free ntfy app at ntfy.sh — no account needed.",
                    "Push notifications")
            self._push.alert(message)
            return text_card("Notification sent to your device.", "Push")

        if intent == "contacts":
            query = message.lower().replace("find","").replace(
                "contact","").replace("who is","").strip()
            contacts = self._contacts.search(query)
            if not contacts:
                return text_card(f"No contact found for '{query}'.", "Contacts")
            c = contacts[0]
            lines = [f"{c.name}"]
            if c.organisation: lines.append(f"  {c.role} at {c.organisation}")
            if c.emails:  lines.append(f"  Email: {', '.join(c.emails)}")
            if c.phones:  lines.append(f"  Phone: {', '.join(c.phones)}")
            if c.notes:   lines.append(f"  Notes: {c.notes[:200]}")
            return text_card("\n".join(lines),
                              f"Contact · {c.source}")

        if intent == "add_task":
            router = getattr(self, '_router', None)
            parsed = None
            if router:
                prompt = (f"Extract task details from: '{message}'. "
                          f"Return JSON: {{\"title\":\"...\",\"notes\":\"...\","
                          f"\"due_date\":\"YYYY-MM-DD or empty\","
                          f"\"priority\":1}}")
                raw, _ = router.call(prompt, min_capability=1, max_tokens=200,
                                      json_mode=True)
                try:
                    import json as _j
                    clean = raw.strip().lstrip("```json").rstrip("```").strip()
                    parsed = _j.loads(clean)
                except Exception: pass
            if parsed:
                task = self._task_mgr.add(
                    title    = parsed.get("title", message[:80]),
                    notes    = parsed.get("notes",""),
                    due_date = parsed.get("due_date",""),
                    priority = parsed.get("priority",1),
                )
                return text_card(
                    f"Added: {task.title}"
                    + (f"  Due: {task.due_date}" if task.due_date else ""),
                    f"Task added · {task.source}")
            task = self._task_mgr.add(title=message[:80])
            return text_card(f"Added: {task.title}", "Task added")

        if intent == "list_tasks":
            tasks = self._task_mgr.list_tasks(done=False)
            if not tasks:
                return text_card("No open tasks.", "Tasks")
            provider = self._task_mgr._resolve_provider()
            lines = "\n".join(
                f"{'⚡' if t.priority>=3 else '·'} {t.title}"
                + (f"  (due {t.due_date})" if t.due_date else "")
                for t in tasks[:15])
            return text_card(lines, f"Tasks ({len(tasks)}) · {provider}")

        if intent == "reminder":
            router = getattr(self, '_router', None)
            parsed_time = None
            if router:
                prompt = (f"Extract reminder details from: '{message}'. "
                          f"Return JSON: {{\"message\":\"...\","
                          f"\"seconds_from_now\": <integer seconds or null>,"
                          f"\"iso_datetime\": \"YYYY-MM-DDTHH:MM or null\"}}")
                raw, _ = router.call(prompt, min_capability=1, max_tokens=150, json_mode=True)
                try:
                    import json as _j
                    clean = raw.strip().lstrip("```json").rstrip("```").strip()
                    parsed_time = _j.loads(clean)
                except Exception:
                    pass
            if parsed_time and self._proactive:
                msg = parsed_time.get("message", message)
                secs = parsed_time.get("seconds_from_now")
                iso  = parsed_time.get("iso_datetime")
                if secs:
                    self._proactive.schedule_in(msg, float(secs))
                    mins = int(float(secs) // 60)
                    return text_card(f"Reminder set: '{msg}' in {mins} minutes.", "Reminder")
                elif iso:
                    from datetime import datetime as _dt
                    try:
                        fire_at = _dt.fromisoformat(iso).timestamp()
                        self._proactive.schedule(msg, fire_at)
                        return text_card(f"Reminder set: '{msg}' at {iso}.", "Reminder")
                    except Exception:
                        pass
            return text_card("Could not parse reminder time. Try: 'remind me in 30 minutes to call Alice'.", "Reminder")

        if intent == "calibrate":
            direction = self._calibration.detect(message)
            if not direction:
                return text_card(
                    "I didn't quite catch that as feedback. "
                    "Try: 'that was too aggressive' or 'good call' "
                    "or 'next time weight cost more heavily'.",
                    "Calibration")
            event = self._calibration.process(
                message       = message,
                direction     = direction,
                last_decision = self._last_decision,
                beam          = self._last_beam if hasattr(self,'_last_beam') else None,
                llm_router    = getattr(self, '_router', None),
            )
            direction_text = {
                "too_aggressive":  "noted — I'll be more conservative next time",
                "too_conservative":"noted — I'll be bolder next time",
                "wrong":           "understood — adjusting the model",
                "correct":         "glad that worked — reinforcing this approach",
            }.get(event.direction, "feedback recorded")
            return text_card(
                f"Calibration {direction_text}.\n"
                f"Factor adjusted: {event.factor_id}  "
                f"by {event.adjustment:+.3f}\n"
                f"{self._calibration.summary()}",
                "Model updated")

        if intent == "calibration_summary":
            summary = self._calibration.summary()
            history = self._calibration.history(n=10)
            lines   = "\n".join(
                f"  [{e.domain}] {e.direction}: {e.message[:60]}"
                for e in history[:8])
            return text_card(f"{summary}\n\n{lines}", "Calibration history")

        if intent == "approve_pending":
            pending = getattr(self, '_pending_approval', None)
            if pending:
                task    = pending.get("task","")
                self._pending_approval = None
                task_id = self._autonomous.execute_async(task, ctx)
                return text_card(
                    f"Approved. Executing autonomously.\nTask ID: `{task_id}`\n"
                    f"I'll notify you when done.",
                    "Authorised — working on it")
            return text_card("No pending task to approve.", "Nothing pending")

        if intent == "cancel_pending":
            self._pending_approval = None
            return text_card("Cancelled. Nothing was executed.", "Cancelled")

        if intent == "list_tools":
            tools = self._autonomous.list_tools()
            if not tools:
                return text_card(
                    "No custom tools acquired yet. Give me a task I don't know how "
                    "to do and I'll build the tool for it.",
                    "Learned tools")
            lines = "\n".join(
                f"• **{t.name}** — {t.description} (used {t.use_count}×)"
                for t in tools[:15])
            return text_card(lines, f"Learned tools ({len(tools)})")

        # Unknown intent — behave like a real PA
        return self._handle_unknown(intent, message, ctx)

    def _handle_unknown(self, intent: str, message: str, ctx: dict) -> PrismCard:
        """
        Managerial PA fallback: PRISM autonomously acquires the capability,
        executes the task, and reports back. Never returns instructions to the user.
        """
        # Check if autonomous engine has a cached tool for this
        if self._autonomous.can_handle(message):
            task_id = self._autonomous.execute_async(
                message, ctx, on_complete=None)
            return text_card(
                f"On it. I have a tool for this — working in the background.\n"
                f"Task ID: {task_id}\n"
                f"I'll notify you when done."
                + (" Check your phone." if self._push.configured else ""),
                "Working on it")

        # No cached tool — synthesise and execute asynchronously
        router = getattr(self, '_router', None)

        # Ask LLM whether this needs approval or is safe to do autonomously
        approval_needed = False
        capability_desc = ""
        if router:
            assess_prompt = (
                f"A personal assistant is about to autonomously handle: '{message}'\n"
                f"Assess:\n"
                f"1. What external service/capability is needed?\n"
                f"2. Does this require user approval before acting "
                f"(e.g. sending emails, making purchases, deleting data)? yes/no\n"
                f"Return JSON: {{\"capability\": \"...\", \"needs_approval\": true/false, "
                f"\"reason\": \"...\"}}"
            )
            raw, _ = router.call(assess_prompt, min_capability=1,
                                  max_tokens=150, json_mode=True)
            try:
                import json as _j
                clean = raw.strip().lstrip("```json").rstrip("```").strip()
                assessment = _j.loads(clean)
                capability_desc  = assessment.get("capability", "")
                approval_needed  = assessment.get("needs_approval", False)
            except Exception:
                pass

        # Gate destructive/external actions behind policy
        if approval_needed:
            # Store pending and ask
            self._pending_approval = {
                "task":   message,
                "reason": f"This requires: {capability_desc}. Approve autonomous execution?"
            }
            return text_card(
                f"I can do this, but it involves **{capability_desc}** which may "
                f"affect external systems.\n\n"
                f"Say **'yes, go ahead'** to authorise, or **'cancel'** to stop.\n\n"
                f"Task: {message}",
                "Approval needed before I proceed")

        # Safe to proceed autonomously
        task_id = self._autonomous.execute_async(message, ctx)

        notify_suffix = ""
        if self._push and self._push.configured:
            notify_suffix = "\nYou'll get a push notification when it's done."

        capability_line = f"\nAcquiring capability: **{capability_desc}**" if capability_desc else ""

        return text_card(
            f"On it — handling this autonomously in the background.{capability_line}\n"
            f"Task ID: `{task_id}`{notify_suffix}\n\n"
            f"I'll synthesise the tool, install any dependencies, execute, "
            f"and report back.",
            "Autonomous execution started")
