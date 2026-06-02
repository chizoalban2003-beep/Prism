from __future__ import annotations

import json
import logging
import re
import urllib.request
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
from prism_documents import PrismDocuments
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
        (r"help|what\.can|commands|options", "help"),
        (r"turn (?:on|off)|set (?:the )?(?:lights?|thermostat|temp)|"
         r"lock|unlock|what(?:'s| is) (?:on|off)|smart home|home assistant",
         "smart_home"),
        (r"(?:check|read|show|open|fetch|get|list).*(?:email|inbox|mail)|"
         r"(?:email|mail).*(?:unread|new|recent)|send.*(?:email|mail)|"
         r"draft.*(?:email|reply)|reply.*email|email.*summary",
         "email"),
        (r"(?:find|search|look for|open|read) (?:my )?(?:document|doc|file|note|page|drive)|"
         r"(?:what(?:'s| is) in|show me) (?:my )?(?:google drive|notion|dropbox)|"
         r"(?:create|write|save) (?:a )?(?:note|document|doc|page)",  "documents"),
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
            self._memory = PrismMemory(ollama_host=ollama_host)
        except Exception as e:
            logger.warning("PrismMemory not available: %s", e)
            self._memory = None
        self._tts = PrismTTS.setup()
        self._smarthome = PrismSmartHome.from_config({})
        self._email    = PrismEmail.from_config({})
        self._calendar = PrismCalendar.from_config({})
        self._docs     = PrismDocuments.from_config({})
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

        if intent == "email":
            if not self._email.configured:
                return text_card(
                    "Email not configured. "
                    "Add email config to prism_config.toml.",
                    "Email")
            msg_lower = message.lower()
            if any(w in msg_lower for w in ("send", "draft", "reply")):
                return text_card(
                    "Use the /email/send endpoint to send emails or "
                    "ask me to draft a reply to a specific message.",
                    "Email")
            messages = self._email.fetch_unread()
            summary = self._email.summarise_inbox(messages, self._router)
            return text_card(summary, "Email Inbox")

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

        if intent == "documents":
            if not self._docs.configured_providers:
                return text_card(
                    "No document providers configured. "
                    "Add gdrive_token, notion_token, or dropbox_token to "
                    "prism_config.toml [documents] section.", "Documents")
            router = getattr(self, '_router', None)
            msg_lower = message.lower()
            if any(w in msg_lower for w in ("create","write","save","new note")):
                parsed = None
                if router:
                    prompt = (f"Extract title and content from: '{message}'. "
                              f"Return JSON: {{\"title\":\"...\",\"content\":\"...\"}}")
                    raw, _ = router.call(prompt, min_capability=1, max_tokens=200,
                                          json_mode=True)
                    from prism_llm_router import parse_llm_json
                    parsed = parse_llm_json(raw)
                if parsed:
                    doc = self._docs.create_note(parsed.get("title","New note"),
                                                  parsed.get("content",""))
                    if doc:
                        return text_card(f"Created: {doc.title}\n{doc.url}", "Document created")
                return text_card("Could not parse note details.", "Documents")
            # Search or recent
            query = message.replace("find","").replace("search","").replace(
                "document","").replace("my","").strip()
            docs = self._docs.search(query) if query else self._docs.recent()
            if not docs:
                return text_card("No documents found.", "Documents")
            lines = "\n".join(f"• [{d.provider}] {d.title}  {d.url}" for d in docs[:8])
            return text_card(lines, f"Documents ({len(docs)} found)")

        return text_card("I'm not sure how to help with that. Try: 'help'")
