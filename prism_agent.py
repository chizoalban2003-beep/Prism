from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

from domain_configs import ALL_DOMAINS, DomainDecisionModel
from prism_autonomous import PrismAutonomous
from prism_browser_agent import PrismBrowserAgent
from prism_calendar import PrismCalendar
from prism_calibration import PrismCalibration
from prism_chain import PrismChain
from prism_chain_expert import PrismChainExpert
from prism_chain_theory import InterceptorPolicy
from prism_composer import PrismComposer
from prism_contacts import PrismContacts
from prism_device_agent import PrismDeviceAgent
from prism_email import PrismEmail
from prism_instructions import PrismInstructions
from prism_llm_router import LLMRouter
from prism_memory import PrismMemory
from prism_organ_loader import OrganLoader
from prism_perception import PrismPerception
from prism_planner import PrismPlanner
from prism_proactive import PrismProactive, build_advanced_triggers, build_default_triggers
from prism_push import PrismPush
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
from prism_service_discovery import PrismServiceDiscovery
from prism_smart_home import PrismSmartHome
from prism_task_queue import TaskQueue
from prism_tasks import PrismTasks
from prism_tts import PrismTTS
from prism_veax import (
    SpectrumGates,
    get_current_gates,
    render_gates,
    save_spectrum_state,
)
from prism_voice import PrismVoice

logger = logging.getLogger(__name__)


class PrismAgent:
    """
    Unified PRISM agent. Routes natural language to sub-agents.
    Returns PrismCard for every request. Never raises.
    """

    INTENTS = [
        # Live financial/crypto data — must precede plan ("today") and wikipedia ("what is")
        (r"stock (?:price|market|quote)|share price|market cap|"
         r"bitcoin|ethereum|crypto (?:price|market)|coin price|"
         r"(?:price|value) of (?:bitcoin|ethereum|[A-Z]{2,5})\b",
         "web_search"),
        # News must precede plan — "today's headlines" contains "today"
        (r"news|headlines|top stories|latest stories|breaking news", "news_headlines"),
        (r"(?!.*\bto (?:french|spanish|german|japanese|chinese|arabic|russian|hindi|italian"
         r"|portuguese)\b)(?:plan|morning|daily|today|schedule)", "plan"),
        (r"how (?:do|can|should) i|plan (?:for|to)|strategy for|help me (?:with|plan)|"
         r"what(?:'s| is) the best way|i want to|i need to|my goal is", "universal_plan"),
        (r"predict|match|fixture|vs|versus", "predict_match"),
        (r"injury risk|squad risk|squad injury|player risk|player fitness|"
         r"\binjury\b|\bsquad\b|\bfitness\b", "squad_risk"),
        (r"moment|1v1|keeper|\bshot\b|attack", "moment"),
        (r"session|footage|video|analyse.*play", "session"),
        (r"transfer|market|value|worth", "transfer"),
        (r"triage|chest|pain|fever|symptom|patient", "domain_medical"),
        (r"portfolio|invest|allocation|bonds|equity", "domain_financial"),
        (r"legal|case|litigat|settle|arbitrat", "domain_legal"),
        (r"hire|hiring|recruit|talent|headcount", "domain_hr"),
        (r"supply chain|procurement|inventory|(?:stock|restock) (?:level|order|management)|out of stock",
         "domain_supply"),
        (r"climate|carbon|emission|energy\.policy", "domain_climate"),
        (r"what (?:do you )?know about me|my profile|who am i|crystallise|persona|how well do you know me",
         "my_profile"),
        (r"my (?:week|weekly|month|monthly) (?:report|summary|narrative|review)|"
         r"what happened this (?:week|month)",
         "my_narrative"),
        (r"how (?:much have you |have you )learned|growth report|"
         r"what have you learned about me|prism growth",
         "my_growth"),
        (r"identity|digital\.dna|who\.am", "identity"),
        (r"artifact|past\.decision|what\.have\.i|my artifacts", "artifacts"),
        (r"\bstatus\b|connected|device|\bsync\b", "status"),
        (r"index|scan\.files|search\.code|grep|find\.file", "ksa_task"),
        (r"resize|(?:convert|compress) (?:file|image|video)|rename|move|copy|delete|create file|"
         r"find file|search (?:in|for)|read file|list files|"
         r"run (?:command|script)|execute|open (?:app|file)|"
         r"install (?:package|app)|git (?:commit|push|pull|status)|"
         r"what(?:'s| is) (?:on|in) my(?! screen)|show me (?:my )?files", "device_task"),
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
        (r"(?:look up|tell me about|what (?:is|was|are)) .+?(?:on |in )?wikipedia|wikipedia",
         "wikipedia_lookup"),
        (r"(?:go to|open|browse|visit|find (?:on|online)|"
         r"look up(?! .+ (?:on|in) wikipedia)|book|reserve|fill (?:in|out)|"
         r"check (?:the )?(?:price|availability)|what(?:'s| is) (?:on|the) website)",
         "browser_task"),
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
        (r"(?:append|add|write|save|take|jot down) (?:a )?note|note(?:pad)?:? ", "note_append"),
        (r"remind me|set (?:a )?reminder|alert me (?:in|at|when)|"
         r"don't let me forget|in (\d+) (?:minute|hour|day)|"
         r"at (\d+(?::\d+)?(?:am|pm)?)", "reminder_set"),
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
        (r"^(?:yes[,.]?|yeah[,.]?|go ahead|approved?|confirm|do it|proceed)[\s!.]*$",
         "approve_pending"),
        (r"^(?:no[,.]?|cancel|stop|don't|abort|never mind)[\s!.]*$",
         "cancel_pending"),
        (r"(?:what tools|learned tools|acquired tools|"
         r"what can you now do|new capabilities|tool list)",
         "list_tools"),
        (r"how did you do that|what steps did you take|"
         r"explain (?:your )?(?:steps?|process|plan)|"
         r"show (?:me )?(?:the )?steps",
         "explain_composition"),
        (r"show (?:me )?(?:the )?chain|chain (?:steps?|history|log)|"
         r"how did (?:the )?chain work|what (?:steps?|logics?) did you use",
         "chain_history"),
        (r"(?:start|begin|enable|use) (?:voice|microphone|listening|speech)|"
         r"(?:stop|disable) (?:voice|listening|speech)|"
         r"(?:voice|speech|microphone) (?:on|off|status|available)|"
         r"(?:transcribe|listen|record) (?:audio|voice|speech|this)",
         "voice"),
        (r"help|what\.can|commands|options", "help"),
        # Horizon Planner — cross-session goal watching
        (r"watch (?:for|out for)|monitor (?:for|when)|track (?:when|until)|"
         r"(?:tell|alert|notify) me when|(?:wait|keep watching) (?:for|until)|"
         r"(?:book|buy|do|send|run) (?:it |that )?when|"
         r"horizon goal|long.?term goal|background goal",
         "horizon_add"),
        (r"(?:show|list|what are) (?:my )?(?:horizon|background|watching|monitored) goals?|"
         r"what (?:are you |is prism )?(?:watching|monitoring|tracking)|"
         r"horizon (?:status|goals?|list)",
         "horizon_list"),
        (r"(?:stop|cancel|abandon) (?:watching|monitoring|tracking|that horizon|horizon goal)|"
         r"(?:forget|remove|delete) (?:that )?(?:goal|watch|monitor)",
         "horizon_abandon"),
        # Organs — loaded capabilities
        (r"(?:what|which|show|list) (?:my )?(?:organs?|loaded (?:capabilities|modules|tools))|"
         r"organ (?:list|status|registry)",
         "list_organs"),
        (r"turn (?:on|off)|set (?:the )?(?:lights?|thermostat|temp)|"
         r"lock|unlock|what(?:'s| is) (?:on|off)(?! (?:my|the) screen)|smart home|home assistant",
         "smart_home"),
        # NOTE: broad email catch-all — maps to email_read to avoid duplication
        # with the more specific email_read/email_send intents above.
        (r"(?:check|read|show|open|fetch|get|list).*(?:email|inbox|mail)|"
         r"(?:email|mail).*(?:unread|new|recent)|send.*(?:email|mail)|"
         r"draft.*(?:email|reply)|reply.*email|email.*summary",
         "email_read"),
        # Organ-mapped intents (broad fallback patterns — do not duplicate entries above)
        (r"weather|temperature|forecast|how (?:hot|cold)|rain|sunny", "weather_check"),
        (r"wikipedia|look up|tell me about|who (?:is|was)|what (?:is|was) (?:a |an |the )?[A-Za-z]",
         "wikipedia_lookup"),
        (r"translate|translation|in (?:spanish|french|german|italian|portuguese|chinese|japanese|arabic|russian|hindi)",
         "translate_text"),
        (r"(?:convert|exchange|how much) .* (?:usd|eur|gbp|jpy|cad|aud|chf|cny|currency)|"
         r"(?:usd|eur|gbp|jpy|cad|aud|chf|cny) (?:to|in|into)|"
         r"(?:dollar|euro|pound|yen|yuan|franc|rupee|peso|won|ruble|lira|krona|"
         r"baht|ringgit|dirham|real|shekel|zloty|forint|koruna|krone|dinar|"
         r"bitcoin|satoshi|ethereum) (?:to|in|into)|"
         r"(?:convert|exchange) .* (?:dollar|euro|pound|yen|yuan|franc|rupee)",
         "currency_convert"),
        (r"(?:convert|how many|how much) .* (?:to|in|into)|"
         r"(?:km|miles|kg|lbs|celsius|fahrenheit|meters?|feet|inches?|liters?|gallons?) (?:to|in|into)",
         "unit_convert"),
        (r"(?:take|capture|grab) (?:a )?screenshot|screenshot", "screenshot_capture"),
        (r"what(?:'s| is) on (?:my |the )?screen|analyse (?:my |the )?screen|"
         r"analyze (?:my |the )?screen|describe (?:my |the )?screen|"
         r"look at (?:my |the )?screen|what do you see|vision query|"
         r"read (?:my |the )?screen|what(?:'s| is) (?:happening|visible) on screen",
         "vision_query"),
        (r"(?:read|what(?:'s| is) on|show|paste|get) (?:my )?clipboard", "clipboard_read"),
        (r"(?:set|start|create) (?:a )?timer|timer (?:for|of)|countdown", "timer_set"),
        (r"(?:read|open|show|cat|display) (?:my |the )?file|file (?:contents?|read)", "file_read"),
        (r"(?:write|save|create|overwrite) (?:to )?(?:the )?file|write (?:this|that) to", "file_write"),
        (r"(?:play|pause|skip|next|previous|volume|stop) (?:music|spotify|song|track|playback)",
         "spotify_control"),
        (r"(?:generate|create|make|qr) (?:a )?qr (?:code)?|qr code for", "qr_generate"),
        (r"(?:run|execute|shell|bash|cmd|terminal|command)(?:\s|:)", "shell_run"),
        (r"(?:make|place|give|dial) (?:a )?(?:phone )?call|"
         r"(?:call|phone|ring) (?:someone|them|him|her|my |the )|"
         r"phone call to|\bcall \d",
         "phone_call"),
        (r"github (?:issue|pr|pull request|repo)|(?:create|list|open) (?:an? )?issue", "github_issue"),
        (r"(?:send|post) (?:a )?(?:message )?(?:to|on) discord|discord", "discord_send"),
        (r"(?:send|post) (?:a )?(?:message )?(?:to|on) telegram|telegram", "telegram_send"),
        (r"(?:control|turn|set|dim) (?:the )?(?:lights?|thermostat|fan|ac|heater|lock|switch)",
         "smart_home_control"),
        (r"my (?:finances?|budget|spending|transactions?|expenses?)|finance (?:summary|report)",
         "finance_summary"),
        (r"my (?:health|steps?|sleep|hrv|heart rate|calories?)|health (?:summary|report|data)",
         "health_summary"),
        (r"(?:brief|briefing|prep|summary) (?:for|before|about) (?:my )?(?:meeting|call|standup)",
         "meeting_brief"),
        (r"(?:overdue|due today|pending|upcoming) (?:tasks?|reminders?|todos?)|task reminder",
         "task_reminder"),
        (r"policy (?:audit|log|history)|audit (?:log|trail)", "policy_audit"),
        # NOTE: entries below are intentionally absent — the following patterns
        # were duplicates of earlier INTENTS entries and have been removed:
        #   news_headlines (subset of the first news entry above)
    ]

    def __init__(
        self,
        kde_agent=None,
        ksa_agent=None,
        ollama_host: str = "http://localhost:11434",
        text_model: str = "mistral",
        claude_api_key: Optional[str] = None,
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

        # Build LLMRouter from local prism_config.toml [llm] section
        _llm_cfg = dict(self._config.get("llm", {}))
        if self._claude_key:
            _llm_cfg["claude_api_key"] = self._claude_key
        # Env vars override config keys
        if not _llm_cfg.get("claude_api_key"):
            _llm_cfg["claude_api_key"] = os.environ.get("ANTHROPIC_API_KEY", "")
        if not _llm_cfg.get("openai_api_key"):
            _llm_cfg["openai_api_key"] = os.environ.get("OPENAI_API_KEY", "")
        self._router = LLMRouter(
            preferred   = _llm_cfg.get("preferred", ""),
            fallback    = _llm_cfg.get("fallback", []),
            ollama_host = _llm_cfg.get("ollama_host", "http://localhost:11434"),
            config      = _llm_cfg,
        )

        self._queue  = TaskQueue()
        self._planner = PrismPlanner(
            ollama_host    = ollama_host,
            ollama_model   = text_model,
            claude_api_key = claude_api_key,
        )

        # PolicyEngine — resource allocation + per-action approval policy
        try:
            from prism_policy import PolicyEngine
            self._policy = PolicyEngine()
        except Exception as e:
            logger.warning("PolicyEngine not available: %s", e)
            self._policy = None

        # PrismCollaborator — external research/synthesis bridge
        try:
            from prism_collaborator import PrismCollaborator
            self._collaborator = PrismCollaborator(router=self._router)
        except Exception as e:
            logger.warning("PrismCollaborator not available: %s", e)
            self._collaborator = None

        self._device = PrismDeviceAgent.setup(
            policy_engine = self._policy,
            on_approval   = self._request_approval,
            collaborator  = self._collaborator,
            user          = self._user,
        )
        self._memory: Optional[PrismMemory] = None
        try:
            self._memory = PrismMemory(ollama_host=ollama_host)
        except Exception as e:
            logger.warning("PrismMemory not available: %s", e)
            self._memory = None
        self._tts   = PrismTTS.setup()
        self._voice = PrismVoice.from_config(self._config)
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
            collaborator  = self._collaborator,
            tool_registry = getattr(
                getattr(self, '_device', None), '_registry', None),
        )
        self._chat_history: list[dict] = []
        self._perception: Optional[PrismPerception] = None
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
                perception    = self._perception,
                policy_engine = self._policy,
                task_queue    = self._queue,
            )
            for t in triggers:
                self._proactive.register(t)
            self._proactive.start()
        except Exception as e:
            logger.warning("PrismProactive not available: %s", e)
            self._proactive = None

        # KineticEngine — compound personal signal aggregator
        # Wires perception → torque accumulation → proactive action windows
        self._kinetic: Optional[Any] = None
        try:
            from prism_kinetic_engine import KineticEngine
            from prism_routes_kinetic import get_or_set_engine
            self._kinetic = KineticEngine.for_prism()
            get_or_set_engine(self._kinetic)
            if self._proactive is not None:
                import time as _time
                def _on_kinetic_action(window: Any) -> None:
                    fire_at = _time.time() + 2.0  # slight delay so context is ready
                    self._proactive.schedule(
                        window.to_proactive_message(), fire_at,
                        trigger_id=f"kinetic_{window.window_id}")
                self._kinetic.on_action(_on_kinetic_action)
            # Wire kinetic into the perception fuser for real-time signal ingestion
            if self._kinetic is not None and self._perception is not None:
                fuser = getattr(self._perception, '_fuser', None)
                if fuser is not None:
                    fuser._kinetic = self._kinetic
            logger.info("KineticEngine ready (%d levers)", len(self._kinetic._levers))
        except Exception as e:
            logger.warning("KineticEngine not available: %s", e)
            self._kinetic = None

        # Surgical ML Assembler — task-profiling algorithm compiler
        self._ml_assembler: Optional[Any] = None
        try:
            from prism_ml_assembler import MLAssembler
            from prism_routes_ml import get_or_set_assembler as _set_asm
            self._ml_assembler = MLAssembler(
                llm_router=self._router,
                outcome_tracker=getattr(self, "_outcome_tracker", None),
            )
            _set_asm(self._ml_assembler)
            logger.info("MLAssembler ready")
        except Exception as e:
            logger.warning("MLAssembler not available: %s", e)

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
            policy_engine= self._policy,
            push         = self._push,
            task_queue   = self._queue,
        )
        self._composer = PrismComposer(
            llm_router    = self._router,
            policy_engine = self._policy,
            push          = self._push,
            task_queue    = self._queue,
        )
        self._chain = PrismChain(
            llm_router         = self._router,
            policy_engine      = self._policy,
            push               = self._push,
            autonomous         = self._autonomous,
            memory             = self._memory,
            interceptor_policy = InterceptorPolicy(),
            # soul omitted — PrismSoul is built ~70 lines later and
            # back-patched via _wire_backpatches(); passing it here is a no-op.
        )
        self._organ_loader = OrganLoader(llm_router=self._router)
        self._chain._organ_loader = self._organ_loader

        # L1 Constitution guard + Bud execution manager
        self._constitution: Optional[Any] = None
        self._bud_mgr: Optional[Any] = None
        try:
            from prism_bud_manager import BudManager
            from prism_constitution import ConstitutionGuard
            self._constitution = ConstitutionGuard()
            self._bud_mgr = BudManager(constitution_guard=self._constitution)
        except Exception as e:
            logger.warning("Constitution/BudManager not available: %s", e)
            self._constitution = None
            self._bud_mgr = None
        self._chain_expert = PrismChainExpert(
            llm_router    = self._router,
            policy_engine = self._policy,
            push          = self._push,
            autonomous    = self._autonomous,
            memory        = self._memory,
        )

        # Re-construct email/calendar/smarthome with the real config now that
        # prism_config.toml has been loaded.  The initial construction above
        # used {} so these modules default to "unconfigured"; this pass wires
        # any credentials the user has provided in prism_config.toml.
        if self._config:
            self._smarthome = PrismSmartHome.from_config(self._config)
            self._email     = PrismEmail.from_config(self._config)
            self._calendar  = PrismCalendar.from_config(self._config)

        # HorizonPlanner — cross-session long-horizon goal persistence
        self._horizon: Optional[Any] = None
        try:
            from prism_horizon import HorizonPlanner
            self._horizon = HorizonPlanner(
                llm_router = self._router,
                task_queue = self._queue,
                push       = self._push,
            )
            triggered = self._horizon.on_session_start()
            if triggered:
                logger.info(
                    "HorizonPlanner: %d goal(s) triggered at startup: %s",
                    len(triggered), triggered,
                )
        except Exception as e:
            logger.warning("HorizonPlanner not available: %s", e)
            self._horizon = None

        if hasattr(self, '_horizon') and self._horizon and hasattr(self, '_chain') and self._chain:
            self._horizon._chain = self._chain

        # OrganBus — LLM-mediated inter-engine communication bus
        self._organ_bus: Optional[Any] = None
        try:
            from prism_organ_bus import OrganBus
            self._organ_bus = OrganBus(llm_router=self._router)
            self._register_organ_subscriptions()
        except Exception as e:
            logger.warning("OrganBus not available: %s", e)
            self._organ_bus = None

        # PrismSoul — living identity document
        try:
            from prism_soul import PrismSoul
            self._soul = PrismSoul(llm_router=self._router)
            if self._organ_bus is not None:
                self._soul.register_with_bus(self._organ_bus)
            # Back-patch soul into chain (chain is constructed before soul)
            if hasattr(self, '_chain'):
                self._chain._soul = self._soul
            if not self._soul.has_seed():
                logger.info("PrismSoul: no soul seed found — run identity ceremony to personalise")
            else:
                logger.info("PrismSoul: loaded (%d beliefs, %d lenses)",
                            len(self._soul.list_beliefs()),
                            len(self._soul.list_lenses()))
        except Exception as e:
            logger.warning("PrismSoul not available: %s", e)
            self._soul = None

        # Living user model — persona, crystalliser, narrative
        self._persona: Any = None
        self._crystalliser: Any = None
        self._narrative: Any = None
        try:
            from prism_crystalliser import PrismCrystalliser
            from prism_narrative import PrismNarrative
            from prism_persona import PrismPersona
            self._persona = PrismPersona()
            self._crystalliser = PrismCrystalliser(
                persona=self._persona,
                memory=getattr(self, '_memory', None),
                outcome_tracker=getattr(self, '_outcome_tracker', None),
                calibration=getattr(self, '_calibration', None),
                llm_router=self._router,
                ml_assembler=getattr(self, '_ml_assembler', None),
            )
            self._narrative = PrismNarrative(
                persona=self._persona,
                memory=getattr(self, '_memory', None),
                outcome_tracker=getattr(self, '_outcome_tracker', None),
                calibration=getattr(self, '_calibration', None),
                soul=getattr(self, '_soul', None),
                llm_router=self._router,
            )
            logger.info("Living user model ready (persona, crystalliser, narrative)")
        except Exception as e:
            logger.warning("Living user model not available: %s", e)
            self._persona = None
            self._crystalliser = None
            self._narrative = None

        # OutcomeTracker — closes the learning loop
        self._outcome_tracker: Optional[Any] = None
        try:
            from prism_outcome_tracker import OutcomeTracker
            self._outcome_tracker = OutcomeTracker(
                soul    = getattr(self, '_soul', None),
                horizon = getattr(self, '_horizon', None),
            )
            if hasattr(self, '_chain'):
                self._chain._outcome_tracker = self._outcome_tracker
            logger.info("OutcomeTracker ready")
        except Exception as e:
            logger.warning("OutcomeTracker not available: %s", e)
            self._outcome_tracker = None

        # Wire living model dependencies now that outcome_tracker exists.
        # All cross-component back-patches are consolidated in _wire_backpatches()
        # and called again at the end of __init__ to ensure nothing is missed even
        # if construction order changes.
        self._wire_backpatches()

        # ContextManager — work/personal/focus context switching
        try:
            from prism_context_profile import ContextManager
            self._context_manager = ContextManager()
            if hasattr(self, '_chain'):
                self._context_manager.inject_into_chain(self._chain)
            logger.info("ContextManager ready (active: %s)", self._context_manager.active_id)
        except Exception as e:
            logger.warning("ContextManager not available: %s", e)
            self._context_manager = None

        # ProactiveBusWatcher — connect OrganBus signals to proactive triggers
        try:
            if self._organ_bus is not None and getattr(self, '_proactive', None) is not None:
                from prism_proactive_bus_watcher import ProactiveBusWatcher
                self._bus_watcher = ProactiveBusWatcher(
                    proactive = self._proactive,
                    organ_bus = self._organ_bus,
                )
                self._bus_watcher.register()
                logger.info("ProactiveBusWatcher registered")
            else:
                self._bus_watcher = None
        except Exception as e:
            logger.warning("ProactiveBusWatcher not available: %s", e)
            self._bus_watcher = None

        # PrismReflection — weekly meta-learning loop
        try:
            from prism_reflection import PrismReflection
            self._reflection = PrismReflection(
                outcome_tracker = getattr(self, '_outcome_tracker', None),
                soul            = getattr(self, '_soul', None),
                horizon         = getattr(self, '_horizon', None),
                llm_router      = self._router,
                auto_apply      = False,
            )
            logger.info("PrismReflection ready")
        except Exception as e:
            logger.warning("PrismReflection not available: %s", e)
            self._reflection = None

        # ChainOrchestrator — prefrontal cortex for multi-step coordination
        self._orchestrator: Optional[Any] = None
        try:
            from prism_orchestrator import ChainOrchestrator
            self._orchestrator = ChainOrchestrator(
                chain           = getattr(self, '_chain', None),
                organ_loader    = getattr(self, '_organ_loader', None),
                outcome_tracker = getattr(self, '_outcome_tracker', None),
                horizon         = getattr(self, '_horizon', None),
                router          = self._router,
                soul            = getattr(self, '_soul', None),
            )
            self._orchestrator._persona = getattr(self, '_persona', None)
            # Resume any graphs that were paused waiting on horizon goals
            resumed = self._orchestrator.resume_waiting(self._execute, {})
            if resumed:
                logger.info("ChainOrchestrator: %d paused graph(s) resumed", len(resumed))
            logger.info("ChainOrchestrator ready")
        except Exception as e:
            logger.warning("ChainOrchestrator not available: %s", e)
            self._orchestrator = None

        # Advanced proactive triggers — registered after all dependencies exist
        if getattr(self, '_proactive', None) is not None:
            try:
                advanced = build_advanced_triggers(
                    organ_loader = getattr(self, '_organ_loader', None),
                    router       = self._router,
                    calendar     = getattr(self, '_calendar', None),
                    persona      = getattr(self, '_persona', None),
                    horizon      = getattr(self, '_horizon', None),
                    config       = self._config,
                )
                for t in advanced:
                    self._proactive.register(t)
                logger.info("Advanced proactive triggers registered: %d", len(advanced))
            except Exception as e:
                logger.warning("Advanced proactive triggers failed: %s", e)

        # Final pass: re-apply all back-patches in one explicit place so the
        # dependency graph is always consistent regardless of construction order.
        self._wire_backpatches()

    def _wire_backpatches(self) -> None:
        """
        Consolidate all cross-component back-patches.

        Construction order constraints in __init__:
          chain → organ_loader  (chain built before soul/persona/outcome_tracker)
          soul  → chain         (soul back-patched into chain after soul is built)
          horizon → chain       (horizon wired into chain for anchor creation)
          outcome_tracker → chain, crystalliser, ml_assembler, kinetic
          orchestrator → persona

        This method is idempotent and called twice: once after OutcomeTracker is
        created (to wire everything that exists at that point) and once at the very
        end of __init__ (to pick up ChainOrchestrator and any late-registered deps).
        """
        _chain       = getattr(self, '_chain',          None)
        _soul        = getattr(self, '_soul',            None)
        _persona     = getattr(self, '_persona',         None)
        _crystalliser= getattr(self, '_crystalliser',    None)
        _horizon     = getattr(self, '_horizon',         None)
        _ot          = getattr(self, '_outcome_tracker', None)
        _kinetic     = getattr(self, '_kinetic',         None)
        _asm         = getattr(self, '_ml_assembler',    None)
        _orch        = getattr(self, '_orchestrator',    None)
        _organ_loader= getattr(self, '_organ_loader',    None)

        if _chain is not None:
            if _soul is not None:
                _chain._soul = _soul
            if _persona is not None:
                _chain._persona = _persona
            if _ot is not None:
                _chain._outcome_tracker = _ot
            if _organ_loader is not None:
                _chain._organ_loader = _organ_loader

        if _horizon is not None and _chain is not None:
            _horizon._chain = _chain
        if _crystalliser is not None and _ot is not None:
            _crystalliser._outcome_tracker = _ot
        if _ot is not None and _crystalliser is not None:
            _ot._crystalliser = _crystalliser
        if _ot is not None and _kinetic is not None:
            _ot._kinetic = _kinetic
        if _asm is not None and _ot is not None:
            _asm._tracker = _ot
        if _orch is not None and _persona is not None:
            _orch._persona = _persona

    def stop(self) -> None:
        """Gracefully shut down all background subsystems."""
        _hz = getattr(self, '_horizon', None)
        if _hz is not None:
            try:
                _hz.on_session_end()
            except Exception:
                pass
        _pr = getattr(self, '_proactive', None)
        if _pr is not None:
            try:
                _pr.stop()
            except Exception:
                pass
        _perc = getattr(self, '_perception', None)
        if _perc is not None:
            try:
                _perc.stop()
            except Exception:
                pass

    def _register_organ_subscriptions(self) -> None:
        """Wire PRISM subsystems into the OrganBus as subscribers."""
        if not self._organ_bus:
            return

        # Policy engine receives load / risk signals and adjusts allowances
        if getattr(self, '_policy', None):
            def _policy_handler(payload: dict):
                try:
                    if hasattr(self._policy, 'on_organ_signal'):
                        self._policy.on_organ_signal(payload)
                except Exception:
                    pass
            self._organ_bus.register(
                organ_name   = "policy_engine",
                signal_types = ["injury_risk_elevated", "performance_plateau",
                                 "load_adjustment_needed"],
                handler      = _policy_handler,
                vocabulary   = (
                    "Understands: adjustment (str: reduce_training_load|rest|continue), "
                    "factor (float 0-1), duration_days (int), reason (str), "
                    "flag_for_physio (bool)"
                ),
            )

        # Calendar receives scheduling-relevant signals
        if getattr(self, '_calendar', None):
            def _calendar_handler(payload: dict):
                try:
                    if hasattr(self._calendar, 'on_organ_signal'):
                        self._calendar.on_organ_signal(payload)
                except Exception:
                    pass
            self._organ_bus.register(
                organ_name   = "calendar_engine",
                signal_types = ["injury_risk_elevated", "recovery_complete",
                                 "session_scheduled"],
                handler      = _calendar_handler,
                vocabulary   = (
                    "Understands: message (str notification text), "
                    "action (str: reschedule_heavy_session|cancel|add_rest_day), "
                    "days_to_defer (int), notify_coach (bool)"
                ),
            )

        # Horizon planner watches for recovery/long-horizon signals
        _hz2 = getattr(self, '_horizon', None)
        if _hz2 is not None:
            def _horizon_handler(payload: dict):
                try:
                    intent = payload.get("intent", "")
                    trigger = payload.get("trigger_condition", "")
                    completion = payload.get("completion_condition", "")
                    if intent and _hz2 is not None:
                        _hz2.add(
                            intent=intent,
                            trigger_condition=trigger or "condition met",
                            completion_condition=completion,
                        )
                except Exception:
                    pass
            self._organ_bus.register(
                organ_name   = "horizon_planner",
                signal_types = ["injury_risk_elevated", "long_term_goal_detected"],
                handler      = _horizon_handler,
                vocabulary   = (
                    "Understands: intent (str: full goal description), "
                    "trigger_condition (str: condition to watch for), "
                    "completion_condition (str: what done looks like)"
                ),
            )

    def _handle_proactive_event(self, event) -> None:
        """Store proactive notification for chat UI polling."""
        if not hasattr(self, '_proactive_buffer'):
            self._proactive_buffer: list = []
        self._proactive_buffer.append(event)

    @classmethod
    def setup(
        cls,
        name: str,
        sport: str = "Football",
        team: str = "",
        db_path: str = "~/.prism/prism.db",
    ) -> PrismAgent:
        try:
            from kde_agent import KDEAgent, KDEConfig
            from sports_pro import Role

            kde = KDEAgent.setup(
                name=name,
                role=Role.ATHLETE,   # Role.UNIVERSAL exists only on UserRole
                sport=sport,
                team=team,
                config=KDEConfig(
                    db_path     = db_path,
                    media_dir   = "~/.prism/media",
                    ollama_model= "mistral",
                    ollama_host = "http://localhost:11434",
                    auto_watch  = False,
                ),
            )
        except Exception as e:
            logger.warning("KDEAgent.setup failed: %s", e)
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

            # 7. Inject active context into the request context dict
            if getattr(self, '_context_manager', None) is not None:
                self._context_manager.inject_into_chain_ctx(context)
                self._context_manager.inject_into_chain(self._chain)

            # Persona context for chain injection
            context["persona_context"] = (
                self._persona.build_context()
                if getattr(self, '_persona', None) is not None else ""
            )

            # 8. Route intent and execute
            # Tier 0:   orchestrator    — conditional / multi-domain / cross-session
            # Tier 0.5: expert chain    — research / evaluation-heavy
            # Tier 1:   general chain   — adaptive multi-step
            # Tier 2:   static composer — known multi-step, predictable
            # Tier 3:   single intent   — direct
            card   = None
            msg_ln = len((message or "").split())
            msg_lw = (message or "").lower()

            def _bad_card(c) -> bool:
                """True when a chain/orchestrator returned a raw dict or planner noise."""
                if c is None:
                    return False
                b = getattr(c, 'body', '') or ''
                return b.startswith('{') or 'replanned' in b or b.startswith('[{')

            # Tier 0: orchestrator
            orch = getattr(self, '_orchestrator', None)
            if card is None and orch and message and orch.should_orchestrate(message):
                try:
                    card = orch.orchestrate(message, self._execute, context)
                    if _bad_card(card):
                        card = None
                except Exception as e:
                    logger.debug("Orchestrator failed: %s", e)
                    card = None

            # Tier 0: expert chain — for research/evaluation-heavy requests
            EXPERT_SIGNALS = [
                "research", "analyse", "analyze", "figure out",
                "decide", "best way", "should i", "compare",
                "comprehensive", "investigate", "evaluate",
            ]
            use_expert = (message and msg_ln > 5
                          and any(s in msg_lw for s in EXPERT_SIGNALS))

            if use_expert:
                try:
                    card = self._chain_expert.run(
                        message, self._execute, context)
                    if _bad_card(card):
                        card = None
                except Exception as e:
                    logger.debug("Expert chain failed: %s", e)
                    card = None

            # Tier 1: general chain — adaptive multi-step
            if card is None and message and msg_ln > 5 and self._chain.should_chain(message):
                try:
                    card = self._chain.run(message, self._execute, context)
                    if _bad_card(card):
                        card = None
                except Exception as e:
                    logger.debug("Chain failed: %s", e)
                    card = None

            # Tier 2: static composition
            if card is None and message and msg_ln > 6 and self._composer.should_compose(message):
                try:
                    plan = self._composer.decompose(message)
                    if plan:
                        card = self._composer.execute(plan, self._execute, context)
                    if _bad_card(card):
                        card = None
                except Exception as e:
                    logger.debug("Composer failed: %s", e)
                    card = None

            # Tier 3: single intent
            if card is None:
                intent = self._route(message or "")
                card   = self._execute(intent, message or "", context)

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

            # Crystallise behavioural signals from this turn
            if card is not None:
                try:
                    crystalliser = getattr(self, '_crystalliser', None)
                    if crystalliser:
                        intent_used = self._route(message or "") if message else ""
                        crystalliser.observe_turn(message, card.body, intent_used, context)
                except Exception:
                    pass

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
        self._pending_approval: dict | None = {"task": task, "reason": reason}
        return False

    def _route(self, message: str) -> str:
        lowered = message.lower()
        for pattern, intent in self.INTENTS:
            if re.search(pattern, lowered):
                return intent
        return self._llm_classify(message) or "general_chat"

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
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read()).get("response", "").strip().lower()
            return result if result in labels else None
        except Exception:
            return None

    def _execute(self, intent: str, message: str, ctx: dict) -> PrismCard:
        if intent == "my_profile":
            persona = getattr(self, '_persona', None)
            soul = getattr(self, '_soul', None)
            narrative = getattr(self, '_narrative', None)
            if persona or soul:
                parts = []
                if persona:
                    parts.append(persona.summary())
                if soul:
                    parts.append("\n**Soul (beliefs & values):**\n" + soul.compress_for_llm(400))
                if narrative:
                    try:
                        parts.append("\n**Current snapshot:**\n" + narrative.snapshot())
                    except Exception:
                        pass
                return text_card("\n\n".join(parts), "Your crystallised profile")
            return text_card("Profile not yet initialised.", "Profile")

        if intent == "my_narrative":
            narrative = getattr(self, '_narrative', None)
            if narrative:
                try:
                    return text_card(narrative.weekly(), "Weekly narrative")
                except Exception as exc:
                    return text_card(f"Could not generate narrative: {exc}", "Narrative")
            return text_card("Narrative engine not available.", "Narrative")

        if intent == "my_growth":
            persona = getattr(self, '_persona', None)
            narrative = getattr(self, '_narrative', None)
            if persona and narrative:
                try:
                    report = narrative.growth_report()
                    return text_card(report, "What PRISM knows about you")
                except Exception as exc:
                    return text_card(f"Growth report failed: {exc}", "Growth")
            return text_card("Not enough data yet.", "Growth")

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

        if intent == "research":
            try:
                result = self._chain._research_logic(message, self._execute, ctx)
                return text_card(result, "Research")
            except Exception as exc:
                return text_card(f"Research failed: {exc}", "Research Error")

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
            factors = {
                factor.id: float(user_context.get(factor.id, ctx.get(factor.id, 0.5)))
                for factor in config.factors
            }
            diagnosis = DomainDecisionModel(config).evaluate(profile, factors)
            return domain_card(domain_key, diagnosis)

        if intent == "squad_risk":
            if self._kde:
                try:
                    result = self._kde.ask(message)
                    output = getattr(result, 'output', result)
                    if isinstance(output, list) and output and hasattr(output[0], 'risk_level'):
                        return squad_card(output)
                    if isinstance(output, str) and output and "No video" not in output:
                        return text_card(output, "Squad Risk")
                except Exception as exc:
                    logger.debug("KDE squad_risk failed: %s", exc)
            # Fallback — route message through prediction engine directly
            try:
                from prediction_engine import InjuryRiskPredictor
                predictor = InjuryRiskPredictor()
                result = predictor.assess(message)
                return text_card(str(result), "Squad Risk")
            except Exception:
                pass
            return text_card(
                "Squad risk assessment requires player data. "
                "Try: 'what is the injury risk for [player name]' or connect a squad data source.",
                "Squad Risk")

        _KDE_INTENTS = {
            "plan", "predict_match", "moment", "session",
            "transfer", "reflect",
        }
        if self._kde and intent in _KDE_INTENTS:
            try:
                result = self._kde.ask(message)
                output = getattr(result, 'output', result)
                try:
                    from prediction_engine import InjuryRiskPrediction, MatchPrediction
                    from sports_pro import DailyPlan
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
            identity_data: dict = {}
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
        if intent == "general_chat":
            router = getattr(self, "_router", None)
            if router is None:
                return text_card(
                    "I don't have a local LLM connected yet, so I can't free-chat. "
                    "Add an Ollama model or a Claude API key to prism_config.toml.",
                    "Chat unavailable",
                )
            try:
                raw, _ = router.call(message, min_capability=1, max_tokens=400)
                return text_card(raw.strip() or "(no response)", "Chat")
            except Exception as exc:
                return text_card(f"LLM call failed: {exc}", "Chat")
        if intent == "status":
            return text_card(
                f"Connected. KDE: {'active' if self._kde else 'offline'}. "
                f"KSA: {'active' if self._ksa else 'offline'}.",
                "Status",
            )
        if intent == "show_policies":
            from prism_responses import policy_view_card
            if self._policy:
                data = self._policy.show_policies(self._user)
            else:
                data = {"allocations": {}, "note": "No policies set yet. "
                        "Try: 'set my food budget to £80'"}
            return policy_view_card(data)

        if intent == "update_policy":
            if self._policy:
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

        if intent == "email_read":
            if not self._email.configured:
                return text_card("Email not configured. "
                                 "Add email settings to prism_config.toml.", "Email")
            messages = self._email.fetch_unread(n=10)
            summary  = self._email.summarise_inbox(
                messages, llm_router=getattr(self, '_router', None))
            return text_card(summary, f"Inbox — {len(messages)} unread")

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
            _q = re.sub(
                r'^(?:search(?:\s+the\s+web|\s+online|\s+for)?|look\s+up|'
                r'find\s+(?:out|info|information)\s+(?:about|on)|'
                r'research|who\s+is|where\s+is|when\s+(?:did|does|is)|'
                r'what(?:\'s| is) (?:the )?(?:latest|current|today))[:\s]+',
                '', message, flags=re.IGNORECASE,
            ).strip().rstrip('?.')
            results = self._search.search(_q or message, n=5)
            if not results:
                answer = self._search.quick_answer(message)
                if answer:
                    return text_card(answer, "Search result")
                # Fall through to web_search organ (DDG Lite)
                organ_fn = self._organ_loader.get("web_search")
                if organ_fn is not None:
                    try:
                        if self._bud_mgr is not None:
                            caps = self._organ_loader.get_organ_capabilities("web_search")
                            handle = self._bud_mgr.spawn("web_search", message, ctx, caps)
                            return self._bud_mgr.execute(handle, organ_fn)
                        return organ_fn("web_search", message, ctx)
                    except Exception:
                        pass
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
            clines: list[str] = [f"{c.name}"]
            if c.organisation:
                clines.append(f"  {c.role} at {c.organisation}")
            if c.emails:
                clines.append(f"  Email: {', '.join(c.emails)}")
            if c.phones:
                clines.append(f"  Phone: {', '.join(c.phones)}")
            if c.notes:
                clines.append(f"  Notes: {c.notes[:200]}")
            return text_card("\n".join(clines),
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
                except Exception:
                    pass
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

            # Snapshot VEAX before calibration
            gates_before = get_current_gates()

            event = self._calibration.process(
                message       = message,
                direction     = direction,
                last_decision = self._last_decision,
                beam          = self._last_beam if hasattr(self, '_last_beam') else None,
                llm_router    = getattr(self, '_router', None),
            )

            # Apply VEAX delta — make the spectrum rewire visibly
            _VEAX_DELTAS: dict[str, dict[str, float]] = {
                "too_aggressive":   {"A": -0.05, "V": +0.03},
                "too_conservative": {"A": +0.05, "V": -0.03},
                "wrong":            {"V": +0.05},
                "correct":          {"X": +0.02},
            }
            deltas = _VEAX_DELTAS.get(direction, {})
            gates_after = None
            if gates_before is not None and deltas:
                gates_after = SpectrumGates(
                    V=max(0.0, min(1.0, gates_before.V + deltas.get("V", 0.0))),
                    E=max(0.0, min(1.0, gates_before.E + deltas.get("E", 0.0))),
                    A=max(0.0, min(1.0, gates_before.A + deltas.get("A", 0.0))),
                    X=max(0.0, min(1.0, gates_before.X + deltas.get("X", 0.0))),
                )
                try:
                    save_spectrum_state(gates_after)
                except Exception:
                    pass

            direction_text = {
                "too_aggressive":  "noted — I'll be more conservative next time",
                "too_conservative": "noted — I'll be bolder next time",
                "wrong":           "understood — adjusting the model",
                "correct":         "glad that worked — reinforcing this approach",
            }.get(event.direction, "feedback recorded")

            # Build rich before/after body
            sep = "─" * 50
            lines2: list[str] = [
                f"Calibration {direction_text}.",
                "",
                f"{sep}",
                f"Domain: {event.domain}  ·  Factor: {event.factor_id}  ·  Δ {event.adjustment:+.3f}",
            ]
            if deltas:
                axis_str = ", ".join(
                    f"{ax} {d:+.2f}" for ax, d in deltas.items()
                )
                lines2.append(f"VEAX: {axis_str}")

            if gates_before is not None and gates_after is not None:
                lines2 += [
                    "",
                    "── Before ──",
                    render_gates(gates_before),
                    "",
                    "── After  ──",
                    render_gates(gates_after),
                ]
            elif gates_before is not None:
                lines2 += ["", "── Current VEAX ──", render_gates(gates_before)]

            lines2 += ["", f"{sep}", self._calibration.summary()]
            return text_card("\n".join(lines2), "Model updated")

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
                if time.time() > pending.get("expires", 0):
                    self._pending_approval = None
                    return text_card(
                        "That approval request expired (5-minute window). "
                        "Repeat your original request to try again.",
                        "Approval expired")
                # Organ-level approval (email_send, phone_call, calendar_write, etc.)
                if "organ_intent" in pending:
                    organ_intent   = pending["organ_intent"]
                    organ_message  = pending["organ_message"]
                    organ_ctx      = pending["organ_ctx"]
                    self._pending_approval = None
                    organ_ctx[f"_approved_{organ_intent}"] = True
                    organ_fn = self._organ_loader.get(organ_intent)
                    if organ_fn:
                        try:
                            return organ_fn(organ_intent, organ_message, organ_ctx)
                        except Exception as exc:
                            return text_card(f"Organ '{organ_intent}' failed: {exc}", organ_intent)
                    return text_card(f"Organ '{organ_intent}' no longer available.", organ_intent)
                # Autonomous task approval (legacy path)
                task = pending.get("task", "")
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

        if intent == "explain_composition":
            # Show last composition plan from history
            last = next(
                (m["content"] for m in reversed(self._chat_history)
                 if "step" in m.get("content","").lower()
                 and m["role"] == "assistant"), None)
            if last:
                return text_card(last[:1000], "Last composition")
            return text_card(
                "No recent composition to explain. "
                "Give me a multi-step request to see the chain in action.",
                "Composition")

        if intent == "chain_history":
            recent = [m["content"] for m in reversed(self._chat_history)
                      if "Chain " in m.get("content","")
                      and m["role"] == "assistant"]
            if recent:
                return text_card(recent[0][:800], "Last chain execution")
            return text_card(
                "No chain execution in recent history. "
                "Ask me something complex to trigger the chain.",
                "Chain history")

        if intent == "autonomous":
            task_id = self._autonomous.execute_async(message, ctx)
            notify  = " I'll push a notification when done." if (
                self._push and self._push.configured) else ""
            return text_card(
                f"Working on it autonomously in the background.\n"
                f"Task ID: `{task_id}`{notify}",
                "Autonomous task started")

        if intent == "voice":
            return self._handle_voice(message, ctx)

        # ── Horizon Planner ───────────────────────────────────────────────
        if intent == "horizon_add":
            if self._horizon is None:
                return text_card("Horizon Planner is unavailable.", "Error")
            # Extract intent/condition from the message via LLM if available
            intent_text = message
            trigger     = message
            completion  = ""
            if self._router:
                try:
                    parse_prompt = (
                        f"Extract a horizon goal from this message.\n"
                        f"Message: {message}\n\n"
                        f"Return JSON with keys:\n"
                        f"  intent: the full goal (what to do)\n"
                        f"  trigger_condition: what condition makes this fire\n"
                        f"  completion_condition: what success looks like\n"
                        f"Example: {{\"intent\": \"book a flight to Lisbon\","
                        f" \"trigger_condition\": \"price drops below 300\","
                        f" \"completion_condition\": \"flight booked\"}}\n"
                        f"Return only valid JSON."
                    )
                    raw, _ = self._router.call(
                        parse_prompt, min_capability=1, max_tokens=200, json_mode=True)
                    import json as _j
                    parsed = _j.loads(raw.strip().lstrip("```json").rstrip("```").strip())
                    intent_text = parsed.get("intent", message)
                    trigger     = parsed.get("trigger_condition", message)
                    completion  = parsed.get("completion_condition", "")
                except Exception:
                    pass
            gid = self._horizon.add(
                intent=intent_text,
                trigger_condition=trigger,
                completion_condition=completion,
            )
            return text_card(
                f"Got it. I'll watch for: **{trigger}**\n"
                f"Goal: {intent_text}\n"
                f"Goal ID: `{gid}`\n\n"
                f"I'll check every session and act as soon as the condition is met. "
                f"Say *'show my horizon goals'* to see status.",
                "Horizon goal registered")

        if intent == "horizon_list":
            if self._horizon is None:
                return text_card("Horizon Planner is unavailable.", "Error")
            goals = self._horizon.list_goals()
            if not goals:
                return text_card(
                    "No horizon goals registered yet.\n\n"
                    "Say something like: *'watch for flight prices to drop below $300 "
                    "and book for me'* to register one.",
                    "No horizon goals")
            from prism_horizon import HorizonGoalStatus
            status_icon = {
                HorizonGoalStatus.WATCHING:   "👁",
                HorizonGoalStatus.TRIGGERED:  "⚡",
                HorizonGoalStatus.PAUSED:     "⏸",
                HorizonGoalStatus.COMPLETED:  "✅",
                HorizonGoalStatus.ABANDONED:  "🚫",
            }
            lines3: list[str] = []
            for g in goals[:10]:
                icon = status_icon.get(g.status, "•")
                lines3.append(
                    f"{icon} **{g.intent[:60]}**\n"
                    f"  Condition: {g.trigger_condition[:50]}\n"
                    f"  Status: {g.status.value} | "
                    f"Sessions checked: {g.session_count} | "
                    f"Steps done: {len(g.completed_steps)} | "
                    f"ID: `{g.goal_id}`"
                )
            return text_card("\n\n".join(lines3), f"Horizon goals ({len(goals)})")

        if intent == "horizon_abandon":
            if self._horizon is None:
                return text_card("Horizon Planner is unavailable.", "Error")
            # Try to find a goal_id in the message, else abandon the most recent watching
            import re as _re
            gid_match = _re.search(r'\b([0-9a-f]{8})\b', message)
            if gid_match:
                gid = gid_match.group(1)
            else:
                watching = self._horizon.list_goals()
                watching = [g for g in watching
                            if g.status.value in ("watching", "triggered", "paused")]
                if not watching:
                    return text_card("No active horizon goals to abandon.", "Nothing to abandon")
                gid = watching[0].goal_id
            goal = self._horizon.get(gid)
            if not goal:
                return text_card(f"No goal found with ID `{gid}`.", "Not found")
            self._horizon.abandon(gid, reason="user requested via chat")
            return text_card(
                f"Stopped watching: **{goal.intent[:60]}**\n"
                f"Goal `{gid}` has been abandoned.",
                "Horizon goal abandoned")

        # ── Context switching ─────────────────────────────────────────────
        if intent in ("switch_context", "context_switch"):
            import re as _re
            cm = getattr(self, '_context_manager', None)
            if cm is None:
                return text_card("Context manager not available.", "Error")
            m = _re.search(r"\b(work|personal|focus|default)\b", message, _re.IGNORECASE)
            if not m:
                profiles = [p.context_id for p in cm.list_profiles()]
                return text_card(
                    f"Available contexts: {', '.join(profiles)}\n"
                    "Say: 'switch to work' / 'switch to personal' / 'switch to focus'",
                    "Context")
            target = m.group(1).lower()
            try:
                profile = cm.switch(target)
                cm.apply_to_policy(self._policy)
                cm.inject_into_chain(self._chain)
                return text_card(
                    f"Switched to **{target}** context.\n{profile.description}",
                    f"Context: {target}")
            except ValueError as exc:
                return text_card(str(exc), "Error")

        if intent == "context_status":
            cm = getattr(self, '_context_manager', None)
            if cm is None:
                return text_card("Context manager not available.", "Error")
            profile = cm.active()
            lines4: list[str] = [f"Active context: **{profile.context_id}**",
                     f"{profile.description}",
                     f"Policy overrides: {profile.policy_overrides or 'none'}",
                     f"Organ priorities: {profile.organ_priorities or 'none'}"]
            return text_card("\n".join(lines4), "Context status")

        # ── Outcome / learning stats ───────────────────────────────────────
        if intent == "outcome_stats":
            tracker = getattr(self, '_outcome_tracker', None)
            if tracker is None:
                return text_card("OutcomeTracker not available.", "Error")
            stats = tracker.stats(days=30)
            lines5: list[str] = [
                "Chain outcomes (last 30 days):",
                f"  Total chains:     {stats['total']}",
                f"  Completed:        {stats['done']}",
                f"  Abandoned:        {stats['abandoned']}",
                f"  User-corrected:   {stats['user_corrected']}",
                f"  Completion rate:  {stats['completion_rate']:.0%}",
                f"  Avg steps/chain:  {stats['avg_steps']}",
                f"  Avg policy flags: {stats['avg_policy_flags']}",
            ]
            return text_card("\n".join(lines5), "Learning stats")

        # ── Weekly reflection ─────────────────────────────────────────────
        if intent == "reflection":
            refl = getattr(self, '_reflection', None)
            if refl is None:
                return text_card("Reflection engine not available.", "Error")
            try:
                summary = refl.summarise_for_chat()
                return text_card(summary, "Weekly reflection")
            except Exception as exc:
                return text_card(f"Reflection failed: {exc}", "Error")

        # ── Organ registry ────────────────────────────────────────────────
        if intent == "list_organs":
            organs = self._organ_loader.list_organs() if hasattr(
                self._organ_loader, 'list_organs') else []
            if not organs:
                return text_card(
                    "No organs loaded yet. Organs are synthesized on demand when "
                    "you ask me to do something I don't have a built-in handler for.",
                    "Loaded organs")
            lines = "\n".join(f"• **{o}**" for o in organs[:20])
            return text_card(lines, f"Loaded organs ({len(organs)})")

        if intent == "vision_query":
            return self._handle_vision_query(message, ctx)

        # Dynamic organ registry — check loaded and synthesized organs
        organ_fn = self._organ_loader.get(intent)
        if organ_fn is not None:
            try:
                ctx.setdefault("organ_loader", self._organ_loader)
                ctx.setdefault("policy_engine", self._policy)
                ctx.setdefault("tasks", getattr(self, "_tasks", None))
                ctx.setdefault("email", getattr(self, "_email", None))
                ctx.setdefault("calendar", getattr(self, "_calendar", None))
                ctx.setdefault("router", getattr(self, "_router", None))
                _tw = dict(self._config.get("twilio", {}))
                import os as _os
                _tw.setdefault("account_sid", _os.environ.get("TWILIO_ACCOUNT_SID", ""))
                _tw.setdefault("auth_token",  _os.environ.get("TWILIO_AUTH_TOKEN", ""))
                _tw.setdefault("from_number", _os.environ.get("TWILIO_FROM", ""))
                ctx.setdefault("twilio_config", _tw)
                ctx.setdefault("contacts", getattr(self, "_contacts", None))

                # L1 Constitution check — validate organ capabilities against L1 rules
                if self._constitution is not None:
                    caps = self._organ_loader.get_organ_capabilities(intent)
                    ok, reason = self._constitution.check(intent, caps)
                    if not ok:
                        logger.warning("[constitution] Blocked %s: %s", intent, reason)
                        return text_card(
                            f"This action is restricted by PRISM's constitution.\n\n{reason}",
                            f"Blocked — {intent}",
                        )

                # L2 Hard approval gate — block irreversible/requires_approval organs
                if not ctx.get(f"_approved_{intent}"):
                    policy = self._organ_loader.get_organ_policy(intent)
                    if policy.get("requires_approval"):
                        self._pending_approval = {
                            "organ_intent":  intent,
                            "organ_message": message,
                            "organ_ctx":     dict(ctx),
                            "expires":       time.time() + 300,
                        }
                        action_desc = message[:200]
                        return text_card(
                            f"**{intent}** requires approval before executing.\n\n"
                            f"Action: {action_desc}\n\n"
                            f"Say **yes** or **approve** to confirm, or **cancel** to abort.",
                            f"Approval required — {intent}",
                        )

                # Execute via BudManager (scoped context, token lifecycle)
                if self._bud_mgr is not None:
                    caps = self._organ_loader.get_organ_capabilities(intent)
                    handle = self._bud_mgr.spawn(intent, message, ctx, caps)
                    try:
                        return self._bud_mgr.execute(handle, organ_fn)
                    except Exception as exc:
                        return text_card(f"Organ '{intent}' failed: {exc}", intent)
                else:
                    return organ_fn(intent, message, ctx)
            except Exception as exc:
                return text_card(f"Organ '{intent}' failed: {exc}", intent)

        # Unknown intent — attempt synthesis before falling back to autonomous
        if intent not in {"autonomous", "approve_pending"}:
            # Check L1 synthesis limit
            synthesis_ok = (
                self._bud_mgr is None or self._bud_mgr.synthesis_allowed()
            )
            if synthesis_ok:
                logger.info("[agent] Unknown intent '%s' — attempting organ synthesis", intent)
                if self._organ_loader.synthesize(intent, message):
                    organ_fn = self._organ_loader.get(intent)
                    if organ_fn is not None:
                        if self._bud_mgr is not None:
                            self._bud_mgr.record_synthesis()
                        try:
                            caps = self._organ_loader.get_organ_capabilities(intent)
                            if self._bud_mgr is not None:
                                handle = self._bud_mgr.spawn(intent, message, ctx, caps)
                                return self._bud_mgr.execute(handle, organ_fn)
                            return organ_fn(intent, message, ctx)
                        except Exception as exc:
                            return text_card(
                                f"Synthesized organ '{intent}' failed: {exc}", intent)
                else:
                    # Synthesis failed — explain what PRISM would need
                    router = getattr(self, '_router', None)
                    if router is None:
                        return text_card(
                            f"I don't have a built-in handler for '{intent}'.\n\n"
                            "To build this capability automatically, connect an LLM "
                            "(Ollama or Claude API key in prism_config.toml). "
                            "PRISM will synthesize a new organ, validate it, and "
                            "register it permanently for future sessions.",
                            f"Capability not found — {intent}",
                        )
            else:
                logger.warning("[agent] Synthesis limit reached for this session")

        return self._handle_unknown(intent, message, ctx)

    def _handle_voice(self, message: str, ctx: dict) -> PrismCard:
        """Handle voice/STT intents: status, enable/disable, transcribe file."""
        msg = message.lower()

        # Status query
        if any(w in msg for w in ("status", "available", "configured")):
            backend  = self._voice._backend or "none"
            rec_lib  = self._voice._record_lib or "none"
            enabled  = self._voice._enabled
            return text_card(
                f"Voice STT: {'enabled' if enabled else 'disabled'}\n"
                f"Transcription backend: {backend}\n"
                f"Recording backend: {rec_lib}\n"
                f"can_record={self._voice.can_record}  available={self._voice.available}",
                "Voice Status")

        # Enable / disable
        if any(w in msg for w in ("enable", "start", "on", "begin", "use")):
            self._voice._enabled = True
            return text_card("Voice input enabled. I'll transcribe audio you send.",
                             "Voice On")
        if any(w in msg for w in ("disable", "stop", "off")):
            self._voice._enabled = False
            return text_card("Voice input disabled.", "Voice Off")

        # Transcribe a file path if mentioned
        import re
        path_match = re.search(r'(/[\w./~-]+\.(?:wav|mp3|flac|m4a|ogg))', message)
        if path_match:
            path = path_match.group(1)
            text = self._voice.transcribe(path)
            if text:
                return text_card(text, f"Transcript — {path}")
            return text_card("Could not transcribe the file. "
                             "Check the path and install openai-whisper.",
                             "Transcription Failed")

        if not self._voice.available:
            return text_card(
                "No STT backend available. Install openai-whisper for local transcription:\n"
                "  pip install openai-whisper\n"
                "Or install faster-whisper for a lighter alternative:\n"
                "  pip install faster-whisper",
                "Voice — not configured")

        return text_card(
            "Voice input is ready. Send an audio file path or use the /voice/transcribe "
            "API endpoint to upload audio.",
            "Voice Ready")

    def _handle_vision_query(self, message: str, ctx: dict) -> PrismCard:
        """Capture a screenshot and feed it to the LLM for visual reasoning."""
        try:
            import base64
            try:
                import mss
                import mss.tools
                with mss.mss() as sct:
                    img = sct.grab(sct.monitors[1])
                    # Convert to JPEG bytes via mss.tools
                    img_bytes = mss.tools.to_png(img.rgb, img.size)
            except ImportError:
                return text_card(
                    "Vision analysis requires the mss library.\n"
                    "Install with: pip install mss\n\n"
                    "Alternatively, I can analyse an image you provide via "
                    "POST /perception/visual/reason",
                    "Vision — mss not installed")
            # Re-encode PNG bytes as JPEG-compatible base64 (PNG is fine for Claude)
            b64_img = base64.b64encode(img_bytes).decode("ascii")
            router = getattr(self, "_router", None)
            if router is None:
                return text_card(
                    "No LLM available to analyse the screenshot.", "Vision")
            result, model = router.call(
                prompt=message,
                images=[b64_img],
                min_capability=2,
                max_tokens=800,
                system="You are a visual assistant. Describe what you see on the screen clearly and concisely.",
            )
            if not result:
                return text_card("LLM returned no response for the screenshot.", "Vision")
            return text_card(result, "Vision Analysis")
        except Exception as exc:
            logger.warning("vision_query failed: %s", exc)
            # Fallback: text-only analysis without screenshot
            router = getattr(self, "_router", None)
            if router:
                result, _ = router.call(
                    prompt=message,
                    min_capability=1,
                    max_tokens=400,
                )
                if result:
                    return text_card(result, "Vision Analysis (text-only)")
            return text_card(
                f"Could not capture or analyse screen: {exc}", "Vision Error")

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
                "task":    message,
                "reason":  f"This requires: {capability_desc}. Approve autonomous execution?",
                "expires": time.time() + 300,   # 5-minute window
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
