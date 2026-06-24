from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

from domain_configs import ALL_DOMAINS, DomainDecisionModel
from prism_agent_bootstrap import (
    build_llm_config,
    load_toml_config,
    safe_init,
    safe_init_class,
)
from prism_browser_agent import PrismBrowserAgent
from prism_calendar import PrismCalendar
from prism_calibration import PrismCalibration
from prism_chat_context import (
    attach_memory_recall,
    attach_perception,
    attach_persona,
    setup_required_short_circuit,
)
from prism_chat_graph_bridge import mirror_turn_to_graph, recall_from_graph
from prism_chat_subsystems import build_chat_subsystems
from prism_chat_tiers import TierDispatcher
from prism_contacts import PrismContacts
from prism_device_agent import PrismDeviceAgent
from prism_email import PrismEmail
from prism_goal_intents import handle_goal_intent
from prism_identity_learning import build_identity_learning
from prism_info_intents import handle_info_intent
from prism_instructions import PrismInstructions
from prism_intents import INTENTS as _ROUTING_INTENTS
from prism_llm_router import LLMRouter
from prism_memory import PrismMemory
from prism_organ_dispatch import dispatch_organ
from prism_pa_intents import handle_pa_intent
from prism_perception_cluster import build_perception_cluster
from prism_planner import PrismPlanner
from prism_proactive import build_advanced_triggers
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
from prism_routing import LLMClassifier, route_intent, should_suppress
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

    # Most-recent user-turn graph node id, used by the durability bridge to
    # link a user turn to the assistant turn that answers it.
    _last_user_turn_node: Optional[str] = None

    INTENTS = _ROUTING_INTENTS

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
        self._config = load_toml_config(Path(__file__).parent / "prism_config.toml")

        # Overlay user-edited settings.db on top of TOML so the chat-driven
        # setup-form card can configure integrations without file edits.
        try:
            from prism_settings_store import get_settings_store
            self._settings_store = get_settings_store()
            self._config = self._settings_store.overlay_on_toml(self._config)
        except Exception as exc:
            logger.debug("settings overlay skipped: %s", exc)
            self._settings_store = None

        self._user = self._config.get("user", {}).get("name", "default")

        _llm_cfg = build_llm_config(self._config, claude_api_key=self._claude_key)
        self._router = LLMRouter(
            preferred   = _llm_cfg["preferred"],
            fallback    = _llm_cfg["fallback"],
            ollama_host = _llm_cfg["ollama_host"],
            config      = _llm_cfg,
        )

        self._queue  = TaskQueue()
        _agent_cfg = self._config.get("agent", {}) if self._config else {}
        _planner_model = _agent_cfg.get("text_model") or _agent_cfg.get("ollama_model") or text_model
        _planner_host  = _agent_cfg.get("ollama_host") or ollama_host
        self._text_model = _planner_model
        self._ollama_host = _planner_host.rstrip('/')
        self._planner = PrismPlanner(
            ollama_host    = _planner_host,
            ollama_model   = _planner_model,
            claude_api_key = claude_api_key,
        )

        # PolicyEngine — resource allocation + per-action approval policy
        self._policy = safe_init_class(
            "PolicyEngine", "prism_policy", "PolicyEngine", logger=logger)

        # PrismCollaborator — external research/synthesis bridge
        self._collaborator = safe_init_class(
            "PrismCollaborator", "prism_collaborator", "PrismCollaborator",
            router=self._router, logger=logger)

        self._device = PrismDeviceAgent.setup(
            policy_engine = self._policy,
            on_approval   = self._request_approval,
            collaborator  = self._collaborator,
            user          = self._user,
        )
        self._memory: Optional[PrismMemory] = safe_init_class(
            "PrismMemory", "prism_memory", "PrismMemory",
            ollama_host=ollama_host, logger=logger)
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

        # Perception / Proactive / Kinetic — awareness layer. The factory
        # builds all three and threads the perception↔kinetic↔proactive
        # wiring in one named place.
        _perc = build_perception_cluster(
            config             = self._config,
            policy             = self._policy,
            task_queue         = self._queue,
            on_voice_command   = self.chat,
            on_proactive_event = self._handle_proactive_event,
            logger             = logger,
        )
        self._perception = _perc.perception
        self._proactive  = _perc.proactive
        self._kinetic    = _perc.kinetic

        # Surgical ML Assembler — task-profiling algorithm compiler
        def _build_ml_assembler():
            from prism_ml_assembler import MLAssembler
            from prism_routes_ml import get_or_set_assembler as _set_asm
            asm = MLAssembler(
                llm_router=self._router,
                outcome_tracker=getattr(self, "_outcome_tracker", None),
            )
            _set_asm(asm)
            logger.info("MLAssembler ready")
            return asm
        self._ml_assembler: Optional[Any] = safe_init(
            "MLAssembler", _build_ml_assembler, logger=logger)

        self._search = PrismSearch.from_config(self._config)
        self._push   = PrismPush.from_config(self._config)
        if self._proactive:
            self._proactive._push = self._push
        self._contacts     = PrismContacts.from_config(self._config)
        self._task_mgr     = PrismTasks.from_config(self._config)
        self._calibration  = PrismCalibration()
        self._last_decision: dict = {}
        _chat_subs = build_chat_subsystems(
            router       = self._router,
            policy       = self._policy,
            push         = self._push,
            task_queue   = self._queue,
            device_agent = self._device,
            memory       = self._memory,
        )
        self._autonomous   = _chat_subs.autonomous
        self._composer     = _chat_subs.composer
        self._chain        = _chat_subs.chain
        self._organ_loader = _chat_subs.organ_loader
        self._chain_expert = _chat_subs.chain_expert

        # MCP (Model Context Protocol) client — connect to any configured
        # external MCP servers and expose their tools as organs. Disabled
        # unless [mcp].enabled is set in prism_config.toml, so this is a no-op
        # for the default install and for tests.
        def _build_mcp():
            import prism_mcp
            mcp = prism_mcp.MCPManager.from_config(self._config)
            prism_mcp.set_manager(mcp)
            if mcp.server_names():
                mcp.connect_all()
                _n = prism_mcp.register_mcp_organs(
                    self._organ_loader, mcp, router=self._router)
                logger.info("MCP: registered %d tool(s) from %d server(s)",
                            _n, len(mcp.server_names()))
            return mcp
        self._mcp: Optional[Any] = safe_init("MCP", _build_mcp, logger=logger)

        # L1 Constitution guard + Bud execution manager
        def _build_constitution_bud():
            from prism_bud_manager import BudManager
            from prism_constitution import ConstitutionGuard
            guard = ConstitutionGuard()
            return guard, BudManager(constitution_guard=guard)
        _const_bud = safe_init(
            "Constitution/BudManager", _build_constitution_bud, logger=logger)
        self._constitution: Optional[Any]
        self._bud_mgr: Optional[Any]
        self._constitution, self._bud_mgr = _const_bud if _const_bud else (None, None)

        # HorizonPlanner — cross-session long-horizon goal persistence
        def _build_horizon():
            from prism_horizon import HorizonPlanner
            planner = HorizonPlanner(
                llm_router = self._router,
                task_queue = self._queue,
                push       = self._push,
            )
            if self._chain is not None:
                planner._chain = self._chain
            triggered = planner.on_session_start()
            if triggered:
                logger.info(
                    "HorizonPlanner: %d goal(s) triggered at startup: %s",
                    len(triggered), triggered,
                )
            return planner
        self._horizon: Optional[Any] = safe_init(
            "HorizonPlanner", _build_horizon, logger=logger)

        # OrganBus — LLM-mediated inter-engine communication bus
        self._organ_bus: Optional[Any] = None
        def _build_organ_bus():
            from prism_organ_bus import OrganBus
            bus = OrganBus(llm_router=self._router)
            self._organ_bus = bus
            self._register_organ_subscriptions()
            return bus
        self._organ_bus = safe_init(
            "OrganBus", _build_organ_bus, logger=logger)

        # Identity & learning cluster — soul, outcome_tracker, persona,
        # crystalliser, narrative, reflection. The factory builds
        # outcome_tracker before the persona block (so crystalliser
        # receives a real tracker) and threads the bi-directional wires
        # (ot↔crystalliser, ot→kinetic, asm→ot, chain→persona) inside
        # the closures that own each side.
        _id_learn = build_identity_learning(
            router       = self._router,
            organ_bus    = self._organ_bus,
            chain        = getattr(self, '_chain', None),
            horizon      = getattr(self, '_horizon', None),
            memory       = self._memory,
            calibration  = self._calibration,
            ml_assembler = getattr(self, '_ml_assembler', None),
            kinetic      = self._kinetic,
            logger       = logger,
        )
        self._soul:            Optional[Any] = _id_learn.soul
        self._outcome_tracker: Optional[Any] = _id_learn.outcome_tracker
        self._persona:         Any           = _id_learn.persona
        self._crystalliser:    Any           = _id_learn.crystalliser
        self._narrative:       Any           = _id_learn.narrative
        self._reflection                     = _id_learn.reflection

        # PlanTelemetry — per-step status for DailyPlans, feeds re-plan signal
        self._last_plan_id: Optional[str] = None
        self._plan_telemetry: Optional[Any] = safe_init_class(
            "PlanTelemetry", "prism_plan_telemetry", "get_plan_telemetry",
            logger=logger, info_on_success="PlanTelemetry ready")

        # ContextManager — work/personal/focus context switching
        def _build_context_manager():
            from prism_context_profile import ContextManager
            cm = ContextManager()
            if hasattr(self, '_chain'):
                cm.inject_into_chain(self._chain)
            logger.info("ContextManager ready (active: %s)", cm.active_id)
            return cm
        self._context_manager = safe_init(
            "ContextManager", _build_context_manager, logger=logger)

        # ProactiveBusWatcher — connect OrganBus signals to proactive triggers
        def _build_bus_watcher():
            if self._organ_bus is None or getattr(self, '_proactive', None) is None:
                return None
            from prism_proactive_bus_watcher import ProactiveBusWatcher
            watcher = ProactiveBusWatcher(
                proactive = self._proactive,
                organ_bus = self._organ_bus,
            )
            watcher.register()
            logger.info("ProactiveBusWatcher registered")
            return watcher
        self._bus_watcher = safe_init(
            "ProactiveBusWatcher", _build_bus_watcher, logger=logger)

        # Budget — CEO-style governance over LLM spend.
        # PRISM (the manager) checks itself before spending the user's money.
        def _build_budget():
            from prism_budget import from_config
            budget = from_config(self._config)
            logger.info("PrismBudget ready: daily=$%.2f", budget.daily_usd)
            return budget
        self._budget = safe_init("PrismBudget", _build_budget, logger=logger)

        # ChainOrchestrator — prefrontal cortex for multi-step coordination
        def _build_orchestrator():
            from prism_orchestrator import ChainOrchestrator
            orch = ChainOrchestrator(
                chain           = getattr(self, '_chain', None),
                organ_loader    = getattr(self, '_organ_loader', None),
                outcome_tracker = getattr(self, '_outcome_tracker', None),
                horizon         = getattr(self, '_horizon', None),
                router          = self._router,
                soul            = getattr(self, '_soul', None),
            )
            orch._persona = getattr(self, '_persona', None)
            # Resume any graphs that were paused waiting on horizon goals
            resumed = orch.resume_waiting(self._execute, {})
            if resumed:
                logger.info("ChainOrchestrator: %d paused graph(s) resumed", len(resumed))
            logger.info("ChainOrchestrator ready")
            return orch
        self._orchestrator: Optional[Any] = safe_init(
            "ChainOrchestrator", _build_orchestrator, logger=logger)

        # Advanced proactive triggers — registered after all dependencies exist
        def _register_advanced_triggers():
            if getattr(self, '_proactive', None) is None:
                return None
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
            return advanced
        safe_init(
            "Advanced proactive triggers",
            _register_advanced_triggers, logger=logger)

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

            # 1b. Check for a personal-fact assertion ("my X is Y") — these
            # belong in PrismMemory, not the standing-rule store, so the
            # later "what is my X" recall can find them.
            fact = self._instructions.parse_fact(message or "")
            if fact and self._memory:
                key, value = fact
                try:
                    self._memory.ingest(
                        f"My {key} is {value}.",
                        source="fact",
                        title=f"my {key}",
                        tags=["fact", key.lower()],
                    )
                    return text_card(
                        f"✓ Got it — your {key} is {value}.",
                        "Fact stored",
                    )
                except Exception:
                    pass

            # 2. Inject relevant instructions into context
            instructions_str = self._instructions.to_context_string(message or "")
            if instructions_str:
                context["standing_instructions"] = instructions_str

            # 3. Inject conversation history
            context["history"] = self._chat_history[-10:]

            # 4. Add to history — unless this is a never-log intent
            # (constitution: e.g. email_send / phone_call must not have their
            # content persisted to conversation history or sessions).
            suppress_log = self._should_suppress_logging(message or "")
            self._suppress_logging = suppress_log
            if not suppress_log:
                self._chat_history.append({"role": "user", "content": message or ""})
                if len(self._chat_history) > 20:
                    self._chat_history = self._chat_history[-20:]

            # 5. Perception context
            attach_perception(context, self._perception)

            # 6. Memory context — flat first, then graph recall, then mirror this turn
            if self._memory and message:
                attach_memory_recall(context, self._memory, message)
                # Recall from the durable conversation graph BEFORE mirroring
                # this turn, so it surfaces *past* turns, not the current one.
                self._graph_recall(message, context)
                if not suppress_log:
                    try:
                        _uid = self._memory.ingest_conversation("user", message)
                        self._mirror_turn_to_graph("user", message, _uid)
                    except Exception:
                        pass

            # 7. Inject active context into the request context dict
            if getattr(self, '_context_manager', None) is not None:
                self._context_manager.inject_into_chain_ctx(context)
                self._context_manager.inject_into_chain(self._chain)

            attach_persona(context, getattr(self, '_persona', None))

            # 8. Pre-tier short-circuit + tiered routing
            initial_card = setup_required_short_circuit(
                message or "", self._calendar, self._email)
            card = self._tier_dispatcher().dispatch(
                message or "", context, initial_card=initial_card)

            # 9. Store response in history (skip for never-log intents)
            if hasattr(card, 'body') and card.body and not suppress_log:
                self._chat_history.append(
                    {"role": "assistant", "content": card.body[:500]})

            # 10. Memory ingestion for response
            if self._memory and card.body and not suppress_log:
                try:
                    _aid = self._memory.ingest_conversation("assistant", card.body)
                    self._mirror_turn_to_graph("assistant", card.body, _aid)
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

    def _graph_recall(self, message: str, context: dict) -> None:
        recall_from_graph(getattr(self, "_memory_graph", None), message, context)

    def _mirror_turn_to_graph(self, role: str, content: str, entry_id) -> None:
        self._last_user_turn_node = mirror_turn_to_graph(
            getattr(self, "_memory_graph", None),
            role,
            content,
            entry_id,
            getattr(self, "_last_user_turn_node", None),
        )

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
        return route_intent(message, self.INTENTS, self._llm_classify)

    def _should_suppress_logging(self, message: str) -> bool:
        """True when the message routes (by regex, no LLM call) to a
        constitution ``never_log`` intent (e.g. email_send / phone_call), so
        its content is kept out of conversation history, memory, and sessions."""
        return should_suppress(message, self.INTENTS, self._constitution)

    def _llm_classify(self, message: str) -> Optional[str]:
        classifier = getattr(self, "_llm_classifier", None)
        if classifier is None:
            classifier = LLMClassifier(
                intents=self.INTENTS,
                router=getattr(self, "_router", None),
                ollama_host=self._ollama_host,
                text_model=self._text_model,
                get_organ_intents=self._organ_intents_for_classifier,
            )
            self._llm_classifier = classifier
        return classifier.classify(message)

    def _organ_intents_for_classifier(self) -> dict[str, str]:
        loader = getattr(self, "_organ_loader", None)
        if loader is None:
            return {}
        try:
            return loader.known_intents()
        except Exception:
            return {}

    def _tier_dispatcher(self) -> TierDispatcher:
        dispatcher = getattr(self, "_tier_dispatcher_cache", None)
        if dispatcher is None:
            dispatcher = TierDispatcher(
                orchestrator=getattr(self, "_orchestrator", None),
                chain_expert=self._chain_expert,
                chain=self._chain,
                composer=self._composer,
                execute=self._execute,
                route=self._route,
            )
            self._tier_dispatcher_cache = dispatcher
        return dispatcher

    def _execute(self, intent: str, message: str, ctx: dict) -> PrismCard:
        info_card = handle_info_intent(self, intent, message, ctx)
        if info_card is not None:
            return info_card
        goal_card = handle_goal_intent(self, intent, message, ctx)
        if goal_card is not None:
            return goal_card
        pa_card = handle_pa_intent(self, intent, message, ctx)
        if pa_card is not None:
            return pa_card
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
            "predict_match", "moment", "session",
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
                    self._last_plan = output
                    self._last_plan_request = message
                    if self._plan_telemetry is not None:
                        try:
                            self._last_plan_id = self._plan_telemetry.record_plan(output, message)
                        except Exception as exc:
                            logger.debug("plan_telemetry.record_plan failed: %s", exc)
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
        if intent == "memory_recall":
            if not self._memory:
                return text_card(
                    "Memory isn't initialised yet, so I can't recall stored "
                    "facts. Try restarting the daemon.",
                    "Memory unavailable",
                )
            try:
                hits = self._memory.search(message or "", top_n=3)
            except Exception as exc:
                return text_card(f"Memory search failed: {exc}", "Memory")
            facts = [h for h in hits if h.entry.source == "fact"]
            top   = facts or hits
            if not top:
                return text_card(
                    "I don't have a stored answer for that yet. Tell me with "
                    "\"remember that my X is Y\" and I'll keep it.",
                    "No memory of that",
                )
            # Prefer terse direct answer for a fact hit; fall back to a list.
            if facts:
                lines = [h.entry.content for h in facts[:3]]
            else:
                lines = [f"- {h.entry.content}" for h in hits[:3]]
            return text_card("\n".join(l for l in lines if l), "Recalled")

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
                answer = (raw or "").strip()
                if not answer:
                    # Router returned nothing — almost always means no LLM
                    # backend is actually reachable (stdlib fallback yields "").
                    return text_card(
                        "I couldn't generate a reply — no LLM backend is reachable. "
                        "Connect Ollama or add a Claude/OpenAI key at "
                        "/settings/llm (or run `python3 prism_daemon.py --setup-llm`).",
                        "LLM not connected",
                    )
                return text_card(answer, "Chat")
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
                            # Re-enter the full gate with the approval flag set:
                            # this runs the organ through L3 BudManager scoped
                            # context instead of handing it the unscoped ctx.
                            return self._execute(organ_intent, organ_message, organ_ctx)
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

        if intent == "vision_query":
            return self._handle_vision_query(message, ctx)

        # Dynamic organ registry — run a loaded organ through the three-layer
        # gate (L1 constitution → L2 approval/rate-limit → L3 BudManager).
        # Extracted to prism_organ_dispatch to keep this module focused.
        organ_card = dispatch_organ(self, intent, message, ctx)
        if organ_card is not None:
            return organ_card

        # Unknown intent — defer to _handle_unknown, which gates organ
        # synthesis behind a synthesis_approval_card. Approval round-trips
        # through /device/approve into handle_synthesis_approval(); that path
        # is the only one allowed to call OrganLoader.synthesize() now.
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
        from prism_unknown_handler import handle_unknown
        return handle_unknown(self, intent, message, ctx)
    def record_denial(self, task: str, params: dict, reason: str = "") -> dict:
        """
        Persist a user denial so future identical requests can be pre-warned or
        auto-declined.

        Two writes happen, both best-effort:

        1. **Task-scoped record** — trigger=task_slug. Surfaces via
           ``prior_denials_for(task)`` so the next approval card for the
           exact same task shows a "you denied this before" banner.
        2. **Standing rule** — if the reason contains a marker like "never"
           or "always", the reason is also stored with a broad trigger
           category (email/calendar/finance/...) so it applies to ALL
           similar requests, not just retries of this one task.

        Returns
        -------
        dict
            ``{"task_scoped": <bool>, "standing_trigger": <str|None>,
               "standing_text": <str|None>}``. Callers can use this to
            phrase a "saved as a rule" confirmation back to the user.
        """
        out: dict = {"task_scoped": False, "standing_trigger": None, "standing_text": None}
        try:
            instr = getattr(self, "_instructions", None)
            if instr is None:
                return out
            # 1. Task-scoped record (one-shot retry guard)
            trigger = (task or "")[:80] or "denial"
            text = (reason or "").strip() or f"User denied '{task}' once. Ask again only if context differs."
            if reason and reason.strip():
                text = f"On '{task}': {reason.strip()[:300]}"
            try:
                instr.add(text=text, trigger=trigger)
                out["task_scoped"] = True
            except TypeError:
                try:
                    instr.add(text)
                    out["task_scoped"] = True
                except Exception:
                    pass

            # 2. Standing rule extraction
            try:
                standing_text, standing_trigger = instr.classify_denial(task, reason)
            except Exception:
                standing_text, standing_trigger = None, None
            if standing_text and standing_trigger:
                try:
                    instr.add(text=standing_text, trigger=standing_trigger)
                    out["standing_trigger"] = standing_trigger
                    out["standing_text"]    = standing_text
                except Exception:
                    pass
        except Exception as exc:
            logger.debug("record_denial best-effort failed: %s", exc)
        return out

    def replan(self, instructions: str = "", tasks: list | None = None) -> PrismCard:
        """
        Re-run plan generation with user refinement instructions and an
        optional edited task list. The instructions are folded into the KDE
        ask message so the planner can honour pins/removals/timing changes
        without the user needing to re-state the whole request.
        """
        base = getattr(self, "_last_plan_request", "") or "plan my day"
        pinned = ""
        if tasks:
            try:
                rows = []
                for t in tasks[:24]:
                    if not isinstance(t, dict):
                        continue
                    title = str(t.get("title", "") or "").strip()[:120]
                    when = str(t.get("time", "") or "").strip()[:32]
                    if title:
                        rows.append(f"- {when} {title}".strip())
                if rows:
                    pinned = "Keep these slots from the current plan:\n" + "\n".join(rows)
            except Exception:
                pinned = ""
        refine = (instructions or "").strip()
        prior_id = getattr(self, "_last_plan_id", None)
        telemetry_line = ""
        if prior_id and self._plan_telemetry is not None:
            try:
                telemetry_line = self._plan_telemetry.telemetry_summary(prior_id)
            except Exception as exc:
                logger.debug("plan_telemetry.summary failed: %s", exc)
        parts = [base, "Re-plan with these refinements:"]
        if telemetry_line:
            parts.append("Previous plan status: " + telemetry_line)
        if refine:
            parts.append(refine)
        if pinned:
            parts.append(pinned)
        message = "\n\n".join(parts)
        if not self._kde:
            return text_card("Re-planning requires the KDE module, which isn't available.", "Re-plan unavailable")
        try:
            result = self._kde.ask(message)
            output = getattr(result, "output", result)
            from sports_pro import DailyPlan
            if isinstance(output, DailyPlan):
                self._last_plan = output
                self._last_plan_request = base
                if self._plan_telemetry is not None:
                    try:
                        new_id = self._plan_telemetry.record_plan(output, base)
                        if prior_id:
                            self._plan_telemetry.supersede(prior_id, new_id)
                        self._last_plan_id = new_id
                    except Exception as exc:
                        logger.debug("plan_telemetry.record_plan (replan) failed: %s", exc)
                return plan_card(output)
            if isinstance(output, str):
                return text_card(output, "Re-plan")
            return text_card(str(output), "Re-plan")
        except Exception as exc:
            return text_card(f"Re-plan failed: {exc}", "Re-plan error")

    def apply_settings_change(self, section: str) -> dict:
        """
        Re-overlay settings.db on top of TOML and hot-rebuild the affected
        integration service. Lets the chat-driven setup-form card configure
        email/calendar/smarthome/search/push without restarting the daemon.
        Returns {"ok": bool, "configured": bool, "section": section}.
        """
        try:
            if getattr(self, "_settings_store", None) is None:
                from prism_settings_store import get_settings_store
                self._settings_store = get_settings_store()
            # Refresh the merged config so from_config sees new values
            try:
                import tomllib
            except ImportError:
                try:
                    import tomli as tomllib  # type: ignore[no-redef]
                except ImportError:
                    tomllib = None  # type: ignore[assignment]
            base: dict = {}
            if tomllib is not None:
                try:
                    with open(Path(__file__).parent / "prism_config.toml", "rb") as f:
                        base = tomllib.load(f)
                except Exception:
                    pass
            self._config = self._settings_store.overlay_on_toml(base)

            configured = False
            if section == "email":
                from prism_email import PrismEmail
                self._email = PrismEmail.from_config(self._config)
                configured = bool(getattr(self._email, "configured", False))
            elif section == "calendar":
                from prism_calendar import PrismCalendar
                self._calendar = PrismCalendar.from_config(self._config)
                configured = bool(getattr(self._calendar, "configured", False))
            elif section == "smarthome":
                from prism_smart_home import PrismSmartHome
                self._smarthome = PrismSmartHome.from_config(self._config)
                configured = bool(getattr(self._smarthome, "configured", False))
            elif section == "search":
                from prism_search import PrismSearch
                self._search = PrismSearch.from_config(self._config)
                configured = bool(getattr(self._search, "configured", False))
            elif section == "push":
                try:
                    from prism_push import PrismPush
                    self._push = PrismPush.from_config(self._config)
                    configured = True
                except Exception:
                    configured = False
            elif section == "contacts":
                from prism_contacts import PrismContacts
                self._contacts = PrismContacts.from_config(self._config)
                configured = True
            else:
                # tasks/twilio: re-rebuild lazily on next use; just mark refreshed
                configured = True
            return {"ok": True, "section": section, "configured": configured}
        except Exception as exc:
            logger.warning("apply_settings_change(%s) failed: %s", section, exc)
            return {"ok": False, "section": section, "error": str(exc)[:200]}

    def _slugify_intent(self, message: str) -> str:
        """
        Turn a free-text request into a Python-safe intent identifier suitable
        for use as an organ file name. 'What is on my calendar today?' →
        'synth_what_is_on_my_calendar_today'.
        """
        base = re.sub(r"[^a-z0-9_]+", "_", (message or "").lower()).strip("_")
        base = re.sub(r"_+", "_", base)[:40] or "new_intent"
        return f"synth_{base}"

    def handle_synthesis_approval(self, params: dict, instructions: str = "") -> PrismCard:
        """
        Called by /device/approve when the user approves synthesising a new
        organ. Augments the synthesis prompt with the user's optional
        instructions, drives OrganLoader.synthesize() through its safety
        pipeline, and runs the freshly registered organ inline. Falls back
        to PrismAutonomous.execute_async for capabilities the organ_loader
        can't satisfy (those needing pip installs, async work, etc.).
        """
        intent     = (params or {}).get("intent", "")
        message    = (params or {}).get("message", "")
        capability = (params or {}).get("capability", "")

        if not intent or not message:
            return text_card(
                "Synthesis approval was missing intent or message.",
                "Synthesis failed")

        refined = message
        if instructions and instructions.strip():
            refined = (f"{message}\n\n"
                       f"User instructions for the implementation:\n"
                       f"{instructions.strip()}")

        # L1 absolute limit — cap how many organs may be synthesised per session.
        if self._bud_mgr is not None and not self._bud_mgr.synthesis_allowed():
            return text_card(
                "PRISM has reached its per-session synthesis limit (constitution L1). "
                "Start a new session before building more tools.",
                "Synthesis blocked")

        # First try the OrganLoader pipeline (sandboxed, AST-checked, persistent).
        try:
            if self._organ_loader and self._organ_loader.synthesize(intent, refined):
                if self._bud_mgr is not None:
                    self._bud_mgr.record_synthesis()
                organ_fn = self._organ_loader.get(intent)
                if organ_fn:
                    try:
                        # Run the freshly-registered organ through the full
                        # security gate (L1 constitution → L2 approval/rate
                        # limit → L3 BudManager scoped ctx) rather than calling
                        # it directly with an unscoped, empty context.
                        return self._execute(intent, message, {})
                    except Exception as exc:
                        logger.warning("Synthesised organ '%s' failed at execution: %s",
                                       intent, exc)
                        return text_card(
                            f"Built `{intent}` but it failed when run: {exc}",
                            intent)
        except Exception as exc:
            logger.warning("OrganLoader.synthesize failed for '%s': %s", intent, exc)

        # Fallback: the autonomous engine handles capabilities that need
        # external packages or async execution.
        if self._autonomous:
            task_id = self._autonomous.execute_async(refined, {})
            cap_line = f"\nCapability: **{capability}**" if capability else ""
            return text_card(
                f"Approved — building and running the tool now.{cap_line}\n"
                f"Task ID: `{task_id}`\n\n"
                f"I'll notify you when it finishes.",
                "Synthesising in background")

        return text_card(
            "Couldn't synthesise a tool — no organ_loader and no autonomous engine.",
            "Synthesis failed")
