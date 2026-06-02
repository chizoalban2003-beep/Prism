"""Universal service integrator — researches, characterises, and builds integrations for any unknown service."""
from __future__ import annotations
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from prism_llm_router import parse_llm_json

logger = logging.getLogger(__name__)

@dataclass
class DiscoveredService:
    service_id:   str
    name:         str
    description:  str
    category:     str        # "messaging"|"productivity"|"media"|"finance"|"other"
    access_method:str        # "official_api"|"unofficial_api"|"browser"|"installed_app"|"webhook"
    setup_steps:  list[str]
    executor_code:str        # Python code for PrismCollaborator to synthesise
    configured:   bool = False
    config_data:  dict = field(default_factory=dict)   # stored credentials/tokens
    created_at:   float = field(default_factory=time.time)
    last_used:    float = 0.0
    auto_buildable: bool = False  # True when PRISM can synthesise the integration autonomously

class PrismServiceDiscovery:
    """
    Universal new service/tool/platform integrator.

    When a user mentions any service PRISM doesn't know:
      Step 1: Research — use LLM + web search to understand the service
      Step 2: Clarify  — ask the user what they want and their constraints
      Step 3: Discover — find the best integration method
      Step 4: Build    — synthesise an executor via prism_collaborator
      Step 5: Store    — register in the tool registry for future use
      Step 6: Confirm  — tell the user what PRISM can now do with it

    Examples that all route through this:
      "Use Telegram to send me a reminder"
      "I want to use Notion for my notes"
      "Connect to my Garmin watch"
      "Book via Booksy"
      "Post to my Mastodon account"
      "Use my new NAS to store files"
    """

    # Integration methods in order of preference
    # (most reliable first, least reliable last)
    INTEGRATION_PREFERENCE = [
        "official_api",       # has documented API with auth
        "webhook",            # send/receive via webhooks
        "unofficial_api",     # reverse-engineered or community API
        "installed_app",      # app is installed, can automate via CLI or AppleScript
        "browser",            # browser automation via Playwright
        "manual_steps",       # PRISM can't automate, provides step-by-step instructions
    ]

    def __init__(
        self,
        collaborator=None,
        tool_registry=None,
        db_path: str = "~/.prism/discovered_services.db",
    ):
        self._collab   = collaborator
        self._registry = tool_registry
        self._db       = Path(db_path).expanduser()
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def is_known(self, service_name: str) -> bool:
        """Check if PRISM already has an integration for this service."""
        with sqlite3.connect(self._db) as c:
            row = c.execute(
                "SELECT id FROM services WHERE "
                "lower(name) LIKE lower(?)",
                (f"%{service_name}%",)).fetchone()
        return row is not None

    def get(self, service_name: str) -> Optional[DiscoveredService]:
        """Retrieve a previously discovered service."""
        with sqlite3.connect(self._db) as c:
            row = c.execute(
                "SELECT * FROM services WHERE lower(name) LIKE lower(?)",
                (f"%{service_name}%",)).fetchone()
        if not row:
            return None
        return DiscoveredService(
            service_id    = row[0], name=row[1], description=row[2],
            category      = row[3], access_method=row[4],
            setup_steps   = json.loads(row[5]),
            executor_code = row[6], configured=bool(row[7]),
            config_data   = json.loads(row[8]),
            created_at    = row[9], last_used=row[10])

    def discover(
        self,
        service_name:  str,
        user_intent:   str,
        constraints:   dict = None,
    ) -> tuple[DiscoveredService, list[str]]:
        """
        Research and characterise a new service.

        Returns (DiscoveredService, questions_to_ask_user).
        The questions are what PRISM needs the user to clarify
        before it can complete the integration.
        """
        constraints = constraints or {}

        # Research the service
        profile = self._research(service_name, user_intent)

        # Determine best integration method given constraints
        method = self._choose_method(profile, constraints)

        # Generate setup steps
        steps = self._generate_setup_steps(profile, method)

        # Generate clarifying questions
        questions = self._clarifying_questions(profile, method, constraints)

        service = DiscoveredService(
            service_id    = service_name.lower().replace(" ","_"),
            name          = service_name,
            description   = profile.get("description",""),
            category      = profile.get("category","other"),
            access_method = method,
            setup_steps   = steps,
            executor_code = "",
            auto_buildable= (method in ("official_api", "webhook", "unofficial_api")),
        )
        self._store(service)
        return service, questions

    def build_integration(
        self,
        service:      DiscoveredService,
        user_answers: dict,
    ) -> bool:
        """
        Given user's answers to clarifying questions, build the integration.
        Stores credentials, synthesises executor code, registers in tool registry.
        Returns True if the integration was successfully built.
        """
        if self._collab is None:
            return False

        # Store user-provided config
        service.config_data.update(user_answers)

        # Ask LLM to synthesise executor code
        prompt = (
            f"Write a Python class that integrates with {service.name}.\n"
            f"Integration method: {service.access_method}\n"
            f"What the user wants to do: {service.description}\n"
            f"Available config: {json.dumps(service.config_data)}\n\n"
            f"Requirements:\n"
            f"- Class name: {service.name.replace(' ','').title()}Executor\n"
            f"- Methods: send(message,**kwargs), receive(n=10), status()\n"
            f"- Handle errors gracefully — never raise, return dicts\n"
            f"- Use only stdlib + requests (if HTTP needed)\n"
            f"- Include a test() method that verifies the connection\n"
            f"Return ONLY the Python class code."
        )

        code, _ = self._collab._router.call(
            prompt, min_capability=2, max_tokens=1500)

        if not code or "class" not in code:
            return False

        service.executor_code = code
        service.configured    = True
        service.config_data   = user_answers
        self._store(service)

        # Register in tool registry if available
        if self._registry:
            from prism_executor_agent import ExecutorRecord
            record = ExecutorRecord(
                executor_id   = service.service_id,
                task_name     = service.service_id,
                description   = service.description,
                safety_class  = "communicate",
                source        = "discovered",
                code_path     = self._save_code(service),
                success_rate  = 0.0,
                n_executions  = 0,
                last_used     = time.time(),
                tags          = [service.name.lower(), service.category,
                                  service.access_method],
            )
            self._registry.register(record)

        logger.info("Integration built for %s via %s",
                    service.name, service.access_method)
        return True

    def confirmation_message(self, service: DiscoveredService) -> str:
        """What PRISM tells the user after building an integration."""
        if not service.configured:
            return (f"I've researched {service.name}. "
                    f"It uses {service.access_method}. "
                    f"To complete the integration I need a few details from you.")
        return (f"I've connected to {service.name} via {service.access_method}. "
                f"I can now: {service.description}. "
                f"Just ask me to use {service.name} for any of these.")

    # ── Research helpers ──────────────────────────────────────────────────

    def _research(self, name: str, intent: str) -> dict:
        """Use LLM to characterise the service."""
        if self._collab is None:
            return {"name":name,"description":intent,
                    "category":"other","has_api":False,
                    "api_url":"","needs_auth":True}
        prompt = (
            f"What is {name}? I want to: {intent}\n"
            f"Return ONLY valid JSON:\n"
            f'{{"description":"what it is in one sentence",'
            f'"category":"messaging|productivity|media|finance|health|other",'
            f'"has_api":true/false,'
            f'"api_url":"official API docs URL or empty",'
            f'"needs_auth":true/false,'
            f'"auth_type":"api_key|oauth|app_password|none",'
            f'"has_webhook":true/false,'
            f'"has_cli_app":true/false}}'
        )
        raw, _ = self._collab._router.call(
            prompt, min_capability=1, max_tokens=300, json_mode=True)
        data = parse_llm_json(raw)
        if data and isinstance(data, dict):
            data["name"] = name
            return data
        return {"name":name,"description":intent,"category":"other",
                "has_api":False,"api_url":"","needs_auth":True}

    def _choose_method(self, profile: dict, constraints: dict) -> str:
        """Pick the best integration method given what's available."""
        prefers_no_browser = constraints.get("no_browser", False)
        prefers_official   = constraints.get("official_only", False)

        if profile.get("has_api"):
            return "official_api"
        if profile.get("has_webhook"):
            return "webhook"
        if not prefers_official:
            if profile.get("has_cli_app"):
                return "installed_app"
            if not prefers_no_browser:
                return "browser"
        return "manual_steps"

    def _generate_setup_steps(self, profile: dict,
                               method: str) -> list[str]:
        """Generate human-readable setup steps for this integration."""
        name      = profile.get("name","")
        auth_type = profile.get("auth_type","api_key")
        steps     = []

        if method == "official_api":
            if auth_type == "api_key":
                steps.append(f"Log into {name} → Settings → Developers → Generate API key")
                steps.append("Paste the API key here when asked")
            elif auth_type == "oauth":
                steps.append(f"I'll open {name}'s authorisation page")
                steps.append("Click Allow to grant PRISM access")
            elif auth_type == "app_password":
                steps.append(f"Log into {name} → Security → App Passwords")
                steps.append("Create a new app password and paste it here")
        elif method == "browser":
            steps.append(f"Make sure {name} is accessible in a browser")
            steps.append("PRISM will automate the browser — "
                         "do not interact while it runs")
        elif method == "installed_app":
            steps.append(f"Make sure {name} app is installed on this device")
        elif method == "webhook":
            steps.append(f"In {name}'s settings, find Webhooks or Integrations")
            steps.append("Create a new webhook and paste the URL here")
        else:
            steps.append(f"PRISM cannot automate {name} directly.")
            steps.append("I can show you the steps to do it manually.")
        return steps

    def _clarifying_questions(self, profile: dict,
                               method: str,
                               constraints: dict) -> list[str]:
        """Questions PRISM needs answered before building the integration."""
        questions = []
        auth_type = profile.get("auth_type","api_key")
        name      = profile.get("name","")

        if method == "official_api":
            if auth_type == "api_key":
                questions.append(f"Please paste your {name} API key:")
            elif auth_type == "app_password":
                questions.append(f"Please paste your {name} app password:")
            elif auth_type == "oauth":
                questions.append(
                    f"What permissions should PRISM have on {name}? "
                    f"(e.g. read-only, read and write, full access)")

        if not constraints.get("scope"):
            questions.append(
                f"What specifically do you want PRISM to do with {name}? "
                f"(e.g. send messages, read notifications, post updates)")

        if not constraints.get("approval_level"):
            questions.append(
                f"Should PRISM ask for approval before sending anything "
                f"via {name}, or can it act automatically?")

        return questions

    def _save_code(self, service: DiscoveredService) -> str:
        """Save synthesised executor code to disk."""
        path = (self._db.parent / "discovered_tools" /
                f"{service.service_id}_executor.py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(service.executor_code, encoding="utf-8")
        return str(path)

    def _store(self, service: DiscoveredService) -> None:
        with sqlite3.connect(self._db) as c:
            c.execute(
                "INSERT OR REPLACE INTO services VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (service.service_id, service.name, service.description,
                 service.category, service.access_method,
                 json.dumps(service.setup_steps),
                 service.executor_code, int(service.configured),
                 json.dumps(service.config_data),
                 service.created_at, service.last_used))

    def list_all(self) -> list[DiscoveredService]:
        with sqlite3.connect(self._db) as c:
            rows = c.execute("SELECT * FROM services ORDER BY last_used DESC").fetchall()
        return [DiscoveredService(
            service_id=r[0],name=r[1],description=r[2],category=r[3],
            access_method=r[4],setup_steps=json.loads(r[5]),
            executor_code=r[6],configured=bool(r[7]),
            config_data=json.loads(r[8]),
            created_at=r[9],last_used=r[10]) for r in rows]

    def _init_db(self) -> None:
        with sqlite3.connect(self._db) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS services(
                id TEXT PRIMARY KEY, name TEXT, description TEXT,
                category TEXT, access_method TEXT, setup_steps_json TEXT,
                executor_code TEXT, configured INTEGER,
                config_data_json TEXT, created_at REAL, last_used REAL)""")
