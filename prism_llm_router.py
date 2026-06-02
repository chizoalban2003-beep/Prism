from __future__ import annotations
import json
import logging
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


def parse_llm_json(raw: str) -> Optional[dict]:
    """Safely parse JSON from an LLM response that may have markdown fences."""
    if not raw:
        return None
    try:
        clean = raw.strip()
        if clean.startswith("```"):
            parts = clean.split("```")
            clean = parts[1].lstrip("json").strip() if len(parts) > 1 else clean
        return json.loads(clean.strip())
    except Exception:
        try:
            start = clean.index("{"); end = clean.rindex("}") + 1
            return json.loads(clean[start:end])
        except Exception:
            return None

@dataclass
class LLMOption:
    """One available LLM."""
    provider:    str          # "claude"|"ollama"|"openai_compat"|"stdlib"
    model:       str          # e.g. "claude-sonnet-4-20250514"|"mistral"|"deepseek-r1"
    endpoint:    str          # base URL
    available:   bool
    latency_ms:  float = 0.0  # measured on last ping
    capability:  int   = 0    # 0=none 1=basic 2=good 3=best — used for ranking
    notes:       str   = ""

# Capability ranking — higher = preferred for complex tasks
MODEL_CAPABILITY: dict[str, int] = {
    "claude-sonnet": 3, "claude-opus": 3,
    "deepseek-r1": 3, "deepseek-v3": 2,
    "llama3": 2, "llama3.1": 2, "llama3.2": 2,
    "mistral": 2, "mixtral": 2,
    "qwen": 1, "phi": 1, "gemma": 1,
    "stdlib": 0,
}

def _rank(option: LLMOption) -> int:
    for key, cap in MODEL_CAPABILITY.items():
        if key in option.model.lower():
            return cap
    return 1 if option.available else 0

class LLMRouter:
    """
    Discovers all available LLMs, ranks them, and routes requests
    to the best available option with automatic fallback.

    Discovery checks (in order, all non-blocking with 2s timeout):
      1. Anthropic API (ANTHROPIC_API_KEY env or config)
      2. Ollama local (localhost:11434 or configured host)
      3. OpenAI-compatible endpoints (OPENAI_API_KEY env or config)
      4. stdlib fallback (always available, no LLM capability)
    """

    def __init__(
        self,
        preferred:    str = "",           # e.g. "ollama/mistral" or "claude"
        fallback:     list[str] = None,   # ordered fallback list
        ollama_host:  str = "http://localhost:11434",
        config:       dict = None,        # from prism_config.toml [llm] section
    ):
        self._preferred   = preferred
        self._fallback    = fallback or []
        self._ollama_host = ollama_host
        self._config      = config or {}
        self._options:    list[LLMOption] = []
        self._discovered  = False
        self._last_scan   = 0.0

    @classmethod
    def from_config(cls, config_path: str = "~/.prism/config.toml",
                    claude_api_key: str = "") -> "LLMRouter":
        """Load LLM preferences from prism_config.toml [llm] section."""
        try:
            from pathlib import Path
            import tomllib
            path = Path(config_path).expanduser()
            if path.exists():
                data = tomllib.loads(path.read_text())
                llm  = data.get("llm", {})
                # Allow caller to override claude_api_key
                if claude_api_key:
                    llm["claude_api_key"] = claude_api_key
                return cls(
                    preferred   = llm.get("preferred", ""),
                    fallback    = llm.get("fallback", []),
                    ollama_host = llm.get("ollama_host","http://localhost:11434"),
                    config      = llm,
                )
        except Exception:
            pass
        if claude_api_key:
            return cls(config={"claude_api_key": claude_api_key})
        return cls()

    def discover(self, force: bool = False) -> list[LLMOption]:
        """
        Discover all available LLMs. Cached for 60 seconds.
        Returns list sorted by capability descending.
        """
        if self._discovered and not force and time.time() - self._last_scan < 60:
            return self._options

        options: list[LLMOption] = []

        # 1. Claude API
        api_key = (self._config.get("claude_api_key")
                   or os.environ.get("ANTHROPIC_API_KEY",""))
        if api_key:
            opt = self._ping_claude(api_key)
            options.append(opt)

        # 2. Ollama — enumerate all installed models
        options.extend(self._discover_ollama())

        # 3. OpenAI-compatible endpoint
        oai_key  = (self._config.get("openai_api_key")
                    or os.environ.get("OPENAI_API_KEY",""))
        oai_host = self._config.get("openai_host","https://api.openai.com")
        if oai_key:
            options.append(self._ping_openai_compat(oai_host, oai_key))

        # 4. stdlib fallback — always available
        options.append(LLMOption(
            provider="stdlib", model="stdlib", endpoint="",
            available=True, capability=0,
            notes="No LLM — limited to Python stdlib operations only"))

        # Rank by capability then latency
        options.sort(key=lambda o: (-_rank(o), o.latency_ms if o.available else 9999))
        # Apply capability field
        for o in options:
            o.capability = _rank(o)

        self._options    = options
        self._discovered = True
        self._last_scan  = time.time()
        return options

    def best(self, min_capability: int = 1) -> Optional[LLMOption]:
        """Return the best available LLM meeting the minimum capability."""
        # Check preferred first
        if self._preferred:
            for opt in self.discover():
                if (self._preferred in f"{opt.provider}/{opt.model}"
                        and opt.available and opt.capability >= min_capability):
                    return opt

        # Then ranked list
        for opt in self.discover():
            if opt.available and opt.capability >= min_capability:
                return opt

        # Fallback chain from config
        for fb in self._fallback:
            for opt in self._options:
                if fb in f"{opt.provider}/{opt.model}" and opt.available:
                    return opt

        return None

    def call(
        self,
        prompt:               str,
        min_capability:       int = 1,
        max_tokens:           int = 1500,
        system:               str = "",
        json_mode:            bool = False,
        conversation_history: list[dict] = None,
        # list of {"role":"user"|"assistant","content":str}
    ) -> tuple[str, str]:
        """
        Call the best available LLM.
        conversation_history: pass recent turns so LLM has session context.
        Returns (response_text, model_used).
        Falls back through the chain automatically.
        """
        for opt in self.discover():
            if not opt.available or opt.capability < min_capability:
                continue
            try:
                text = self._call_option(
                    opt, prompt, max_tokens, system,
                    json_mode, conversation_history or [])
                if text:
                    logger.debug("LLM call via %s/%s", opt.provider, opt.model)
                    return text, f"{opt.provider}/{opt.model}"
            except Exception as e:
                logger.warning("LLM %s/%s failed: %s", opt.provider, opt.model, e)
                opt.available = False  # mark down until next discovery
                continue

        return "", "none"

    def status_summary(self) -> dict:
        """For /llm/status endpoint and sidebar display."""
        options = self.discover()
        best    = self.best()
        return {
            "best":      f"{best.provider}/{best.model}" if best else "none",
            "available": [{"provider":o.provider,"model":o.model,
                           "available":o.available,"capability":o.capability,
                           "latency_ms":round(o.latency_ms,1)}
                          for o in options if o.provider != "stdlib"],
            "stdlib_only": best is None or best.capability == 0,
        }

    # ── Discovery helpers ────────────────────────────────────────────────

    def _ping_claude(self, api_key: str) -> LLMOption:
        t = time.time()
        try:
            payload = json.dumps({"model":"claude-haiku-4-5-20251001",
                "max_tokens":1,
                "messages":[{"role":"user","content":"hi"}]}).encode()
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={"Content-Type":"application/json",
                         "anthropic-version":"2023-06-01",
                         "x-api-key":api_key})
            urllib.request.urlopen(req, timeout=3)
            return LLMOption("claude","claude-sonnet-4-20250514",
                "https://api.anthropic.com",True,
                (time.time()-t)*1000,3,"Claude API configured")
        except Exception as e:
            return LLMOption("claude","claude-sonnet-4-20250514",
                "https://api.anthropic.com",False,0,0,str(e)[:80])

    def _discover_ollama(self) -> list[LLMOption]:
        try:
            resp = urllib.request.urlopen(
                f"{self._ollama_host}/api/tags", timeout=2)
            data = json.loads(resp.read())
            opts = []
            for m in data.get("models",[]):
                name = m.get("name","")
                t = time.time()
                opts.append(LLMOption(
                    "ollama", name, self._ollama_host, True,
                    (time.time()-t)*1000, _rank(
                        LLMOption("ollama",name,"",True))))
            return opts if opts else [LLMOption(
                "ollama","none",self._ollama_host,False,0,0,"No models installed")]
        except Exception:
            return [LLMOption("ollama","none",self._ollama_host,
                              False,0,0,"Ollama not running")]

    def _ping_openai_compat(self, host: str, api_key: str) -> LLMOption:
        try:
            req = urllib.request.Request(f"{host}/v1/models",
                headers={"Authorization": "Bearer " + api_key})
            urllib.request.urlopen(req, timeout=2)
            return LLMOption("openai_compat","gpt-4",host,True,0,2)
        except Exception as e:
            return LLMOption("openai_compat","unknown",host,False,0,0,str(e)[:80])

    def _call_option(self, opt: LLMOption, prompt: str,
                     max_tokens: int, system: str,
                     json_mode: bool, history: list = None) -> str:
        if opt.provider == "claude":
            return self._call_claude(opt, prompt, max_tokens, system, json_mode, history)
        if opt.provider == "ollama":
            return self._call_ollama(opt, prompt, max_tokens, system, json_mode, history)
        if opt.provider == "openai_compat":
            return self._call_openai(opt, prompt, max_tokens, system, json_mode, history)
        return ""

    def _call_claude(self, opt: LLMOption, prompt: str,
                     max_tokens: int, system: str,
                     json_mode: bool, history: list = None) -> str:
        api_key = (self._config.get("claude_api_key")
                   or os.environ.get("ANTHROPIC_API_KEY",""))
        # Build messages: history + current prompt
        msgs = list(history or [])
        msgs.append({"role": "user", "content": prompt})
        body: dict = {"model":opt.model,"max_tokens":max_tokens,"messages":msgs}
        if system: body["system"] = system
        payload = json.dumps(body).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=payload,
            headers={"Content-Type":"application/json",
                     "anthropic-version":"2023-06-01",
                     "x-api-key":api_key})
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read())["content"][0]["text"]

    def _call_ollama(self, opt: LLMOption, prompt: str,
                     max_tokens: int, system: str,
                     json_mode: bool, history: list = None) -> str:
        # Prepend history as conversation context in the prompt
        if history:
            ctx = "\n".join(
                f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
                for m in history[-6:])  # last 3 turns
            full_prompt = f"Previous conversation:\n{ctx}\n\nUser: {prompt}"
        else:
            full_prompt = prompt
        body: dict = {"model":opt.model,"prompt":full_prompt,"stream":False}
        if json_mode: body["format"] = "json"
        payload = json.dumps(body).encode()
        req = urllib.request.Request(f"{opt.endpoint}/api/generate",
            data=payload,headers={"Content-Type":"application/json"})
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read()).get("response","")

    def _call_openai(self, opt: LLMOption, prompt: str,
                     max_tokens: int, system: str,
                     json_mode: bool, history: list = None) -> str:
        api_key = (self._config.get("openai_api_key")
                   or os.environ.get("OPENAI_API_KEY",""))
        msgs = []
        if system: msgs.append({"role":"system","content":system})
        msgs.extend(history or [])
        msgs.append({"role":"user","content":prompt})
        body: dict = {"model":"gpt-4o-mini","max_tokens":max_tokens,"messages":msgs}
        if json_mode: body["response_format"] = {"type":"json_object"}
        payload = json.dumps(body).encode()
        req = urllib.request.Request(f"{opt.endpoint}/v1/chat/completions",
            data=payload,
            headers={"Content-Type":"application/json",
                     "Authorization": "Bearer " + api_key})
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read())["choices"][0]["message"]["content"]
