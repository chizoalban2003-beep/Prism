from __future__ import annotations
import json
import logging
import os
import time
import urllib.request
from dataclasses import dataclass, field
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


# Cost per 1K tokens (input, output) in USD — update as pricing changes
PROVIDER_COSTS: dict[str, tuple[float, float]] = {
    # provider_id: (cost_per_1k_input, cost_per_1k_output)
    "claude-haiku-4-5-20251001":          (0.0008,  0.004),
    "claude-sonnet-4-20250514":           (0.003,   0.015),
    "claude-opus-4-20250514":             (0.015,   0.075),
    "gpt-4o-mini":                        (0.00015, 0.0006),
    "gpt-4o":                             (0.005,   0.015),
    "gpt-4-turbo":                        (0.01,    0.03),
    "gemini-1.5-flash":                   (0.00035, 0.00105),
    "gemini-1.5-pro":                     (0.0035,  0.0105),
    "deepseek-chat":                      (0.00014, 0.00028),
    "deepseek-reasoner":                  (0.00055, 0.00219),
    "mistral-small-latest":               (0.0002,  0.0006),
    "mistral-large-latest":               (0.002,   0.006),
    "llama-3.1-70b-versatile":            (0.00059, 0.00079),  # Groq
    "llama-3.1-8b-instant":               (0.00005, 0.00008),  # Groq
    "ollama/any":                         (0.0,     0.0),      # free local
}

# All supported providers with discovery logic
# (provider_id, env_key, base_url, default_model, capability)
PROVIDER_CATALOGUE = [
    ("anthropic",  "ANTHROPIC_API_KEY",  "https://api.anthropic.com",                  "claude-haiku-4-5-20251001", 3),
    ("openai",     "OPENAI_API_KEY",     "https://api.openai.com",                      "gpt-4o-mini",               3),
    ("google",     "GOOGLE_API_KEY",     "https://generativelanguage.googleapis.com",   "gemini-1.5-flash",          3),
    ("deepseek",   "DEEPSEEK_API_KEY",   "https://api.deepseek.com",                    "deepseek-chat",             3),
    ("mistral",    "MISTRAL_API_KEY",    "https://api.mistral.ai",                      "mistral-small-latest",      2),
    ("groq",       "GROQ_API_KEY",       "https://api.groq.com/openai",                 "llama-3.1-8b-instant",      2),
    ("together",   "TOGETHER_API_KEY",   "https://api.together.xyz",                    "mistralai/Mistral-7B-v0.1", 2),
    ("ollama",     "",                   "http://localhost:11434",                       "",                          1),
    ("custom",     "CUSTOM_LLM_API_KEY", "",                                             "",                          1),
]


@dataclass
class LLMOption:
    """One available LLM."""
    provider:        str          # e.g. "anthropic"|"openai"|"ollama"|"stdlib"
    model:           str          # e.g. "claude-sonnet-4-20250514"|"mistral"|"gpt-4o-mini"
    endpoint:        str          # base URL
    available:       bool
    latency_ms:      float = 0.0  # measured on last ping
    capability:      int   = 0    # 0=none 1=basic 2=good 3=best — used for ranking
    notes:           str   = ""
    cost_per_1k_in:  float = 0.0  # USD per 1K input tokens
    cost_per_1k_out: float = 0.0  # USD per 1K output tokens
    monthly_budget:  float = 0.0  # set from policy engine (0 = unlimited)
    monthly_spent:   float = 0.0  # tracked from policy engine

    def estimated_cost(self, approx_tokens: int = 500) -> float:
        """Estimated cost in USD for a typical query."""
        return (approx_tokens * self.cost_per_1k_in / 1000 +
                approx_tokens * self.cost_per_1k_out / 1000)

    def budget_remaining(self) -> float:
        return max(0.0, self.monthly_budget - self.monthly_spent)

    def within_budget(self, approx_tokens: int = 500) -> bool:
        if self.monthly_budget == 0:
            return True   # no budget set = unlimited
        return self.budget_remaining() >= self.estimated_cost(approx_tokens)


# Capability ranking — higher = preferred for complex tasks
MODEL_CAPABILITY: dict[str, int] = {
    "claude-sonnet": 3, "claude-opus": 3,
    "deepseek-r1": 3, "deepseek-v3": 2,
    "llama3": 2, "llama3.1": 2, "llama3.2": 2,
    "mistral": 2, "mixtral": 2,
    "qwen": 1, "phi": 1, "gemma": 1,
    "stdlib": 0,
}

# Map provider id → env variable name for API key lookup
PROVIDER_ENV_KEYS: dict[str, str] = {
    "openai":       "OPENAI_API_KEY",
    "deepseek":     "DEEPSEEK_API_KEY",
    "mistral":      "MISTRAL_API_KEY",
    "groq":         "GROQ_API_KEY",
    "together":     "TOGETHER_API_KEY",
    "openai_compat":"CUSTOM_LLM_API_KEY",
}

# Multiplier applied to word-split token count to approximate real token usage.
# LLMs use subword (BPE/WordPiece) tokenization; words average ~1.3 tokens.
_TOKEN_WORD_RATIO = 1.3

def _rank(option: LLMOption) -> int:
    for key, cap in MODEL_CAPABILITY.items():
        if key in option.model.lower():
            return cap
    return 1 if option.available else 0

class LLMRouter:
    """
    Discovers all available LLMs, ranks them, and routes requests
    to the best available option with automatic fallback.

    Discovery checks all providers in PROVIDER_CATALOGUE (non-blocking, 3s
    timeout each). Providers are only contacted when an API key is configured.
    stdlib fallback is always available.
    """

    def __init__(
        self,
        preferred:    str = "",           # e.g. "ollama/mistral" or "anthropic"
        fallback:     list[str] = None,   # ordered fallback list
        ollama_host:  str = "http://localhost:11434",
        config:       dict = None,        # from prism_config.toml [llm] section
    ):
        self._preferred    = preferred
        self._fallback     = fallback or []
        self._ollama_host  = ollama_host
        self._config       = config or {}
        self._options:     list[LLMOption] = []
        self._discovered   = False
        self._last_scan    = 0.0
        self._policy_engine = None  # injected externally if spend tracking needed

    @classmethod
    def from_config(cls, config_path: str = "~/.prism/config.toml") -> "LLMRouter":
        """Load LLM preferences from prism_config.toml [llm] section."""
        try:
            from pathlib import Path
            import tomllib
            path = Path(config_path).expanduser()
            if path.exists():
                data = tomllib.loads(path.read_text())
                llm  = data.get("llm", {})
                return cls(
                    preferred   = llm.get("preferred", ""),
                    fallback    = llm.get("fallback", []),
                    ollama_host = llm.get("ollama_host","http://localhost:11434"),
                    config      = llm,
                )
        except Exception:
            pass
        return cls()

    def discover(self, force: bool = False) -> list[LLMOption]:
        """
        Discover all available LLMs. Cached for 60 seconds.
        Returns list sorted by capability descending, then cost ascending.
        """
        if self._discovered and not force and time.time() - self._last_scan < 60:
            return self._options

        options: list[LLMOption] = []

        for pid, env_key, default_url, default_model, cap in PROVIDER_CATALOGUE:
            # Resolve API key, base URL, and model from config or env/defaults
            api_key  = (self._config.get(f"{pid}_api_key")
                        or os.environ.get(env_key, ""))
            base_url = (self._config.get(f"{pid}_host") or default_url)
            model    = (self._config.get(f"{pid}_model") or default_model)

            if pid == "ollama":
                options.extend(self._discover_ollama(base_url))
                continue

            if pid == "custom":
                if api_key and base_url:
                    options.append(self._ping_openai_compat_full(
                        base_url, api_key,
                        self._config.get("custom_model", "gpt-3.5-turbo"), 1))
                continue

            if not api_key:
                continue   # skip unconfigured providers

            opt = self._ping_provider(pid, base_url, api_key, model, cap)
            if opt.available:
                costs = PROVIDER_COSTS.get(model, (0.001, 0.002))
                opt.cost_per_1k_in  = costs[0]
                opt.cost_per_1k_out = costs[1]
            options.append(opt)

        # stdlib fallback — always available, always last
        options.append(LLMOption("stdlib", "stdlib", "",
                                 True, 0, 0, "No LLM — stdlib only"))

        options.sort(key=lambda o: (-o.capability, o.cost_per_1k_in))
        # Apply capability field from MODEL_CAPABILITY table for Ollama models
        for o in options:
            if o.capability == 0 and o.provider == "ollama":
                o.capability = _rank(o)

        self._options    = options
        self._discovered = True
        self._last_scan  = time.time()
        return options

    def best(
        self,
        min_capability:  int   = 1,
        max_cost_per_q:  float = None,    # USD ceiling per query
        preferred:       str   = None,    # override: "anthropic/claude-sonnet-4-20250514"
        task_complexity: str   = "medium", # "simple"|"medium"|"complex"
    ) -> Optional[LLMOption]:
        """
        Return best available LLM within capability, cost, and budget constraints.

        task_complexity routing:
          simple  → prefer cheap fast models (Haiku, Flash, Ollama)
          medium  → prefer balanced models (Sonnet, GPT-4o-mini, Gemini Flash)
          complex → prefer strongest models within budget (Sonnet, GPT-4o, Gemini Pro)
        """
        complexity_cap = {"simple": 1, "medium": 2, "complex": 3}
        min_cap = max(min_capability, complexity_cap.get(task_complexity, 1))

        # Explicit preference (caller arg or instance default)
        pref = preferred or self._preferred
        if pref:
            for opt in self.discover():
                if (pref in f"{opt.provider}/{opt.model}"
                        and opt.available and opt.capability >= min_cap
                        and opt.within_budget()):
                    return opt

        # Budget-constrained selection
        for opt in self.discover():
            if not opt.available:
                continue
            if opt.capability < min_cap:
                continue
            if not opt.within_budget():
                continue
            if max_cost_per_q and opt.estimated_cost() > max_cost_per_q:
                continue
            return opt

        # Fallback chain from config
        for fb in self._fallback:
            for opt in self._options:
                if fb in f"{opt.provider}/{opt.model}" and opt.available:
                    return opt

        # Last resort: any available option above floor
        for opt in self.discover():
            if opt.available and opt.capability >= 1:
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
        task_complexity:      str = "medium",
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
            if not opt.within_budget():
                logger.info("Skipping %s/%s — budget exhausted",
                            opt.provider, opt.model)
                continue
            try:
                text = self._call_option(
                    opt, prompt, max_tokens, system,
                    json_mode, conversation_history or [])
                if text:
                    # Approximate token count using word count with correction factor
                    # (subword tokenizers average ~1.3 tokens per whitespace-split word)
                    approx_tokens = int(
                        (len(prompt.split()) + len(text.split())) * _TOKEN_WORD_RATIO
                    )
                    spend = (approx_tokens * opt.cost_per_1k_in / 1000 +
                             approx_tokens * opt.cost_per_1k_out / 1000)
                    self._record_spend(opt.provider, opt.model, spend)
                    logger.debug("LLM call via %s/%s", opt.provider, opt.model)
                    return text, f"{opt.provider}/{opt.model}"
            except Exception as e:
                logger.warning("LLM %s/%s failed: %s", opt.provider, opt.model, e)
                opt.available = False  # mark down until next discovery
                continue

        return "", "none"

    def _record_spend(self, provider: str, model: str, usd: float) -> None:
        """Record spend — notify policy engine if configured."""
        if self._policy_engine is not None:
            self._policy_engine._log_spend(
                "default", "llm", f"{provider}/{model}", usd, True)

    def set_preferred(self, provider_model: str) -> None:
        """Set the preferred provider/model for future calls."""
        self._preferred = provider_model
        self._config["preferred"] = provider_model

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

    def _ping_provider(self, pid: str, url: str, api_key: str,
                       model: str, cap: int) -> LLMOption:
        """Ping a provider with a minimal request to verify availability."""
        try:
            if pid == "anthropic":
                return self._ping_claude(api_key)
            elif pid == "google":
                return self._ping_gemini(url, api_key, model)
            else:
                return self._ping_openai_compat_full(url, api_key, model, cap)
        except Exception as e:
            return LLMOption(pid, model, url, False, 0, 0, str(e)[:80])

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
            return LLMOption("anthropic","claude-sonnet-4-20250514",
                "https://api.anthropic.com",True,
                (time.time()-t)*1000,3,"Claude API configured")
        except Exception as e:
            return LLMOption("anthropic","claude-sonnet-4-20250514",
                "https://api.anthropic.com",False,0,0,str(e)[:80])

    def _ping_gemini(self, url: str, api_key: str, model: str) -> LLMOption:
        t = time.time()
        try:
            req_url = (f"{url}/v1beta/models/{model}:generateContent"
                       f"?key={api_key}")
            payload = json.dumps({
                "contents": [{"parts": [{"text": "hi"}]}],
                "generationConfig": {"maxOutputTokens": 1}
            }).encode()
            req = urllib.request.Request(req_url, data=payload,
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=3)
            return LLMOption("google", model, url, True,
                             (time.time()-t)*1000, 3)
        except Exception as e:
            return LLMOption("google", model, url, False, 0, 0, str(e)[:80])

    def _ping_openai_compat_full(self, url: str, api_key: str,
                                 model: str, cap: int) -> LLMOption:
        t = time.time()
        try:
            req = urllib.request.Request(f"{url}/v1/models",
                headers={"Authorization": "Bearer " + api_key,
                         "Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=3)
            # Infer provider from URL for correct call routing
            pid = self._pid_from_url(url)
            return LLMOption(pid, model, url, True,
                             (time.time()-t)*1000, cap)
        except Exception as e:
            pid = self._pid_from_url(url)
            return LLMOption(pid, model, url, False, 0, 0, str(e)[:80])

    def _pid_from_url(self, url: str) -> str:
        """Derive a provider id string from a base URL using hostname matching."""
        from urllib.parse import urlparse
        try:
            host = urlparse(url).hostname or ""
        except Exception:
            host = ""
        if host == "api.openai.com" or host.endswith(".openai.com"):
            return "openai"
        if host == "api.deepseek.com" or host.endswith(".deepseek.com"):
            return "deepseek"
        if host == "api.mistral.ai" or host.endswith(".mistral.ai"):
            return "mistral"
        if host == "api.groq.com" or host.endswith(".groq.com"):
            return "groq"
        if host == "api.together.xyz" or host.endswith(".together.xyz"):
            return "together"
        return "openai_compat"

    def _discover_ollama(self, host: str = "") -> list[LLMOption]:
        ollama_host = host or self._ollama_host
        try:
            resp = urllib.request.urlopen(
                f"{ollama_host}/api/tags", timeout=2)
            data = json.loads(resp.read())
            opts = []
            for m in data.get("models",[]):
                name = m.get("name","")
                t = time.time()
                opts.append(LLMOption(
                    "ollama", name, ollama_host, True,
                    (time.time()-t)*1000, _rank(
                        LLMOption("ollama",name,"",True))))
            return opts if opts else [LLMOption(
                "ollama","none",ollama_host,False,0,0,"No models installed")]
        except Exception:
            return [LLMOption("ollama","none",ollama_host,
                              False,0,0,"Ollama not running")]

    def _ping_openai_compat(self, host: str, api_key: str) -> LLMOption:
        """Legacy helper kept for backward compatibility."""
        try:
            req = urllib.request.Request(f"{host}/v1/models",
                headers={"Authorization": "Bearer " + api_key})
            urllib.request.urlopen(req, timeout=2)
            return LLMOption("openai_compat","gpt-4",host,True,0,2)
        except Exception as e:
            return LLMOption("openai_compat","unknown",host,False,0,0,str(e)[:80])

    # ── Call helpers ─────────────────────────────────────────────────────

    def _call_option(self, opt: LLMOption, prompt: str,
                     max_tokens: int, system: str,
                     json_mode: bool, history: list = None) -> str:
        if opt.provider == "anthropic":
            return self._call_claude(opt, prompt, max_tokens, system, json_mode, history)
        if opt.provider == "ollama":
            return self._call_ollama(opt, prompt, max_tokens, system, json_mode, history)
        if opt.provider == "google":
            return self._call_gemini(opt, prompt, max_tokens, system, json_mode, history)
        # All others: OpenAI-compatible (openai, deepseek, mistral, groq, together, openai_compat)
        return self._call_openai_compat_full(opt, prompt, max_tokens, system, json_mode, history)

    def _call_claude(self, opt: LLMOption, prompt: str,
                     max_tokens: int, system: str,
                     json_mode: bool, history: list = None) -> str:
        api_key = (self._config.get("anthropic_api_key")
                   or self._config.get("claude_api_key")
                   or os.environ.get("ANTHROPIC_API_KEY",""))
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
        if history:
            ctx = "\n".join(
                f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
                for m in history[-6:])
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

    def _call_gemini(self, opt: LLMOption, prompt: str,
                     max_tokens: int, system: str,
                     json_mode: bool, history: list = None) -> str:
        api_key = (self._config.get("google_api_key")
                   or os.environ.get("GOOGLE_API_KEY", ""))
        req_url = (f"{opt.endpoint}/v1beta/models/{opt.model}:generateContent"
                   f"?key={api_key}")
        contents = []
        for h in (history or []):
            role = "user" if h["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": h["content"]}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        body: dict = {"contents": contents,
                      "generationConfig": {"maxOutputTokens": max_tokens}}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        payload = json.dumps(body).encode()
        req = urllib.request.Request(req_url, data=payload,
            headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read())
        return (data.get("candidates", [{}])[0]
                .get("content", {}).get("parts", [{}])[0]
                .get("text", ""))

    def _call_openai_compat_full(self, opt: LLMOption, prompt: str,
                                  max_tokens: int, system: str,
                                  json_mode: bool, history: list = None) -> str:
        provider = opt.provider
        api_key = (self._config.get(f"{provider}_api_key")
                   or os.environ.get(PROVIDER_ENV_KEYS.get(provider, ""), ""))
        msgs = []
        if system: msgs.append({"role": "system", "content": system})
        for h in (history or []):
            msgs.append({"role": h["role"], "content": h["content"]})
        msgs.append({"role": "user", "content": prompt})
        body: dict = {"model": opt.model, "max_tokens": max_tokens, "messages": msgs}
        if json_mode: body["response_format"] = {"type": "json_object"}
        payload = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{opt.endpoint}/v1/chat/completions", data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + api_key})
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read())["choices"][0]["message"]["content"]

    def _call_openai(self, opt: LLMOption, prompt: str,
                     max_tokens: int, system: str,
                     json_mode: bool, history: list = None) -> str:
        """Legacy helper kept for backward compatibility."""
        return self._call_openai_compat_full(opt, prompt, max_tokens,
                                              system, json_mode, history)
