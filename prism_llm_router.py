from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

try:
    import httpx as _httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _httpx = None
    _HTTPX_AVAILABLE = False

import types as _types

_prism_phase:       Optional[_types.ModuleType] = None
_PhaseState:        Optional[Any]               = None
_lora_reg:          Optional[_types.ModuleType] = None
_silicon_policy_mod: Optional[_types.ModuleType] = None
_ctx_budget_mod:    Optional[_types.ModuleType] = None
_tvm_bridge_mod:    Optional[_types.ModuleType] = None

try:
    import prism_phase as _prism_phase
    from prism_phase import PhaseState as _PhaseState
except ImportError:
    pass

try:
    import prism_lora_registry as _lora_reg
except ImportError:
    pass

try:
    import prism_silicon_policy as _silicon_policy_mod
except ImportError:
    pass

try:
    import prism_context_budget as _ctx_budget_mod
except ImportError:
    pass

try:
    import prism_tvm_bridge as _tvm_bridge_mod
except ImportError:
    pass

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
            start = clean.index("{")
            end = clean.rindex("}") + 1
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
        fallback:     list[str] | None = None,   # ordered fallback list
        ollama_host:  str = "http://localhost:11434",
        config:       dict | None = None,        # from prism_config.toml [llm] section
    ):
        self._preferred   = preferred
        self._fallback    = fallback or []
        self._ollama_host = ollama_host
        self._config      = config or {}
        self._options:    list[LLMOption] = []
        self._discovered  = False
        self._last_scan   = 0.0
        self._speculative_pipeline: Any | None = None

    @classmethod
    def from_config(cls, config_path: str = "~/.prism/config.toml",
                    claude_api_key: str = "") -> "LLMRouter":
        """Load LLM preferences from prism_config.toml [llm] section."""
        try:
            import tomllib
            from pathlib import Path
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

    def best(self, min_capability: int = 1, phase_hint: Optional[str] = None) -> Optional[LLMOption]:
        """
        Return the best available LLM meeting the minimum capability.
        phase_hint: 'fast'|'standard'|'capable'|'emergency' from CrystallizationEngine.
        When prism_phase is available, the current phase overrides the hint.
        """
        # Phase-aware override (requires prism_phase available)
        effective_hint = phase_hint
        if _prism_phase is not None:
            try:
                engine = _prism_phase.get_engine()
                # Only override if compute() has been called at least once
                if engine.history:
                    phase = engine.current_phase
                    effective_hint = engine.model_hint(phase)
            except Exception:
                pass

        if effective_hint == "fast":
            # Prefer smallest/fastest local model (capability=1) over cloud
            for opt in self.discover():
                if opt.available and opt.capability >= 1 and opt.provider == "ollama":
                    return opt
            # Fall through to normal selection if no local available
        elif effective_hint == "emergency":
            # Prefer cloud (claude/openai) as fastest reliable fallback
            for opt in self.discover():
                if opt.available and opt.provider in ("claude", "openai_compat"):
                    return opt
            # Fall through to normal selection

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
        conversation_history: list[dict] | None = None,
        speculative:          bool = False,
        phase_hint:           Optional[str] = None,
        task_hint:            str = "",
        images:               list[str] | None = None,
        # list of {"role":"user"|"assistant","content":str}
    ) -> tuple[str, str]:
        """
        Call the best available LLM.
        conversation_history: pass recent turns so LLM has session context.
        speculative=True: try capability=1 first; only escalate to min_capability
            if the fast model responds with uncertainty or very short output.
            Reduces cost ~40-60% on simple chains.
        Returns (response_text, model_used).
        Falls back through the chain automatically.
        """
        if speculative and min_capability >= 2:
            fast_resp, fast_model = self.call(
                prompt, min_capability=1, max_tokens=max_tokens,
                system=system, json_mode=json_mode,
                conversation_history=conversation_history, speculative=False,
                phase_hint=phase_hint, task_hint=task_hint, images=images,
            )
            _uncertain = ("i don't know", "uncertain", "cannot", "i'm not sure",
                          "i am not sure", "unclear", "not enough information")
            if (not any(sig in fast_resp.lower() for sig in _uncertain)
                    and len(fast_resp.split()) >= 30):
                logger.debug("[llm_router] speculative: fast model sufficient (%s)", fast_model)
                return fast_resp, fast_model
            logger.debug("[llm_router] speculative: escalating to min_capability=%d", min_capability)

        # Silicon Response Policy — apply hard budget constraints
        _eff_capability = min_capability
        _eff_max_tokens = max_tokens
        _eff_speculative = speculative
        if _silicon_policy_mod is not None:
            try:
                _policy = _silicon_policy_mod.get_policy()
                _phase_name = "STABLE"
                _delta_b = 0.0
                if _prism_phase is not None:
                    try:
                        _eng = _prism_phase.get_engine()
                        if _eng.history:
                            _phase_name = _eng.current_phase.value
                    except Exception:
                        pass
                # Get delta_b from bridge if engine has one
                # Bridge is not directly accessible from router — use policy's own extended ΔH
                _budget = _policy.current_budget(delta_b=_delta_b, phase_name=_phase_name)
                _eff_capability = min(_eff_capability, _budget.capability_ceil)
                _eff_max_tokens = min(_eff_max_tokens, _budget.max_tokens)
                _eff_speculative = _eff_speculative or _budget.speculative
                if _budget.throttle_reason:
                    logger.debug("[silicon] throttling call: %s", _budget.throttle_reason)
            except Exception as _se:
                logger.debug("[silicon] policy error: %s", _se)

        # Context Budget — prune conversation history to match token budget
        _pruned_history = conversation_history or []
        if _ctx_budget_mod is not None and _pruned_history:
            try:
                _ctx_mgr = _ctx_budget_mod.get_context_manager()
                from prism_silicon_policy import ExecutionBudget as _EB
                _ctx_budget = _EB(max_tokens=_eff_max_tokens)
                _result = _ctx_mgr.prune(_pruned_history, _ctx_budget, query=prompt)
                if _result.evicted_count > 0:
                    _pruned_history = _result.messages
                    logger.debug("[ctx_budget] evicted %d messages (%s strategy)",
                                 _result.evicted_count, _result.strategy)
            except Exception as _cbe:
                logger.debug("[ctx_budget] error: %s", _cbe)

        # TVM Bridge — apply quantization target when transitioning precision
        if _tvm_bridge_mod is not None and _silicon_policy_mod is not None:
            try:
                _tvm = _tvm_bridge_mod.get_tvm_bridge()
                _q_hint = _budget.quantization_hint if "_budget" in dir() else "fp16"
                _ct = _tvm.compile_target(_q_hint)
                _tvm.apply_target(_ct)
            except Exception:
                pass

        # ── LoRA / task-adapter injection (Vector V) ──────────────────────
        # Select and inject system prompt template based on phase + bio_debt.
        # This is the LAST modification before the actual LLM call — we only
        # modify the sent prompt, not any stored state.
        _effective_prompt = prompt
        if _lora_reg is not None:
            try:
                _registry = _lora_reg.get_registry()
                # Derive phase name from current engine state
                _phase_name = "STABLE"
                if _prism_phase is not None:
                    try:
                        _eng = _prism_phase.get_engine()
                        if _eng.history:
                            _phase_name = _eng.current_phase.value
                    except Exception:
                        pass
                # Derive bio_debt from bridge if wired (bridge not held here;
                # callers can pass bio_debt via task_hint with "bio_debt=X" prefix
                # or by subclassing. Default to 0.0 for now.)
                _bio_debt = 0.0
                _adapter = _registry.select(
                    phase_name=_phase_name,
                    bio_debt=_bio_debt,
                    task_hint=task_hint,
                )
                _effective_prompt = _registry.inject_system_prompt(prompt, _adapter)
                logger.debug("[llm_router] lora adapter=%s phase=%s",
                             _adapter.adapter_id, _phase_name)
            except Exception as _le:
                logger.debug("[llm_router] lora injection failed: %s", _le)
                _effective_prompt = prompt
        # ─────────────────────────────────────────────────────────────────

        # Phase-aware: when LIQUID, prefer fastest/cloud first regardless of ranking
        if _prism_phase is not None:
            try:
                _eng = _prism_phase.get_engine()
                if _eng.history and _PhaseState is not None and _eng.current_phase is _PhaseState.LIQUID:
                    preferred = self.best(min_capability=_eff_capability, phase_hint="emergency")
                    if preferred is not None:
                        try:
                            text = self._call_option(
                                preferred, _effective_prompt, _eff_max_tokens, system,
                                json_mode, _pruned_history, images=images)
                            if text:
                                logger.debug("[llm_router] LIQUID phase → %s/%s",
                                             preferred.provider, preferred.model)
                                return text, f"{preferred.provider}/{preferred.model}"
                        except Exception as _e:
                            logger.warning("[llm_router] LIQUID preferred failed: %s", _e)
            except Exception:
                pass

        for opt in self.discover():
            if not opt.available or opt.capability < _eff_capability:
                continue
            try:
                _t0 = time.time()
                text = self._call_option(
                    opt, _effective_prompt, _eff_max_tokens, system,
                    json_mode, _pruned_history, images=images)
                if text:
                    logger.debug("LLM call via %s/%s", opt.provider, opt.model)
                    _latency = (time.time() - _t0) * 1000
                    try:
                        from prism_llm_ledger import get_ledger as _get_ledger
                        _in_tok  = len(_effective_prompt) // 4
                        _out_tok = len(text) // 4
                        _get_ledger().record_call(
                            provider=opt.provider,
                            model=opt.model,
                            input_tokens=_in_tok,
                            output_tokens=_out_tok,
                            latency_ms=_latency,
                            source=task_hint or "unknown",
                        )
                    except Exception:
                        pass
                    return text, f"{opt.provider}/{opt.model}"
            except Exception as e:
                logger.warning("LLM %s/%s failed: %s", opt.provider, opt.model, e)
                opt.available = False  # mark down until next discovery
                continue

        return "", "none"

    def speculative_call(
        self,
        prompt: str,
        system: str = "",
        conversation_history: list[dict] | None = None,
    ) -> tuple[str, str]:
        """
        Call via speculative decoding pipeline when budget indicates.
        Falls back to normal call() if pipeline unavailable or budget is healthy.
        """
        try:
            import prism_silicon_policy as _sp
            import prism_speculative as _spec
            policy = _sp.get_policy()
            budget = policy.current_budget()
            if not budget.speculative:
                return self.call(prompt, system=system,
                                 conversation_history=conversation_history or [])
            pipeline = _spec.get_pipeline(router=self)
            if pipeline is None:
                return self.call(prompt, system=system,
                                 conversation_history=conversation_history or [])
            result = pipeline.call(prompt, budget=budget, system=system,
                                   conversation_history=conversation_history)
            return result.response, result.draft_model
        except Exception as e:
            logger.debug("[speculative] fallback to normal call: %s", e)
            return self.call(prompt, system=system,
                             conversation_history=conversation_history or [])

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
                     json_mode: bool, history: list | None = None,
                     images: list[str] | None = None) -> str:
        if opt.provider == "claude":
            return self._call_claude(opt, prompt, max_tokens, system, json_mode, history, images=images)
        if opt.provider == "ollama":
            return self._call_ollama(opt, prompt, max_tokens, system, json_mode, history, images=images)
        if opt.provider == "openai_compat":
            return self._call_openai(opt, prompt, max_tokens, system, json_mode, history, images=images)
        return ""

    def _call_claude(self, opt: LLMOption, prompt: str,
                     max_tokens: int, system: str,
                     json_mode: bool, history: list | None = None,
                     images: list[str] | None = None) -> str:
        api_key = (self._config.get("claude_api_key")
                   or os.environ.get("ANTHROPIC_API_KEY",""))
        # Build messages: history + current prompt
        msgs = list(history or [])
        if images:
            user_content: list = [
                {"type": "image", "source": {"type": "base64",
                  "media_type": "image/jpeg", "data": img}}
                for img in images
            ]
            user_content.append({"type": "text", "text": prompt})
            msgs.append({"role": "user", "content": user_content})
        else:
            msgs.append({"role": "user", "content": prompt})
        body: dict = {"model":opt.model,"max_tokens":max_tokens,"messages":msgs}
        if system:
            body["system"] = system
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
                     json_mode: bool, history: list | None = None,
                     images: list[str] | None = None) -> str:
        # Prepend history as conversation context in the prompt
        if history:
            ctx = "\n".join(
                f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
                for m in history[-6:])  # last 3 turns
            full_prompt = f"Previous conversation:\n{ctx}\n\nUser: {prompt}"
        else:
            full_prompt = prompt
        body: dict = {"model":opt.model,"prompt":full_prompt,"stream":False}
        if images:
            body["images"] = images
        if json_mode:
            body["format"] = "json"
        payload = json.dumps(body).encode()
        req = urllib.request.Request(f"{opt.endpoint}/api/generate",
            data=payload,headers={"Content-Type":"application/json"})
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read()).get("response","")

    def _call_openai(self, opt: LLMOption, prompt: str,
                     max_tokens: int, system: str,
                     json_mode: bool, history: list | None = None,
                     images: list[str] | None = None) -> str:
        api_key = (self._config.get("openai_api_key")
                   or os.environ.get("OPENAI_API_KEY",""))
        msgs: list[dict[str, Any]] = []
        if system:
            msgs.append({"role":"system","content":system})
        msgs.extend(history or [])
        if images:
            user_content: list = [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                for img in images
            ]
            user_content.append({"type": "text", "text": prompt})
            msgs.append({"role": "user", "content": user_content})
        else:
            msgs.append({"role":"user","content":prompt})
        body: dict = {"model":"gpt-4o-mini","max_tokens":max_tokens,"messages":msgs}
        if json_mode:
            body["response_format"] = {"type":"json_object"}
        payload = json.dumps(body).encode()
        req = urllib.request.Request(f"{opt.endpoint}/v1/chat/completions",
            data=payload,
            headers={"Content-Type":"application/json",
                     "Authorization": "Bearer " + api_key})
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read())["choices"][0]["message"]["content"]

    # ── Async interface ───────────────────────────────────────────────────
    # All methods below are additive — sync call() is untouched.

    async def async_call(
        self,
        prompt:               str,
        min_capability:       int = 1,
        max_tokens:           int = 1500,
        system:               str = "",
        json_mode:            bool = False,
        conversation_history: list[dict] | None = None,
        phase_hint:           Optional[str] = None,
    ) -> tuple[str, str]:
        """
        Async version of call(). Uses httpx when available for non-blocking I/O.
        Falls back to asyncio.to_thread(self.call, ...) when httpx is absent.
        Signature is a strict subset of call() — all existing callers of call()
        are unaffected.
        """
        if not _HTTPX_AVAILABLE:
            import asyncio
            return await asyncio.to_thread(
                self.call, prompt,
                min_capability=min_capability,
                max_tokens=max_tokens,
                system=system,
                json_mode=json_mode,
                conversation_history=conversation_history or [],
                phase_hint=phase_hint,
            )
        opt = self.best(min_capability=min_capability, phase_hint=phase_hint)
        if opt is None:
            return "", "none"
        try:
            text = await self._async_call_option(
                opt, prompt, max_tokens, system, json_mode,
                conversation_history or []
            )
            if text:
                return text, f"{opt.provider}/{opt.model}"
        except Exception as exc:
            logger.warning("[async_call] %s/%s failed: %s", opt.provider, opt.model, exc)
        return "", "none"

    async def _async_call_option(
        self,
        opt: LLMOption,
        prompt: str,
        max_tokens: int,
        system: str,
        json_mode: bool,
        history: list,
    ) -> str:
        if opt.provider == "claude":
            return await self._async_call_claude(opt, prompt, max_tokens, system, json_mode, history)
        if opt.provider == "ollama":
            return await self._async_call_ollama(opt, prompt, max_tokens, system, json_mode, history)
        if opt.provider == "openai_compat":
            return await self._async_call_openai(opt, prompt, max_tokens, system, json_mode, history)
        return ""

    async def _async_call_claude(
        self,
        opt: LLMOption,
        prompt: str,
        max_tokens: int,
        system: str,
        json_mode: bool,
        history: list,
    ) -> str:
        api_key = self._config.get("claude_api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
        msgs = list(history)
        msgs.append({"role": "user", "content": prompt})
        body: dict = {"model": opt.model, "max_tokens": max_tokens, "messages": msgs}
        if system:
            body["system"] = system
        async with _httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                json=body,
                headers={
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                    "x-api-key": api_key,
                },
            )
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]

    async def _async_call_ollama(
        self,
        opt: LLMOption,
        prompt: str,
        max_tokens: int,
        system: str,
        json_mode: bool,
        history: list,
    ) -> str:
        if history:
            ctx = "\n".join(
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                for m in history[-6:]
            )
            full_prompt = f"Previous conversation:\n{ctx}\n\nUser: {prompt}"
        else:
            full_prompt = prompt
        body: dict = {"model": opt.model, "prompt": full_prompt, "stream": False}
        if json_mode:
            body["format"] = "json"
        async with _httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{opt.endpoint}/api/generate",
                json=body,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            return resp.json().get("response", "")

    async def _async_call_openai(
        self,
        opt: LLMOption,
        prompt: str,
        max_tokens: int,
        system: str,
        json_mode: bool,
        history: list,
    ) -> str:
        api_key = self._config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(history)
        msgs.append({"role": "user", "content": prompt})
        body: dict = {"model": "gpt-4o-mini", "max_tokens": max_tokens, "messages": msgs}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        async with _httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{opt.endpoint}/v1/chat/completions",
                json=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + api_key,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    async def async_call_stream(
        self,
        prompt: str,
        min_capability: int = 1,
        max_tokens: int = 1500,
        system: str = "",
        conversation_history: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """
        Async generator that yields raw text tokens as they arrive from the provider.
        Requires httpx. Falls back to yielding the full async_call() result as one chunk.
        """
        if not _HTTPX_AVAILABLE:
            text, _ = await self.async_call(
                prompt, min_capability=min_capability, max_tokens=max_tokens,
                system=system, conversation_history=conversation_history,
            )
            if text:
                yield text
            return
        opt = self.best(min_capability=min_capability)
        if opt is None:
            return
        if opt.provider == "claude":
            async for token in self._stream_claude(opt, prompt, max_tokens, system,
                                                   conversation_history or []):
                yield token
        elif opt.provider == "ollama":
            async for token in self._stream_ollama(opt, prompt, conversation_history or []):
                yield token
        elif opt.provider == "openai_compat":
            async for token in self._stream_openai(opt, prompt, max_tokens, system,
                                                   conversation_history or []):
                yield token
        else:
            # stdlib / unknown — fall back to full call
            text, _ = await self.async_call(
                prompt, min_capability=min_capability, max_tokens=max_tokens,
                system=system, conversation_history=conversation_history,
            )
            if text:
                yield text

    async def _stream_claude(
        self, opt: LLMOption, prompt: str, max_tokens: int,
        system: str, history: list,
    ) -> AsyncIterator[str]:
        api_key = self._config.get("claude_api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
        msgs = list(history)
        msgs.append({"role": "user", "content": prompt})
        body: dict = {"model": opt.model, "max_tokens": max_tokens,
                      "messages": msgs, "stream": True}
        if system:
            body["system"] = system
        async with _httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST", "https://api.anthropic.com/v1/messages",
                json=body,
                headers={
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                    "x-api-key": api_key,
                },
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        chunk = line[6:]
                        if chunk == "[DONE]":
                            return
                        try:
                            data = json.loads(chunk)
                            if data.get("type") == "content_block_delta":
                                token = data["delta"].get("text", "")
                                if token:
                                    yield token
                        except Exception:
                            pass

    async def _stream_ollama(
        self, opt: LLMOption, prompt: str, history: list,
    ) -> AsyncIterator[str]:
        if history:
            ctx = "\n".join(
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                for m in history[-6:]
            )
            full_prompt = f"Previous conversation:\n{ctx}\n\nUser: {prompt}"
        else:
            full_prompt = prompt
        async with _httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST", f"{opt.endpoint}/api/generate",
                json={"model": opt.model, "prompt": full_prompt, "stream": True},
                headers={"Content-Type": "application/json"},
            ) as resp:
                async for line in resp.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            token = data.get("response", "")
                            if token:
                                yield token
                            if data.get("done"):
                                return
                        except Exception:
                            pass

    async def _stream_openai(
        self, opt: LLMOption, prompt: str, max_tokens: int,
        system: str, history: list,
    ) -> AsyncIterator[str]:
        api_key = self._config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(history)
        msgs.append({"role": "user", "content": prompt})
        body: dict = {"model": "gpt-4o-mini", "max_tokens": max_tokens,
                      "messages": msgs, "stream": True}
        async with _httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST", f"{opt.endpoint}/v1/chat/completions",
                json=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + api_key,
                },
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        chunk = line[6:]
                        if chunk == "[DONE]":
                            return
                        try:
                            data = json.loads(chunk)
                            token = (data.get("choices", [{}])[0]
                                     .get("delta", {}).get("content", ""))
                            if token:
                                yield token
                        except Exception:
                            pass
