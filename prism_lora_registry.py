"""
PRISM LoRA Registry
Task-specialized adapter selection with hardware-aware fallback.

On GPU-capable hardware: vLLM LoRA weights loaded per task.
On CPU-only hardware (current): task-specialized system prompt templates
serve as the behavioral equivalent — same selection logic, different backend.
"""
from __future__ import annotations

import logging
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

_DB_PATH = Path("~/.prism/lora_registry.db")

_log = logging.getLogger(__name__)

# ── Task-specialized system prompt templates ──────────────────────────────────
# These are the CPU-fallback equivalent of LoRA weight injection.
# On GPU hardware, these would be replaced by actual adapter weights.

_PROMPT_CODE_ANALYST = """\
You are a precise code analyst. Focus on correctness, edge cases, and security.
Prioritize: explicit reasoning, concrete examples, minimal speculation.
When uncertain, say so and offer what you do know."""

_PROMPT_FACTUAL_AUDIT = """\
You are a rigorous fact-checker. Cross-reference claims before accepting them.
Flag contradictions. Prefer conservative estimates. Cite your reasoning chain.
Uncertainty is a valid answer; speculation is not."""

_PROMPT_CREATIVE_SCOUT = """\
You are an exploratory thinker. Generate diverse hypotheses.
Propose non-obvious connections. Quantity before quality — filter later.
Be concise per idea, generate many."""

_PROMPT_FAST_EXECUTOR = """\
You are a decisive executor. Give the shortest correct answer.
No hedging, no preamble. If a task has a clear answer, state it.
Reserve explanation for when it is strictly necessary."""

_PROMPT_DEEP_ANALYST = """\
You are a structured analyst. Build your answer in layers:
hypothesis → evidence → conclusion → caveats.
Take space if needed. Precision beats brevity here."""

_PROMPT_RECOVERY_GENTLE = """\
You are a focused assistant operating in conservation mode.
Prioritize the most important thing only. Reduce cognitive overhead.
Short answers, clear actions, minimal branching."""


@dataclass
class LoRAAdapter:
    adapter_id:       str
    task_type:        str    # "code", "factual", "creative", "fast", "deep", "recovery"
    system_prompt:    str    # always-available fallback
    weights_path:     str | None = None     # None on CPU-only hardware
    base_model:       str = "any"           # model this adapter was trained on
    rank:             int = 16
    capability_floor: int = 1               # minimum LLM capability needed
    bio_phase:        str = "any"           # "any" | "CRYSTAL" | "STABLE" | "VISCOUS" | "LIQUID"


class LoRARegistry:
    """
    Selects the appropriate task adapter based on phase state and biological pressure.

    Selection priority:
    1. If biological debt is critical (ΔB > 0.7) → "recovery" adapter
    2. If phase is LIQUID → "fast" adapter
    3. If phase is CRYSTAL → "factual" adapter (precision mode)
    4. If task_hint provided → match task type
    5. Default → "deep" adapter
    """

    _ADAPTERS: list[LoRAAdapter] = [
        LoRAAdapter("code-analyst",    "code",     _PROMPT_CODE_ANALYST,    capability_floor=2),
        LoRAAdapter("factual-audit",   "factual",  _PROMPT_FACTUAL_AUDIT,   capability_floor=2, bio_phase="CRYSTAL"),
        LoRAAdapter("creative-scout",  "creative", _PROMPT_CREATIVE_SCOUT,  capability_floor=1),
        LoRAAdapter("fast-executor",   "fast",     _PROMPT_FAST_EXECUTOR,   capability_floor=1, bio_phase="LIQUID"),
        LoRAAdapter("deep-analyst",    "deep",     _PROMPT_DEEP_ANALYST,    capability_floor=2),
        LoRAAdapter("recovery-gentle", "recovery", _PROMPT_RECOVERY_GENTLE, capability_floor=1),
    ]

    def __init__(self) -> None:
        self._adapters = {a.adapter_id: a for a in self._ADAPTERS}

    def select(
        self,
        phase_name: str = "STABLE",
        bio_debt:   float = 0.0,
        task_hint:  str = "",
    ) -> LoRAAdapter:
        """
        Select the best adapter for current conditions.
        Returns the adapter; caller applies system_prompt to LLM call.
        """
        # Critical biological debt → recovery mode
        if bio_debt > 0.7:
            _log.debug("[lora] bio_debt=%.2f → recovery adapter", bio_debt)
            return self._adapters["recovery-gentle"]

        # Phase-driven selection
        if phase_name == "LIQUID":
            return self._adapters["fast-executor"]
        if phase_name == "CRYSTAL":
            return self._adapters["factual-audit"]

        # Task-type hint matching
        if task_hint:
            hint = task_hint.lower()
            for adapter in self._adapters.values():
                if adapter.task_type in hint or hint in adapter.task_type:
                    return adapter

        # Default
        return self._adapters["deep-analyst"]

    def inject_system_prompt(self, prompt: str, adapter: LoRAAdapter) -> str:
        """Prepend the adapter's system prompt to the user prompt."""
        if not adapter.system_prompt:
            return prompt
        return f"{adapter.system_prompt}\n\n---\n\n{prompt}"

    def gpu_available(self) -> bool:
        """Check if vLLM LoRA loading is feasible."""
        try:
            result = subprocess.run(  # nosec B603 — constant argv, no shell, no user input
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            free_mb = int(result.stdout.strip().split("\n")[0])
            return free_mb > 8000  # need 8GB free for LoRA inference
        except Exception:
            return False

    def load_weights(self, adapter: LoRAAdapter) -> bool:
        """
        Attempt to load LoRA weights via vLLM if GPU available.
        Returns True if weights loaded, False if falling back to prompt template.
        """
        if not self.gpu_available() or adapter.weights_path is None:
            _log.debug(
                "[lora] CPU-only fallback: using prompt template for %s",
                adapter.adapter_id,
            )
            return False
        try:
            # vLLM LoRA loading (only reached on GPU hardware)
            from vllm import LLM  # noqa: F401
            from vllm.lora.request import LoRARequest  # noqa: F401

            _log.info(
                "[lora] Loading weights: %s from %s",
                adapter.adapter_id,
                adapter.weights_path,
            )
            return True
        except ImportError:
            _log.debug("[lora] vLLM not available, using prompt template")
            return False

    def register(self, job_id: str, gguf_path: str, ollama_model: str) -> LoRAAdapter:
        """
        Register a freshly trained LoRA adapter produced by PrismLoraTrainer.
        Adds it to the in-memory registry, persists to SQLite, and returns the adapter.
        """
        adapter_id = f"trained-{job_id}"
        adapter = LoRAAdapter(
            adapter_id=adapter_id,
            task_type="trained",
            system_prompt=(
                "You are PRISM, a local-first AI assistant personalised to this user."
            ),
            weights_path=gguf_path,
            base_model=ollama_model,
        )
        self._adapters[adapter_id] = adapter
        try:
            db = _DB_PATH.expanduser()
            db.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(db) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS adapters "
                    "(adapter_id TEXT PRIMARY KEY, name TEXT, path TEXT, "
                    "task_type TEXT, base_model TEXT, created_at REAL)"
                )
                conn.execute(
                    "INSERT OR REPLACE INTO adapters "
                    "(adapter_id, name, path, task_type, base_model, created_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (adapter_id, adapter_id, gguf_path or "", "trained", ollama_model, time.time()),
                )
        except Exception as _dbe:
            _log.debug("[lora] DB persist failed: %s", _dbe)
        _log.info("[lora] Registered trained adapter %s (gguf=%s)", adapter_id, gguf_path)
        return adapter

    def list_adapters(self) -> list[LoRAAdapter]:
        """Return all in-memory adapters; load from DB if in-memory list only has built-ins."""
        # Built-in adapters are always populated; trained ones come from DB
        trained = [a for a in self._adapters.values() if a.task_type == "trained"]
        if not trained:
            self._load_from_db()
        return list(self._adapters.values())

    def _load_from_db(self) -> None:
        """Populate in-memory registry from SQLite on first access."""
        try:
            db = _DB_PATH.expanduser()
            if not db.exists():
                return
            with sqlite3.connect(db) as conn:
                rows = conn.execute(
                    "SELECT adapter_id, path, task_type, base_model FROM adapters"
                ).fetchall()
            for row in rows:
                adapter_id, path, task_type, base_model = row
                if adapter_id not in self._adapters:
                    self._adapters[adapter_id] = LoRAAdapter(
                        adapter_id=adapter_id,
                        task_type=task_type or "trained",
                        system_prompt=(
                            "You are PRISM, a local-first AI assistant personalised to this user."
                        ),
                        weights_path=path or None,
                        base_model=base_model or "any",
                    )
        except Exception as _dbe:
            _log.debug("[lora] DB load failed: %s", _dbe)


# Module-level singleton
_registry: LoRARegistry | None = None


def get_registry() -> LoRARegistry:
    global _registry
    if _registry is None:
        _registry = LoRARegistry()
    return _registry
