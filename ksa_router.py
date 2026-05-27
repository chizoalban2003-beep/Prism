"""
ksa_router.py
=============
Kinetic State Agent — Master Fulcrum (Router)

The entry point of the KSA system. Parses a user prompt or OS event string,
resolves it to a registered task name, then loads and returns the best-known
ThreeBarSystem for that task from the SnapshotRegistry.

Resolution strategy (fast-path first):
    1. Keyword scoring  — zero-LLM, sub-millisecond, deterministic
    2. LLM resolver     — optional Ollama slot, only called when keyword
                          confidence is below threshold
    3. Bootstrap        — if no match at all, create a default system,
                          save it to the registry, and return it

Usage:
    router = MasterFulcrum(registry)
    router.register_intent(
        task_name = "file_index_stealth",
        keywords  = ["index", "scan", "files", "directory", "stealth", "background"],
    )

    result = router.route("quietly index my project folder")
    print(result.task_name)       # "file_index_stealth"
    print(result.method)          # "keyword"
    result.system.simulate()
"""

from __future__ import annotations

import re
import sys
import os
import time
import json
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(__file__))
from ksa_lever import ThreeBarSystem
from ksa_registry import SnapshotRegistry, PerformanceMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IntentPattern:
    """
    A registered mapping from a task name to a set of keywords and
    an optional pre-configured default ThreeBarSystem to bootstrap with.
    """
    task_name:      str
    keywords:       list[str]           # matched case-insensitively
    aliases:        list[str] = field(default_factory=list)  # exact task-name aliases
    default_system: Optional[ThreeBarSystem] = None
    description:    str = ""


@dataclass
class RouteResult:
    """
    The fully resolved output of a route() call.
    Contains everything needed to start execution.
    """
    task_name:   str
    system:      ThreeBarSystem
    version:     int
    confidence:  float              # 0.0–1.0 keyword match ratio
    method:      str                # "keyword" | "llm" | "bootstrap" | "alias"
    elapsed_ms:  float
    prompt_raw:  str

    def __str__(self) -> str:
        return (
            f"RouteResult("
            f"task='{self.task_name}', "
            f"v{self.version}, "
            f"method={self.method}, "
            f"conf={self.confidence:.0%}, "
            f"{self.elapsed_ms:.1f}ms)"
        )


# ---------------------------------------------------------------------------
# MasterFulcrum
# ---------------------------------------------------------------------------

class MasterFulcrum:
    """
    The top-level lever of the KSA cascade.

    Receives raw intent strings and returns a ready-to-simulate ThreeBarSystem
    loaded with the best snapshot for the resolved task. Acts as the single
    choke-point between the outside world and the physics engine.

    Keyword scoring:
        Each registered keyword that appears in the normalised prompt adds
        1 point. The task with the most points wins. Ties are broken by
        pattern registration order (first registered wins). Confidence is
        (matched_keywords / total_keywords_in_pattern).

    LLM slot (optional):
        Set llm_resolver to any callable that accepts a prompt str and
        returns a task_name str (or None). The built-in Ollama helper is
        provided as MasterFulcrum.ollama_resolver(model, host).

    Bootstrap:
        If no pattern matches and no LLM result, a default ThreeBarSystem
        is created for the inferred task name (snake_cased prompt), saved
        to the registry, and returned. Confidence = 0.0, method = "bootstrap".
    """

    # Keyword confidence must meet this to skip the LLM slot
    KEYWORD_CONFIDENCE_THRESHOLD: float = 0.25

    def __init__(
        self,
        registry:        SnapshotRegistry,
        llm_resolver:    Optional[Callable[[str], Optional[str]]] = None,
        confidence_floor: float = KEYWORD_CONFIDENCE_THRESHOLD,
    ):
        self.registry          = registry
        self.llm_resolver      = llm_resolver
        self.confidence_floor  = confidence_floor
        self._patterns:  list[IntentPattern] = []
        self._alias_map: dict[str, str]      = {}  # alias → canonical task_name

    # ── Pattern registration ─────────────────────────────────────────────────

    def register_intent(
        self,
        task_name:      str,
        keywords:       list[str],
        aliases:        Optional[list[str]]       = None,
        default_system: Optional[ThreeBarSystem]  = None,
        description:    str                       = "",
    ) -> None:
        """
        Register a task name with its intent keywords.

        Args:
            task_name:      Canonical name used as the registry key.
            keywords:       Words/phrases that signal this task in a prompt.
                            Matched case-insensitively, whole-word preferred.
            aliases:        Exact synonyms for task_name (e.g. short names).
            default_system: ThreeBarSystem to save if no snapshot exists yet.
                            If None, ThreeBarSystem.from_defaults() is used.
            description:    Human-readable purpose of this task.
        """
        pattern = IntentPattern(
            task_name      = task_name,
            keywords       = [k.lower() for k in keywords],
            aliases        = [a.lower() for a in (aliases or [])],
            default_system = default_system,
            description    = description,
        )
        self._patterns.append(pattern)
        for alias in pattern.aliases:
            self._alias_map[alias] = task_name
        logger.debug("Registered intent: %s (%d keywords)", task_name, len(keywords))

    def unregister_intent(self, task_name: str) -> bool:
        """Remove a registered intent pattern. Returns True if found."""
        before = len(self._patterns)
        self._patterns = [p for p in self._patterns if p.task_name != task_name]
        self._alias_map = {k: v for k, v in self._alias_map.items() if v != task_name}
        return len(self._patterns) < before

    # ── Routing ──────────────────────────────────────────────────────────────

    def route(self, prompt: str) -> RouteResult:
        """
        Resolve a prompt to a task and return a hot-swapped RouteResult.

        Resolution order:
            1. Alias exact match   (instant, O(1))
            2. Keyword scoring     (O(patterns × keywords))
            3. LLM resolver        (only if keyword confidence < floor)
            4. Bootstrap           (last resort)
        """
        t0 = time.perf_counter()
        normalised = self._normalise(prompt)

        # ── 1. Alias check ───────────────────────────────────────────────────
        for token in normalised.split():
            if token in self._alias_map:
                task_name = self._alias_map[token]
                system, version = self._load_or_bootstrap(task_name)
                return RouteResult(
                    task_name  = task_name,
                    system     = system,
                    version    = version,
                    confidence = 1.0,
                    method     = "alias",
                    elapsed_ms = (time.perf_counter() - t0) * 1000,
                    prompt_raw = prompt,
                )

        # ── 2. Keyword scoring ───────────────────────────────────────────────
        best_task, best_conf = self._keyword_score(normalised)

        if best_task and best_conf >= self.confidence_floor:
            system, version = self._load_or_bootstrap(best_task)
            return RouteResult(
                task_name  = best_task,
                system     = system,
                version    = version,
                confidence = best_conf,
                method     = "keyword",
                elapsed_ms = (time.perf_counter() - t0) * 1000,
                prompt_raw = prompt,
            )

        # ── 3. LLM resolver (optional slot) ─────────────────────────────────
        if self.llm_resolver is not None:
            try:
                llm_task = self.llm_resolver(prompt)
                if llm_task:
                    system, version = self._load_or_bootstrap(llm_task)
                    return RouteResult(
                        task_name  = llm_task,
                        system     = system,
                        version    = version,
                        confidence = 0.6,   # LLM result is treated as moderate confidence
                        method     = "llm",
                        elapsed_ms = (time.perf_counter() - t0) * 1000,
                        prompt_raw = prompt,
                    )
            except Exception as exc:
                logger.warning("LLM resolver failed: %s — falling back to bootstrap", exc)

        # ── 4. Bootstrap: infer task name from prompt, create default ────────
        inferred_name = self._infer_task_name(normalised)
        system, version = self._load_or_bootstrap(inferred_name)
        return RouteResult(
            task_name  = inferred_name,
            system     = system,
            version    = version,
            confidence = 0.0,
            method     = "bootstrap",
            elapsed_ms = (time.perf_counter() - t0) * 1000,
            prompt_raw = prompt,
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _normalise(text: str) -> str:
        """Lowercase, strip punctuation, collapse whitespace."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _keyword_score(self, normalised: str) -> tuple[Optional[str], float]:
        """
        Score every registered pattern against the normalised prompt.
        Returns (best_task_name, confidence) or (None, 0.0).

        Confidence = matched_count / total_keywords_in_pattern
        (capped at 1.0 so patterns with many keywords aren't penalised).
        """
        tokens = set(normalised.split())
        best_task  = None
        best_score = 0.0
        best_conf  = 0.0

        for pattern in self._patterns:
            if not pattern.keywords:
                continue
            matched = sum(
                1 for kw in pattern.keywords
                if kw in tokens or kw in normalised   # whole-word + substring
            )
            if matched == 0:
                continue
            score = matched                                   # raw match count
            conf  = matched / len(pattern.keywords)           # normalised confidence
            if score > best_score or (score == best_score and conf > best_conf):
                best_score = score
                best_conf  = conf
                best_task  = pattern.task_name

        return best_task, best_conf

    def _load_or_bootstrap(self, task_name: str) -> tuple[ThreeBarSystem, int]:
        """
        Try to load the current snapshot for task_name from the registry.
        If not found, create a default system, save it, and return it.
        """
        try:
            system = self.registry.load(task_name)
            # Retrieve the current version number
            tasks = {t["task_name"]: t for t in self.registry.list_tasks()}
            version = tasks[task_name]["current_version"] if task_name in tasks else 1
            return system, version
        except KeyError:
            return self._bootstrap(task_name)

    def _bootstrap(self, task_name: str) -> tuple[ThreeBarSystem, int]:
        """
        No snapshot exists. Find the registered default_system for this task
        (if any), otherwise use ThreeBarSystem.from_defaults(). Save and return.
        """
        default = None
        for pattern in self._patterns:
            if pattern.task_name == task_name and pattern.default_system is not None:
                default = pattern.default_system
                break
        system  = default if default is not None else ThreeBarSystem.from_defaults()
        version = self.registry.save(task_name, system)
        logger.info("Bootstrapped new snapshot for task '%s' at v%d", task_name, version)
        return system, version

    @staticmethod
    def _infer_task_name(normalised: str) -> str:
        """
        Derive a snake_case task name from the first 4 significant words
        of the normalised prompt. Used as a last-resort registry key.
        """
        stop_words = {"the", "a", "an", "my", "me", "i", "it", "is", "in",
                      "on", "of", "to", "for", "and", "or", "please", "can",
                      "you", "with", "at", "this", "that", "do", "run"}
        words = [w for w in normalised.split() if w not in stop_words][:4]
        return "_".join(words) if words else "unknown_task"

    # ── Built-in LLM resolver: Ollama ────────────────────────────────────────

    @staticmethod
    def ollama_resolver(
        model: str = "mistral",
        host:  str = "http://localhost:11434",
        registered_tasks: Optional[list[str]] = None,
    ) -> Callable[[str], Optional[str]]:
        """
        Factory that returns an LLM resolver using a local Ollama instance.

        The resolver sends the prompt + registered task list to Ollama and
        asks it to return only the single best-matching task name.

        Args:
            model:             Ollama model tag (e.g. "mistral", "llama3").
            host:              Ollama server URL.
            registered_tasks:  If provided, included in the system prompt so
                               the LLM can pick from the known list.

        Returns a callable: prompt:str → task_name:str | None
        """
        try:
            import urllib.request
        except ImportError:
            raise RuntimeError("urllib.request not available.")

        task_list_str = (
            "\n".join(f"  - {t}" for t in registered_tasks)
            if registered_tasks else "  (none registered yet)"
        )

        def resolver(prompt: str) -> Optional[str]:
            system_msg = (
                "You are a task classifier for a local AI agent. "
                "Given a user prompt, return ONLY the single most appropriate "
                "task name from the list below, as a bare snake_case string. "
                "No explanation, no punctuation, no quotes — just the task name.\n\n"
                f"Registered tasks:\n{task_list_str}"
            )
            payload = json.dumps({
                "model":  model,
                "prompt": f"{system_msg}\n\nUser prompt: {prompt}",
                "stream": False,
            }).encode()
            req = urllib.request.Request(
                f"{host}/api/generate",
                data    = payload,
                headers = {"Content-Type": "application/json"},
                method  = "POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data     = json.loads(resp.read())
                raw      = data.get("response", "").strip().lower()
                cleaned  = re.sub(r"[^\w]", "_", raw).strip("_")
                return cleaned if cleaned else None

        return resolver

    # ── Inspection ───────────────────────────────────────────────────────────

    def list_intents(self) -> list[dict]:
        """Return a summary of all registered intents."""
        return [
            {
                "task_name":   p.task_name,
                "keywords":    p.keywords,
                "aliases":     p.aliases,
                "description": p.description,
            }
            for p in self._patterns
        ]

    def __repr__(self) -> str:
        llm = "Ollama" if self.llm_resolver else "none"
        return (
            f"MasterFulcrum("
            f"patterns={len(self._patterns)}, "
            f"llm={llm}, "
            f"floor={self.confidence_floor:.0%})"
        )


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, os

    print("=== KSA Master Fulcrum Demo ===\n")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        registry = SnapshotRegistry(db_path)
        router   = MasterFulcrum(registry)

        # ── Register task intents ────────────────────────────────────────────
        router.register_intent(
            task_name   = "file_index_stealth",
            keywords    = ["index", "scan", "files", "directory", "folder",
                           "stealth", "background", "quiet", "silently"],
            aliases     = ["index"],
            description = "Background file indexing without UI interference",
        )
        router.register_intent(
            task_name   = "local_search",
            keywords    = ["search", "find", "locate", "grep", "query", "lookup"],
            aliases     = ["search", "find"],
            description = "Low-priority local file/content search",
        )
        router.register_intent(
            task_name   = "code_gen_assist",
            keywords    = ["write", "generate", "code", "function", "class",
                           "script", "implement", "draft"],
            aliases     = ["codegen"],
            description = "Code generation with local LLM assist",
        )

        print(f"Router: {router}\n")

        # ── Route several prompts ────────────────────────────────────────────
        prompts = [
            "quietly scan my project directory in the background",
            "find all TODO comments in my codebase",
            "write a Python function to parse JSON logs",
            "index",                                            # alias exact match
            "do something completely unrecognised please",      # bootstrap path
        ]

        for p in prompts:
            result = router.route(p)
            print(f"  Prompt : {p!r}")
            print(f"  Result : {result}")
            eq = result.system.simulate()
            print(f"  Decision: {eq.final_tilt.value.upper()} "
                  f"(confidence {eq.confidence:.0%})\n")

        # ── Inspect registry after routing ───────────────────────────────────
        print("Registry tasks after routing:")
        for t in registry.list_tasks():
            print(f"  {t['task_name']:35s} v{t['current_version']}")

    finally:
        os.unlink(db_path)
        print("\nTemp DB cleaned up. ✓")
