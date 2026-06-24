"""
prism_routing.py
================
Intent routing helpers extracted from PrismAgent.

Three pieces:

* :func:`route_intent` — first-match regex sweep over ``INTENTS``, falling
  back to an LLM classifier callback if no pattern matches.
* :func:`should_suppress` — true when a message routes (by regex, no LLM
  call) to a constitution ``never_log`` intent. Used to keep sensitive
  content out of conversation history.
* :class:`LLMClassifier` — wraps the router + Ollama-fallback used by
  :meth:`PrismAgent._llm_classify`. Pulls organ intents through a callback
  so synthesized organs from prior sessions stay reachable.

Behaviour is unchanged from the original in-line implementation.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.request
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

IntentTable = list[tuple[str, str]]


# Reasoning/explanation/listing questions — these are open-ended chat, not
# tool invocations. We use this to short-circuit the LLM classifier so it
# can't latch onto an ambient topic keyword in the message (e.g. "explain
# database deadlocks" → devices_list because of "deadlocks"). Regex stays
# narrow: it must be the *frame* of the question, not just a substring.
_REASONING_PATTERNS = re.compile(
    r"^\s*(?:please\s+)?"
    r"(?:"
    r"explain\b|describe\b|define\b|"
    r"what (?:is|are|was|were) (?:a |an |the )?(?!my\b|your\b)"
    r"|why (?:do|does|did|is|are|was|were|should|would|can|could)\b"
    r"|how (?:do|does|did|is|are|was|were|should|would|can|could|come)\b"
    r"|when (?:do|does|did|is|are|was|were|should|would)\b"
    r"|where (?:do|does|did|is|are|was|were|should|would)\b"
    r"|who (?:is|are|was|were)\b(?!.*\bcontact\b)"
    r"|name (?:three|five|ten|\d+|some|a few)\b"
    r"|tell me (?:about|why|how|the difference)\b"
    r"|compare\b|contrast\b|"
    r"give me (?:an? )?(?:example|overview|summary)\b"
    r")",
    re.IGNORECASE,
)


def _is_reasoning_question(message: str) -> bool:
    """True when the message reads as an explanation/definition/comparison.

    Used to keep the LLM classifier from misrouting open-ended questions to
    a tool-execution intent just because the question text mentions a token
    that overlaps an intent keyword. Returns False fast for empty input.
    """
    if not message:
        return False
    return bool(_REASONING_PATTERNS.search(message[:200]))


def route_intent(
    message: str,
    intents: IntentTable,
    llm_fallback: Callable[[str], Optional[str]],
) -> str:
    lowered = message.lower()
    for pattern, intent in intents:
        if re.search(pattern, lowered):
            return intent
    return llm_fallback(message) or "general_chat"


def should_suppress(
    message: str,
    intents: IntentTable,
    constitution: Any,
) -> bool:
    if not message or constitution is None:
        return False
    lowered = message.lower()
    for pattern, intent in intents:
        if re.search(pattern, lowered):
            try:
                return bool(constitution.is_never_log(intent))
            except Exception:
                return False
    return False


class LLMClassifier:
    """LLM-backed intent picker. Uses the configured router; falls back to a
    raw Ollama call when the router isn't wired up yet during bootstrap."""

    def __init__(
        self,
        *,
        intents: IntentTable,
        router: Any,
        ollama_host: str,
        text_model: str,
        get_organ_intents: Callable[[], dict[str, str]],
    ) -> None:
        self._intents = intents
        self._router = router
        self._ollama_host = ollama_host
        self._text_model = text_model
        self._get_organ_intents = get_organ_intents

    def classify(self, message: str) -> Optional[str]:
        # Reasoning/explanation/definition questions route to chat — the
        # classifier LLM is too eager to pick a tool intent when the
        # question text mentions a topic keyword (e.g. "explain deadlocks"
        # → devices_list because of the word "deadlocks").
        if _is_reasoning_question(message):
            return None
        regex_labels = [intent for _, intent in self._intents]
        try:
            organ_intents = self._get_organ_intents() or {}
        except Exception:
            organ_intents = {}
        organ_only = sorted(set(organ_intents) - set(regex_labels))
        labels = sorted(set(regex_labels) | set(organ_intents))

        organ_block = ""
        if organ_only:
            organ_lines = "\n".join(
                f"      {i} — {organ_intents.get(i, '')[:80]}"
                for i in organ_only[:50]
            )
            organ_block = (
                "\n  - Loaded organs (prefer these when they fit; "
                "they execute immediately):\n" + organ_lines
            )

        prompt = (
            "You are a routing classifier. Pick ONE label that best fits the user's message.\n\n"
            "Labels:\n"
            "  - One of these specific intent names: " + ", ".join(labels) + "\n"
            + organ_block +
            "\n  - 'novel_capability' if the user is asking PRISM to DO a concrete action "
            "(run, send, control, generate, fetch, transform, automate something) and "
            "no specific intent above fits.\n"
            "  - 'chat' for questions, explanations, opinions, or anything conversational "
            "that doesn't require a tool.\n\n"
            f"Message: {message}\n\n"
            "Reply with ONLY the label. No quotes, no punctuation, no explanation."
        )

        result: str = ""
        if self._router is not None:
            try:
                raw, _ = self._router.call(prompt, min_capability=1, max_tokens=24)
                result = (raw or "").strip().lower().strip("'\"`.,!?")
            except Exception as exc:
                logger.debug("LLMClassifier via router failed: %s", exc)

        if not result:
            try:
                payload = json.dumps(
                    {"model": self._text_model, "prompt": prompt, "stream": False}
                ).encode()
                request = urllib.request.Request(
                    f"{self._ollama_host}/api/generate",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=30) as response:
                    result = (
                        json.loads(response.read())
                        .get("response", "")
                        .strip()
                        .lower()
                        .strip("'\"`.,!?")
                    )
            except Exception:
                return None

        if result in labels:
            return result
        if result == "novel_capability":
            return "novel_capability"
        return None
