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
