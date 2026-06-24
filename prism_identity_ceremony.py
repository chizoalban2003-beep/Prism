"""
prism_identity_ceremony.py
==========================
One-time conversational onboarding that populates the soul seed.

Not a form — a real LLM-facilitated dialogue. Seven questions, each
building on the previous answer. The LLM acts as an interviewer that
listens carefully, asks follow-up questions, and extracts structured
facts while preserving the narrative voice.

Usage
-----
    ceremony = IdentityCeremony(soul=soul, llm_router=router)

    # Run interactively (yields questions, accepts answers)
    for question in ceremony.questions():
        answer = input(question + "  > ")
        ceremony.answer(answer)

    seed = ceremony.complete()  # saves to soul

    # Or run from a dict of pre-written answers (testing / batch)
    seed = ceremony.run_from_answers({
        "identity": "I'm a product manager at a startup...",
        "decisions": "I want better support on hiring and product prioritisation...",
        ...
    })
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prism_soul import SoulSeed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------

CEREMONY_QUESTIONS: dict[str, str] = {
    "identity": (
        "Let's start simply. Who are you — not your job title, "
        "but what actually defines you and what you're building?"
    ),
    "decisions": (
        "What kinds of decisions do you want better support with? "
        "Be as specific as you can — the more concrete, the more useful I become."
    ),
    "values": (
        "What do you genuinely care about? Not what sounds good — "
        "what you actually make sacrifices for."
    ),
    "obstacles": (
        "What tends to get in your way? Patterns you've noticed in yourself, "
        "things you keep having to fight."
    ),
    "success": (
        "A year from now, if I've genuinely helped you, what will be different? "
        "What will you have done or become?"
    ),
    "misunderstand": (
        "What do people — or AI systems — tend to get wrong about you? "
        "What assumption should I never make?"
    ),
    "boundaries": (
        "Is there anything you want to keep private, "
        "or areas where you want me to ask before acting?"
    ),
}

_QUESTION_ORDER = ["identity", "decisions", "values", "obstacles", "success", "misunderstand", "boundaries"]

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
Given these answers from an identity ceremony, extract:
1. stated_values: list of 3–6 short phrases (what they genuinely value)
2. stated_goals: list of 2–4 specific goals for the next year
3. stated_constraints: list of 1–3 constraints or boundaries they set
4. initial_beliefs: list of dicts with {{text, belief_type, confidence}} — \
pick 4–8 concrete beliefs/patterns/preferences you can extract from their words
5. suggested_lenses: list of dicts with {{name, description}} — \
2–4 observation dimensions that would be useful to track for this person

Return ONLY valid JSON with these 5 keys.

Answers:
{answers_text}
"""


# ---------------------------------------------------------------------------
# IdentityCeremony
# ---------------------------------------------------------------------------


class IdentityCeremony:
    """Conversational identity ceremony that populates the soul seed."""

    def __init__(self, soul, llm_router=None):
        self._soul = soul
        self._llm_router = llm_router
        self._answers: dict[str, str] = {}  # key -> answer

    def questions(self) -> list[str]:
        """Return the question texts in order."""
        return [CEREMONY_QUESTIONS[k] for k in _QUESTION_ORDER]

    def answer(self, text: str) -> None:
        """Store the next unanswered question's answer."""
        for key in _QUESTION_ORDER:
            if key not in self._answers:
                self._answers[key] = text
                return
        logger.warning("All questions already answered.")

    def is_complete(self) -> bool:
        """True if all 7 questions have answers."""
        return all(k in self._answers for k in _QUESTION_ORDER)

    def _build_narrative(self) -> str:
        """Join questions + answers into a readable narrative."""
        parts = []
        for key in _QUESTION_ORDER:
            q = CEREMONY_QUESTIONS[key]
            a = self._answers.get(key, "")
            parts.append(f"Q: {q}\nA: {a}")
        return "\n\n".join(parts)

    def _extract_structured(self) -> dict:
        """
        Call LLM with EXTRACTION_PROMPT.
        If no LLM, use simple heuristics.
        Always returns a valid dict with all 5 keys.
        """
        answers_text = self._build_narrative()
        empty = {
            "stated_values": [],
            "stated_goals": [],
            "stated_constraints": [],
            "initial_beliefs": [],
            "suggested_lenses": [],
        }

        if self._llm_router is not None:
            try:
                prompt = EXTRACTION_PROMPT.format(answers_text=answers_text)
                response = self._llm_router.chat(prompt)
                # Extract JSON from response
                raw = response.strip()
                # Strip markdown code fences if present
                raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
                raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
                data = json.loads(raw)
                # Validate all keys present
                for k in empty:
                    if k not in data:
                        data[k] = []
                return data
            except Exception as e:
                logger.warning("LLM extraction failed: %s — falling back to heuristics", e)

        # Heuristics fallback — extract per question key so a separator-less
        # answer doesn't collapse into a single giant string. Each ceremony
        # question maps to a soul-seed field semantically:
        #   values     → stated_values
        #   success    → stated_goals  ("a year from now... what will be different")
        #   boundaries → stated_constraints
        # Anything else (identity/decisions/obstacles/misunderstand) feeds
        # initial_beliefs.
        def _phrases(key: str, *, min_len: int = 5,
                     limit: int = 6) -> list[str]:
            text = self._answers.get(key, "") or ""
            parts = re.split(r"[,;.!?\n]", text)
            seen: set[str] = set()
            out: list[str] = []
            for p in parts:
                p = p.strip()
                if len(p) < min_len or p.lower() in seen:
                    continue
                seen.add(p.lower())
                out.append(p)
                if len(out) >= limit:
                    break
            return out

        stated_values = _phrases("values", limit=6) or [
            "authenticity", "growth", "clarity",
        ]
        stated_goals = _phrases("success", limit=4) or [
            "make progress in key areas",
        ]
        stated_constraints = _phrases("boundaries", limit=3) or [
            "respect my privacy",
        ]

        # Build a few initial beliefs from the answers
        initial_beliefs = []
        for phrase in _phrases("values", limit=4):
            initial_beliefs.append({
                "text": phrase, "belief_type": "value", "confidence": 0.7,
            })
        for phrase in _phrases("obstacles", limit=3):
            initial_beliefs.append({
                "text": phrase, "belief_type": "pattern", "confidence": 0.6,
            })

        suggested_lenses = [
            {"name": "Focus", "description": "Track depth of focused work sessions"},
            {"name": "Energy", "description": "Track daily energy and momentum levels"},
        ]

        return {
            "stated_values": [v for v in stated_values if v][:6],
            "stated_goals": [g for g in stated_goals if g][:4],
            "stated_constraints": [c for c in stated_constraints if c][:3],
            "initial_beliefs": initial_beliefs[:8],
            "suggested_lenses": suggested_lenses[:4],
        }

    def complete(self) -> SoulSeed:
        """Extract structured data, build SoulSeed, save to soul."""
        from prism_soul import SoulSeed

        extracted = self._extract_structured()
        narrative = self._build_narrative()
        now = time.time()

        seed = SoulSeed(
            narrative=narrative,
            stated_values=extracted.get("stated_values", []),
            stated_goals=extracted.get("stated_goals", []),
            stated_constraints=extracted.get("stated_constraints", []),
            created_at=now,
            updated_at=now,
        )
        self._soul.set_seed(seed)

        # Add initial beliefs to soul
        for belief_data in extracted.get("initial_beliefs", []):
            text = belief_data.get("text", "")
            if not text:
                continue
            belief_type = belief_data.get("belief_type", "value")
            confidence = float(belief_data.get("confidence", 0.7))
            self._soul.add_belief(
                text=text,
                belief_type=belief_type,
                source="stated",
                confidence=confidence,
            )

        # Add suggested lenses to soul
        for lens_data in extracted.get("suggested_lenses", []):
            name = lens_data.get("name", "")
            description = lens_data.get("description", "")
            if name:
                self._soul.add_lens(name=name, description=description)

        return seed

    def run_from_answers(self, answers: dict[str, str]) -> SoulSeed:
        """
        Accept a dict mapping question keys to answer strings,
        call complete(), return seed. Used in tests and batch mode.
        """
        self._answers = {k: v for k, v in answers.items()}
        return self.complete()
