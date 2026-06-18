"""
prism_crystalliser.py
=====================
Extracts behavioural signals from interactions and writes to PrismPersona.

Two modes:
  Real-time  — heuristic extraction per turn, no LLM, called after every chat turn
  Deep       — LLM batch analysis of recent interactions, called hourly by daemon
"""
from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from prism_calibration import PrismCalibration
    from prism_memory import PrismMemory
    from prism_outcome_tracker import OutcomeTracker
    from prism_persona import PrismPersona

logger = logging.getLogger(__name__)

# Technical vocabulary signals for depth detection
_TECH_TERMS = frozenset({
    "function", "class", "method", "variable", "loop", "api", "endpoint",
    "database", "query", "index", "schema", "config", "deploy", "docker",
    "git", "commit", "branch", "merge", "pull", "push", "pip", "npm",
    "json", "yaml", "toml", "regex", "async", "await", "thread", "socket",
    "http", "https", "ssl", "tls", "auth", "token", "jwt", "oauth",
    "cpu", "ram", "disk", "kernel", "process", "daemon", "cron",
    "algorithm", "complexity", "recursion", "iterator", "generator",
})

_CORRECTION_EXPLICIT = re.compile(
    r"\b(prefer|want|like|need|use|always|never|stop|don'?t)\b", re.I
)

# LLM placeholders / meta-comments we've seen poisoning the persona.
# Reject anything that's just the prompt's example echoed back, or a
# "no preferences identified" meta-statement.
_PREF_BLACKLIST = re.compile(
    r"^(?:prefers?\s+x|dislikes?\s+y|"
    r"no(?:ne)?\s+(?:explicit\s+)?preferences?(?:\s+identified)?|"
    r"none\s+provided|n/?a|unknown|null|undefined|"
    r"pattern\s*\d+|example\s*\d*|placeholder)\s*[.!]?$",
    re.I,
)


def _looks_like_real_preference(pref: str) -> bool:
    """Filter out placeholder echoes and meta-comments before they hit the persona."""
    s = pref.strip()
    if len(s) < 8:                       # too short to carry signal
        return False
    if _PREF_BLACKLIST.match(s):
        return False
    return True


class PrismCrystalliser:
    """
    Observes interactions and crystallises behavioural traits into PrismPersona.
    Gracefully degrades when LLM is unavailable.
    """

    def __init__(
        self,
        persona: PrismPersona,
        memory: Optional[PrismMemory] = None,
        outcome_tracker: Optional[OutcomeTracker] = None,
        calibration: Optional[PrismCalibration] = None,
        llm_router=None,
        ml_assembler=None,
        soul=None,
    ):
        self._persona = persona
        self._memory = memory
        self._outcome_tracker = outcome_tracker
        self._calibration = calibration
        self._router = llm_router
        self._ml_assembler = ml_assembler
        self._soul = soul

    # ── Real-time heuristics ──────────────────────────────────────────────────

    def observe_turn(
        self,
        message: str,
        response: str,
        intent: str,
        ctx: dict,
    ) -> None:
        """Heuristic extraction from one chat turn. No LLM. Called after every turn."""
        hour = time.localtime().tm_hour
        self._persona.record_active_hour(hour)

        words = message.split() if message else []
        word_count = len(words)
        if word_count < 15:
            length_val, length_conf = "concise", 0.6
        elif word_count <= 50:
            length_val, length_conf = "medium", 0.5
        else:
            length_val, length_conf = "detailed", 0.6
        self._persona.update_trait(
            "response_length_preference", length_val, length_conf, delta=1
        )

        msg_lower = message.lower() if message else ""
        tech_hits = sum(1 for t in _TECH_TERMS if t in msg_lower)
        if tech_hits >= 3:
            self._persona.update_trait("technical_depth", "high", 0.65, delta=1)
        elif tech_hits == 1:
            self._persona.update_trait("technical_depth", "medium", 0.5, delta=1)
        elif word_count > 5:
            self._persona.update_trait("technical_depth", "low", 0.4, delta=1)

        approval_action = ctx.get("_last_approval_action", "")
        if approval_action == "approved":
            self._persona.update_trait("risk_tolerance", "willing", 0.55, delta=1)
        elif approval_action == "cancelled":
            self._persona.update_trait("risk_tolerance", "cautious", 0.55, delta=1)

        if message:
            if msg_lower.endswith("?") or msg_lower.startswith(("what", "how", "why", "when", "where")):
                style_val = "inquisitive"
            elif msg_lower.startswith(("do ", "run ", "set ", "create ", "delete ", "send ")):
                style_val = "directive"
            elif any(t in msg_lower for t in ("wrong", "incorrect", "not what", "actually")):
                style_val = "corrective"
            else:
                style_val = "conversational"
            self._persona.update_trait("communication_style", style_val, 0.45, delta=1)

    def observe_outcome(
        self,
        intent: str,
        outcome: str,
        goal: str,
        correction: str = "",
    ) -> None:
        """Called when an outcome is recorded. No LLM."""
        if outcome == "user_corrected" and correction:
            if _CORRECTION_EXPLICIT.search(correction):
                snippet = correction[:120].strip()
                self._persona.update_trait(
                    "correction_pattern",
                    snippet,
                    0.75,
                    source="explicit",
                    delta=1,
                )
        elif outcome == "abandoned":
            self._persona.bump_pattern(
                f"abandoned {intent} tasks",
                example=goal[:80],
            )
        elif outcome == "done":
            self._persona.bump_pattern(
                f"completes {intent} tasks",
                example=goal[:80],
            )

    # ── Deep LLM analysis ────────────────────────────────────────────────────

    def deep_analyse(self, lookback_hours: int = 24) -> int:
        """LLM batch extraction from recent interactions. Returns count of updates made."""
        if self._router is None:
            return 0

        conversations = self._fetch_recent_conversations(20)

        outcome_stats = {}
        if self._outcome_tracker is not None:
            try:
                outcome_stats = self._outcome_tracker.stats(days=max(1, (lookback_hours + 23) // 24))
            except Exception as exc:
                logger.debug("[crystalliser] outcome stats failed: %s", exc)

        calibration_notes = []
        if self._calibration is not None:
            try:
                since = time.time() - lookback_hours * 3600
                events = [
                    e for e in self._calibration.history(n=20)
                    if e.timestamp >= since
                ]
                calibration_notes = [
                    f"{e.direction} ({e.domain})" for e in events[:10]
                ]
            except Exception as exc:
                logger.debug("[crystalliser] calibration fetch failed: %s", exc)

        prompt = self._build_extraction_prompt(
            conversations, outcome_stats, calibration_notes
        )
        try:
            raw, _ = self._router.call(
                prompt, min_capability=1, max_tokens=600, json_mode=True
            )
        except Exception as exc:
            logger.debug("[crystalliser] LLM call failed: %s", exc)
            return 0

        from prism_llm_router import parse_llm_json

        parsed = parse_llm_json(raw)
        if not isinstance(parsed, dict):
            return 0

        n_updates = self._apply_extraction(parsed)

        # Write high-confidence traits to Soul belief graph (cap at 3 per run)
        if self._soul is not None:
            _trait_map = {
                "communication_style": parsed.get("communication_style"),
                "technical_depth": parsed.get("technical_depth"),
                "decision_style": parsed.get("decision_style"),
            }
            _written = 0
            for _key, _value in _trait_map.items():
                if _written >= 3:
                    break
                if _value and isinstance(_value, str) and _value.strip() and _value.strip() != "unknown":
                    try:
                        self._soul.add_belief(
                            text=f"User's {_key} is {_value.strip()}",
                            belief_type="pattern",
                            source="crystalliser",
                            confidence=0.72,
                        )
                        _written += 1
                    except Exception as _se:
                        logger.debug("[crystalliser] soul add_belief failed: %s", _se)

        self._run_ml_sweep()
        return n_updates

    def _run_ml_sweep(self) -> None:
        """Run nightly ML hyperparameter sweep on failed outcomes (conf < 1-ERROR_THRESHOLD)."""
        if self._ml_assembler is None or self._outcome_tracker is None:
            return
        try:
            from prism_ml_assembler import run_nightly_sweep
            updated = run_nightly_sweep(self._ml_assembler, self._outcome_tracker)
            if updated:
                logger.info("[crystalliser] ML nightly sweep updated params: %s", list(updated))
        except Exception as exc:
            logger.debug("[crystalliser] ML nightly sweep failed: %s", exc)

    def crystallise(self, force: bool = False) -> dict:
        """Full recrystallisation — called weekly by daemon."""
        n_updates = self.deep_analyse(lookback_hours=168)
        peaks = self._persona.peak_hours()
        traits = self._persona.list_traits()
        avg_conf = (
            round(sum(t.confidence for t in traits) / len(traits), 3)
            if traits else 0.0
        )
        patterns = self._persona.growth_since(days=7)
        return {
            "traits_updated": n_updates,
            "patterns_found": patterns["new_patterns"],
            "confidence_avg": avg_conf,
            "peak_hours": peaks,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _fetch_recent_conversations(self, n: int) -> list[str]:
        if self._memory is None:
            return []
        try:
            results = self._memory.search("", top_n=n)
            snippets = []
            for r in results:
                source = getattr(r.entry, "source", "")
                if source in ("conversation", "user", "assistant"):
                    snippets.append(r.excerpt[:200])
            return snippets[:n]
        except Exception as exc:
            logger.debug("[crystalliser] memory fetch failed: %s", exc)
            return []

    def _build_extraction_prompt(
        self,
        conversations: list[str],
        outcome_stats: dict,
        calibration_notes: list[str],
    ) -> str:
        conv_block = "\n".join(f"- {c}" for c in conversations) or "(none)"
        stats_block = (
            f"completion_rate={outcome_stats.get('completion_rate', 'n/a')}, "
            f"total={outcome_stats.get('total', 0)}"
            if outcome_stats
            else "(none)"
        )
        calib_block = ", ".join(calibration_notes) or "(none)"

        return (
            "Analyze these recent user interactions and extract behavioral patterns.\n\n"
            f"Recent conversation excerpts:\n{conv_block}\n\n"
            f"Outcome stats: {stats_block}\n"
            f"Calibration events: {calib_block}\n\n"
            "Return JSON only. Use EMPTY string/array fields when you have no\n"
            "evidence — do not invent or guess.\n"
            "{\n"
            '  "communication_style": "direct/elaborate/technical/casual",\n'
            '  "response_length": "concise/medium/detailed",\n'
            '  "technical_depth": "low/medium/high",\n'
            '  "decision_style": "quick/deliberate/risk_averse/experimental",\n'
            '  "patterns": [],\n'
            '  "explicit_preferences": []\n'
            "}\n"
            "Only fill explicit_preferences with verbatim user statements like\n"
            '"prefers Postgres over MySQL". Do NOT include meta-comments such as\n'
            '"no preferences identified" — leave the array empty in that case.'
        )

    def _apply_extraction(self, data: dict) -> int:
        updates = 0

        field_map = {
            "communication_style": ("communication_style", 0.6),
            "response_length": ("response_length_preference", 0.6),
            "technical_depth": ("technical_depth", 0.65),
            "decision_style": ("decision_style", 0.6),
        }
        for json_key, (trait_name, conf) in field_map.items():
            val = data.get(json_key, "")
            if isinstance(val, str) and val.strip():
                self._persona.update_trait(
                    trait_name, val.strip(), conf, source="inferred", delta=2
                )
                updates += 1

        for patt in data.get("patterns", [])[:5]:
            if isinstance(patt, str) and patt.strip():
                self._persona.bump_pattern(patt.strip())
                updates += 1

        for pref in data.get("explicit_preferences", [])[:5]:
            if not (isinstance(pref, str) and pref.strip()):
                continue
            if not _looks_like_real_preference(pref):
                logger.debug("[crystalliser] dropping placeholder pref: %r", pref)
                continue
            self._persona.update_trait(
                f"pref_{pref[:30].lower().replace(' ', '_')}",
                pref.strip(),
                0.8,
                source="explicit",
                delta=1,
            )
            updates += 1

        return updates
