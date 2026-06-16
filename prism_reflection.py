"""
prism_reflection.py
===================
Weekly self-audit: summarises completed chains, extracts behaviour patterns,
proposes soul belief updates, and flags unresolved horizon goals.

Run automatically by the daemon worker every 7 days, or on demand:

    reflection = PrismReflection(
        outcome_tracker = tracker,
        soul            = soul,
        horizon         = horizon,
        llm_router      = router,
    )
    report = reflection.run()

Output
------
A ReflectionReport containing:
  - summary          prose paragraph about the week's activity
  - patterns         list of extracted behaviour patterns
  - belief_proposals list of {node_id, new_confidence, rationale} dicts
  - unresolved_goals list of stale HorizonGoal IDs with age in days
  - applied          bool — whether belief updates were auto-applied
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from prism_horizon import HorizonPlanner
    from prism_outcome_tracker import OutcomeTracker
    from prism_soul import PrismSoul

logger = logging.getLogger(__name__)

_REFLECTION_PROMPT = """\
You are PRISM's self-reflection engine. Analyse the following chain execution
outcomes from the past 7 days and produce a structured reflection.

OUTCOME STATS:
{stats}

RECENT GOALS (last 20):
{goals}

CURRENT SOUL BELIEFS (stated, top 10):
{beliefs}

Produce a JSON response with this exact shape:
{{
  "summary": "2-3 sentence prose paragraph about overall activity and progress",
  "patterns": [
    "pattern 1 (e.g. user frequently asks about X)",
    "pattern 2"
  ],
  "belief_proposals": [
    {{
      "node_id": "<existing belief node_id or 'new'>",
      "text": "<belief text if new>",
      "new_confidence": 0.0,
      "rationale": "why this update is warranted"
    }}
  ],
  "flag_goal_ids": ["goal_id_1", "goal_id_2"]
}}

Rules:
- Only propose belief updates when outcome evidence genuinely supports it
- Keep proposals conservative (confidence delta ≤ 0.15)
- flag_goal_ids should list goals that appear stale (no progress in 14+ days)
- Return valid JSON only — no markdown, no prose outside the JSON
"""


@dataclass
class ReflectionReport:
    ran_at:           float = field(default_factory=time.time)
    summary:          str = ""
    patterns:         list[str] = field(default_factory=list)
    belief_proposals: list[dict] = field(default_factory=list)
    unresolved_goals: list[dict] = field(default_factory=list)  # {goal_id, intent, age_days}
    applied:          bool = False
    error:            str = ""


class PrismReflection:
    """
    Weekly meta-learning loop.

    auto_apply=True  — applies LLM-proposed belief updates automatically
    auto_apply=False — stores proposals in the report for user review
    """

    def __init__(
        self,
        outcome_tracker: Optional[OutcomeTracker] = None,
        soul:            Optional[PrismSoul]       = None,
        horizon:         Optional[HorizonPlanner]  = None,
        llm_router=None,
        auto_apply: bool = False,
        days: int = 7,
    ):
        self._tracker   = outcome_tracker
        self._soul      = soul
        self._horizon   = horizon
        self._router    = llm_router
        self._auto      = auto_apply
        self._days      = days
        # Reflection is a weekly meta-loop but the endpoint may be polled.
        # Cache the last report for an hour so probes don't trigger a fresh
        # LLM pass each time.
        self._cache_ttl: float = 3600.0
        self._cached:    Optional[ReflectionReport] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, force: bool = False) -> ReflectionReport:
        """Run the weekly reflection and return a ReflectionReport.

        Returns the cached report if one was generated within `_cache_ttl`
        seconds. Pass `force=True` to skip the cache (e.g. nightly cron).
        """
        if not force and self._cached is not None:
            if time.time() - self._cached.ran_at < self._cache_ttl:
                return self._cached

        report = ReflectionReport()

        # 1. Outcome stats
        stats = self._get_stats()

        # 2. Recent goals
        goals = self._get_recent_goals()

        # 3. Current beliefs
        beliefs = self._get_beliefs()

        # 4. Stale horizon goals (always computed, no LLM needed)
        report.unresolved_goals = self._find_stale_goals()

        if not self._router:
            report.summary = (
                f"Reflection (no LLM): {stats.get('total', 0)} chains in {self._days} days, "
                f"completion rate {stats.get('completion_rate', 0):.0%}. "
                f"{len(report.unresolved_goals)} stale goal(s)."
            )
            report.patterns = self._heuristic_patterns(stats, goals)
            logger.info("[reflection] completed (no LLM)")
            self._cached = report
            return report

        # 5. LLM reflection
        try:
            report = self._llm_reflect(report, stats, goals, beliefs)
        except Exception as exc:
            report.error = str(exc)
            logger.warning("[reflection] LLM reflection failed: %s", exc)

        # 6. Apply belief updates
        if self._auto and self._soul and report.belief_proposals:
            report.applied = self._apply_proposals(report.belief_proposals)

        self._cached = report
        return report

    def summarise_for_chat(self) -> str:
        """Run reflection and return a human-readable summary string."""
        report = self.run()
        lines = [f"Weekly reflection ({time.strftime('%Y-%m-%d')}):", ""]
        if report.summary:
            lines.append(report.summary)
        if report.patterns:
            lines.append("\nPatterns observed:")
            for p in report.patterns:
                lines.append(f"  • {p}")
        if report.unresolved_goals:
            lines.append(f"\nStale goals ({len(report.unresolved_goals)}):")
            for g in report.unresolved_goals[:5]:
                lines.append(f"  — [{g['age_days']}d] {g['intent'][:80]}")
        if report.belief_proposals:
            applied = "applied" if report.applied else "pending review"
            lines.append(f"\n{len(report.belief_proposals)} belief update(s) {applied}.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_stats(self) -> dict:
        if self._tracker is None:
            return {}
        try:
            return self._tracker.stats(days=self._days)
        except Exception:
            return {}

    def _get_recent_goals(self) -> list[str]:
        if self._tracker is None:
            return []
        try:
            records = self._tracker.recent(n=20)
            return [f"[{r.outcome}] {r.goal[:80]}" for r in records]
        except Exception:
            return []

    def _get_beliefs(self) -> list[str]:
        if self._soul is None:
            return []
        try:
            beliefs = self._soul.list_beliefs(belief_type="stated")[:10]
            return [f"[{b.node_id}] {b.text[:80]} (conf={b.confidence:.2f})" for b in beliefs]
        except Exception:
            return []

    def _find_stale_goals(self) -> list[dict]:
        if self._horizon is None:
            return []
        stale_threshold = time.time() - 14 * 86400
        stale = []
        try:
            goals = self._horizon.list_goals(status="active")
            for g in goals:
                age_days = (time.time() - g.created_at) / 86400
                if g.created_at < stale_threshold:
                    stale.append({
                        "goal_id":  g.goal_id,
                        "intent":   g.intent,
                        "age_days": round(age_days, 1),
                    })
        except Exception:
            pass
        return stale

    def _llm_reflect(
        self, report: ReflectionReport, stats: dict, goals: list[str], beliefs: list[str]
    ) -> ReflectionReport:
        import json as _json
        stats_str   = _json.dumps(stats, indent=2)
        goals_str   = "\n".join(goals) or "(none)"
        beliefs_str = "\n".join(beliefs) or "(none)"

        prompt = _REFLECTION_PROMPT.format(
            stats   = stats_str,
            goals   = goals_str,
            beliefs = beliefs_str,
        )
        raw, _ = self._router.call(prompt, min_capability=2, max_tokens=800)

        # Strip markdown fences
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        data = _json.loads(raw)

        report.summary          = data.get("summary", "")
        report.patterns         = data.get("patterns", [])
        report.belief_proposals = data.get("belief_proposals", [])

        # Merge LLM-flagged stale goals with locally computed ones
        llm_flags = set(data.get("flag_goal_ids", []))
        existing_ids = {g["goal_id"] for g in report.unresolved_goals}
        if self._horizon:
            for gid in llm_flags:
                if gid not in existing_ids:
                    goal = self._horizon.get(gid)
                    if goal:
                        age_days = (time.time() - goal.created_at) / 86400
                        report.unresolved_goals.append({
                            "goal_id":  gid,
                            "intent":   goal.intent,
                            "age_days": round(age_days, 1),
                        })
        return report

    def _heuristic_patterns(self, stats: dict, goals: list[str]) -> list[str]:
        import re
        from collections import Counter
        patterns = []
        # Keyword frequency from goal text
        words: list[str] = []
        for g in goals:
            words.extend(re.findall(r"[a-z]{5,}", g.lower()))
        stop = {"chain", "about", "check", "with", "done", "this", "that", "from", "what"}
        counts = Counter(w for w in words if w not in stop)
        for word, cnt in counts.most_common(5):
            if cnt >= 2:
                patterns.append(f"Recurring topic: '{word}' appeared in {cnt} chains")
        # Completion rate signal — always check regardless of goals list
        rate = stats.get("completion_rate", 0)
        total = stats.get("total", 0)
        if total >= 3:
            if rate < 0.5:
                patterns.append(f"Low completion rate ({rate:.0%}) — chains may be too complex")
            elif rate > 0.9:
                patterns.append(f"High completion rate ({rate:.0%}) — chains running efficiently")
        return patterns

    def _apply_proposals(self, proposals: list[dict]) -> bool:
        applied = 0
        for p in proposals:
            try:
                node_id    = p.get("node_id", "")
                new_conf   = float(p.get("new_confidence", 0.0))
                rationale  = p.get("rationale", "")
                if node_id == "new":
                    text = p.get("text", "")
                    if text:
                        self._soul.add_belief(
                            text        = text,
                            belief_type = "observed",
                            source      = "reflection",
                            confidence  = max(0.0, min(1.0, new_conf)),
                            notes       = rationale[:200],
                        )
                        applied += 1
                elif node_id:
                    existing = self._soul.get_belief(node_id)
                    if existing:
                        # Guard: max delta ±0.15
                        delta = new_conf - existing.confidence
                        if abs(delta) <= 0.15:
                            self._soul.update_belief(
                                node_id    = node_id,
                                confidence = max(0.0, min(1.0, new_conf)),
                                notes      = rationale[:200],
                            )
                            applied += 1
            except Exception as exc:
                logger.debug("[reflection] apply proposal failed: %s", exc)
        logger.info("[reflection] applied %d/%d belief proposals", applied, len(proposals))
        return applied > 0
