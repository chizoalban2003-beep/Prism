"""
prism_outcome_tracker.py
========================
Closes the learning loop by recording every chain completion and feeding
outcome deltas back into PrismSoul beliefs and HorizonPlanner goal confidence.

Outcome types
-------------
done            — chain reached a final answer the LLM judged complete
abandoned       — chain hit MAX_STEPS without completing
user_corrected  — user explicitly marked the result as wrong/incomplete

The tracker writes to ~/.prism/outcomes.db and exposes:

  tracker.record(state, outcome, correction=None)
  tracker.recent(n=20)
  tracker.stats()
  tracker.pattern_stats(keyword)
  tracker.feed_soul(soul)        # update belief observations from recent outcomes
  tracker.feed_horizon(horizon)  # update goal context from completion patterns
"""
from __future__ import annotations

import logging
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from prism_horizon import HorizonPlanner
    from prism_soul import PrismSoul

logger = logging.getLogger(__name__)

OUTCOME_DONE      = "done"
OUTCOME_ABANDONED = "abandoned"
OUTCOME_CORRECTED = "user_corrected"


@dataclass
class OutcomeRecord:
    record_id:         str
    chain_id:          str
    goal:              str
    outcome:           str          # done / abandoned / user_corrected
    steps_count:       int
    duration_ms:       float
    policy_flags:      int          # how many _policy_node flags fired
    final_answer:      str
    correction:        str          # user-supplied correction text, if any
    context_id:        str          # active context at time of chain
    timestamp:         float = field(default_factory=time.time)


class OutcomeTracker:
    """
    Persistent store of chain execution outcomes.

    Soul feedback
    -------------
    Every recorded outcome calls feed_soul() incrementally:
    - 'done': observation_count += 1 on any belief whose text overlaps
      with the goal keywords (Bayesian Beta(alpha, beta) update: each
      completed chain is a success; confidence converges toward 0.95)
    - 'user_corrected': observation on a "_corrections" lens if present
    - 'abandoned': no soul update (noise)

    Horizon feedback
    ----------------
    feed_horizon() scans recent outcomes and calls
    horizon.update_context(goal_id, outcome_rate=...) for goals whose
    intent text matches recent chain goals, so the LLM evaluator has
    richer signal on whether a goal type is reliably achievable.
    """

    def __init__(
        self,
        db_path: str = "~/.prism/outcomes.db",
        soul: Optional[PrismSoul] = None,
        horizon: Optional[HorizonPlanner] = None,
    ):
        self._db           = Path(db_path).expanduser()
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._soul         = soul
        self._horizon      = horizon
        self._crystalliser = None
        self._kinetic      = None   # KineticEngine — wired by prism_agent at startup
        self._init_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        chain_id: str,
        goal: str,
        outcome: str,
        steps_count: int = 0,
        duration_ms: float = 0.0,
        policy_flags: int = 0,
        final_answer: str = "",
        correction: str = "",
        context_id: str = "default",
    ) -> OutcomeRecord:
        """Persist one outcome and trigger incremental soul/horizon feedback."""
        rec = OutcomeRecord(
            record_id   = str(uuid.uuid4())[:8],
            chain_id    = chain_id,
            goal        = goal[:400],
            outcome     = outcome,
            steps_count = steps_count,
            duration_ms = duration_ms,
            policy_flags= policy_flags,
            final_answer= final_answer[:400],
            correction  = correction[:400],
            context_id  = context_id,
        )
        with sqlite3.connect(self._db) as con:
            con.execute(
                "INSERT INTO outcomes VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (rec.record_id, rec.chain_id, rec.goal, rec.outcome,
                 rec.steps_count, rec.duration_ms, rec.policy_flags,
                 rec.final_answer, rec.correction, rec.context_id, rec.timestamp),
            )
        logger.debug("[outcome_tracker] recorded %s → %s (chain %s)", goal[:40], outcome, chain_id)

        self._incremental_soul_update(rec)
        self._fulcrum_feedback(rec)

        # Notify crystalliser
        crystalliser = self._crystalliser
        if crystalliser is not None:
            try:
                crystalliser.observe_outcome(
                    intent=goal[:100],
                    outcome=outcome,
                    goal=goal,
                    correction=correction,
                )
            except Exception:
                pass

        self._kinetic_feedback(rec)
        return rec

    def recent(self, n: int = 20, context_id: Optional[str] = None) -> list[OutcomeRecord]:
        """Return the n most recent records, optionally filtered by context."""
        with sqlite3.connect(self._db) as con:
            if context_id:
                rows = con.execute(
                    "SELECT * FROM outcomes WHERE context_id=? ORDER BY timestamp DESC LIMIT ?",
                    (context_id, n),
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT * FROM outcomes ORDER BY timestamp DESC LIMIT ?", (n,)
                ).fetchall()
        return [self._row(r) for r in rows]

    def stats(self, days: int = 30) -> dict:
        """Aggregate stats over the last `days` days."""
        since = time.time() - days * 86400
        with sqlite3.connect(self._db) as con:
            total, done, abandoned, corrected, avg_steps, avg_flags = con.execute(
                """
                SELECT
                  COUNT(*),
                  SUM(outcome='done'),
                  SUM(outcome='abandoned'),
                  SUM(outcome='user_corrected'),
                  AVG(steps_count),
                  AVG(policy_flags)
                FROM outcomes WHERE timestamp >= ?
                """,
                (since,),
            ).fetchone()
        total = total or 0
        return {
            "total": total,
            "done": done or 0,
            "abandoned": abandoned or 0,
            "user_corrected": corrected or 0,
            "completion_rate": round((done or 0) / total, 2) if total else 0.0,
            "avg_steps": round(avg_steps or 0, 1),
            "avg_policy_flags": round(avg_flags or 0, 2),
            "days": days,
        }

    def pattern_stats(self, keyword: str, days: int = 30) -> dict:
        """Stats for chains whose goal contains `keyword`."""
        since = time.time() - days * 86400
        kw = f"%{keyword.lower()}%"
        with sqlite3.connect(self._db) as con:
            rows = con.execute(
                "SELECT outcome FROM outcomes WHERE lower(goal) LIKE ? AND timestamp >= ?",
                (kw, since),
            ).fetchall()
        total = len(rows)
        done  = sum(1 for r in rows if r[0] == OUTCOME_DONE)
        return {
            "keyword": keyword,
            "total": total,
            "done": done,
            "completion_rate": round(done / total, 2) if total else 0.0,
        }

    def feed_soul(self, soul: PrismSoul, days: int = 7) -> int:
        """
        Scan recent outcomes and update soul beliefs/lenses.
        Returns the number of belief updates applied.
        """
        self._soul = soul
        records = self.recent(n=100)
        recent_ts = time.time() - days * 86400
        records = [r for r in records if r.timestamp >= recent_ts]
        updates = 0
        for rec in records:
            updates += self._soul_update_for_record(rec, soul)
        return updates

    def feed_horizon(self, horizon: HorizonPlanner, days: int = 14) -> None:
        """
        Update HorizonPlanner goal context with outcome rate data for each
        active goal whose intent overlaps with recently completed chains.
        """
        self._horizon = horizon
        active_goals = horizon.list_goals()
        for goal in active_goals:
            keywords = _extract_keywords(goal.intent)
            if not keywords:
                continue
            for kw in keywords[:3]:
                ps = self.pattern_stats(kw, days=days)
                if ps["total"] >= 3:
                    horizon.update_context(
                        goal.goal_id,
                        outcome_rate=ps["completion_rate"],
                        sample_size=ps["total"],
                    )
                    break

    def record_ml_result(
        self,
        result_id: str,
        task: str,
        algorithm: str,
        confidence: float,
        duration_ms: float,
        error: Optional[str] = None,
    ) -> None:
        """Persist one MLAssembler result for nightly Grid Search review."""
        with sqlite3.connect(self._db) as con:
            con.execute(
                "INSERT OR IGNORE INTO ml_results VALUES (?,?,?,?,?,?,?)",
                (result_id, task[:400], algorithm, confidence,
                 duration_ms, error or "", time.time()),
            )

    def get_ml_results(self, min_error: float = 0.15, days: int = 7) -> list[dict]:
        """Return ML results whose confidence is below (1 - min_error), for nightly sweep."""
        since = time.time() - days * 86400
        threshold = 1.0 - min_error
        with sqlite3.connect(self._db) as con:
            rows = con.execute(
                "SELECT result_id, task, algorithm, confidence, duration_ms, error, timestamp "
                "FROM ml_results WHERE confidence < ? AND timestamp >= ? ORDER BY timestamp DESC",
                (threshold, since),
            ).fetchall()
        return [
            {"result_id": r[0], "task": r[1], "algorithm": r[2],
             "confidence": r[3], "duration_ms": r[4], "error": r[5], "timestamp": r[6]}
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _kinetic_feedback(self, rec: OutcomeRecord) -> None:
        """
        After each chain outcome, update KineticEngine cross-domain link confidence.

        Done outcomes → slight boost (+0.05, capped 1.0) on active lever links.
        User-corrected → slight decay (−0.07, floored 0.0) — the compound signal
        fired but the answer was wrong, so the cross-domain diagnosis was off.
        Abandoned outcomes are treated as noise and ignored.
        """
        kinetic = self._kinetic
        if kinetic is None:
            return
        if rec.outcome == OUTCOME_ABANDONED:
            return
        try:
            windows = kinetic.active_windows(max_age_seconds=120.0)
            if not windows:
                return

            delta = +0.05 if rec.outcome == OUTCOME_DONE else -0.07
            # Only update links that contributed to recently active levers
            active_lever_ids = {w.lever_id for w in windows}
            for link in kinetic._links:
                # Check if this link's source domain contributed to an active lever
                contributing = any(
                    w.source_signal.domain in (link.source_domain, link.target_domain)
                    for w in windows
                    if w.lever_id in active_lever_ids
                )
                if contributing:
                    new_conf = max(0.0, min(1.0, link.confidence + delta))
                    kinetic.update_link_confidence(
                        link.source_domain, link.target_domain, new_conf
                    )
        except Exception as exc:
            logger.debug("[outcome_tracker] kinetic feedback failed: %s", exc)

    def _incremental_soul_update(self, rec: OutcomeRecord) -> None:
        if self._soul is None:
            return
        try:
            self._soul_update_for_record(rec, self._soul)
        except Exception as exc:
            logger.debug("[outcome_tracker] soul update failed: %s", exc)

    def _fulcrum_feedback(self, rec: OutcomeRecord) -> None:
        """Feed real outcome payoff into the live DecisionNetwork's AdaptiveFulcrums."""
        try:
            from prism_veax import get_current_network, observe_outcome
            net = get_current_network()
            if net is None:
                return
            # Map outcome string → actual payoff in [0, 1]
            payoff_map = {
                "done":           1.0,
                "abandoned":      0.1,
                "user_corrected": 0.3,
                "error":          0.0,
            }
            actual = payoff_map.get(rec.outcome, 0.5)
            # Predict from step efficiency: fewer steps for same goal → higher payoff
            predicted = min(1.0, 1.0 / max(1, rec.steps_count / 5)) if rec.steps_count else 0.5
            # Use policy_flags as a proxy for how cautious the position was (0=exec, 1=full_oversight)
            position = min(1.0, rec.policy_flags / 10.0) if rec.policy_flags else 0.5
            observe_outcome(net, logic="outcome_feedback", actual_payoff=actual,
                            predicted_payoff=predicted, chosen_position=position)
        except Exception as exc:
            logger.debug("[outcome_tracker] fulcrum feedback failed: %s", exc)

    def _soul_update_for_record(self, rec: OutcomeRecord, soul: PrismSoul) -> int:
        updates = 0
        if rec.outcome == OUTCOME_DONE:
            lenses = soul.list_lenses()
            keywords = _extract_keywords(rec.goal)
            for ln in lenses:
                if any(kw in ln.description.lower() for kw in keywords):
                    soul.record_observation(ln.lens_id, 1.0, f"chain completed: {rec.goal[:60]}")
                    updates += 1
            # Bayesian Beta(alpha, beta) confidence update for related stated beliefs
            beliefs = soul.list_beliefs(belief_type="stated")
            for b in beliefs:
                if any(kw in b.text.lower() for kw in keywords):
                    obs = getattr(b, "observation_count", 0) or 0
                    # Beta(successes+1, failures+1) mean — treat each completed chain as a success
                    alpha = obs + 2  # successes + 1 (prior)
                    beta_param = max(1, round(obs * (1 - b.confidence))) + 1  # failures + 1
                    new_conf = round(min(0.95, alpha / (alpha + beta_param)), 4)
                    soul.update_belief(b.node_id, confidence=new_conf, observation_count_delta=1)
                    updates += 1

        elif rec.outcome == OUTCOME_CORRECTED:
            lenses = soul.list_lenses()
            for ln in lenses:
                if "correction" in ln.name.lower() or "feedback" in ln.name.lower():
                    soul.record_observation(ln.lens_id, -1.0, f"user corrected: {rec.goal[:60]}")
                    updates += 1
        return updates

    def _row(self, row: tuple) -> OutcomeRecord:
        return OutcomeRecord(
            record_id   = row[0],
            chain_id    = row[1],
            goal        = row[2],
            outcome     = row[3],
            steps_count = row[4],
            duration_ms = row[5],
            policy_flags= row[6],
            final_answer= row[7],
            correction  = row[8],
            context_id  = row[9],
            timestamp   = row[10],
        )

    def _migrate(self, con: sqlite3.Connection) -> None:
        ver = con.execute("PRAGMA user_version").fetchone()[0]
        if ver < 1:
            cols = {r[1] for r in con.execute("PRAGMA table_info(outcomes)")}
            if "context_id" not in cols:
                con.execute("ALTER TABLE outcomes ADD COLUMN context_id TEXT NOT NULL DEFAULT 'default'")
            con.execute("PRAGMA user_version = 1")
        if ver < 2:
            con.execute("""
                CREATE TABLE IF NOT EXISTS ml_results (
                    result_id   TEXT PRIMARY KEY,
                    task        TEXT NOT NULL DEFAULT '',
                    algorithm   TEXT NOT NULL DEFAULT '',
                    confidence  REAL NOT NULL DEFAULT 0.0,
                    duration_ms REAL NOT NULL DEFAULT 0.0,
                    error       TEXT NOT NULL DEFAULT '',
                    timestamp   REAL NOT NULL
                )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_ml_ts ON ml_results(timestamp)")
            con.execute("PRAGMA user_version = 2")

    def _init_db(self) -> None:
        with sqlite3.connect(self._db) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS outcomes (
                    record_id    TEXT PRIMARY KEY,
                    chain_id     TEXT NOT NULL,
                    goal         TEXT NOT NULL,
                    outcome      TEXT NOT NULL,
                    steps_count  INTEGER NOT NULL DEFAULT 0,
                    duration_ms  REAL NOT NULL DEFAULT 0,
                    policy_flags INTEGER NOT NULL DEFAULT 0,
                    final_answer TEXT NOT NULL DEFAULT '',
                    correction   TEXT NOT NULL DEFAULT '',
                    context_id   TEXT NOT NULL DEFAULT 'default',
                    timestamp    REAL NOT NULL
                )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_ts ON outcomes(timestamp)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_ctx ON outcomes(context_id)")
            self._migrate(con)


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from a goal string for overlap matching."""
    stop = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "have", "has", "had", "will", "would", "could",
        "should", "may", "might", "can", "shall", "to", "of", "in", "on",
        "for", "with", "at", "by", "from", "as", "into", "about", "what",
        "when", "how", "why", "who", "which", "that", "this", "it", "i",
        "my", "me", "get", "set", "give", "make", "let", "go", "use",
    }
    words = re.findall(r"[a-z]{4,}", text.lower())
    return [w for w in words if w not in stop][:8]
