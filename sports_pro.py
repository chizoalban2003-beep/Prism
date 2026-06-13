"""
sports_pro.py
=============
KDE Sports Agent — Sports Practitioner Profiles & Daily Planning

Manages athlete/coach/practitioner profiles, plans daily sessions using
the KDE decision model, and learns from every session through fixed-fulcrum
drift (exponential moving average, α = 0.05).

SQLite tables:
    profiles      — registered practitioner profiles
    daily_ratings — day ratings that drive fulcrum learning
    perf_metrics  — performance metric time-series
    session_plans — generated daily plan records
"""

from __future__ import annotations

import json
import logging
import os
import random
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from ksa_lever import EquilibriumResult, ThreeBarSystem, TiltDirection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALPHA = 0.05          # EMA learning rate for fixed-fulcrum drift
_DEFAULT_DB = "~/.kde/sports_pro.db"


# ---------------------------------------------------------------------------
# Role definitions
# ---------------------------------------------------------------------------

class Role(str, Enum):
    ATHLETE          = "athlete"
    COACH            = "coach"
    PHYSIOTHERAPIST  = "physiotherapist"
    ANALYST          = "analyst"
    PERFORMANCE_DIR  = "performance_director"
    AGENT            = "agent"
    NUTRITIONIST     = "nutritionist"
    PSYCHOLOGIST     = "psychologist"


# Default fixed_fulcrum per role (0.3–0.7 range; lower = more proactive)
ROLE_BASES: dict[Role, float] = {
    Role.ATHLETE:         0.40,
    Role.COACH:           0.50,
    Role.PHYSIOTHERAPIST: 0.62,
    Role.ANALYST:         0.45,
    Role.PERFORMANCE_DIR: 0.50,
    Role.AGENT:           0.55,
    Role.NUTRITIONIST:    0.52,
    Role.PSYCHOLOGIST:    0.60,
}

# Daily task pools per role: (time_slot, title, duration_min, category)
ROLE_PLANKS: dict[Role, list[tuple]] = {
    Role.ATHLETE: [
        ("06:00", "Morning activation", 20,  "movement_prep"),
        ("07:00", "Strength & conditioning", 60, "physical"),
        ("09:00", "Tactical film review", 45,  "cognitive"),
        ("10:00", "Technical drills", 60,      "skill"),
        ("12:00", "Nutrition & hydration", 30, "recovery"),
        ("14:00", "Team training session", 90, "physical"),
        ("17:00", "Cool-down & stretch", 30,   "recovery"),
        ("18:00", "Recovery protocol", 45,     "recovery"),
        ("20:00", "Mental preparation", 20,    "cognitive"),
        ("21:30", "Sleep prep routine", 30,    "recovery"),
    ],
    Role.COACH: [
        ("06:30", "Session planning", 45,      "planning"),
        ("08:00", "Staff briefing", 30,        "communication"),
        ("09:00", "Individual player meetings", 60, "communication"),
        ("10:30", "Training session delivery", 90,  "physical"),
        ("13:00", "Video analysis", 60,        "cognitive"),
        ("15:00", "Performance review", 45,    "analysis"),
        ("16:30", "Tactical preparation", 60,  "planning"),
        ("18:30", "Scouting review", 45,       "analysis"),
        ("20:00", "Match preparation notes", 30, "planning"),
    ],
    Role.PHYSIOTHERAPIST: [
        ("07:00", "Morning assessments", 60,   "assessment"),
        ("08:30", "Injury rehabilitation", 90, "treatment"),
        ("11:00", "Pre-training screening", 45,"assessment"),
        ("13:00", "Treatment sessions", 90,    "treatment"),
        ("15:30", "Post-training recovery", 60,"treatment"),
        ("17:00", "Load monitoring review", 30,"analysis"),
        ("18:00", "Documentation", 30,         "admin"),
    ],
    Role.ANALYST: [
        ("07:00", "Data ingestion & cleaning", 60, "data"),
        ("09:00", "Performance metrics review", 90,"analysis"),
        ("11:00", "Opposition scouting", 60,   "analysis"),
        ("14:00", "Model validation", 90,       "analysis"),
        ("16:00", "Report generation", 60,      "reporting"),
        ("17:30", "Stakeholder briefing", 45,   "communication"),
        ("19:00", "Next-day data prep", 30,     "data"),
    ],
    Role.PERFORMANCE_DIR: [
        ("07:30", "KPI review", 45,            "analysis"),
        ("09:00", "Staff coordination", 60,    "communication"),
        ("10:30", "Budget & resource planning", 60, "planning"),
        ("13:00", "Athlete review meetings", 90,"communication"),
        ("15:30", "Stakeholder report", 60,    "reporting"),
        ("17:00", "Strategic planning", 60,    "planning"),
    ],
    Role.AGENT: [
        ("08:00", "Client check-in", 30,       "communication"),
        ("09:00", "Contract review", 60,       "admin"),
        ("11:00", "Media & brand management", 45, "communication"),
        ("14:00", "Club negotiations", 90,     "negotiation"),
        ("16:30", "Market analysis", 45,       "analysis"),
        ("18:00", "Network building", 30,      "communication"),
    ],
    Role.NUTRITIONIST: [
        ("07:00", "Morning nutrition check", 30,  "assessment"),
        ("08:30", "Meal planning & prep", 60,     "planning"),
        ("10:30", "Athlete consultations", 90,    "communication"),
        ("13:00", "Supplement protocol review", 45,"analysis"),
        ("15:00", "Post-training nutrition", 30,  "treatment"),
        ("16:30", "Data analysis", 60,            "analysis"),
        ("18:30", "Documentation", 30,            "admin"),
    ],
    Role.PSYCHOLOGIST: [
        ("08:00", "Morning mindfulness session", 30, "wellness"),
        ("09:30", "Individual sessions", 90,        "treatment"),
        ("12:00", "Group session", 60,              "treatment"),
        ("14:00", "Performance anxiety work", 60,   "treatment"),
        ("16:00", "Case notes & reports", 60,       "admin"),
        ("18:00", "Relaxation protocol", 30,        "wellness"),
    ],
}

# Contextual factor schemas per role: list of {name, key, scale, direction}
# key maps to a DailyContext field; direction is "reduce" (high value → fewer tasks)
# or "expand" (high value → more tasks); scale is the sensitivity weight.
ROLE_FACTORS: dict[Role, list[dict]] = {
    Role.ATHLETE: [
        {"name": "recovery",      "key": "recovery_score",   "scale": 0.02, "direction": "expand"},
        {"name": "soreness",      "key": "muscle_soreness",  "scale": 0.08, "direction": "reduce"},
        {"name": "sleep_quality", "key": "sleep_quality",    "scale": 1.5,  "direction": "expand"},
        {"name": "load",          "key": "load_7d",          "scale": 0.01, "direction": "reduce"},
        {"name": "readiness",     "key": "mental_readiness", "scale": 0.08, "direction": "expand"},
        {"name": "proximity",     "key": "_match_proximity", "scale": 2.0,  "direction": "reduce"},
    ],
    Role.COACH: [
        {"name": "energy",    "key": "mental_readiness", "scale": 0.08, "direction": "expand"},
        {"name": "sleep",     "key": "sleep_quality",    "scale": 1.5,  "direction": "expand"},
        {"name": "load",      "key": "load_7d",          "scale": 0.005,"direction": "reduce"},
        {"name": "proximity", "key": "_match_proximity", "scale": 2.5,  "direction": "reduce"},
    ],
    Role.PHYSIOTHERAPIST: [
        {"name": "readiness", "key": "mental_readiness", "scale": 0.08, "direction": "expand"},
        {"name": "sleep",     "key": "sleep_quality",    "scale": 1.5,  "direction": "expand"},
        {"name": "load",      "key": "load_7d",          "scale": 0.005,"direction": "reduce"},
    ],
    Role.ANALYST: [
        {"name": "focus",     "key": "mental_readiness", "scale": 0.10, "direction": "expand"},
        {"name": "sleep",     "key": "sleep_quality",    "scale": 1.5,  "direction": "expand"},
        {"name": "proximity", "key": "_match_proximity", "scale": 2.0,  "direction": "reduce"},
    ],
    Role.PERFORMANCE_DIR: [
        {"name": "readiness", "key": "mental_readiness", "scale": 0.08, "direction": "expand"},
        {"name": "sleep",     "key": "sleep_quality",    "scale": 1.2,  "direction": "expand"},
        {"name": "proximity", "key": "_match_proximity", "scale": 1.5,  "direction": "reduce"},
    ],
    Role.AGENT: [
        {"name": "energy",    "key": "mental_readiness", "scale": 0.08, "direction": "expand"},
        {"name": "sleep",     "key": "sleep_quality",    "scale": 1.2,  "direction": "expand"},
    ],
    Role.NUTRITIONIST: [
        {"name": "readiness", "key": "mental_readiness", "scale": 0.08, "direction": "expand"},
        {"name": "sleep",     "key": "sleep_quality",    "scale": 1.5,  "direction": "expand"},
        {"name": "load",      "key": "load_7d",          "scale": 0.005,"direction": "reduce"},
    ],
    Role.PSYCHOLOGIST: [
        {"name": "wellbeing", "key": "mental_readiness", "scale": 0.10, "direction": "expand"},
        {"name": "sleep",     "key": "sleep_quality",    "scale": 1.5,  "direction": "expand"},
        {"name": "recovery",  "key": "recovery_score",   "scale": 0.01, "direction": "expand"},
    ],
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SportsProProfile:
    name:          str
    role:          Role
    sport:         str
    team:          str
    fixed_fulcrum: float = 0.0

    def __post_init__(self) -> None:
        if self.fixed_fulcrum == 0.0:
            self.fixed_fulcrum = ROLE_BASES.get(self.role, 0.50)


@dataclass
class DailyContext:
    """All biometric and contextual signals that shape the day's plan."""
    recovery_score:   float = 70.0   # 0–100
    sleep_quality:    float = 0.75   # 0–1
    sleep_hrs:        float = 7.5
    muscle_soreness:  float = 3.0    # 0–10
    load_7d:          float = 250.0  # arbitrary load units
    mental_readiness: float = 7.0    # 0–10
    days_to_match:    float = 7.0
    season_phase:     str   = "in_season"   # pre_season|in_season|off_season|tournament

    def match_proximity(self) -> float:
        """0.0 = far away, 1.0 = match today."""
        days = max(0.0, self.days_to_match)
        if days == 0:
            return 1.0
        return max(0.0, 1.0 - days / 14.0)

    def to_factor_values(self, role: Role) -> dict:
        """Convert this context to factor values for the given role's factors."""
        raw = {
            "recovery_score":   self.recovery_score,
            "sleep_quality":    self.sleep_quality,
            "sleep_hrs":        self.sleep_hrs,
            "muscle_soreness":  self.muscle_soreness,
            "load_7d":          self.load_7d,
            "mental_readiness": self.mental_readiness,
            "days_to_match":    self.days_to_match,
            "_match_proximity": self.match_proximity(),
        }
        factors = ROLE_FACTORS.get(role, [])
        result: dict = {}
        for f in factors:
            key = f["key"]
            result[f["name"]] = raw.get(key, 0.0) * f["scale"]
        return result


@dataclass
class DailyTask:
    time_slot:    str
    duration_min: int
    category:     str
    title:        str
    notes:        str = ""


@dataclass
class DailyPlan:
    primary_focus: str
    activation:    float         # 0–1: how active the day should be
    fulcrum:       float         # lever fulcrum used to compute this plan
    tasks:         list[DailyTask] = field(default_factory=list)
    warnings:      list[str]     = field(default_factory=list)
    rationale:     str           = ""


@dataclass
class WearableReading:
    source:          str
    hrv_ms:          float
    hrv_baseline_ms: float
    sleep_hrs:       float
    sleep_score:     float
    body_battery:    float

    def to_daily_context(self, existing: Optional[DailyContext] = None) -> DailyContext:
        """Convert wearable data to a DailyContext, merging with any existing context."""
        ctx = existing or DailyContext()

        # HRV ratio → recovery score
        ratio = min(2.0, self.hrv_ms / max(1.0, self.hrv_baseline_ms))
        ctx.recovery_score = min(100.0, ratio * 70.0)

        ctx.sleep_hrs     = self.sleep_hrs
        ctx.sleep_quality = min(1.0, self.sleep_score / 100.0)
        ctx.mental_readiness = min(10.0, (self.body_battery / 100.0) * 10.0)

        # Derive muscle soreness proxy from body battery (inverse)
        ctx.muscle_soreness = max(0.0, 10.0 - (self.body_battery / 100.0) * 10.0)

        return ctx


# ---------------------------------------------------------------------------
# WearableReader
# ---------------------------------------------------------------------------

class WearableReader:
    """Factory for WearableReading objects."""

    @staticmethod
    def mock(seed: int = 42) -> WearableReading:
        """Return a deterministic mock reading for testing."""
        rng = random.Random(seed)
        baseline = 60.0
        hrv      = baseline + rng.uniform(-15, 25)
        return WearableReading(
            source          = "mock",
            hrv_ms          = hrv,
            hrv_baseline_ms = baseline,
            sleep_hrs       = rng.uniform(6.5, 9.0),
            sleep_score     = rng.uniform(55.0, 95.0),
            body_battery    = rng.uniform(40.0, 95.0),
        )

    @staticmethod
    def manual(
        hrv_ms:       float,
        sleep_hrs:    float,
        soreness:     float,
        energy:       float,
        baseline_hrv: float = 60.0,
    ) -> WearableReading:
        """Build a reading from manually entered values."""
        sleep_score   = min(100.0, (sleep_hrs / 9.0) * 100.0)
        body_battery  = min(100.0, energy * 10.0)
        return WearableReading(
            source          = "manual",
            hrv_ms          = hrv_ms,
            hrv_baseline_ms = baseline_hrv,
            sleep_hrs       = sleep_hrs,
            sleep_score     = sleep_score,
            body_battery    = body_battery,
        )


# ---------------------------------------------------------------------------
# DailyPlanner
# ---------------------------------------------------------------------------

class DailyPlanner:
    """
    Computes a DailyPlan from a SportsProProfile and DailyContext.

    Uses the lever system to calculate an activation level, then selects
    and orders tasks from the role's plank pool.
    """

    _FOCUS_MAP = {
        "in_season":   "Match performance",
        "pre_season":  "Fitness building",
        "off_season":  "Rest & recovery",
        "tournament":  "Peak performance",
    }

    def plan(self, profile: SportsProProfile, ctx: DailyContext) -> DailyPlan:
        system   = ThreeBarSystem.from_defaults()
        fv       = ctx.to_factor_values(profile.role)
        factors  = ROLE_FACTORS.get(profile.role, [])

        left_w  = 0.0
        right_w = 0.0
        for f in factors:
            val = fv.get(f["name"], 0.0)
            if f["direction"] == "expand":
                left_w += val
            else:
                right_w += val

        # Apply profile's learned fulcrum bias
        system.levers[1].fulcrum_bias = max(-4.0, min(4.0, (profile.fixed_fulcrum - 0.5) * 8.0))

        left_w  = max(0.1, left_w)
        right_w = max(0.1, right_w)
        system.levers[0].set_weights(left=left_w, right=right_w)

        eq: EquilibriumResult = system.simulate()

        # Activation: LEFT tilt → high activity, RIGHT → recovery, BALANCED → moderate
        if eq.final_tilt == TiltDirection.LEFT:
            activation = 0.7 + eq.confidence * 0.3
        elif eq.final_tilt == TiltDirection.RIGHT:
            activation = 0.3 - eq.confidence * 0.2
        else:
            activation = 0.5

        activation = max(0.0, min(1.0, activation))

        warnings = self._build_warnings(ctx, profile.role)
        tasks    = self._select_tasks(profile.role, ctx, activation)
        focus    = self._primary_focus(profile.role, ctx, activation)
        rationale = (
            f"Lever tilt: {eq.final_tilt.value} (confidence {eq.confidence:.0%}). "
            f"Recovery: {ctx.recovery_score:.0f}%, "
            f"Sleep: {ctx.sleep_hrs:.1f}h, "
            f"Soreness: {ctx.muscle_soreness:.1f}/10. "
            f"Season phase: {ctx.season_phase}."
        )

        return DailyPlan(
            primary_focus = focus,
            activation    = activation,
            fulcrum       = profile.fixed_fulcrum,
            tasks         = tasks,
            warnings      = warnings,
            rationale     = rationale,
        )

    # ── private helpers ──────────────────────────────────────────────────

    def _build_warnings(self, ctx: DailyContext, role: Role) -> list[str]:
        w: list[str] = []
        if ctx.recovery_score < 40:
            w.append("⚠ Low recovery score — consider reducing load")
        if ctx.sleep_hrs < 6.0:
            w.append("⚠ Short sleep — prioritise rest today")
        if ctx.muscle_soreness > 7:
            w.append("⚠ High muscle soreness — avoid intense physical work")
        if ctx.load_7d > 400 and role == Role.ATHLETE:
            w.append("⚠ High 7-day load — injury risk elevated")
        if ctx.match_proximity() > 0.85:
            w.append("⚠ Match tomorrow — use activation protocol only")
        return w

    def _primary_focus(
        self, role: Role, ctx: DailyContext, activation: float
    ) -> str:
        base = self._FOCUS_MAP.get(ctx.season_phase, "Training")
        if activation < 0.35:
            return f"Recovery & restoration ({base})"
        if activation > 0.75 and ctx.match_proximity() > 0.7:
            return f"Match preparation ({base})"
        return base

    def _select_tasks(
        self, role: Role, ctx: DailyContext, activation: float
    ) -> list[DailyTask]:
        planks  = ROLE_PLANKS.get(role, [])
        recovery_day = activation < 0.35 or ctx.recovery_score < 45

        selected: list[DailyTask] = []
        for slot, title, duration, category in planks:
            skip = False
            if recovery_day and category == "physical" and ctx.muscle_soreness > 6:
                skip = True
            if ctx.match_proximity() > 0.9 and category == "physical":
                # Day before match: activation only
                duration = min(duration, 30)
            if skip:
                continue
            notes = ""
            if category == "recovery" and ctx.recovery_score < 50:
                notes = "Priority: low recovery score"
            selected.append(
                DailyTask(
                    time_slot    = slot,
                    duration_min = duration,
                    category     = category,
                    title        = title,
                    notes        = notes,
                )
            )

        return selected


# ---------------------------------------------------------------------------
# SportsProAssistant
# ---------------------------------------------------------------------------

class SportsProAssistant:
    """
    Top-level sports practitioner assistant.

    Manages profiles in SQLite, plans days using DailyPlanner,
    and learns from session ratings via fixed-fulcrum drift.
    """

    def __init__(self, db_path: str = _DEFAULT_DB) -> None:
        self._db_path = str(Path(db_path).expanduser())
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._planner  = DailyPlanner()
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS profiles (
                name          TEXT PRIMARY KEY,
                role          TEXT NOT NULL,
                sport         TEXT NOT NULL,
                team          TEXT DEFAULT '',
                fixed_fulcrum REAL NOT NULL,
                created_at    TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS daily_ratings (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                date_str    TEXT NOT NULL,
                rating      REAL NOT NULL,
                notes       TEXT DEFAULT '',
                created_at  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS perf_metrics (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                metric      TEXT NOT NULL,
                value       REAL NOT NULL,
                unit        TEXT DEFAULT '',
                recorded_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS session_plans (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                date_str    TEXT NOT NULL,
                plan_json   TEXT NOT NULL,
                context_json TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );
            """)

    # ── profile management ────────────────────────────────────────────────

    def register(self, profile: SportsProProfile) -> str:
        """Register a practitioner profile. Returns the name."""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO profiles
                    (name, role, sport, team, fixed_fulcrum, created_at)
                VALUES (?,?,?,?,?,?)
                """,
                (
                    profile.name,
                    profile.role.value,
                    profile.sport,
                    profile.team,
                    profile.fixed_fulcrum,
                    _now_iso(),
                ),
            )
        logger.info("Registered profile: %s (%s, %s)", profile.name, profile.role.value, profile.sport)
        return profile.name

    def get_profile(self, name: str) -> tuple[str, SportsProProfile]:
        """Return (status, profile). Status is 'ok' or 'not_found'."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM profiles WHERE name=?", (name,)
            ).fetchone()
        if row is None:
            return ("not_found", None)
        profile = SportsProProfile(
            name          = row["name"],
            role          = Role(row["role"]),
            sport         = row["sport"],
            team          = row["team"],
            fixed_fulcrum = row["fixed_fulcrum"],
        )
        return ("ok", profile)

    # ── daily planning ────────────────────────────────────────────────────

    def plan_day(
        self,
        name:    str,
        ctx:     Optional[DailyContext]     = None,
        reading: Optional[WearableReading]  = None,
    ) -> DailyPlan:
        """Generate today's plan for a profile.

        If *reading* is provided it is converted to a DailyContext (and merged
        with any supplied *ctx*). If neither is provided, a default context is used.
        """
        status, profile = self.get_profile(name)
        if status != "ok":
            raise KeyError(f"Profile '{name}' not found")

        if reading is not None:
            ctx = reading.to_daily_context(ctx)
        if ctx is None:
            ctx = DailyContext()

        plan = self._planner.plan(profile, ctx)

        # Persist the plan
        date_str = date.today().isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO session_plans (id, name, date_str, plan_json, context_json, created_at)
                VALUES (?,?,?,?,?,?)
                """,
                (
                    str(uuid.uuid4()),
                    name,
                    date_str,
                    json.dumps(asdict(plan), default=str),
                    json.dumps(asdict(ctx)),
                    _now_iso(),
                ),
            )
        return plan

    # ── learning & feedback ───────────────────────────────────────────────

    def rate_day(
        self,
        name:     str,
        date_str: str,
        rating:   float,
        notes:    str = "",
    ) -> None:
        """Log a day rating (0–5) and update the profile's fixed_fulcrum via EMA.

        Rating  > 3.0 → plan was too light  → nudge fulcrum toward more activity
        Rating  < 3.0 → plan was too heavy  → nudge fulcrum toward more recovery
        """
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO daily_ratings (id, name, date_str, rating, notes, created_at)
                VALUES (?,?,?,?,?,?)
                """,
                (str(uuid.uuid4()), name, date_str, rating, notes, _now_iso()),
            )
            # Drift the fixed_fulcrum
            row = conn.execute(
                "SELECT fixed_fulcrum FROM profiles WHERE name=?", (name,)
            ).fetchone()
            if row:
                current   = row["fixed_fulcrum"]
                # rating >= 3.0 → plan was well-tolerated → target more activity (0.4)
                # rating <  3.0 → plan was too hard/stressful → target more recovery (0.6)
                target    = 0.4 if rating >= 3.0 else 0.6
                new_value = current + _ALPHA * (target - current)
                new_value = max(0.2, min(0.8, new_value))
                conn.execute(
                    "UPDATE profiles SET fixed_fulcrum=? WHERE name=?",
                    (new_value, name),
                )
                logger.debug(
                    "Fulcrum drift for '%s': %.3f → %.3f (rating=%.1f)",
                    name, current, new_value, rating,
                )

    def log_metric(
        self,
        name:   str,
        metric: str,
        value:  float,
        unit:   str = "",
    ) -> None:
        """Store a single performance metric data point."""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO perf_metrics (id, name, metric, value, unit, recorded_at)
                VALUES (?,?,?,?,?,?)
                """,
                (str(uuid.uuid4()), name, metric, value, unit, _now_iso()),
            )

    # ── history & reflection ──────────────────────────────────────────────

    def history(self, name: str, days: int = 14) -> list[dict]:
        """Return daily_ratings and session_plans for the last *days* days."""
        with self._conn() as conn:
            ratings = conn.execute(
                """
                SELECT date_str, rating, notes FROM daily_ratings
                WHERE name=? ORDER BY created_at DESC LIMIT ?
                """,
                (name, days),
            ).fetchall()
            plans = conn.execute(
                """
                SELECT date_str, plan_json FROM session_plans
                WHERE name=? ORDER BY created_at DESC LIMIT ?
                """,
                (name, days),
            ).fetchall()

        result: list[dict] = []
        for r in ratings:
            result.append(
                {
                    "type":     "rating",
                    "date":     r["date_str"],
                    "rating":   r["rating"],
                    "notes":    r["notes"],
                }
            )
        for p in plans:
            plan_data = json.loads(p["plan_json"])
            result.append(
                {
                    "type":   "plan",
                    "date":   p["date_str"],
                    "focus":  plan_data.get("primary_focus", ""),
                    "tasks":  len(plan_data.get("tasks", [])),
                }
            )
        result.sort(key=lambda x: x.get("date", ""), reverse=True)
        return result

    def reflect(self, name: str) -> dict:
        """Return a summary of what the system has learned about the practitioner."""
        status, profile = self.get_profile(name)
        if status != "ok":
            return {"error": f"Profile '{name}' not found"}

        with self._conn() as conn:
            ratings = conn.execute(
                "SELECT rating FROM daily_ratings WHERE name=? ORDER BY created_at DESC LIMIT 30",
                (name,),
            ).fetchall()
            n_plans = conn.execute(
                "SELECT COUNT(*) as c FROM session_plans WHERE name=?", (name,)
            ).fetchone()["c"]

        avg_rating = (
            sum(r["rating"] for r in ratings) / len(ratings)
            if ratings else None
        )
        return {
            "profile":       profile.name,
            "role":          profile.role.value,
            "sport":         profile.sport,
            "fixed_fulcrum": round(profile.fixed_fulcrum, 4),
            "fulcrum_trend": (
                "high_activity" if profile.fixed_fulcrum < 0.45
                else "recovery_focused" if profile.fixed_fulcrum > 0.55
                else "balanced"
            ),
            "avg_day_rating": round(avg_rating, 2) if avg_rating is not None else None,
            "total_ratings":  len(ratings),
            "total_plans":    n_plans,
        }

    # ── helpers ───────────────────────────────────────────────────────────

    def list_profiles(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT name FROM profiles ORDER BY name").fetchall()
        return [r["name"] for r in rows]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    print("=== SportsProAssistant Demo ===\n")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    assistant = SportsProAssistant(db_path)

    profile = SportsProProfile(name="Marcus", role=Role.ATHLETE, sport="Football", team="City FC")
    assistant.register(profile)

    reading = WearableReader.mock(seed=7)
    print(f"HRV: {reading.hrv_ms:.1f} ms  Sleep: {reading.sleep_hrs:.1f}h  Battery: {reading.body_battery:.0f}%")

    plan = assistant.plan_day("Marcus", reading=reading)
    print(f"\nPrimary focus : {plan.primary_focus}")
    print(f"Activation    : {plan.activation:.0%}")
    print(f"Tasks         : {len(plan.tasks)}")
    for t in plan.tasks[:3]:
        print(f"  {t.time_slot}  {t.title}  ({t.duration_min} min)")
    if plan.warnings:
        for w in plan.warnings:
            print(f"  {w}")
    print(f"\nRationale: {plan.rationale}")

    assistant.rate_day("Marcus", date.today().isoformat(), rating=4.2)
    print("\nReflection:", assistant.reflect("Marcus"))
    os.unlink(db_path)
