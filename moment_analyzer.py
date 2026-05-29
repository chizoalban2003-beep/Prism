"""
moment_analyzer.py
==================
KDE Moment Platform — Core Moment Analysis Engine

Uses Kernel Density Estimation (KDE) to evaluate decision options at critical
game moments.  For each moment the analyzer places a Gaussian kernel at the
player's contextual focal point and scores every option by its kernel weight
multiplied by the option's contextual expected value.

Public API
----------
MomentOption       — one selectable action (name / position / payoff / cost / risk / base_prob)
MomentSportConfig  — kernel config for one (sport, moment_type) pair
ALL_MOMENT_CONFIGS — global mutable registry: dict[(sport, moment_type), MomentSportConfig]
NearbyPlayer       — a player in the focal player's vicinity
Moment             — snapshot of a live game situation
ActionOutcome      — what actually happened (used for calibration)
MomentResult       — full analysis output
MomentAnalyzer     — the analysis engine
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from decision_spectrum import (
    AdaptiveFulcrum,
    DecisionBeam,
    DecisionPlank,
    Factor,
    OutcomeDiagnosis,
)


# ---------------------------------------------------------------------------
# Core data classes
# ---------------------------------------------------------------------------

@dataclass
class MomentOption:
    """One selectable decision at a game moment."""
    name:       str
    position:   float   # 0.0 = safest / most conservative → 1.0 = most aggressive
    payoff:     float   # reward units if action succeeds
    cost:       float   # cost units to attempt the action
    risk:       float   # additional penalty loading (0–100)
    base_prob:  float   # baseline success probability (0–1)


@dataclass
class MomentSportConfig:
    """KDE configuration for one (sport, moment_type) pair."""
    sport:       str
    moment_type: str
    options:     list[MomentOption]
    bandwidth:   float = 0.15   # Gaussian bandwidth (0.12–0.22)


@dataclass
class NearbyPlayer:
    """A player near the focal player at moment time."""
    name:          str
    team:          str
    distance:      float        # yards (batch) or metres (live)
    arrival_time:  float        # estimated seconds to reach ball position
    x:             float = 0.0
    y:             float = 0.0
    speed:         float = 0.0
    is_goalkeeper: bool  = False


@dataclass
class Moment:
    """Full snapshot of a live game situation to be analysed."""
    moment_id:           str
    match_id:            str
    sport:               str
    moment_type:         str
    timestamp:           float
    focal_player:        str
    focal_profile:       str    # role / position label (e.g. "Winger", "Striker")
    focal_team:          str
    focal_base:          float  # 0–1: player's baseline aggressiveness from profile
    pitch_x:             float  # 0–1 (0 = own goal end, 1 = opponent goal end)
    pitch_y:             float  # 0–1
    primary_opponent:    Optional[NearbyPlayer] = None
    secondary_opponents: list[NearbyPlayer] = field(default_factory=list)
    teammates:           list[NearbyPlayer] = field(default_factory=list)
    fatigue:             float = 0.0          # 0–1 (0 = fully fresh)
    confidence:          float = 0.5          # 0–1
    score_pressure:      float = 0.0          # negative = losing, positive = winning / urgent
    xg_raw:              float = 0.0          # raw xG from position alone
    custom_factors:      dict  = field(default_factory=dict)


@dataclass
class ActionOutcome:
    """What actually happened after the moment resolved (for calibration)."""
    action_taken: str
    success:      bool
    xg_delta:     float = 0.0
    notes:        str   = ""


@dataclass
class MomentResult:
    """Full output of MomentAnalyzer.analyze()."""
    moment:         Moment
    recommended:    str                  # name of the highest-scoring option
    xg_contextual:  float                # KDE-adjusted expected value of best option
    option_scores:  dict[str, float]     # option_name → score
    focal_position: float                # computed focal point used for the kernel
    config:         MomentSportConfig
    activations:    list[tuple[str, float, float]] = field(default_factory=list)
    time_pressure:  float = 0.0


# ---------------------------------------------------------------------------
# Analyser
# ---------------------------------------------------------------------------

class MomentAnalyzer:
    def __init__(self) -> None:
        self._fulcrums: dict[str, AdaptiveFulcrum] = {}
        self._calibration: dict[str, dict] = {}

    def _get_fulcrum(self, player: str, moment_type: str) -> AdaptiveFulcrum:
        key = f"{player}:{moment_type}"
        if key not in self._fulcrums:
            self._fulcrums[key] = AdaptiveFulcrum(learning_rate=0.04,
                                                   weight_min=0.10, weight_max=8.0)
        return self._fulcrums[key]

    def analyze(self, moment) -> MomentResult:
        cfg = ALL_MOMENT_CONFIGS.get((moment.sport, moment.moment_type))
        if cfg is None:
            raise KeyError(f"No config for ({moment.sport}, {moment.moment_type})")
        focal = self._compute_focal(moment)
        adaptive = self._get_fulcrum(moment.focal_player, moment.moment_type)
        adaptive.add_factor(Factor("_focal_anchor", 1.0, 2.0, focal))
        beam = DecisionBeam(moment.moment_id, bandwidth=cfg.bandwidth, fulcrum=adaptive)
        for opt in cfg.options:
            beam.add_plank(DecisionPlank(opt.name, opt.position, opt.payoff,
                                          opt.cost, opt.risk, opt.base_prob))
        diag = beam.evaluate()
        option_scores = {a.plank.name: a.activation * a.plank.payoff * a.plank.probability
                         for a in diag.activations}
        activations = [(a.plank.name, a.activation,
                        a.activation * a.plank.payoff * a.plank.probability)
                       for a in diag.activations]
        shoot_acts = [a for a in diag.activations
                      if any(w in a.plank.name.lower()
                             for w in ("shoot", "shot", "cross", "post", "corner", "driven"))]
        xg = 0.0
        if shoot_acts:
            tot = sum(a.activation for a in shoot_acts)
            wp = sum(a.activation * a.plank.probability for a in shoot_acts) / max(tot, 1e-9)
            xg = (moment.xg_raw + wp) / 2.0 if moment.xg_raw > 0 else wp
        tp = 0.0
        if moment.secondary_opponents:
            fastest = min(moment.secondary_opponents, key=lambda p: p.arrival_time)
            tp = max(0.0, 1.0 - fastest.arrival_time / 4.0)
        return MomentResult(
            moment=moment,
            recommended=diag.primary_plank.name,
            xg_contextual=round(xg, 3),
            option_scores=option_scores,
            focal_position=focal,
            config=cfg,
            activations=activations,
            time_pressure=tp,
        )

    def calibrate(self, moment, outcome) -> None:
        cfg = ALL_MOMENT_CONFIGS.get((moment.sport, moment.moment_type))
        if cfg:
            chosen = next((o for o in cfg.options if o.name == outcome.action_taken), None)
            if chosen:
                actual = chosen.payoff * (1.0 if outcome.success else 0.15)
                if hasattr(outcome, "xg_realized") and outcome.xg_realized > 0:
                    actual = outcome.xg_realized * chosen.payoff
                self._get_fulcrum(moment.focal_player, moment.moment_type).observe(
                    actual, chosen.payoff * chosen.base_prob, chosen.position)
        player = moment.focal_player
        if player not in self._calibration:
            self._calibration[player] = {"total": 0, "success": 0, "last_action": None}
        c = self._calibration[player]
        c["total"] += 1
        if outcome.success:
            c["success"] += 1
        c["last_action"] = outcome.action_taken

    def player_stats(self, player: str) -> dict:
        c = self._calibration.get(player, {"total": 0, "success": 0, "last_action": None})
        return {**c, "success_rate": c["success"] / c["total"] if c["total"] > 0 else 0.0}

    def _compute_focal(self, moment: Moment) -> float:
        """Derive the KDE focal position (0–1) from moment context."""
        focal = moment.focal_base
        # Confidence nudges towards more aggressive choices
        focal += (moment.confidence - 0.5) * 0.20
        # Fatigue reduces willingness to take risk
        focal -= moment.fatigue * 0.15
        # Score pressure: positive (winning/urgent) encourages action
        focal += moment.score_pressure * 0.10
        # Attacking position encourages more aggressive decisions
        focal += (moment.pitch_x - 0.5) * 0.10
        # High raw xG means a good position → favour decisive options
        focal += moment.xg_raw * 0.15
        # Very tight marking by primary opponent forces conservative choice
        if (
            moment.primary_opponent is not None
            and moment.primary_opponent.distance < 2.0
        ):
            focal -= 0.08
        return max(0.02, min(0.98, focal))

    def _adjusted_prob(self, opt: MomentOption, moment: Moment) -> float:
        """Return base_prob adjusted for context and custom factors."""
        cf = moment.custom_factors
        prob = opt.base_prob
        # Confidence modifier
        prob *= 1.0 + (moment.confidence - 0.5) * 0.40
        # Fatigue penalty
        prob *= 1.0 - moment.fatigue * 0.30
        # Custom factor boosts ----------------------------------------
        # Open roll man in basketball boosts roll-related options
        if cf.get("open_roll_man") and "roll" in opt.name.lower():
            prob *= 1.15
        # Close support players boost offload actions (rugby/football)
        if cf.get("support_close") and "offload" in opt.name.lower():
            prob *= 1.20
        # Open court in tennis boosts decisive (non-defensive) shots
        if cf.get("open_court") and opt.position > 0.40:
            prob *= 1.05
        return max(0.05, min(0.99, prob))


# ---------------------------------------------------------------------------
# Base configuration registry
# ---------------------------------------------------------------------------

ALL_MOMENT_CONFIGS: dict[tuple[str, str], MomentSportConfig] = {
    # ── Football ─────────────────────────────────────────────────────────
    ("Football", "1v1_keeper"): MomentSportConfig(
        sport="Football", moment_type="1v1_keeper", bandwidth=0.15,
        options=[
            MomentOption("Chip keeper",       0.00, 120, 6, 35, 0.45),
            MomentOption("Near post low",     0.17,  90, 5, 20, 0.65),
            MomentOption("Far post driven",   0.33, 100, 5, 25, 0.58),
            MomentOption("Side-foot placed",  0.50,  85, 5, 20, 0.70),
            MomentOption("Driven near post",  0.67,  95, 6, 30, 0.60),
            MomentOption("Cut back pass",     0.83,  80, 4, 15, 0.72),
            MomentOption("Power shot",        1.00, 110, 6, 40, 0.50),
        ],
    ),
    ("Football", "winger_cross"): MomentSportConfig(
        sport="Football", moment_type="winger_cross", bandwidth=0.16,
        options=[
            MomentOption("Hold and recycle",  0.00,  20, 3, 10, 0.94),
            MomentOption("Low driven cross",  0.20,  75, 5, 22, 0.72),
            MomentOption("Whipped in-swinger",0.40,  90, 5, 28, 0.64),
            MomentOption("Cut-back",          0.60,  85, 5, 24, 0.68),
            MomentOption("Lofted far post",   0.78,  80, 5, 30, 0.60),
            MomentOption("Pull-back penalty", 1.00, 130, 6, 40, 0.50),
        ],
    ),
    ("Football", "penalty"): MomentSportConfig(
        sport="Football", moment_type="penalty", bandwidth=0.14,
        options=[
            MomentOption("Down the middle",   0.00,  80, 4, 15, 0.78),
            MomentOption("Low left",          0.20,  90, 4, 20, 0.74),
            MomentOption("Low right",         0.40,  90, 4, 20, 0.74),
            MomentOption("High left",         0.60, 100, 4, 28, 0.70),
            MomentOption("High right",        0.80, 100, 4, 28, 0.68),
            MomentOption("Panenka",           1.00, 120, 5, 45, 0.76),
        ],
    ),
    # ── Basketball ───────────────────────────────────────────────────────
    ("Basketball", "drive_to_basket"): MomentSportConfig(
        sport="Basketball", moment_type="drive_to_basket", bandwidth=0.16,
        options=[
            MomentOption("Pull back dribble", 0.00,  20, 3, 10, 0.94),
            MomentOption("Kick out 3",        0.17,  90, 4, 22, 0.76),
            MomentOption("Floater",           0.33,  70, 5, 24, 0.58),
            MomentOption("Lay-up",            0.50,  85, 4, 20, 0.72),
            MomentOption("Euro step",         0.67,  90, 5, 26, 0.66),
            MomentOption("Dunk",              0.83, 110, 5, 30, 0.62),
            MomentOption("And-one",           1.00, 130, 6, 38, 0.50),
        ],
    ),
    ("Basketball", "isolation"): MomentSportConfig(
        sport="Basketball", moment_type="isolation", bandwidth=0.17,
        options=[
            MomentOption("Pass out reset",    0.00,  15, 3,  8, 0.96),
            MomentOption("Dribble pull-up",   0.20,  65, 5, 22, 0.55),
            MomentOption("Mid-range",         0.40,  75, 5, 25, 0.52),
            MomentOption("Drive kick",        0.60,  90, 5, 28, 0.62),
            MomentOption("Step-back 3",       0.80, 115, 6, 38, 0.42),
            MomentOption("Fadeway",           1.00,  90, 6, 35, 0.46),
        ],
    ),
    # ── Tennis ───────────────────────────────────────────────────────────
    ("Tennis", "approach_shot"): MomentSportConfig(
        sport="Tennis", moment_type="approach_shot", bandwidth=0.18,
        options=[
            MomentOption("Stay back",           0.00,  10, 3,  5, 0.96),
            MomentOption("Defensive slice",     0.20,  40, 4, 12, 0.86),
            MomentOption("Deep topspin",        0.40,  70, 5, 20, 0.76),
            MomentOption("Approach slice",      0.58,  90, 5, 26, 0.68),
            MomentOption("Aggressive approach", 0.75, 110, 6, 34, 0.58),
            MomentOption("Outright winner",     1.00, 140, 7, 50, 0.42),
        ],
    ),
    # ── Boxing ───────────────────────────────────────────────────────────
    ("Boxing", "in_range"): MomentSportConfig(
        sport="Boxing", moment_type="in_range", bandwidth=0.16,
        options=[
            MomentOption("Clinch",              0.00,  15, 3, 10, 0.92),
            MomentOption("Jab to create space", 0.20,  50, 4, 15, 0.80),
            MomentOption("Jab-cross combo",     0.40,  80, 5, 22, 0.68),
            MomentOption("Body shot",           0.58,  90, 5, 26, 0.62),
            MomentOption("Hook",                0.75, 110, 6, 34, 0.54),
            MomentOption("Uppercut",            1.00, 130, 7, 44, 0.44),
        ],
    ),
    ("Boxing", "counter"): MomentSportConfig(
        sport="Boxing", moment_type="counter", bandwidth=0.15,
        options=[
            MomentOption("Slip and clinch",     0.00,  20, 3, 10, 0.88),
            MomentOption("Parry and jab",       0.20,  60, 4, 16, 0.78),
            MomentOption("Duck under hook",     0.40,  80, 5, 22, 0.70),
            MomentOption("Counter right cross", 0.60, 110, 6, 30, 0.60),
            MomentOption("Left hook counter",   0.80, 120, 6, 36, 0.52),
            MomentOption("Overhand right",      1.00, 140, 7, 46, 0.40),
        ],
    ),
    # ── MMA ──────────────────────────────────────────────────────────────
    ("MMA", "clinch"): MomentSportConfig(
        sport="MMA", moment_type="clinch", bandwidth=0.17,
        options=[
            MomentOption("Disengage",           0.00,  10, 3,  8, 0.94),
            MomentOption("Knee to body",        0.20,  70, 5, 20, 0.72),
            MomentOption("Dirty boxing",        0.40,  85, 5, 26, 0.64),
            MomentOption("Single leg attempt",  0.58, 100, 6, 32, 0.56),
            MomentOption("Hip throw",           0.75, 120, 7, 38, 0.48),
            MomentOption("Slam attempt",        1.00, 150, 8, 52, 0.36),
        ],
    ),
    ("MMA", "ground_top_position"): MomentSportConfig(
        sport="MMA", moment_type="ground_top_position", bandwidth=0.18,
        options=[
            MomentOption("Control and pace",    0.00,  15, 3,  8, 0.96),
            MomentOption("Ground and pound",    0.20,  80, 5, 24, 0.68),
            MomentOption("Pass guard",          0.40,  95, 6, 30, 0.58),
            MomentOption("Armbar attempt",      0.60, 130, 7, 40, 0.46),
            MomentOption("Choke attempt",       0.80, 140, 7, 44, 0.42),
            MomentOption("Heavy ground pound",  1.00, 150, 8, 50, 0.38),
        ],
    ),
    # ── Wrestling ────────────────────────────────────────────────────────
    ("Wrestling", "takedown_attempt"): MomentSportConfig(
        sport="Wrestling", moment_type="takedown_attempt", bandwidth=0.16,
        options=[
            MomentOption("Feint / setup",       0.00,  15, 3,  8, 0.94),
            MomentOption("Ankle pick",          0.20,  70, 5, 22, 0.68),
            MomentOption("Single leg",          0.40,  90, 6, 28, 0.60),
            MomentOption("Double leg",          0.60, 110, 6, 34, 0.54),
            MomentOption("High crotch",         0.80, 100, 6, 30, 0.56),
            MomentOption("Blast double",        1.00, 120, 7, 42, 0.46),
        ],
    ),
}
