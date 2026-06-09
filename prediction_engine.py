"""
prediction_engine.py
====================
KDE Sports Platform — Prediction Engine

Provides sport-intelligence predictions using the existing KDE decision model
(ksa_lever, ksa_registry). All computation is local — no external APIs.

Predictors:
    MatchPredictor         → win/draw/loss probabilities
    InjuryRiskPredictor    → injury risk level and timeline
    PerformancePredictor   → expected performance rating / form trend
    TransferValuePredictor → market value band estimation
    TacticalPredictor      → matchup analysis and style prediction
    PredictionPlatform     → unified interface combining all predictors
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from ksa_lever import EquilibriumResult, ThreeBarSystem
from ksa_registry import SnapshotRegistry

# ---------------------------------------------------------------------------
# Core prediction dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    """Base prediction result."""
    subject:          str
    prediction:       str
    confidence:       float                      # 0–1
    distribution:     dict[str, float]           # outcome → probability
    expected_value:   float
    risk:             float                      # 0–1
    risk_adj:         float                      # expected_value * (1 − risk)
    fulcrum:          float                      # KDE lever fulcrum used
    key_factors:      list[tuple[str, float, str]]  # (name, weight, direction)


@dataclass
class MatchPrediction(Prediction):
    home_team:        str   = ""
    away_team:        str   = ""
    p_home_win:       float = 0.0
    p_draw:           float = 0.0
    p_away_win:       float = 0.0
    predicted_margin: float = 0.0


@dataclass
class InjuryRiskPrediction(Prediction):
    athlete_name:  str        = ""
    risk_level:    str        = "low"    # low|moderate|high|critical
    days_to_risk:  int        = 30
    recommendations: list[str] = field(default_factory=list)


@dataclass
class PerformancePrediction(Prediction):
    athlete_name:    str   = ""
    period:          str   = "next_7_days"
    expected_rating: float = 5.0
    form_trend:      str   = "stable"    # improving|stable|declining


@dataclass
class TransferPrediction(Prediction):
    athlete_name: str   = ""
    value_band:   str   = "3-5M"
    value_low_m:  float = 0.0
    value_high_m: float = 0.0

    def price_range(self) -> str:
        return f"£{self.value_low_m:.1f}M – £{self.value_high_m:.1f}M"


@dataclass
class TacticalPrediction:
    home_team:              str
    away_team:              str
    sport:                  str
    matchup_summary:        str
    home_advantage:         str
    away_advantage:         str
    key_duels:              list[dict] = field(default_factory=list)
    home_predicted_style:   str = ""
    away_predicted_style:   str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _lever_fulcrum(
    weights: list[float],
    registry: Optional[SnapshotRegistry] = None,
    task_name: str = "prediction",
) -> tuple[float, EquilibriumResult]:
    """Run a minimal three-bar simulation and return (fulcrum, eq_result)."""
    system = ThreeBarSystem.from_defaults()
    for i, w in enumerate(weights[:3]):
        left  = _clamp(w)
        right = _clamp(1.0 - w)
        system.levers[i].set_weights(left=left * 10, right=right * 10)
    eq = system.simulate()
    fulcrum = sum(lw * 0.33 for lw in weights[:3]) if weights else 0.5
    return fulcrum, eq


# ---------------------------------------------------------------------------
# MatchPredictor
# ---------------------------------------------------------------------------

class MatchPredictor:
    """
    Predicts win/draw/loss probabilities for a match between two teams.

    Factors accepted (all optional, 0–1 scale):
        home_form, away_form, home_fitness, away_fitness,
        home_advantage (default 0.6), head_to_head_home
    """

    def predict(
        self,
        home_team:       str,
        away_team:       str,
        sport:           str         = "football",
        home_form:       float       = 0.5,
        away_form:       float       = 0.5,
        home_fitness:    float       = 0.7,
        away_fitness:    float       = 0.7,
        home_advantage:  float       = 0.6,
        head_to_head:    float       = 0.5,
        **kwargs,
    ) -> MatchPrediction:

        # Raw scores (higher = stronger)
        home_score = (home_form * 0.35 + home_fitness * 0.25 +
                      home_advantage * 0.25 + head_to_head * 0.15)
        away_score = (away_form * 0.40 + away_fitness * 0.30 +
                      (1.0 - home_advantage) * 0.20 + (1.0 - head_to_head) * 0.10)

        total = home_score + away_score + 0.3  # 0.3 → draw mass
        p_home = home_score / total
        p_away = away_score / total
        p_draw = 0.3 / total

        # Normalise
        t = p_home + p_draw + p_away
        p_home /= t
        p_draw /= t
        p_away /= t

        confidence  = _clamp(abs(p_home - p_away) + 0.3)
        risk        = _clamp(1.0 - confidence)
        ev          = p_home * 1.0 + p_draw * 0.5 + p_away * 0.0
        risk_adj    = ev * (1.0 - risk * 0.5)
        margin      = (p_home - p_away) * 3.0   # rough goal-margin proxy

        fulcrum, _ = _lever_fulcrum([home_form, home_fitness, home_advantage])

        if p_home > p_draw and p_home > p_away:
            pred_str = f"{home_team} win"
        elif p_away > p_home and p_away > p_draw:
            pred_str = f"{away_team} win"
        else:
            pred_str = "Draw"

        key_factors = [
            ("home_form",      home_form,      "positive" if home_form > 0.5 else "negative"),
            ("away_form",      away_form,      "positive" if away_form > 0.5 else "negative"),
            ("home_advantage", home_advantage, "positive"),
        ]

        return MatchPrediction(
            subject          = f"{home_team} vs {away_team}",
            prediction       = pred_str,
            confidence       = round(confidence, 3),
            distribution     = {"home_win": round(p_home, 3),
                                 "draw":     round(p_draw, 3),
                                 "away_win": round(p_away, 3)},
            expected_value   = round(ev, 3),
            risk             = round(risk, 3),
            risk_adj         = round(risk_adj, 3),
            fulcrum          = round(fulcrum, 3),
            key_factors      = key_factors,
            home_team        = home_team,
            away_team        = away_team,
            p_home_win       = round(p_home, 3),
            p_draw           = round(p_draw, 3),
            p_away_win       = round(p_away, 3),
            predicted_margin = round(margin, 2),
        )


# ---------------------------------------------------------------------------
# InjuryRiskPredictor
# ---------------------------------------------------------------------------

class InjuryRiskPredictor:
    """
    Predicts injury risk for an athlete based on load, recovery, and fitness.

    Factors (all optional, 0–1 scale):
        recovery_score, load_7d, muscle_soreness, sleep_quality, age_factor
    """

    _THRESHOLDS = [
        (0.75, "critical",  7,  ["Immediate rest", "Physio assessment today",
                                  "No training for 48h"]),
        (0.55, "high",      14, ["Reduce load by 40%", "Daily mobility work",
                                  "Monitor soreness"]),
        (0.35, "moderate",  21, ["Reduce intensity by 20%", "Add recovery session",
                                  "Monitor HRV"]),
        (0.0,  "low",       30, ["Maintain current load", "Continue monitoring"]),
    ]

    def predict(
        self,
        athlete_name:   str,
        recovery_score: float = 0.7,
        load_7d:        float = 0.5,
        muscle_soreness: float = 0.3,
        sleep_quality:  float = 0.7,
        age_factor:     float = 0.5,
        **kwargs,
    ) -> InjuryRiskPrediction:

        # Higher risk when: low recovery, high load, high soreness, poor sleep
        risk_raw = (
            (1.0 - recovery_score) * 0.30 +
            load_7d               * 0.30 +
            muscle_soreness       * 0.25 +
            (1.0 - sleep_quality) * 0.10 +
            age_factor            * 0.05
        )
        risk = _clamp(risk_raw)

        level, days, recs = "low", 30, []
        for threshold, lv, d, rs in self._THRESHOLDS:
            if risk >= threshold:
                level, days, recs = lv, d, rs
                break

        confidence = _clamp(0.5 + abs(risk - 0.5))
        ev         = 1.0 - risk
        risk_adj   = ev * (1.0 - risk * 0.3)
        fulcrum, _ = _lever_fulcrum([recovery_score, 1.0 - load_7d, sleep_quality])

        return InjuryRiskPrediction(
            subject          = athlete_name,
            prediction       = f"{level} injury risk",
            confidence       = round(confidence, 3),
            distribution     = {"low": round(1 - risk, 3), "high": round(risk, 3)},
            expected_value   = round(ev, 3),
            risk             = round(risk, 3),
            risk_adj         = round(risk_adj, 3),
            fulcrum          = round(fulcrum, 3),
            key_factors      = [
                ("recovery_score",   recovery_score,   "positive" if recovery_score > 0.5 else "negative"),
                ("load_7d",          load_7d,          "negative" if load_7d > 0.6 else "neutral"),
                ("muscle_soreness",  muscle_soreness,  "negative" if muscle_soreness > 0.4 else "neutral"),
            ],
            athlete_name     = athlete_name,
            risk_level       = level,
            days_to_risk     = days,
            recommendations  = recs,
        )


# ---------------------------------------------------------------------------
# PerformancePredictor
# ---------------------------------------------------------------------------

class PerformancePredictor:
    """
    Predicts athlete performance rating for the coming period.

    Factors (all optional, 0–1 scale):
        recent_form, fitness_level, recovery_score,
        motivation, training_load_quality
    """

    def predict(
        self,
        athlete_name:          str,
        period:                str   = "next_7_days",
        recent_form:           float = 0.6,
        fitness_level:         float = 0.7,
        recovery_score:        float = 0.7,
        motivation:            float = 0.7,
        training_load_quality: float = 0.6,
        **kwargs,
    ) -> PerformancePrediction:

        raw = (
            recent_form           * 0.35 +
            fitness_level         * 0.25 +
            recovery_score        * 0.20 +
            motivation            * 0.12 +
            training_load_quality * 0.08
        )
        raw = _clamp(raw)

        expected_rating = round(raw * 10.0, 1)   # 0–10 scale

        if recent_form > 0.65:
            trend = "improving"
        elif recent_form < 0.45:
            trend = "declining"
        else:
            trend = "stable"

        confidence = _clamp(0.4 + abs(raw - 0.5))
        risk       = _clamp(1.0 - raw)
        ev         = raw
        risk_adj   = ev * (1.0 - risk * 0.2)
        fulcrum, _ = _lever_fulcrum([recent_form, fitness_level, recovery_score])

        return PerformancePrediction(
            subject          = athlete_name,
            prediction       = f"Rating {expected_rating}/10 ({trend})",
            confidence       = round(confidence, 3),
            distribution     = {"below_avg": round(risk * 0.5, 3),
                                 "average":   round(0.3, 3),
                                 "above_avg": round(raw * 0.5, 3)},
            expected_value   = round(ev, 3),
            risk             = round(risk, 3),
            risk_adj         = round(risk_adj, 3),
            fulcrum          = round(fulcrum, 3),
            key_factors      = [
                ("recent_form",   recent_form,   "positive" if recent_form > 0.5 else "negative"),
                ("fitness_level", fitness_level, "positive" if fitness_level > 0.5 else "negative"),
                ("motivation",    motivation,    "positive" if motivation > 0.6 else "neutral"),
            ],
            athlete_name     = athlete_name,
            period           = period,
            expected_rating  = expected_rating,
            form_trend       = trend,
        )


# ---------------------------------------------------------------------------
# TransferValuePredictor
# ---------------------------------------------------------------------------

class TransferValuePredictor:
    """
    Estimates transfer market value band for an athlete.

    Factors:
        performance_score (0–1), age (int), contract_years (int),
        injury_record (0–1; 0=clean, 1=frequent), league_tier (0–1)
    """

    _BANDS = [
        (0.85, "50M+",   50.0, 150.0),
        (0.70, "20-50M", 20.0,  50.0),
        (0.55, "10-20M", 10.0,  20.0),
        (0.40, "5-10M",   5.0,  10.0),
        (0.25, "2-5M",    2.0,   5.0),
        (0.0,  "0-2M",    0.0,   2.0),
    ]

    def predict(
        self,
        athlete_name:     str,
        performance_score: float = 0.6,
        age:              int   = 24,
        contract_years:   int   = 2,
        injury_record:    float = 0.2,
        league_tier:      float = 0.5,
        **kwargs,
    ) -> TransferPrediction:

        # Age penalty: prime 22–27, reduces outside
        age_factor = 1.0 - (abs(age - 24.5) / 20.0)
        age_factor = _clamp(age_factor)

        raw = (
            performance_score * 0.40 +
            age_factor        * 0.20 +
            league_tier       * 0.20 +
            (1.0 - injury_record) * 0.15 +
            min(contract_years / 5.0, 1.0) * 0.05
        )
        raw = _clamp(raw)

        band, low, high = "0-2M", 0.0, 2.0
        for threshold, b, lo, hi in self._BANDS:
            if raw >= threshold:
                band, low, high = b, lo, hi
                break

        confidence = _clamp(0.4 + performance_score * 0.3 + league_tier * 0.3)
        risk       = _clamp(injury_record * 0.5 + (1.0 - age_factor) * 0.3)
        ev         = (low + high) / 2.0
        risk_adj   = ev * (1.0 - risk * 0.4)
        fulcrum, _ = _lever_fulcrum([performance_score, age_factor, league_tier])

        return TransferPrediction(
            subject        = athlete_name,
            prediction     = f"Transfer value band: £{band}",
            confidence     = round(confidence, 3),
            distribution   = {"value_band": band},
            expected_value = round(ev, 1),
            risk           = round(risk, 3),
            risk_adj       = round(risk_adj, 1),
            fulcrum        = round(fulcrum, 3),
            key_factors    = [
                ("performance_score", performance_score, "positive" if performance_score > 0.5 else "negative"),
                ("age",              float(age),         "positive" if 21 <= age <= 27 else "neutral"),
                ("injury_record",    injury_record,      "negative" if injury_record > 0.3 else "positive"),
            ],
            athlete_name   = athlete_name,
            value_band     = band,
            value_low_m    = low,
            value_high_m   = high,
        )


# ---------------------------------------------------------------------------
# TacticalPredictor
# ---------------------------------------------------------------------------

class TacticalPredictor:
    """
    Generates a tactical matchup analysis between two teams.

    All analysis is rule-based using the supplied factor dicts.
    Typical factor keys (all optional):
        pressing_intensity, possession_style, counter_attack,
        set_piece_quality, defensive_line
    """

    _STYLES = {
        "pressing":   ("high pressing, quick transitions, compact shape",
                       "gegenpressing, vertical play"),
        "possession": ("patient build-up, positional play, high line",
                       "tiki-taka, recycling possession"),
        "counter":    ("deep block, fast counters, direct play",
                       "4-4-2 low block, long balls"),
        "default":    ("balanced approach, mixed transitions",
                       "flexible 4-3-3 or 4-2-3-1"),
    }

    def _style_key(self, factors: dict) -> str:
        pressing   = factors.get("pressing_intensity", 0.5)
        possession = factors.get("possession_style", 0.5)
        counter    = factors.get("counter_attack", 0.5)
        if pressing > 0.65:
            return "pressing"
        if possession > 0.65:
            return "possession"
        if counter > 0.65:
            return "counter"
        return "default"

    def predict(
        self,
        home_team:     str,
        away_team:     str,
        sport:         str  = "football",
        home_factors: Optional[dict] = None,
        away_factors: Optional[dict] = None,
    ) -> TacticalPrediction:
        hf = home_factors or {}
        af = away_factors or {}

        h_style_key = self._style_key(hf)
        a_style_key = self._style_key(af)
        h_style, _  = self._STYLES.get(h_style_key, self._STYLES["default"])
        a_style, _  = self._STYLES.get(a_style_key, self._STYLES["default"])

        # Home advantage
        h_press = hf.get("pressing_intensity", 0.5)
        a_press = af.get("pressing_intensity", 0.5)
        h_poss  = hf.get("possession_style", 0.5)

        home_advantage = (
            "High pressing and home crowd support"
            if h_press > 0.6 else
            "Possession control and tactical flexibility"
            if h_poss > 0.6 else
            "Home environment and set-piece threat"
        )
        away_advantage = (
            "Counter-attacking pace on transition"
            if af.get("counter_attack", 0.5) > 0.6 else
            "Defensive organisation and compactness"
            if af.get("defensive_line", 0.5) < 0.4 else
            "Away form and individual quality"
        )

        key_duels = [
            {"zone": "midfield", "home_advantage": round(h_poss, 2),
             "away_advantage": round(a_press, 2), "importance": "high"},
            {"zone": "defensive_line", "home_advantage": round(hf.get("defensive_line", 0.5), 2),
             "away_advantage": round(af.get("counter_attack", 0.5), 2), "importance": "medium"},
        ]

        summary = (
            f"{home_team} ({h_style_key} style) host {away_team} "
            f"({a_style_key} style) in a {sport} fixture. "
            f"Key battle expected in midfield."
        )

        return TacticalPrediction(
            home_team             = home_team,
            away_team             = away_team,
            sport                 = sport,
            matchup_summary       = summary,
            home_advantage        = home_advantage,
            away_advantage        = away_advantage,
            key_duels             = key_duels,
            home_predicted_style  = h_style,
            away_predicted_style  = a_style,
        )


# ---------------------------------------------------------------------------
# PredictionPlatform
# ---------------------------------------------------------------------------

class PredictionPlatform:
    """
    Unified interface combining all sport predictors.

    Usage:
        platform = PredictionPlatform()
        mp = platform.match.predict("Arsenal", "Chelsea", home_form=0.8)
        brief = platform.pre_match_brief("Arsenal", "Chelsea", "football", squad, hf, af)
    """

    def __init__(self, registry: Optional[SnapshotRegistry] = None) -> None:
        self._registry  = registry
        self.match      = MatchPredictor()
        self.injury     = InjuryRiskPredictor()
        self.performance = PerformancePredictor()
        self.transfer   = TransferValuePredictor()
        self.tactical   = TacticalPredictor()

    def pre_match_brief(
        self,
        home_team:     str,
        away_team:     str,
        sport:         str        = "football",
        squad: Optional[list[dict]] = None,
        home_factors: Optional[dict]       = None,
        away_factors: Optional[dict]       = None,
    ) -> dict:
        """
        Generate a full pre-match intelligence brief.

        squad: list of dicts with keys:
            name, recovery_score, load_7d, muscle_soreness,
            recent_form, fitness_level, performance_score, age

        Returns a dict with keys:
            match_prediction, tactical_analysis,
            squad_risk, squad_performance, generated_at
        """
        squad    = squad    or []
        home_factors  = home_factors  or {}
        away_factors  = away_factors  or {}

        match_pred = self.match.predict(
            home_team, away_team, sport,
            home_form   = home_factors.get("form", 0.5),
            away_form   = away_factors.get("form", 0.5),
            home_fitness= home_factors.get("fitness", 0.7),
            away_fitness= away_factors.get("fitness", 0.7),
            home_advantage = home_factors.get("home_advantage", 0.6),
        )

        tactical_pred = self.tactical.predict(
            home_team, away_team, sport, home_factors, away_factors
        )

        squad_risk:  list[InjuryRiskPrediction]  = []
        squad_perf:  list[PerformancePrediction]  = []

        for player in squad:
            name = player.get("name", "Unknown")
            irp  = self.injury.predict(
                name,
                recovery_score  = player.get("recovery_score",  0.7),
                load_7d         = player.get("load_7d",         0.5),
                muscle_soreness = player.get("muscle_soreness", 0.3),
            )
            pp   = self.performance.predict(
                name,
                recent_form   = player.get("recent_form",   0.6),
                fitness_level = player.get("fitness_level", 0.7),
                recovery_score= player.get("recovery_score", 0.7),
            )
            squad_risk.append(irp)
            squad_perf.append(pp)

        return {
            "match_prediction":  match_pred,
            "tactical_analysis": tactical_pred,
            "squad_risk":        squad_risk,
            "squad_performance": squad_perf,
            "generated_at":      datetime.now(tz=timezone.utc).isoformat(),
        }
