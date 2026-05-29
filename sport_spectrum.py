from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from decision_spectrum import (
    DecisionBeam,
    DecisionPlank,
    Factor,
    OutcomeDiagnosis,
    SpectrumFulcrum,
)

logger = logging.getLogger(__name__)


@dataclass
class PlankTemplate:
    name: str
    position: float
    payoff: float
    cost: float
    risk: float
    probability: float = 0.7


@dataclass
class FactorTemplate:
    id: str
    label: str
    weight: float
    direction: float
    range: float
    description: str = ""


@dataclass
class SportProfile:
    name: str
    fixed_fulcrum: float
    description: str = ""


@dataclass
class SportConfig:
    name: str
    planks: list[PlankTemplate]
    factors: list[FactorTemplate]
    profiles: list[SportProfile]
    bandwidth: float = 0.18


class SportDecisionModel:
    def __init__(self, config: SportConfig):
        self.config = copy.deepcopy(config)

    def make_beam(
        self,
        profile_name: str,
        factor_values: dict[str, float] = None,
    ) -> DecisionBeam:
        profile = next(p for p in self.config.profiles if p.name == profile_name)
        beam = DecisionBeam(
            f"{self.config.name}_{profile_name}",
            bandwidth=self.config.bandwidth,
            fulcrum=SpectrumFulcrum(),
        )
        for pt in self.config.planks:
            beam.add_plank(
                DecisionPlank(
                    pt.name,
                    pt.position,
                    pt.payoff,
                    pt.cost,
                    pt.risk,
                    pt.probability,
                )
            )
        fv = factor_values or {}
        for ft in self.config.factors:
            val = max(0.0, min(1.0, fv.get(ft.id, 0.5)))
            if ft.direction > 0:
                target = min(1.0, profile.fixed_fulcrum + val * ft.range)
            else:
                target = max(0.0, profile.fixed_fulcrum - val * ft.range)
            beam.fulcrum.add_factor(Factor(ft.id, val, ft.weight, target, ft.description))
        beam.fulcrum.add_factor(
            Factor(
                "_base",
                1.0,
                2.0,
                profile.fixed_fulcrum,
                f"Profile: {profile.description}",
            )
        )
        return beam

    def evaluate(
        self,
        profile_name: str,
        factor_values: dict[str, float] = None,
    ) -> OutcomeDiagnosis:
        return self.make_beam(profile_name, factor_values).evaluate()


@dataclass
class DuelOutcome:
    attacker_fulcrum: float
    defender_fulcrum: float
    attacker_decision: str
    defender_decision: str
    attacker_activation: float
    defender_activation: float
    attacker_ev: float
    defender_ev: float
    iterations: int
    advantage: str


class DuelModel:
    def __init__(
        self,
        sport_model: SportDecisionModel,
        coupling_strength: float = 0.7,
        max_iterations: int = 20,
        tolerance: float = 1e-5,
    ):
        self.sport_model = sport_model
        self.coupling_strength = coupling_strength
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def simulate(
        self,
        attacker_profile: str,
        defender_profile: str,
        attacker_context: dict = None,
        defender_context: dict = None,
    ) -> DuelOutcome:
        a_ctx = dict(attacker_context or {})
        d_ctx = dict(defender_context or {})
        a_beam = self.sport_model.make_beam(attacker_profile, a_ctx)
        d_beam = self.sport_model.make_beam(defender_profile, d_ctx)
        a_pos = a_beam.fulcrum.position()
        d_pos = d_beam.fulcrum.position()
        iteration = 0
        for iteration in range(self.max_iterations):
            a_beam = self.sport_model.make_beam(attacker_profile, a_ctx)
            d_beam = self.sport_model.make_beam(defender_profile, d_ctx)
            cs = self.coupling_strength
            a_beam.fulcrum.add_factor(
                Factor(
                    "_duel_pressure",
                    d_pos * cs,
                    cs * 1.5,
                    max(0, a_pos - d_pos * 0.4),
                    "Defender pressure",
                )
            )
            d_beam.fulcrum.add_factor(
                Factor(
                    "_duel_threat",
                    a_pos * cs,
                    cs * 1.2,
                    min(1, d_pos + a_pos * 0.3),
                    "Attacker threat",
                )
            )
            new_a = a_beam.fulcrum.position()
            new_d = d_beam.fulcrum.position()
            if abs(new_a - a_pos) + abs(new_d - d_pos) < self.tolerance:
                a_pos, d_pos = new_a, new_d
                break
            a_pos, d_pos = new_a, new_d
        a_diag = a_beam.evaluate()
        d_diag = d_beam.evaluate()
        gap = a_pos - d_pos
        if gap > 0.15:
            advantage = "attacker"
        elif gap < -0.15:
            advantage = "defender"
        else:
            advantage = "contested"
        return DuelOutcome(
            attacker_fulcrum=a_pos,
            defender_fulcrum=d_pos,
            attacker_decision=a_diag.primary_plank.name,
            defender_decision=d_diag.primary_plank.name,
            attacker_activation=a_diag.activations[0].activation,
            defender_activation=d_diag.activations[0].activation,
            attacker_ev=a_diag.expected_net,
            defender_ev=d_diag.expected_net,
            iterations=iteration + 1,
            advantage=advantage,
        )


@dataclass
class ChainLink:
    player_name: str
    profile: str
    fulcrum: float
    decision: str
    activation: float
    expected_net: float
    context_out: dict


@dataclass
class ChainOutcome:
    links: list[ChainLink]
    final_decision: str
    chain_ev: float


class PossessionChain:
    AGGRESSION = 0.60
    MEDIUM = 0.38

    def __init__(self, sport_model: SportDecisionModel):
        self.sport_model = sport_model

    def simulate(
        self,
        sequence: list[tuple[str, str]],
        initial_context: dict = None,
    ) -> ChainOutcome:
        ctx = dict(initial_context or {})
        links = []
        for player_name, profile in sequence:
            beam = self.sport_model.make_beam(profile, ctx)
            diag = beam.evaluate()
            pos = beam.fulcrum.position()
            cat = (
                "aggressive"
                if pos >= self.AGGRESSION
                else "medium"
                if pos >= self.MEDIUM
                else "conservative"
            )
            new_ctx = dict(ctx)
            if cat == "aggressive":
                new_ctx["pitch_zone"] = min(1.0, ctx.get("pitch_zone", 0.3) + 0.30)
                new_ctx["xg"] = min(1.0, ctx.get("xg", 0.1) + 0.20)
            elif cat == "medium":
                new_ctx["pitch_zone"] = min(1.0, ctx.get("pitch_zone", 0.3) + 0.12)
            else:
                new_ctx["pitch_zone"] = max(0.0, ctx.get("pitch_zone", 0.3) - 0.08)
            links.append(
                ChainLink(
                    player_name,
                    profile,
                    pos,
                    diag.primary_plank.name,
                    diag.activations[0].activation,
                    diag.expected_net,
                    new_ctx,
                )
            )
            ctx = new_ctx
        return ChainOutcome(
            links,
            links[-1].decision if links else "",
            sum(l.expected_net for l in links),
        )


FOOTBALL = SportConfig(
    name="Football",
    bandwidth=0.17,
    planks=[
        PlankTemplate("Defend/clear", 0.00, 15, 3, 8, 0.98),
        PlankTemplate("Back pass", 0.14, 55, 5, 10, 0.93),
        PlankTemplate("Lateral", 0.28, 80, 7, 18, 0.88),
        PlankTemplate("Safe fwd", 0.43, 110, 9, 25, 0.82),
        PlankTemplate("Carry", 0.57, 145, 11, 38, 0.65),
        PlankTemplate("Through ball", 0.71, 195, 14, 58, 0.48),
        PlankTemplate("Shot", 0.86, 265, 17, 65, 0.38),
        PlankTemplate("Finish", 1.00, 350, 20, 45, 0.70),
    ],
    factors=[
        FactorTemplate("pitch_zone", "Pitch zone", 3.5, +1, 0.80),
        FactorTemplate("xg", "Expected goals", 2.5, +1, 0.60),
        FactorTemplate("press", "Pressure", 1.5, -1, 0.40),
        FactorTemplate("support", "Support", 1.2, -1, 0.35),
        FactorTemplate("scoreline", "Scoreline", 1.0, +1, 0.25),
        FactorTemplate("fatigue", "Fatigue", 0.8, -1, 0.15),
    ],
    profiles=[
        SportProfile("Goalkeeper", 0.04),
        SportProfile("Centre back", 0.13),
        SportProfile("Full back", 0.27),
        SportProfile("Def. mid", 0.33),
        SportProfile("Box-to-box", 0.50),
        SportProfile("Att. mid", 0.63),
        SportProfile("Winger", 0.71),
        SportProfile("Striker", 0.82),
    ],
)

BASKETBALL = SportConfig(
    name="Basketball",
    bandwidth=0.18,
    planks=[
        PlankTemplate("Dribble back", 0.00, 10, 2, 5, 0.99),
        PlankTemplate("Kick out", 0.14, 60, 4, 12, 0.90),
        PlankTemplate("Corner 3 pass", 0.28, 90, 5, 22, 0.82),
        PlankTemplate("Mid-range", 0.43, 120, 8, 35, 0.65),
        PlankTemplate("Drive base", 0.57, 150, 9, 42, 0.60),
        PlankTemplate("Drive rim", 0.71, 200, 11, 50, 0.55),
        PlankTemplate("Logo 3", 0.86, 240, 14, 80, 0.35),
        PlankTemplate("And-one", 1.00, 280, 16, 55, 0.50),
    ],
    factors=[
        FactorTemplate("shot_clock", "Shot clock", 3.0, +1, 0.75),
        FactorTemplate("coverage", "Coverage", 2.5, -1, 0.55),
        FactorTemplate("court_zone", "Court zone", 2.0, +1, 0.70),
        FactorTemplate("score_margin", "Score margin", 1.5, +1, 0.30),
        FactorTemplate("foul_trouble", "Foul trouble", 1.0, -1, 0.25),
        FactorTemplate("hot_hand", "Hot hand", 0.9, +1, 0.20),
    ],
    profiles=[
        SportProfile("Point guard", 0.52),
        SportProfile("Shooting guard", 0.70),
        SportProfile("Small forward", 0.62),
        SportProfile("Power forward", 0.45),
        SportProfile("Centre", 0.38),
    ],
)

TENNIS = SportConfig(
    name="Tennis",
    bandwidth=0.20,
    planks=[
        PlankTemplate("Def lob", 0.00, 20, 3, 8, 0.96),
        PlankTemplate("Deep cross", 0.16, 55, 5, 15, 0.88),
        PlankTemplate("Slice", 0.32, 65, 6, 18, 0.85),
        PlankTemplate("Neutral", 0.47, 85, 7, 22, 0.80),
        PlankTemplate("Down line", 0.62, 140, 10, 38, 0.62),
        PlankTemplate("Drop shot", 0.75, 160, 12, 55, 0.55),
        PlankTemplate("Aggressive cross", 0.87, 200, 14, 60, 0.48),
        PlankTemplate("Winner", 1.00, 250, 16, 70, 0.38),
    ],
    factors=[
        FactorTemplate("score_pressure", "Score pressure", 3.0, +1, 0.65),
        FactorTemplate("surface", "Surface", 2.0, +1, 0.45),
        FactorTemplate("opp_position", "Opponent position", 2.5, +1, 0.55),
        FactorTemplate("energy", "Energy", 1.5, -1, 0.35),
        FactorTemplate("rally_length", "Rally length", 1.2, +1, 0.30),
        FactorTemplate("wind", "Wind", 0.8, -1, 0.20),
    ],
    profiles=[
        SportProfile("Defensive baseliner", 0.22),
        SportProfile("Aggressive baseliner", 0.62),
        SportProfile("All-court", 0.50),
        SportProfile("Serve-volley", 0.78),
        SportProfile("Counter-puncher", 0.30),
    ],
)

RUGBY_UNION = SportConfig(
    name="Rugby Union",
    bandwidth=0.19,
    planks=[
        PlankTemplate("Kick to touch", 0.00, 25, 4, 10, 0.92),
        PlankTemplate("Pick and drive", 0.14, 50, 5, 18, 0.88),
        PlankTemplate("Recycle", 0.28, 70, 6, 22, 0.84),
        PlankTemplate("Pop pass", 0.42, 100, 8, 28, 0.78),
        PlankTemplate("Flat pass", 0.56, 135, 9, 35, 0.70),
        PlankTemplate("Switch", 0.68, 165, 11, 45, 0.60),
        PlankTemplate("Cross-kick", 0.82, 200, 14, 58, 0.48),
        PlankTemplate("Chip", 1.00, 240, 16, 70, 0.40),
    ],
    factors=[
        FactorTemplate("field_pos", "Field position", 3.5, +1, 0.80),
        FactorTemplate("phase_count", "Phase count", 2.0, -1, 0.40),
        FactorTemplate("def_line", "Defensive line", 2.0, -1, 0.45),
        FactorTemplate("score", "Score", 1.5, +1, 0.30),
        FactorTemplate("fatigue", "Fatigue", 1.0, -1, 0.20),
        FactorTemplate("setpiece", "Set piece", 0.8, +1, 0.25),
    ],
    profiles=[
        SportProfile("Loosehead prop", 0.12),
        SportProfile("Hooker", 0.18),
        SportProfile("Number 8", 0.42),
        SportProfile("Scrum half", 0.38),
        SportProfile("Fly half", 0.58),
        SportProfile("Inside centre", 0.50),
        SportProfile("Outside centre", 0.60),
        SportProfile("Wing", 0.72),
    ],
)

AMERICAN_FOOTBALL = SportConfig(
    name="American Football",
    bandwidth=0.19,
    planks=[
        PlankTemplate("Kneel", 0.00, 5, 1, 2, 0.99),
        PlankTemplate("QB sneak", 0.14, 40, 4, 12, 0.85),
        PlankTemplate("Short run", 0.28, 70, 5, 18, 0.80),
        PlankTemplate("Screen", 0.42, 95, 7, 25, 0.75),
        PlankTemplate("Play action", 0.56, 145, 9, 35, 0.65),
        PlankTemplate("Crossing route", 0.68, 190, 11, 45, 0.58),
        PlankTemplate("Deep route", 0.82, 280, 14, 75, 0.38),
        PlankTemplate("Hail Mary", 1.00, 420, 18, 95, 0.18),
    ],
    factors=[
        FactorTemplate("down_distance", "Down and distance", 3.5, +1, 0.80),
        FactorTemplate("field_pos", "Field position", 2.5, +1, 0.65),
        FactorTemplate("clock", "Clock", 2.0, +1, 0.50),
        FactorTemplate("coverage", "Coverage", 1.8, -1, 0.45),
        FactorTemplate("pass_rush", "Pass rush", 1.5, -1, 0.35),
        FactorTemplate("mobility", "Mobility", 1.0, +1, 0.25),
    ],
    profiles=[
        SportProfile("Pocket passer QB", 0.42),
        SportProfile("Dual-threat QB", 0.62),
        SportProfile("Running back", 0.35),
        SportProfile("Wide receiver", 0.55),
        SportProfile("Tight end", 0.40),
        SportProfile("Offensive lineman", 0.08),
    ],
)

CRICKET = SportConfig(
    name="Cricket",
    bandwidth=0.21,
    planks=[
        PlankTemplate("Defensive block", 0.00, 10, 2, 5, 0.97),
        PlankTemplate("Leave", 0.12, 15, 2, 8, 0.95),
        PlankTemplate("Nudge leg", 0.25, 35, 4, 10, 0.90),
        PlankTemplate("Drive", 0.40, 65, 5, 18, 0.80),
        PlankTemplate("Cut/pull", 0.55, 95, 8, 28, 0.72),
        PlankTemplate("Lofted drive", 0.68, 140, 11, 45, 0.58),
        PlankTemplate("Sweep", 0.82, 180, 13, 58, 0.50),
        PlankTemplate("Slog", 1.00, 240, 16, 75, 0.40),
    ],
    factors=[
        FactorTemplate("required_rate", "Required rate", 3.0, +1, 0.75),
        FactorTemplate("wickets", "Wickets", 2.5, -1, 0.55),
        FactorTemplate("pitch", "Pitch", 2.0, -1, 0.45),
        FactorTemplate("field_set", "Field set", 1.8, +1, 0.40),
        FactorTemplate("overs", "Overs", 1.5, +1, 0.35),
        FactorTemplate("bowler_type", "Bowler type", 1.0, +1, 0.25),
    ],
    profiles=[
        SportProfile("Opener anchor", 0.15),
        SportProfile("Opener aggressor", 0.60),
        SportProfile("No.3", 0.38),
        SportProfile("Middle order", 0.50),
        SportProfile("Lower order", 0.72),
        SportProfile("Tail-ender", 0.20),
    ],
)

ALL_SPORTS: dict[str, SportConfig] = {
    "Football": FOOTBALL,
    "Basketball": BASKETBALL,
    "Tennis": TENNIS,
    "Rugby Union": RUGBY_UNION,
    "American Football": AMERICAN_FOOTBALL,
    "Cricket": CRICKET,
}
