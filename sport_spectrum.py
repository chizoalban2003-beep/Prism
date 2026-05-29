from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

from decision_spectrum import (
    DecisionBeam,
    DecisionPlank,
    Factor,
    OutcomeDiagnosis,
)


def _normalize_key(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


class SportCatalog(dict[str, "SportConfig"]):
    def _resolve(self, key: str) -> str:
        wanted = _normalize_key(key)
        for existing in self.keys():
            if _normalize_key(existing) == wanted:
                return existing
        raise KeyError(key)

    def __getitem__(self, key: str) -> "SportConfig":
        return super().__getitem__(self._resolve(key))

    def get(self, key: str, default=None) -> Optional["SportConfig"]:
        if key is None:
            return default
        try:
            return super().__getitem__(self._resolve(key))
        except KeyError:
            return default

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        try:
            self._resolve(key)
        except KeyError:
            return False
        return True


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

    def _resolve_profile(self, profile_name: str) -> SportProfile:
        wanted = _normalize_key(profile_name)
        for profile in self.config.profiles:
            if _normalize_key(profile.name) == wanted:
                return profile
        raise KeyError(profile_name)

    def make_beam(
        self,
        profile_name: str,
        factor_values: dict | None = None,
    ) -> DecisionBeam:
        profile = self._resolve_profile(profile_name)
        beam = DecisionBeam(
            f"{self.config.name}_{profile.name}",
            bandwidth=self.config.bandwidth,
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

        values = factor_values or {}
        for ft in self.config.factors:
            val = max(0.0, min(1.0, values.get(ft.id, 0.5)))
            if ft.direction > 0:
                target = min(1.0, profile.fixed_fulcrum + val * ft.range)
            else:
                target = max(0.0, profile.fixed_fulcrum - val * ft.range)
            beam.fulcrum.add_factor(
                Factor(ft.id, val, ft.weight, target, ft.description)
            )

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
        factor_values: dict | None = None,
    ) -> OutcomeDiagnosis:
        return self.make_beam(profile_name, factor_values).evaluate()

    def sensitivity_sweep(
        self,
        profile_name: str,
        factor_id: str,
        steps: int = 5,
    ) -> list[OutcomeDiagnosis]:
        beam = self.make_beam(profile_name)
        return beam.sensitivity_sweep(factor_id, steps=steps)


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
    """Two coupled SportDecisionModel beams iterated to convergence."""

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
        attacker_context: dict | None = None,
        defender_context: dict | None = None,
    ) -> DuelOutcome:
        a_ctx = dict(attacker_context or {})
        d_ctx = dict(defender_context or {})
        a_pos = self.sport_model.make_beam(attacker_profile, a_ctx).fulcrum.position()
        d_pos = self.sport_model.make_beam(defender_profile, d_ctx).fulcrum.position()
        cs = self.coupling_strength
        iteration = 0

        for iteration in range(self.max_iterations):
            a_beam = self.sport_model.make_beam(attacker_profile, a_ctx)
            d_beam = self.sport_model.make_beam(defender_profile, d_ctx)
            a_beam.fulcrum.add_factor(
                Factor(
                    "_duel_pressure",
                    d_pos * cs,
                    cs * 1.5,
                    max(0.0, a_pos - d_pos * 0.4),
                    "Defender pressure",
                )
            )
            d_beam.fulcrum.add_factor(
                Factor(
                    "_duel_threat",
                    a_pos * cs,
                    cs * 1.2,
                    min(1.0, d_pos + a_pos * 0.3),
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
        advantage = (
            "attacker"
            if gap > 0.15
            else "defender"
            if gap < -0.15
            else "contested"
        )
        return DuelOutcome(
            a_pos,
            d_pos,
            a_diag.primary_plank.name,
            d_diag.primary_plank.name,
            a_diag.activations[0].activation,
            d_diag.activations[0].activation,
            a_diag.expected_net,
            d_diag.expected_net,
            iteration + 1,
            advantage,
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
        initial_context: dict | None = None,
    ) -> ChainOutcome:
        ctx = dict(initial_context or {})
        links: list[ChainLink] = []
        for player_name, profile in sequence:
            beam = self.sport_model.make_beam(profile, ctx)
            diag = beam.evaluate()
            pos = beam.fulcrum.position()
            category = (
                "aggressive"
                if pos >= self.AGGRESSION
                else "medium"
                if pos >= self.MEDIUM
                else "conservative"
            )
            next_ctx = dict(ctx)
            if category == "aggressive":
                next_ctx["pitch_zone"] = min(1.0, ctx.get("pitch_zone", 0.3) + 0.30)
                next_ctx["xg"] = min(1.0, ctx.get("xg", 0.1) + 0.20)
            elif category == "medium":
                next_ctx["pitch_zone"] = min(1.0, ctx.get("pitch_zone", 0.3) + 0.12)
            else:
                next_ctx["pitch_zone"] = max(0.0, ctx.get("pitch_zone", 0.3) - 0.08)
            links.append(
                ChainLink(
                    player_name,
                    profile,
                    pos,
                    diag.primary_plank.name,
                    diag.activations[0].activation,
                    diag.expected_net,
                    next_ctx,
                )
            )
            ctx = next_ctx
        return ChainOutcome(
            links,
            links[-1].decision if links else "",
            sum(link.expected_net for link in links),
        )


def _planks(*items: tuple[str, float, float, float, float, float]) -> list[PlankTemplate]:
    return [PlankTemplate(*item) for item in items]


def _factors(*items: tuple[str, str, float, float, float]) -> list[FactorTemplate]:
    return [FactorTemplate(*item) for item in items]


def _profiles(*items: tuple[str, float]) -> list[SportProfile]:
    return [SportProfile(*item) for item in items]


ALL_SPORTS: SportCatalog = SportCatalog(
    {
        "FOOTBALL": SportConfig(
            name="Football",
            bandwidth=0.17,
            planks=_planks(
                ("Defend", 0.00, 15, 3, 8, 0.98),
                ("BackPass", 0.14, 55, 5, 10, 0.93),
                ("Lateral", 0.28, 80, 7, 18, 0.88),
                ("SafeFwd", 0.43, 110, 9, 25, 0.82),
                ("Carry", 0.57, 145, 11, 38, 0.65),
                ("ThroughBall", 0.71, 195, 14, 58, 0.48),
                ("Shot", 0.86, 265, 17, 65, 0.38),
                ("Finish", 1.00, 350, 20, 45, 0.70),
            ),
            factors=_factors(
                ("pitch_zone", "Pitch zone", 3.5, +1, 0.80),
                ("xg", "Expected goals", 2.5, +1, 0.60),
                ("press", "Pressure", 1.5, -1, 0.40),
                ("support", "Support", 1.2, -1, 0.35),
                ("scoreline", "Scoreline", 1.0, +1, 0.25),
                ("fatigue", "Fatigue", 0.8, -1, 0.15),
            ),
            profiles=_profiles(
                ("Goalkeeper", 0.04),
                ("Centre_back", 0.13),
                ("Full_back", 0.27),
                ("Def_mid", 0.33),
                ("Box_to_box", 0.50),
                ("Att_mid", 0.63),
                ("Winger", 0.71),
                ("Striker", 0.82),
            ),
        ),
        "BASKETBALL": SportConfig(
            name="Basketball",
            bandwidth=0.18,
            planks=_planks(
                ("Reset", 0.00, 10, 2, 5, 0.99),
                ("Swing", 0.14, 60, 4, 12, 0.90),
                ("KickOut", 0.28, 90, 5, 22, 0.82),
                ("PullUp", 0.43, 120, 8, 35, 0.65),
                ("Drive", 0.57, 150, 9, 42, 0.60),
                ("AttackRim", 0.71, 200, 11, 50, 0.55),
                ("StepBack3", 0.86, 240, 14, 80, 0.35),
                ("AndOne", 1.00, 280, 16, 55, 0.50),
            ),
            factors=_factors(
                ("shot_clock", "Shot clock", 3.0, +1, 0.75),
                ("coverage", "Coverage", 2.5, -1, 0.55),
                ("court_zone", "Court zone", 2.0, +1, 0.70),
                ("score_margin", "Score margin", 1.5, +1, 0.30),
                ("foul_trouble", "Foul trouble", 1.0, -1, 0.25),
                ("hot_hand", "Hot hand", 0.9, +1, 0.20),
            ),
            profiles=_profiles(
                ("Point_guard", 0.52),
                ("Shooting_guard", 0.70),
                ("Small_forward", 0.62),
                ("Power_forward", 0.45),
                ("Centre", 0.38),
            ),
        ),
        "TENNIS": SportConfig(
            name="Tennis",
            bandwidth=0.20,
            planks=_planks(
                ("Block", 0.00, 20, 3, 8, 0.96),
                ("DeepCross", 0.16, 55, 5, 15, 0.88),
                ("Slice", 0.32, 65, 6, 18, 0.85),
                ("Neutral", 0.47, 85, 7, 22, 0.80),
                ("DownLine", 0.62, 140, 10, 38, 0.62),
                ("DropShot", 0.75, 160, 12, 55, 0.55),
                ("AggressiveCross", 0.87, 200, 14, 60, 0.48),
                ("Winner", 1.00, 250, 16, 70, 0.38),
            ),
            factors=_factors(
                ("score_pressure", "Score pressure", 3.0, +1, 0.65),
                ("surface", "Surface", 2.0, +1, 0.45),
                ("opp_position", "Opponent position", 2.5, +1, 0.55),
                ("energy", "Energy", 1.5, -1, 0.35),
                ("rally_length", "Rally length", 1.2, +1, 0.30),
                ("wind", "Wind", 0.8, -1, 0.20),
            ),
            profiles=_profiles(
                ("Defensive_baseliner", 0.22),
                ("Aggressive_baseliner", 0.62),
                ("All_court", 0.50),
                ("Serve_volley", 0.78),
                ("Counter_puncher", 0.30),
            ),
        ),
        "RUGBY_UNION": SportConfig(
            name="Rugby Union",
            bandwidth=0.19,
            planks=_planks(
                ("KickTouch", 0.00, 25, 4, 10, 0.92),
                ("PickDrive", 0.14, 50, 5, 18, 0.88),
                ("Recycle", 0.28, 70, 6, 22, 0.84),
                ("PopPass", 0.42, 100, 8, 28, 0.78),
                ("FlatPass", 0.56, 135, 9, 35, 0.70),
                ("Switch", 0.68, 165, 11, 45, 0.60),
                ("CrossKick", 0.82, 200, 14, 58, 0.48),
                ("Chip", 1.00, 240, 16, 70, 0.40),
            ),
            factors=_factors(
                ("field_pos", "Field position", 3.5, +1, 0.80),
                ("phase_count", "Phase count", 2.0, -1, 0.40),
                ("def_line", "Defensive line", 2.0, -1, 0.45),
                ("score", "Score", 1.5, +1, 0.30),
                ("fatigue", "Fatigue", 1.0, -1, 0.20),
                ("setpiece", "Set piece", 0.8, +1, 0.25),
            ),
            profiles=_profiles(
                ("Prop", 0.12),
                ("Hooker", 0.18),
                ("Number_8", 0.42),
                ("Scrum_half", 0.38),
                ("Fly_half", 0.58),
                ("Inside_centre", 0.50),
                ("Outside_centre", 0.60),
                ("Wing", 0.72),
            ),
        ),
        "AMERICAN_FOOTBALL": SportConfig(
            name="American Football",
            bandwidth=0.19,
            planks=_planks(
                ("Kneel", 0.00, 5, 1, 2, 0.99),
                ("QBSneak", 0.14, 40, 4, 12, 0.85),
                ("ShortRun", 0.28, 70, 5, 18, 0.80),
                ("Screen", 0.42, 95, 7, 25, 0.75),
                ("PlayAction", 0.56, 145, 9, 35, 0.65),
                ("CrossingRoute", 0.68, 190, 11, 45, 0.58),
                ("DeepRoute", 0.82, 280, 14, 75, 0.38),
                ("HailMary", 1.00, 420, 18, 95, 0.18),
            ),
            factors=_factors(
                ("down_distance", "Down and distance", 3.5, +1, 0.80),
                ("field_pos", "Field position", 2.5, +1, 0.65),
                ("clock", "Clock", 2.0, +1, 0.50),
                ("coverage", "Coverage", 1.8, -1, 0.45),
                ("pass_rush", "Pass rush", 1.5, -1, 0.35),
                ("mobility", "Mobility", 1.0, +1, 0.25),
            ),
            profiles=_profiles(
                ("Pocket_passer", 0.42),
                ("Dual_threat_QB", 0.62),
                ("Running_back", 0.35),
                ("Wide_receiver", 0.55),
                ("Tight_end", 0.40),
                ("Lineman", 0.08),
            ),
        ),
        "CRICKET": SportConfig(
            name="Cricket",
            bandwidth=0.21,
            planks=_planks(
                ("Block", 0.00, 10, 2, 5, 0.97),
                ("Leave", 0.12, 15, 2, 8, 0.95),
                ("Nudge", 0.25, 35, 4, 10, 0.90),
                ("Drive", 0.40, 65, 5, 18, 0.80),
                ("CutPull", 0.55, 95, 8, 28, 0.72),
                ("LoftedDrive", 0.68, 140, 11, 45, 0.58),
                ("Sweep", 0.82, 180, 13, 58, 0.50),
                ("Slog", 1.00, 240, 16, 75, 0.40),
            ),
            factors=_factors(
                ("required_rate", "Required rate", 3.0, +1, 0.75),
                ("wickets", "Wickets", 2.5, -1, 0.55),
                ("pitch", "Pitch", 2.0, -1, 0.45),
                ("field_set", "Field set", 1.8, +1, 0.40),
                ("overs", "Overs", 1.5, +1, 0.35),
                ("bowler_type", "Bowler type", 1.0, +1, 0.25),
            ),
            profiles=_profiles(
                ("Opener_anchor", 0.15),
                ("Opener_aggressor", 0.60),
                ("No3", 0.38),
                ("Middle_order", 0.50),
                ("Lower_order", 0.72),
                ("Tail_ender", 0.20),
            ),
        ),
    }
)
