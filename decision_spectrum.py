from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Factor:
    name: str
    value: float
    weight: float
    target: float
    description: str = ""

    def contribution(self) -> float:
        return self.weight * self.value * self.target

    def mass(self) -> float:
        return self.weight * self.value


@dataclass
class DecisionPlank:
    name: str
    position: float
    payoff: float
    cost: float
    risk: float
    probability: float = 1.0
    metadata: dict = field(default_factory=dict)

    @property
    def net_value(self) -> float:
        return self.payoff - self.cost


@dataclass
class PlankActivation:
    plank: DecisionPlank
    activation: float


@dataclass
class OutcomeDiagnosis:
    fulcrum_position: float
    activations: list[PlankActivation]
    primary_plank: DecisionPlank
    expected_payoff: float
    expected_cost: float
    expected_net: float
    variance: float
    risk_adjusted_return: float
    bandwidth: float

    @property
    def std_dev(self) -> float:
        return math.sqrt(max(self.variance, 0))

    @property
    def confidence_interval(self) -> tuple[float, float]:
        m = 1.96 * self.std_dev
        return (self.expected_payoff - m, self.expected_payoff + m)

    def top(self, n: int = 3) -> list[PlankActivation]:
        return self.activations[:n]


class SpectrumFulcrum:
    def __init__(self, factors: list[Factor] = None):
        self._factors: dict[str, Factor] = {}
        for f in (factors or []):
            self._factors[f.name] = f

    def add_factor(self, f: Factor) -> None:
        self._factors[f.name] = f

    def update(self, name: str, value: float) -> None:
        if name in self._factors:
            self._factors[name].value = max(0.0, min(1.0, value))

    def remove_factor(self, name: str) -> None:
        self._factors.pop(name, None)

    @property
    def factors(self) -> list[Factor]:
        return list(self._factors.values())

    def position(self) -> float:
        num = sum(f.contribution() for f in self._factors.values())
        den = sum(f.mass() for f in self._factors.values())
        if den < 1e-12:
            return 0.5
        return max(0.0, min(1.0, num / den))

    def dominant_factor(self) -> Optional[Factor]:
        if not self._factors:
            return None
        return max(self._factors.values(), key=lambda f: abs(f.contribution()))


class AdaptiveFulcrum(SpectrumFulcrum):
    def __init__(
        self,
        factors: list[Factor] = None,
        learning_rate: float = 0.05,
        weight_min: float = 0.01,
        weight_max: float = 10.0,
    ):
        super().__init__(factors)
        self.learning_rate = learning_rate
        self.weight_min = weight_min
        self.weight_max = weight_max
        self._update_count = 0

    def observe(
        self,
        actual_payoff: float,
        predicted_payoff: float,
        chosen_position: float,
    ) -> None:
        error = actual_payoff - predicted_payoff
        if abs(error) < 1e-9:
            return
        scale = max(abs(predicted_payoff), abs(actual_payoff), 1.0)
        norm_error = error / scale
        for f in self._factors.values():
            alignment = 1.0 - abs(f.target - chosen_position)
            delta = self.learning_rate * norm_error * alignment * f.value
            f.weight = max(self.weight_min, min(self.weight_max, f.weight + delta))
        self._update_count += 1


class DecisionBeam:
    def __init__(
        self,
        name: str,
        bandwidth: float = 0.25,
        fulcrum: SpectrumFulcrum = None,
    ):
        self.name = name
        self.bandwidth = bandwidth
        self.fulcrum = fulcrum if fulcrum is not None else SpectrumFulcrum()
        self._planks: list[DecisionPlank] = []

    def add_plank(self, p: DecisionPlank) -> None:
        self._planks.append(p)
        self._planks.sort(key=lambda x: x.position)

    def remove_plank(self, name: str) -> bool:
        before = len(self._planks)
        self._planks = [p for p in self._planks if p.name != name]
        return len(self._planks) < before

    @property
    def planks(self) -> list[DecisionPlank]:
        return list(self._planks)

    def _activations(self, pos: float) -> list[PlankActivation]:
        if not self._planks:
            return []
        raws = [
            math.exp(-0.5 * ((pos - p.position) / self.bandwidth) ** 2)
            for p in self._planks
        ]
        total = sum(raws) or 1.0
        result = [PlankActivation(p, r / total) for p, r in zip(self._planks, raws)]
        return sorted(result, key=lambda a: a.activation, reverse=True)

    def evaluate(self) -> OutcomeDiagnosis:
        if not self._planks:
            raise ValueError(f"Beam '{self.name}' has no planks")
        pos = self.fulcrum.position()
        acts = self._activations(pos)
        exp_pay = sum(
            a.activation * a.plank.payoff * a.plank.probability for a in acts
        )
        exp_cost = sum(a.activation * a.plank.cost for a in acts)
        exp_net = exp_pay - exp_cost
        var = sum(a.activation * (a.plank.payoff - exp_pay) ** 2 for a in acts)
        rar = exp_net / max(math.sqrt(var), 1e-9)
        return OutcomeDiagnosis(
            fulcrum_position=pos,
            activations=acts,
            primary_plank=acts[0].plank,
            expected_payoff=exp_pay,
            expected_cost=exp_cost,
            expected_net=exp_net,
            variance=var,
            risk_adjusted_return=rar,
            bandwidth=self.bandwidth,
        )

    def sensitivity_sweep(
        self,
        factor_name: str,
        steps: int = 5,
    ) -> list[OutcomeDiagnosis]:
        results = []
        orig = self.fulcrum._factors[factor_name].value
        for v in [i / (steps - 1) for i in range(steps)]:
            self.fulcrum.update(factor_name, v)
            results.append(self.evaluate())
        self.fulcrum.update(factor_name, orig)
        return results


class DecisionNetwork:
    def __init__(self, max_iterations: int = 20, tolerance: float = 1e-4):
        self._beams: dict[str, DecisionBeam] = {}
        self._deps: list[tuple] = []
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def add_beam(self, beam: DecisionBeam) -> None:
        self._beams[beam.name] = beam

    def add_dependency(
        self,
        source: str,
        target: str,
        strength: float,
        factor_name: str = None,
    ) -> None:
        fname = factor_name or f"_{source}_influence"
        self._deps.append((source, target, strength, fname))

    def solve(self) -> dict[str, OutcomeDiagnosis]:
        prev: dict[str, float] = {}
        for _iteration in range(self.max_iterations):
            for src, tgt, strength, fname in self._deps:
                if src not in self._beams or tgt not in self._beams:
                    continue
                src_pos = self._beams[src].fulcrum.position()
                tgt_beam = self._beams[tgt]
                tgt_beam.fulcrum.add_factor(
                    Factor(fname, src_pos, strength, src_pos, f"Propagated from '{src}'")
                )
            positions = {n: b.fulcrum.position() for n, b in self._beams.items()}
            if prev and all(
                abs(positions[k] - prev.get(k, 0)) < self.tolerance for k in positions
            ):
                break
            prev = positions
        return {n: b.evaluate() for n, b in self._beams.items()}
