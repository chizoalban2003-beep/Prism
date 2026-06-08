"""
PRISM Silicon Response Policy

Maps (ΔB biological debt, ΔH hardware pressure, phase) → ExecutionBudget.
The budget is applied as hard constraints on every LLM call.

On GPU hardware: quantization_hint drives TVM/vLLM compilation target.
On CPU hardware: capability_ceil + max_tokens + speculative are the levers.

Biological rest state propagates to silicon: when your CNS is depleted,
Prism reduces its own computational intensity to match.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

_log = logging.getLogger(__name__)


@dataclass
class ExecutionBudget:
    capability_ceil: int = 3  # max LLM capability (0-3); 3=best
    max_tokens: int = 1500  # hard token ceiling
    speculative: bool = False  # force speculative routing
    disable_evolution: bool = False  # block PrismSoul new node creation
    quantization_hint: str = "fp16"  # "fp32"|"fp16"|"int8"|"int4" (GPU future)
    throttle_reason: str = ""  # empty = no throttle active


# Quantization order: lower index = more aggressive (less precision, faster)
_Q_ORDER = ["int4", "int8", "fp16", "fp32"]


class SiliconResponsePolicy:
    """
    Reads (ΔB, phase, extended ΔH) and emits a cached ExecutionBudget.
    TTL = 10s — budget doesn't need to update faster than that.
    """

    _TTL: float = 10.0

    def __init__(self) -> None:
        self._last_budget = ExecutionBudget()
        self._last_ts: float = 0.0

    def current_budget(
        self,
        delta_b: float = 0.0,
        phase_name: str = "STABLE",
        delta_h: float | None = None,
    ) -> ExecutionBudget:
        now = time.monotonic()
        if now - self._last_ts < self._TTL:
            return self._last_budget
        budget = self._compute(delta_b, phase_name, delta_h)
        self._last_budget = budget
        self._last_ts = now
        if budget.throttle_reason:
            _log.debug("[silicon] budget: %s", budget)
        return budget

    def _compute(self, delta_b: float, phase_name: str, delta_h_override: float | None) -> ExecutionBudget:
        pressure = max(delta_b, self._extended_delta_h(delta_h_override))

        # Base budget from biological/combined pressure
        budget = ExecutionBudget()
        if pressure >= 0.70:
            budget = ExecutionBudget(
                capability_ceil=1,
                max_tokens=400,
                speculative=True,
                disable_evolution=True,
                quantization_hint="int4",
                throttle_reason="critical pressure",
            )
        elif pressure >= 0.50:
            budget = ExecutionBudget(
                capability_ceil=2,
                max_tokens=700,
                speculative=True,
                disable_evolution=False,
                quantization_hint="int8",
                throttle_reason="high pressure",
            )
        elif pressure >= 0.30:
            budget = ExecutionBudget(
                capability_ceil=3,
                max_tokens=1000,
                speculative=False,
                quantization_hint="fp16",
                throttle_reason="moderate pressure",
            )

        # Phase overrides — only tighten, never loosen
        _PHASE_OVERRIDES: dict[str, dict] = {
            "LIQUID": {"capability_ceil": 1, "max_tokens": 400, "speculative": True, "quantization_hint": "int4"},
            "VISCOUS": {"max_tokens": 900, "speculative": True, "quantization_hint": "int8"},
            "CRYSTAL": {},  # precision mode — no override; CRYSTAL = healthy system
        }
        for k, v in _PHASE_OVERRIDES.get(phase_name, {}).items():
            if k == "capability_ceil":
                budget.capability_ceil = min(budget.capability_ceil, v)
            elif k == "max_tokens":
                budget.max_tokens = min(budget.max_tokens, v)
            elif k == "speculative":
                budget.speculative = budget.speculative or v
            elif k == "quantization_hint":
                cur_i = _Q_ORDER.index(budget.quantization_hint) if budget.quantization_hint in _Q_ORDER else 2
                new_i = _Q_ORDER.index(v) if v in _Q_ORDER else 2
                budget.quantization_hint = _Q_ORDER[min(cur_i, new_i)]

        if budget.throttle_reason and phase_name not in ("STABLE", "CRYSTAL", ""):
            budget.throttle_reason += f" + {phase_name.lower()} phase"

        return budget

    def _extended_delta_h(self, override: float | None) -> float:
        if override is not None:
            return override
        try:
            import psutil

            load_5m = os.getloadavg()[1]
            n_cores = psutil.cpu_count(logical=True) or 1
            cpu_sustained = min(1.0, load_5m / n_cores)
            ram_pressure = psutil.virtual_memory().percent / 100.0
            thermal_throttle = self._thermal_throttle_ratio()
            return min(1.0, cpu_sustained * 0.4 + ram_pressure * 0.4 + thermal_throttle * 0.2)
        except Exception:
            return 0.0

    @staticmethod
    def _thermal_throttle_ratio() -> float:
        """Detect CPU thermal throttling via cpufreq scaling on Linux."""
        try:
            import glob as _g

            max_f = [int(open(p).read()) for p in _g.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq")]
            cur_f = [int(open(p).read()) for p in _g.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq")]
            if not max_f or not cur_f:
                return 0.0
            return max(0.0, 1.0 - sum(cur_f) / len(cur_f) / (sum(max_f) / len(max_f)))
        except Exception:
            return 0.0

    def gpu_quantization_target(self, budget: ExecutionBudget) -> str | None:
        """
        Return TVM/vLLM quantization target if GPU available, else None.
        Future GPU wire-up:
            target = policy.gpu_quantization_target(budget)
            if target: lora_req = LoRARequest(..., quantization=target)
        """
        return budget.quantization_hint if self._gpu_available() else None

    @staticmethod
    def _gpu_available() -> bool:
        try:
            import subprocess

            return subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=2).returncode == 0
        except Exception:
            return False


_policy: SiliconResponsePolicy | None = None


def get_policy() -> SiliconResponsePolicy:
    global _policy
    if _policy is None:
        _policy = SiliconResponsePolicy()
    return _policy
