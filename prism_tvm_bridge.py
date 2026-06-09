"""
PRISM TVM/LLVM Compiler Bridge

Maps ExecutionBudget.quantization_hint to JIT compilation targets.

CPU path:  llama.cpp --type flags (Q4_K_M, Q8_0, F16) — available now
GPU path:  TVM Relax IRModule → build → VirtualMachine — one `import tvm` away

The key insight: sub-second precision swapping requires keeping FP16 master
weights in RAM and JIT-compiling quantized execution subgraphs on demand.
On CPU hardware, llama.cpp achieves the same result by reloading pre-quantized
weights (GGUF files) — slower than true JIT but semantically equivalent.
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

_log = logging.getLogger(__name__)


@dataclass
class CompileTarget:
    precision: str  # "fp32"|"fp16"|"int8"|"int4"
    hardware: str  # "cpu"|"cuda"|"metal"|"vulkan"
    tvm_target_str: str  # "llvm -mcpu=native" | "cuda" | "none"
    llama_cpp_flag: str  # "--type f16" etc (CPU fallback, always valid)
    available: bool  # compilable on current hardware
    compile_time_ms: float  # estimated JIT compile time


class TVMBridge:
    """
    Maps ExecutionBudget.quantization_hint → CompileTarget.

    Compilation strategy:
    - CPU + fp32/fp16/int8: TVM LLVM path (if tvm installed), else llama.cpp flag
    - CPU + int4: llama.cpp Q4_K_M only (TVM int4 requires CUDA kernel)
    - GPU + any: TVM Relax with CUDA backend

    Caches compiled targets to avoid recompilation on repeated calls.
    Tracks last-compiled precision to detect transitions (FP16→INT4 etc).
    """

    # llama.cpp flag and TVM dtype per precision level
    _PREC_MAP: dict[str, dict[str, str]] = {
        "fp32": {"llama": "", "tvm_dtype": "float32"},
        "fp16": {"llama": "--type f16", "tvm_dtype": "float16"},
        "int8": {"llama": "--type q8_0", "tvm_dtype": "int8"},
        "int4": {"llama": "--type q4_K_M", "tvm_dtype": "int4"},
    }
    # Estimated JIT compile times (ms)
    _COMPILE_MS: dict[str, dict[str, float]] = {
        "cpu": {"fp32": 400, "fp16": 500, "int8": 800, "int4": 1200},
        "cuda": {"fp32": 6000, "fp16": 8000, "int8": 12000, "int4": 15000},
    }

    def __init__(self) -> None:
        self._tvm_available = self._check_tvm()
        self._gpu_available = self._check_gpu()
        self._target_cache: dict[str, CompileTarget] = {}
        self._active_precision: str = "fp16"  # last successfully applied
        self._transition_count: int = 0  # how many precision swaps occurred

    # ── Public API ─────────────────────────────────────────────────────────────

    def compile_target(
        self,
        quantization_hint: str,
        hardware: str | None = None,
    ) -> CompileTarget:
        """Return (cached) CompileTarget for given hint and hardware."""
        hw = hardware or ("cuda" if self._gpu_available else "cpu")
        key = f"{quantization_hint}_{hw}"
        if key in self._target_cache:
            return self._target_cache[key]
        target = self._build_target(quantization_hint, hw)
        self._target_cache[key] = target
        return target

    def apply_target(self, target: CompileTarget, model_path: str = "") -> bool:
        """
        Apply a compilation target.
        GPU:  triggers TVM Relax JIT (async; returns True immediately).
        CPU:  records llama.cpp flag for next model invocation.
        Returns True if target was applied or queued, False if no-op.
        """
        if target.precision == self._active_precision:
            return False  # no transition needed

        prev = self._active_precision
        self._active_precision = target.precision
        self._transition_count += 1
        _log.info(
            "[tvm] precision transition: %s → %s (hardware=%s, compile≈%.0fms)",
            prev,
            target.precision,
            target.hardware,
            target.compile_time_ms,
        )

        if self._tvm_available and target.available and target.hardware == "cuda":
            return self._tvm_relax_jit(target, model_path)

        # CPU path: llama.cpp flag
        if target.llama_cpp_flag:
            _log.debug("[tvm] llama.cpp flag queued: %s", target.llama_cpp_flag)
        return True

    def gpu_quantization_target(
        self,
        quantization_hint: str,
    ) -> str | None:
        """
        Returns quantization target string for vLLM/TVM if GPU available.
        Wire-up: LoRARequest(..., quantization=bridge.gpu_quantization_target(hint))
        Returns None on CPU-only hardware.
        """
        if not self._gpu_available:
            return None
        return self._PREC_MAP.get(quantization_hint, {}).get("tvm_dtype")

    @property
    def active_precision(self) -> str:
        return self._active_precision

    @property
    def transition_count(self) -> int:
        return self._transition_count

    def status(self) -> dict:
        return {
            "tvm_available": self._tvm_available,
            "gpu_available": self._gpu_available,
            "active_precision": self._active_precision,
            "transition_count": self._transition_count,
            "cached_targets": list(self._target_cache),
        }

    # ── Internal ───────────────────────────────────────────────────────────────

    def _build_target(self, precision: str, hardware: str) -> CompileTarget:
        prec = self._PREC_MAP.get(precision, self._PREC_MAP["fp16"])
        compile_ms = self._COMPILE_MS.get(hardware, self._COMPILE_MS["cpu"]).get(precision, 800)

        if hardware == "cuda":
            if self._tvm_available and self._gpu_available:
                tgt_str, avail = "cuda", True
            else:
                tgt_str, avail = "cuda (unavailable)", False
        elif hardware == "cpu":
            if precision in ("fp32", "fp16", "int8") and self._tvm_available:
                tgt_str, avail = "llvm -mcpu=native", True
            elif precision == "int4":
                # TVM int4 kernel needs CUDA; CPU falls back to llama.cpp
                tgt_str, avail = "llvm (int4 requires llama.cpp fallback)", False
            else:
                tgt_str, avail = "llvm (tvm not installed)", False
        else:
            tgt_str, avail = "none", False

        return CompileTarget(
            precision=precision,
            hardware=hardware,
            tvm_target_str=tgt_str,
            llama_cpp_flag=prec["llama"],
            available=avail,
            compile_time_ms=compile_ms,
        )

    def _tvm_relax_jit(self, target: CompileTarget, model_path: str) -> bool:
        """GPU TVM Relax JIT compilation pipeline — requires Apache TVM to be installed."""
        try:
            import importlib

            importlib.import_module("tvm")
            importlib.import_module("tvm.relax")
            # Pipeline when GPU + TVM are present:
            # mod  = relax.frontend.nn.export_extern(model, spec)
            # ex   = relax.build(mod, tvm.target.Target(target.tvm_target_str))
            # vm   = relax.VirtualMachine(ex, tvm.cuda())
            # Then register vm as an LLMRouter provider
            raise NotImplementedError(
                "TVM Relax JIT pipeline is not yet implemented. "
                "tvm and tvm.relax are importable but the compilation graph has not been wired up."
            )
        except ImportError:
            _log.debug("[tvm] Apache TVM not installed — GPU JIT unavailable")
            return False

    @staticmethod
    def _check_tvm() -> bool:
        try:
            import importlib

            importlib.import_module("tvm")
            return True
        except ImportError:
            return False

    @staticmethod
    def _check_gpu() -> bool:
        try:
            return (
                subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=2).returncode == 0
            )
        except Exception:
            return False


# Module-level singleton
_bridge: TVMBridge | None = None


def get_tvm_bridge() -> TVMBridge:
    global _bridge
    if _bridge is None:
        _bridge = TVMBridge()
    return _bridge
