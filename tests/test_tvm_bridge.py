"""Tests for prism_tvm_bridge — 12 tests."""
from __future__ import annotations

from prism_tvm_bridge import CompileTarget, TVMBridge, get_tvm_bridge


class TestCompileTarget:
    def test_dataclass_fields(self):
        ct = CompileTarget(
            precision="fp16",
            hardware="cpu",
            tvm_target_str="llvm -mcpu=native",
            llama_cpp_flag="--type f16",
            available=True,
            compile_time_ms=500.0,
        )
        assert ct.precision == "fp16"
        assert ct.hardware == "cpu"
        assert ct.tvm_target_str == "llvm -mcpu=native"
        assert ct.llama_cpp_flag == "--type f16"
        assert ct.available is True
        assert ct.compile_time_ms == 500.0


class TestTVMBridge:
    def setup_method(self):
        self.bridge = TVMBridge()

    def test_cpu_fp16_target(self):
        t = self.bridge.compile_target("fp16", "cpu")
        assert t.hardware == "cpu"
        assert t.precision == "fp16"

    def test_cpu_int4_uses_llama_fallback(self):
        t = self.bridge.compile_target("int4", "cpu")
        flag = t.llama_cpp_flag.lower()
        assert "q4" in flag or "llama" in flag or t.llama_cpp_flag == "--type q4_K_M"

    def test_cpu_int4_tvm_not_available(self):
        # TVM int4 on CPU requires CUDA kernel — never available on cpu
        t = self.bridge.compile_target("int4", "cpu")
        assert not t.available

    def test_apply_target_logs_transition(self):
        bridge = TVMBridge()
        t = bridge.compile_target("int4", "cpu")
        result = bridge.apply_target(t)
        assert result is True
        assert bridge.active_precision == "int4"

    def test_no_transition_same_precision(self):
        bridge = TVMBridge()
        # Bridge starts with active_precision="fp16"
        t = bridge.compile_target("fp16", "cpu")
        bridge.apply_target(t)
        count = bridge.transition_count
        bridge.apply_target(t)
        assert bridge.transition_count == count

    def test_transition_count_increments(self):
        bridge = TVMBridge()
        t1 = bridge.compile_target("fp16", "cpu")
        t2 = bridge.compile_target("int4", "cpu")
        bridge.apply_target(t1)
        # fp16 → int4 is a real transition (active starts as fp16)
        bridge.apply_target(t2)
        assert bridge.transition_count >= 1

    def test_gpu_target_unavailable_no_gpu(self):
        bridge = TVMBridge()
        t = bridge.compile_target("int4", "cuda")
        # No GPU in test env
        assert not t.available

    def test_gpu_quantization_target_cpu_none(self):
        bridge = TVMBridge()
        # No GPU available in test environment
        assert bridge.gpu_quantization_target("int4") is None

    def test_cache_hit(self):
        bridge = TVMBridge()
        t1 = bridge.compile_target("fp16", "cpu")
        t2 = bridge.compile_target("fp16", "cpu")
        assert t1 is t2

    def test_status_dict(self):
        bridge = TVMBridge()
        s = bridge.status()
        assert "tvm_available" in s
        assert "active_precision" in s
        assert "gpu_available" in s
        assert "transition_count" in s
        assert "cached_targets" in s

    def test_singleton(self):
        assert get_tvm_bridge() is get_tvm_bridge()

    def test_all_precisions_have_llama_flag(self):
        bridge = TVMBridge()
        for p in ("fp32", "fp16", "int8", "int4"):
            t = bridge.compile_target(p, "cpu")
            assert isinstance(t.llama_cpp_flag, str)
