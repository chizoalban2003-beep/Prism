"""
Tests for prism_ml_assembler — Surgical ML Assembler.
All sklearn/xgboost imports are lazy so tests run without those deps.
"""
from __future__ import annotations

import pytest

from prism_ml_assembler import (
    AssemblyResult,
    DataProfile,
    MLAssembler,
    run_nightly_sweep,
)

# ---------------------------------------------------------------------------
# DataProfile
# ---------------------------------------------------------------------------

class TestDataProfile:
    def test_fields(self):
        p = DataProfile(
            n_samples=100, n_features=5, has_labels=True,
            is_linear=True, is_high_dim=False,
            label_is_continuous=True, sparsity=0.1,
        )
        assert p.n_samples == 100
        assert p.is_linear is True
        assert p.is_high_dim is False


# ---------------------------------------------------------------------------
# MLAssembler._select — decision DAG
# ---------------------------------------------------------------------------

class TestMLAssemblerSelect:
    def _asm(self):
        return MLAssembler()

    def _profile(self, **kwargs):
        defaults = dict(
            n_samples=100, n_features=5, has_labels=True,
            is_linear=False, is_high_dim=False,
            label_is_continuous=True, sparsity=0.1,
        )
        defaults.update(kwargs)
        return DataProfile(**defaults)

    def test_linear_sparse_selects_lasso(self):
        asm = self._asm()
        p = self._profile(is_linear=True, sparsity=0.5)
        algo, _ = asm._select(p)
        assert algo == "lasso"

    def test_linear_dense_selects_ridge(self):
        asm = self._asm()
        p = self._profile(is_linear=True, sparsity=0.1)
        algo, _ = asm._select(p)
        assert algo == "ridge"

    def test_nonlinear_large_continuous_selects_xgboost(self):
        asm = self._asm()
        p = self._profile(is_linear=False, n_samples=200, label_is_continuous=True)
        algo, _ = asm._select(p)
        assert algo == "xgboost"

    def test_nonlinear_large_categorical_selects_lgbm(self):
        asm = self._asm()
        p = self._profile(is_linear=False, n_samples=200, label_is_continuous=False)
        algo, _ = asm._select(p)
        assert algo == "lightgbm"

    def test_nonlinear_small_selects_svm(self):
        asm = self._asm()
        p = self._profile(is_linear=False, n_samples=10)
        algo, _ = asm._select(p)
        assert algo == "svm"

    def test_svm_params_have_kernel(self):
        asm = self._asm()
        p = self._profile(is_linear=False, n_samples=10)
        _, params = asm._select(p)
        assert "kernel" in params
        assert "C" in params

    def test_unlabelled_small_selects_dbscan(self):
        asm = self._asm()
        p = self._profile(has_labels=False, n_samples=100)
        algo, _ = asm._select(p)
        assert algo == "dbscan"

    def test_unlabelled_large_selects_kmeans(self):
        asm = self._asm()
        p = self._profile(has_labels=False, n_samples=6000)
        algo, _ = asm._select(p)
        assert algo == "kmeans"

    def test_nightly_params_override(self):
        asm = self._asm()
        asm._nightly_params["ridge"] = {"alpha": 42.0}
        p = self._profile(is_linear=True, sparsity=0.0)
        _, params = asm._select(p)
        assert params["alpha"] == 42.0


# ---------------------------------------------------------------------------
# MLAssembler._profile
# ---------------------------------------------------------------------------

class TestMLAssemblerProfile:
    def test_profile_from_numpy_array(self):
        pytest.importorskip("numpy")
        import numpy as np
        asm = MLAssembler()
        X = np.random.randn(50, 4)
        y = np.random.randn(50)
        p = asm._profile(X, y)
        assert p.n_samples == 50
        assert p.n_features == 4
        assert p.has_labels is True

    def test_profile_unlabelled(self):
        pytest.importorskip("numpy")
        import numpy as np
        asm = MLAssembler()
        X = np.zeros((20, 3))
        p = asm._profile(X, None)
        assert p.has_labels is False
        assert p.sparsity == 1.0  # all zeros

    def test_profile_sparse(self):
        pytest.importorskip("numpy")
        import numpy as np
        asm = MLAssembler()
        X = np.array([[0, 1], [0, 0], [1, 0]])
        p = asm._profile(X, None)
        assert p.sparsity > 0.4


# ---------------------------------------------------------------------------
# MLAssembler.run — fallback path (no sklearn)
# ---------------------------------------------------------------------------

class TestMLAssemblerRunFallback:
    def test_run_no_numpy_returns_fallback(self, monkeypatch):
        """When numpy is not available, run() returns a fallback result."""
        asm = MLAssembler()
        # Monkeypatch numpy import to fail
        import builtins
        real_import = builtins.__import__

        def fail_numpy(name, *args, **kwargs):
            if name == "numpy":
                raise ImportError("no numpy")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fail_numpy)
        result = asm.run("test task", X=[[1, 2], [3, 4]])
        assert result.algorithm == "fallback_mean"
        assert result.error is not None

    def test_run_with_numpy_returns_result(self):
        pytest.importorskip("numpy")
        import numpy as np
        asm = MLAssembler()
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        result = asm.run("predict", X=X, y=y, translate=False)
        assert isinstance(result, AssemblyResult)
        assert result.algorithm in {"ridge", "lasso", "xgboost", "lightgbm",
                                    "svm", "random_forest", "dbscan", "kmeans",
                                    "fallback_mean"}
        assert 0.0 <= result.confidence <= 1.0
        assert result.duration_ms >= 0.0


# ---------------------------------------------------------------------------
# MLAssembler.apply_grid_search_params
# ---------------------------------------------------------------------------

class TestMLAssemblerGridSearch:
    def test_apply_params_stored(self):
        asm = MLAssembler()
        asm.apply_grid_search_params("ridge", {"alpha": 99.9})
        assert asm._nightly_params["ridge"]["alpha"] == 99.9


# ---------------------------------------------------------------------------
# AssemblyResult
# ---------------------------------------------------------------------------

class TestAssemblyResult:
    def test_fields(self):
        r = AssemblyResult(
            result_id="abc", task="t", algorithm="ridge",
            prediction=[1.0, 2.0], confidence=0.88,
            params={"alpha": 1.0}, explanation="ok", duration_ms=5.0,
        )
        assert r.result_id == "abc"
        assert r.confidence == 0.88
        assert r.error is None


# ---------------------------------------------------------------------------
# run_nightly_sweep — no-ops cleanly when tracker returns empty
# ---------------------------------------------------------------------------

class TestNightlySweep:
    def test_sweep_no_op_empty_tracker(self):
        class FakeTracker:
            def get_ml_results(self, min_error=0.15):
                return []

        asm = MLAssembler()
        updated = run_nightly_sweep(asm, FakeTracker())
        assert updated == {}

    def test_sweep_applies_grid_params(self):
        class FakeTracker:
            def get_ml_results(self, min_error=0.15):
                return [{"algorithm": "ridge", "confidence": 0.5,
                         "task": "t", "duration_ms": 10.0}]

        asm = MLAssembler()
        updated = run_nightly_sweep(asm, FakeTracker(), error_threshold=0.15)
        # Ridge should have received a new param set
        assert "ridge" in updated


# ---------------------------------------------------------------------------
# Hyperband successive halving (_grid_search / _score_candidate)
# ---------------------------------------------------------------------------

class TestHyperband:
    def test_grid_search_returns_dict(self):
        from prism_ml_assembler import _grid_search
        result = _grid_search("ridge", [{"confidence": 0.2, "algorithm": "ridge"}])
        assert isinstance(result, dict)
        assert "alpha" in result

    def test_grid_search_varies_by_confidence(self):
        from prism_ml_assembler import _grid_search
        low_conf = [{"confidence": 0.1, "algorithm": "ridge"}]
        high_conf = [{"confidence": 0.9, "algorithm": "ridge"}]
        low_params = _grid_search("ridge", low_conf)
        high_params = _grid_search("ridge", high_conf)
        # Different confidence levels should yield different alpha values
        assert low_params["alpha"] != high_params["alpha"]

    def test_grid_search_lgbm_returns_num_leaves(self):
        from prism_ml_assembler import _grid_search
        result = _grid_search("lightgbm", [{"confidence": 0.3, "algorithm": "lightgbm"}])
        assert "num_leaves" in result

    def test_grid_search_svm_returns_kernel(self):
        from prism_ml_assembler import _grid_search
        result = _grid_search("svm", [{"confidence": 0.2, "algorithm": "svm"}])
        assert "kernel" in result
        assert "C" in result

    def test_grid_search_unknown_algo_returns_empty(self):
        from prism_ml_assembler import _grid_search
        result = _grid_search("nonexistent_algo", [{"confidence": 0.5}])
        assert result == {}

    def test_score_candidate_ridge_low_conf(self):
        from prism_ml_assembler import _score_candidate
        low = _score_candidate("ridge", {"alpha": 0.001}, avg_conf=0.1)
        high = _score_candidate("ridge", {"alpha": 100.0}, avg_conf=0.1)
        assert low > high  # low conf → prefer low alpha


# ---------------------------------------------------------------------------
# PCA preprocessing path
# ---------------------------------------------------------------------------

class TestPCAPreprocessing:
    def test_high_dim_profile_sets_flag(self):
        pytest.importorskip("numpy")
        import numpy as np
        asm = MLAssembler()
        X = np.random.randn(30, 25)  # n_features=25 > PCA_MIN_FEATURES=20
        y = np.random.randn(30)
        p = asm._profile(X, y)
        assert p.is_high_dim is True

    def test_apply_pca_reduces_dims(self):
        pytest.importorskip("sklearn")
        import numpy as np

        from prism_ml_assembler import DataProfile
        asm = MLAssembler()
        X = np.random.randn(30, 25)
        p = DataProfile(n_samples=30, n_features=25, has_labels=True,
                        is_linear=False, is_high_dim=True,
                        label_is_continuous=True, sparsity=0.0)
        X_reduced = asm._apply_pca(X, p)
        assert X_reduced.shape[1] < 25

    def test_run_high_dim_succeeds(self):
        pytest.importorskip("numpy")
        import numpy as np
        asm = MLAssembler()
        X = np.random.randn(60, 30)  # high-dim, large enough for heavy classifier
        y = np.random.randn(60)
        result = asm.run("high dim predict", X=X, y=y, translate=False)
        assert isinstance(result, AssemblyResult)
        assert result.confidence >= 0.0


# ---------------------------------------------------------------------------
# SVM and LightGBM run paths
# ---------------------------------------------------------------------------

class TestSVMRun:
    def test_svm_run_small_nonlinear(self):
        pytest.importorskip("sklearn")
        import numpy as np
        asm = MLAssembler()
        # n=20 < HEAVY_N_THRESHOLD=50 → SVM branch
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        result = asm.run("small predict", X=X, y=y, translate=False)
        assert isinstance(result, AssemblyResult)
        # SVM or fallback (if sklearn not available in env)
        assert result.algorithm in {"svm", "fallback_mean"}


class TestLightGBMRun:
    def test_lgbm_run_large_categorical(self):
        pytest.importorskip("lightgbm")
        import numpy as np
        asm = MLAssembler()
        rng = np.random.default_rng(0)
        X = rng.standard_normal((80, 4))
        y = rng.integers(0, 3, size=80).astype(float)
        result = asm.run("classify", X=X, y=y, translate=False)
        assert isinstance(result, AssemblyResult)
        assert result.algorithm in {"lightgbm", "fallback_mean"}
