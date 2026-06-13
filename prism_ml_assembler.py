"""
prism_ml_assembler.py
=====================
Surgical ML Assembler — PRISM's task-profiling algorithm compiler.

Instead of routing every analytical task to the LLM, the Assembler profiles
incoming data constraints and compiles the minimal, correct algorithm ensemble:

  Linear Scalpel      → Ridge / Lasso            (linear, labelled, explainability)
  Heavy Classifier    → XGBoost / LightGBM        (nonlinear, labelled, n > 50)
  Boundary Kernel     → SVM (SVC / SVR)           (nonlinear, labelled, n ≤ 50, high-dim)
  Clustering Sieve    → DBSCAN / K-Means          (unlabelled, structure discovery)
  Dim Reduction       → PCA / SVD                 (preprocessing when n_features > 20)
  Semantic Translator → Ollama (LLM)              (strictly last node, text only)

The LLM is never the primary solver for math or prediction. It only translates
final numerical results into human language at the end of the DAG.

Integration points
------------------
  1. prism_orchestrator.py — DAG nodes declare profile="ml" to invoke Assembler
  2. prism_outcome_tracker.py — AssemblerRecord logged per prediction
  3. prism_crystalliser.py — nightly Grid Search on failed outcomes (>15% error)

Usage
-----
    assembler = MLAssembler()

    result = assembler.run(
        task="predict next week's focus score",
        X=feature_matrix,           # numpy array or list[list[float]]
        y=labels,                   # None → unsupervised
        feature_names=["hrv", ...],
    )

    # result.prediction     — numeric output
    # result.algorithm      — which algo ran
    # result.explanation    — LLM-translated summary (if translate=True)
    # result.confidence     — model R² or silhouette score
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task profile
# ---------------------------------------------------------------------------

@dataclass
class DataProfile:
    """Constraints inferred from the raw data — drives algorithm selection."""
    n_samples: int
    n_features: int
    has_labels: bool
    is_linear: bool          # True if correlation |r| > 0.7 on majority of features
    is_high_dim: bool        # n_features > 20
    label_is_continuous: bool  # regression vs classification
    sparsity: float          # 0.0–1.0 fraction of zeros
    is_sequential: bool = False


@dataclass
class AssemblyResult:
    result_id: str
    task: str
    algorithm: str           # ridge|lasso|xgboost|lightgbm|svm|random_forest|dbscan|kmeans|mlp|lstm|gru
    prediction: Any          # array or scalar
    confidence: float        # R², accuracy, or silhouette score (0–1)
    params: dict             # hyperparameters used
    explanation: str         # LLM-translated summary (empty if translate=False)
    duration_ms: float
    triggered_at: float = field(default_factory=time.time)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------

class MLAssembler:
    """
    Profiles a task, selects and fits the right algorithm, optionally
    translates numeric result to language via Ollama.

    All sklearn/xgboost imports are lazy — if unavailable the Assembler
    gracefully degrades to a mean-predictor fallback and logs a warning.
    """

    # Thresholds that Grid Search mutates nightly
    LINEAR_R_THRESHOLD: float = 0.70   # |r| above this → treat as linear
    HEAVY_N_THRESHOLD:  int   = 50     # n_samples above this → allow heavy classifiers
    LASSO_SPARSITY:     float = 0.30   # sparsity above this → prefer Lasso over Ridge
    DBSCAN_MAX_N:       int   = 5000   # above this K-Means is cheaper than DBSCAN
    ERROR_THRESHOLD:    float = 0.15   # >15% prediction error → flag for nightly sweep
    PCA_MIN_FEATURES:   int   = 20     # apply PCA when n_features exceeds this
    TORCH_N_THRESHOLD:  int   = 100

    def __init__(
        self,
        llm_router: Optional[Any] = None,
        outcome_tracker: Optional[Any] = None,
    ) -> None:
        self._llm = llm_router
        self._tracker = outcome_tracker
        self._nightly_params: dict[str, dict] = {}   # algo → overrides from Grid Search

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        task: str,
        X: Any,
        y: Optional[Any] = None,
        feature_names: Optional[list[str]] = None,
        translate: bool = True,
        sequential: bool = False,
    ) -> AssemblyResult:
        """Profile data → select algorithm → fit → optionally translate."""
        t0 = time.time()
        result_id = str(uuid.uuid4())[:8]

        try:
            import numpy as np
            X_arr = np.array(X, dtype=float)
            y_arr = np.array(y, dtype=float) if y is not None else None
        except Exception as exc:
            return self._fallback(task, result_id, t0, f"numpy unavailable: {exc}")

        profile = self._profile(X_arr, y_arr)
        profile.is_sequential = sequential
        algo, params = self._select(profile)
        prediction, confidence = self._fit(algo, params, X_arr, y_arr, profile)

        explanation = ""
        if translate and self._llm is not None:
            explanation = self._translate(task, algo, prediction, confidence, feature_names)

        duration_ms = (time.time() - t0) * 1000
        result = AssemblyResult(
            result_id=result_id,
            task=task,
            algorithm=algo,
            prediction=prediction,
            confidence=confidence,
            params=params,
            explanation=explanation,
            duration_ms=duration_ms,
        )

        self._log_outcome(result, task)
        return result

    def apply_grid_search_params(self, algo: str, params: dict) -> None:
        """Called by nightly crystalliser sweep after Grid Search on failed outcomes."""
        self._nightly_params[algo] = params
        logger.info("MLAssembler: nightly params updated for %s → %s", algo, params)

    # ── Profiling ─────────────────────────────────────────────────────────────

    def _profile(self, X: Any, y: Optional[Any]) -> DataProfile:
        import numpy as np

        n, d = X.shape if X.ndim == 2 else (len(X), 1)
        has_labels = y is not None and len(y) == n

        sparsity = float(np.mean(X == 0)) if n > 0 else 0.0

        is_linear = False
        label_is_continuous = False
        if has_labels and y is not None:
            try:
                corrs = [abs(float(np.corrcoef(X[:, i], y)[0, 1]))
                         for i in range(min(d, 50))]
                is_linear = sum(c > self.LINEAR_R_THRESHOLD for c in corrs) > len(corrs) * 0.5
                unique_ratio = len(set(y.tolist())) / max(len(y), 1)
                label_is_continuous = unique_ratio > 0.1
            except Exception:
                pass

        return DataProfile(
            n_samples=n,
            n_features=d,
            has_labels=has_labels,
            is_linear=is_linear,
            is_high_dim=d > 20,
            label_is_continuous=label_is_continuous,
            sparsity=sparsity,
        )

    # ── Selection ─────────────────────────────────────────────────────────────

    def _select(self, p: DataProfile) -> tuple[str, dict]:
        """
        Decision DAG — Surgical ML selection logic.

          Labelled + linear + sparse  → Lasso
          Labelled + linear           → Ridge
          Labelled + nonlinear + large + continuous    → XGBoost
          Labelled + nonlinear + large + categorical   → LightGBM
          Labelled + nonlinear + small                 → SVM (SVC/SVR)
          Unlabelled + n > DBSCAN_MAX → K-Means
          Unlabelled                  → DBSCAN
        """
        overrides = self._nightly_params  # Grid Search mutations

        if p.has_labels:
            if p.is_sequential:
                if p.label_is_continuous:
                    algo = "lstm"
                    params = {"hidden_size": 64, "num_layers": 2, "bidirectional": False,
                              "dropout": 0.1, **overrides.get("lstm", {})}
                else:
                    algo = "gru"
                    params = {"hidden_size": 64, "num_layers": 2, "bidirectional": False,
                              "dropout": 0.1, **overrides.get("gru", {})}
            elif p.is_linear:
                if p.sparsity > self.LASSO_SPARSITY:
                    algo = "lasso"
                    params = {"alpha": 0.1, **overrides.get("lasso", {})}
                else:
                    algo = "ridge"
                    params = {"alpha": 1.0, **overrides.get("ridge", {})}
            elif p.n_samples >= self.HEAVY_N_THRESHOLD:
                # XGBoost for regression; LightGBM for classification (faster, lower memory)
                if p.is_high_dim and p.n_samples >= self.TORCH_N_THRESHOLD and p.label_is_continuous:
                    algo = "mlp"
                    params = {"hidden_units": 128, "num_layers": 3, "activation": "relu",
                              "dropout": 0.2, **overrides.get("mlp", {})}
                elif p.label_is_continuous:
                    algo = "xgboost"
                    params = {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1,
                              **overrides.get("xgboost", {})}
                else:
                    algo = "lightgbm"
                    params = {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.05,
                              "num_leaves": 31, **overrides.get("lightgbm", {})}
            else:
                # Small nonlinear datasets: SVM finds tight decision boundaries
                kernel = overrides.get("svm", {}).pop("kernel", "rbf") if overrides.get("svm") else "rbf"
                algo = "svm"
                params = {"C": 1.0, "kernel": kernel, "gamma": "scale",
                          **overrides.get("svm", {})}
        else:
            if p.n_samples > self.DBSCAN_MAX_N:
                algo = "kmeans"
                params = {"n_clusters": max(2, min(8, p.n_samples // 200)),
                          **overrides.get("kmeans", {})}
            else:
                algo = "dbscan"
                params = {"eps": 0.5, "min_samples": 5, **overrides.get("dbscan", {})}

        return algo, params

    # ── Fitting ───────────────────────────────────────────────────────────────

    def _fit(
        self, algo: str, params: dict, X: Any, y: Optional[Any], p: DataProfile
    ) -> tuple[Any, float]:
        # Apply PCA preprocessing for high-dimensional feature spaces
        X_fit = self._apply_pca(X, p) if p.is_high_dim else X
        try:
            if algo == "mlp":
                return self._fit_mlp(params, X_fit, y, p.label_is_continuous)
            if algo == "lstm":
                return self._fit_lstm(params, X, y, p.label_is_continuous)
            if algo == "gru":
                return self._fit_gru(params, X, y, p.label_is_continuous)
            if algo == "ridge":
                return self._fit_ridge(params, X_fit, y)
            if algo == "lasso":
                return self._fit_lasso(params, X_fit, y)
            if algo == "xgboost":
                return self._fit_xgboost(params, X_fit, y)
            if algo == "lightgbm":
                return self._fit_lgbm(params, X_fit, y, p.label_is_continuous)
            if algo == "random_forest":
                return self._fit_rf(params, X_fit, y, p.label_is_continuous)
            if algo == "svm":
                return self._fit_svm(params, X_fit, y, p.label_is_continuous)
            if algo == "dbscan":
                return self._fit_dbscan(params, X_fit)
            if algo == "kmeans":
                return self._fit_kmeans(params, X_fit)
        except ImportError as exc:
            logger.warning("MLAssembler: %s unavailable (%s), using mean fallback", algo, exc)
        except Exception as exc:
            logger.warning("MLAssembler: %s failed (%s), using mean fallback", algo, exc)

        import numpy as np
        fallback = float(np.mean(y)) if y is not None else 0.0
        return fallback, 0.0

    def _apply_pca(self, X: Any, p: DataProfile) -> Any:
        """Reduce high-dim feature space to sqrt(n_features) components before fitting."""
        try:
            from sklearn.decomposition import PCA
            n_components = max(2, min(int(p.n_features ** 0.5), p.n_samples - 1, p.n_features - 1))
            pca = PCA(n_components=n_components, svd_solver="auto")
            return pca.fit_transform(X)
        except Exception as exc:
            logger.debug("MLAssembler: PCA failed (%s), using raw features", exc)
            return X

    def _fit_ridge(self, params: dict, X: Any, y: Any) -> tuple[Any, float]:
        import numpy as np
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_score
        m = Ridge(alpha=params.get("alpha", 1.0))
        if len(X) >= 5:
            scores = cross_val_score(m, X, y, cv=min(5, len(X)), scoring="r2")
            conf = float(np.mean(scores))
        else:
            conf = 0.0
        m.fit(X, y)
        return m.predict(X), max(0.0, conf)

    def _fit_lasso(self, params: dict, X: Any, y: Any) -> tuple[Any, float]:
        import numpy as np
        from sklearn.linear_model import Lasso
        from sklearn.model_selection import cross_val_score
        m = Lasso(alpha=params.get("alpha", 0.1), max_iter=2000)
        if len(X) >= 5:
            scores = cross_val_score(m, X, y, cv=min(5, len(X)), scoring="r2")
            conf = float(np.mean(scores))
        else:
            conf = 0.0
        m.fit(X, y)
        return m.predict(X), max(0.0, conf)

    def _fit_xgboost(self, params: dict, X: Any, y: Any) -> tuple[Any, float]:
        import numpy as np
        from sklearn.model_selection import cross_val_score
        from xgboost import XGBRegressor
        m = XGBRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 4),
            learning_rate=params.get("learning_rate", 0.1),
            verbosity=0,
        )
        if len(X) >= 5:
            scores = cross_val_score(m, X, y, cv=min(5, len(X)), scoring="r2")
            conf = float(np.mean(scores))
        else:
            conf = 0.0
        m.fit(X, y)
        return m.predict(X), max(0.0, conf)

    def _fit_rf(
        self, params: dict, X: Any, y: Any, continuous: bool
    ) -> tuple[Any, float]:
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.model_selection import cross_val_score
        cls = RandomForestRegressor if continuous else RandomForestClassifier
        m = cls(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth"),
            n_jobs=-1,
        )
        scoring = "r2" if continuous else "accuracy"
        if len(X) >= 5:
            scores = cross_val_score(m, X, y, cv=min(5, len(X)), scoring=scoring)
            conf = float(np.mean(scores))
        else:
            conf = 0.0
        m.fit(X, y)
        return m.predict(X), max(0.0, conf)

    def _fit_dbscan(self, params: dict, X: Any) -> tuple[Any, float]:
        from sklearn.cluster import DBSCAN
        from sklearn.metrics import silhouette_score
        m = DBSCAN(
            eps=params.get("eps", 0.5),
            min_samples=params.get("min_samples", 5),
            n_jobs=-1,
        )
        labels = m.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        conf = 0.0
        if n_clusters >= 2:
            try:
                conf = float(silhouette_score(X, labels))
            except Exception:
                pass
        return labels, max(0.0, conf)

    def _fit_kmeans(self, params: dict, X: Any) -> tuple[Any, float]:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        k = params.get("n_clusters", 4)
        m = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = m.fit_predict(X)
        conf = 0.0
        try:
            conf = float(silhouette_score(X, labels))
        except Exception:
            pass
        return labels, max(0.0, conf)

    def _fit_svm(
        self, params: dict, X: Any, y: Any, continuous: bool
    ) -> tuple[Any, float]:
        import numpy as np
        from sklearn.model_selection import cross_val_score
        from sklearn.svm import SVC, SVR
        cls = SVR if continuous else SVC
        m = cls(
            C=params.get("C", 1.0),
            kernel=params.get("kernel", "rbf"),
            gamma=params.get("gamma", "scale"),
        )
        scoring = "r2" if continuous else "accuracy"
        if len(X) >= 5:
            scores = cross_val_score(m, X, y, cv=min(5, len(X)), scoring=scoring)
            conf = float(np.mean(scores))
        else:
            conf = 0.0
        m.fit(X, y)
        return m.predict(X), max(0.0, conf)

    def _fit_lgbm(
        self, params: dict, X: Any, y: Any, continuous: bool
    ) -> tuple[Any, float]:
        import numpy as np
        from lightgbm import LGBMClassifier, LGBMRegressor
        from sklearn.model_selection import cross_val_score
        cls = LGBMRegressor if continuous else LGBMClassifier
        m = cls(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            num_leaves=params.get("num_leaves", 31),
            verbose=-1,
        )
        scoring = "r2" if continuous else "accuracy"
        if len(X) >= 5:
            scores = cross_val_score(m, X, y, cv=min(5, len(X)), scoring=scoring)
            conf = float(np.mean(scores))
        else:
            conf = 0.0
        m.fit(X, y)
        return m.predict(X), max(0.0, conf)

    def _fit_mlp(
        self, params: dict, X: Any, y: Any, continuous: bool
    ) -> tuple[Any, float]:
        import torch

        from prism_torch_models import PrismMLP, TorchTrainer
        in_features = X.shape[1] if X.ndim == 2 else 1
        out_features = 1 if continuous else max(2, len(set(y.tolist())))
        model = PrismMLP(
            in_features=in_features,
            out_features=out_features,
            hidden_units=params.get("hidden_units", 128),
            num_layers=params.get("num_layers", 3),
            activation=params.get("activation", "relu"),
            dropout=params.get("dropout", 0.2),
        )
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        task_type = "regression" if continuous else "classification"
        conf = TorchTrainer().fit(
            model, X_t, y_t, task_type=task_type,
            epochs=params.get("epochs", 50),
            batch_size=params.get("batch_size", 32),
        )
        model.eval()
        with torch.no_grad():
            pred = model(X_t).squeeze(-1).numpy()
        return pred, conf

    def _fit_lstm(
        self, params: dict, X: Any, y: Any, continuous: bool
    ) -> tuple[Any, float]:
        import torch

        from prism_torch_models import PrismLSTM, TorchTrainer
        X_3d = X.reshape(len(X), 1, -1) if X.ndim == 2 else X.reshape(len(X), 1, 1)
        in_features = X_3d.shape[2]
        out_features = 1 if continuous else max(2, len(set(y.tolist())))
        model = PrismLSTM(
            input_size=in_features,
            out_features=out_features,
            hidden_size=params.get("hidden_size", 64),
            num_layers=params.get("num_layers", 2),
            bidirectional=params.get("bidirectional", False),
            dropout=params.get("dropout", 0.1),
        )
        X_t = torch.tensor(X_3d, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        task_type = "regression" if continuous else "classification"
        conf = TorchTrainer().fit(
            model, X_t, y_t, task_type=task_type,
            epochs=params.get("epochs", 50),
            batch_size=params.get("batch_size", 32),
        )
        model.eval()
        with torch.no_grad():
            pred = model(X_t).squeeze(-1).numpy()
        return pred, conf

    def _fit_gru(
        self, params: dict, X: Any, y: Any, continuous: bool
    ) -> tuple[Any, float]:
        import torch

        from prism_torch_models import PrismGRU, TorchTrainer
        X_3d = X.reshape(len(X), 1, -1) if X.ndim == 2 else X.reshape(len(X), 1, 1)
        in_features = X_3d.shape[2]
        out_features = 1 if continuous else max(2, len(set(y.tolist())))
        model = PrismGRU(
            input_size=in_features,
            out_features=out_features,
            hidden_size=params.get("hidden_size", 64),
            num_layers=params.get("num_layers", 2),
            bidirectional=params.get("bidirectional", False),
            dropout=params.get("dropout", 0.1),
        )
        X_t = torch.tensor(X_3d, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        task_type = "regression" if continuous else "classification"
        conf = TorchTrainer().fit(
            model, X_t, y_t, task_type=task_type,
            epochs=params.get("epochs", 50),
            batch_size=params.get("batch_size", 32),
        )
        model.eval()
        with torch.no_grad():
            pred = model(X_t).squeeze(-1).numpy()
        return pred, conf

    # ── Semantic translation (end-of-DAG only) ────────────────────────────────

    def _translate(
        self,
        task: str,
        algo: str,
        prediction: Any,
        confidence: float,
        feature_names: Optional[list[str]],
    ) -> str:
        if self._llm is None:
            return ""
        try:
            import numpy as np
            pred_summary = (
                f"mean={float(np.mean(prediction)):.3f}, std={float(np.std(prediction)):.3f}"
                if hasattr(prediction, "__len__") and len(prediction) > 1
                else str(prediction)
            )
            feat_hint = f"Features: {', '.join(feature_names[:8])}." if feature_names else ""
            prompt = (
                f"Translate this ML result into one clear sentence for the user.\n"
                f"Task: {task}\n"
                f"Algorithm: {algo} (confidence/R²={confidence:.2f})\n"
                f"Prediction summary: {pred_summary}\n"
                f"{feat_hint}\n"
                f"Be direct. No jargon. No mention of algorithm names."
            )
            response, _ = self._llm.call(prompt)
            return response.strip()
        except Exception as exc:
            logger.debug("MLAssembler: translation failed: %s", exc)
            return ""

    # ── Outcome logging ───────────────────────────────────────────────────────

    def _log_outcome(self, result: AssemblyResult, task: str) -> None:
        if self._tracker is None:
            return
        try:
            self._tracker.record_ml_result(
                result_id=result.result_id,
                task=task,
                algorithm=result.algorithm,
                confidence=result.confidence,
                duration_ms=result.duration_ms,
                error=result.error,
            )
        except AttributeError:
            pass  # outcome tracker version without record_ml_result
        except Exception as exc:
            logger.debug("MLAssembler: outcome log failed: %s", exc)

    def _fallback(
        self, task: str, result_id: str, t0: float, reason: str
    ) -> AssemblyResult:
        logger.warning("MLAssembler fallback: %s", reason)
        return AssemblyResult(
            result_id=result_id,
            task=task,
            algorithm="fallback_mean",
            prediction=0.0,
            confidence=0.0,
            params={},
            explanation="",
            duration_ms=(time.time() - t0) * 1000,
            error=reason,
        )


# ---------------------------------------------------------------------------
# Nightly Grid Search sweep (called by prism_crystalliser.py daemon)
# ---------------------------------------------------------------------------

def run_nightly_sweep(
    assembler: MLAssembler,
    outcome_tracker: Any,
    error_threshold: float = MLAssembler.ERROR_THRESHOLD,
) -> dict[str, dict]:
    """
    Retrieve failed ML outcomes (confidence < 1-error_threshold) and run
    Grid Search to improve hyperparameters. Mutates assembler._nightly_params.
    Returns updated params dict for logging.
    """
    try:
        failed = _get_failed_outcomes(outcome_tracker, error_threshold)
        if not failed:
            return {}

        updated: dict[str, dict] = {}
        for algo, records in failed.items():
            best = _grid_search(algo, records)
            if best:
                assembler.apply_grid_search_params(algo, best)
                updated[algo] = best

        return updated
    except Exception as exc:
        logger.warning("Nightly ML sweep failed: %s", exc)
        return {}


def _get_failed_outcomes(tracker: Any, threshold: float) -> dict[str, list[dict]]:
    failed: dict[str, list[dict]] = {}
    try:
        rows = tracker.get_ml_results(min_error=threshold)
        for row in rows:
            algo = row.get("algorithm", "unknown")
            failed.setdefault(algo, []).append(row)
    except Exception:
        pass
    return failed


def _grid_search(algo: str, records: list[dict]) -> dict:
    """
    Hyperband successive halving over param candidates, scored by confidence proxy.

    Without stored X/y matrices we proxy fitness via algorithm-specific heuristics
    derived from the failed records' average confidence. Successive halving
    eliminates the bottom half each round until one candidate remains.
    """
    grids: dict[str, list[dict]] = {
        "ridge":         [{"alpha": a} for a in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]],
        "lasso":         [{"alpha": a} for a in [0.0001, 0.001, 0.01, 0.1, 1.0]],
        "xgboost":       [{"n_estimators": n, "max_depth": d, "learning_rate": lr}
                          for n in [50, 100, 200]
                          for d in [3, 4, 6]
                          for lr in [0.05, 0.1]],
        "lightgbm":      [{"n_estimators": n, "max_depth": d, "learning_rate": lr,
                           "num_leaves": nl}
                          for n in [50, 100, 200]
                          for d in [4, 6, 8]
                          for lr in [0.05, 0.1]
                          for nl in [15, 31, 63]],
        "svm":           [{"C": c, "kernel": k, "gamma": "scale"}
                          for c in [0.1, 1.0, 10.0, 100.0]
                          for k in ["rbf", "linear", "poly"]],
        "random_forest": [{"n_estimators": n, "max_depth": d}
                          for n in [50, 100, 200]
                          for d in [None, 5, 10]],
        "dbscan":        [{"eps": e, "min_samples": m}
                          for e in [0.3, 0.5, 1.0]
                          for m in [3, 5, 10]],
        "kmeans":        [{"n_clusters": k} for k in range(2, 10)],
        "mlp":  [{"hidden_units": h, "num_layers": nl, "activation": a, "dropout": d}
                 for h in [64, 128, 256]
                 for nl in [2, 3, 4]
                 for a in ["relu", "gelu"]
                 for d in [0.1, 0.2, 0.3]],
        "lstm": [{"hidden_size": h, "num_layers": nl, "dropout": d}
                 for h in [32, 64, 128]
                 for nl in [1, 2, 3]
                 for d in [0.1, 0.2]],
        "gru":  [{"hidden_size": h, "num_layers": nl, "dropout": d}
                 for h in [32, 64, 128]
                 for nl in [1, 2, 3]
                 for d in [0.1, 0.2]],
    }

    candidates = grids.get(algo, [])
    if not candidates:
        return {}

    avg_conf = float(sum(r.get("confidence", 0.0) for r in records)) / max(len(records), 1)

    # Successive halving: score all, eliminate bottom half each round
    scored = [(c, _score_candidate(algo, c, avg_conf)) for c in candidates]
    while len(scored) > 1:
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[: max(1, len(scored) // 2)]

    return scored[0][0]


def _score_candidate(algo: str, params: dict, avg_conf: float) -> float:
    """
    Heuristic fitness for a param candidate given average confidence of failed records.
    Low avg_conf → prefer simpler / more-regularized params.
    High avg_conf → prefer more expressive params.
    """
    if algo in ("ridge", "lasso"):
        alpha = params.get("alpha", 1.0)
        # Failed with low confidence → underfitting; try less regularization
        target = 0.001 if avg_conf < 0.25 else (0.1 if avg_conf < 0.55 else 1.0)
        return -abs(alpha - target)
    if algo == "xgboost":
        n = params.get("n_estimators", 100)
        d = params.get("max_depth", 4)
        lr = params.get("learning_rate", 0.1)
        if avg_conf < 0.3:
            return -(n / 200.0) - (d / 6.0) + lr  # prefer shallow + high lr
        return (n / 200.0) - (d / 6.0)             # prefer many shallow trees
    if algo == "lightgbm":
        n = params.get("n_estimators", 100)
        nl = params.get("num_leaves", 31)
        lr = params.get("learning_rate", 0.05)
        return (n / 200.0) - (nl / 63.0) + lr if avg_conf >= 0.3 else -(n / 200.0) + lr
    if algo == "svm":
        c = params.get("C", 1.0)
        # Low confidence → lower C (less overfit); high confidence → higher C
        target_c = 0.1 if avg_conf < 0.4 else 10.0
        return -abs(c - target_c)
    if algo == "random_forest":
        n = params.get("n_estimators", 100)
        d = params.get("max_depth") or 15
        return (n / 200.0) - (d / 15.0)
    if algo == "dbscan":
        eps = params.get("eps", 0.5)
        return eps if avg_conf < 0.4 else -eps
    if algo == "kmeans":
        k = params.get("n_clusters", 4)
        return -abs(k - 5)  # prefer moderate cluster count
    if algo == "mlp":
        h = params.get("hidden_units", 128)
        nl = params.get("num_layers", 3)
        d = params.get("dropout", 0.2)
        if avg_conf < 0.3:
            return (h / 256.0) + (nl / 4.0) - d
        return -(h / 256.0) - (nl / 4.0) + d
    if algo in ("lstm", "gru"):
        h = params.get("hidden_size", 64)
        nl = params.get("num_layers", 2)
        if avg_conf < 0.3:
            return (h / 128.0) + (nl / 3.0)
        return -(h / 128.0) - (nl / 3.0)
    return 0.0
