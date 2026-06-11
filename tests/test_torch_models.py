"""
Tests for prism_torch_models and PyTorch integration in prism_ml_assembler.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import torch as th
import torch.nn as nn

from prism_torch_models import PrismGRU, PrismLSTM, PrismMLP, TorchTrainer, _r2

# ---------------------------------------------------------------------------
# PrismMLP
# ---------------------------------------------------------------------------


class TestPrismMLP:
    def test_forward_shape_regression(self):
        model = PrismMLP(in_features=10, out_features=1)
        out = model(th.randn(8, 10))
        assert out.shape == (8, 1)

    def test_forward_shape_multiclass(self):
        model = PrismMLP(in_features=5, out_features=3)
        out = model(th.randn(4, 5))
        assert out.shape == (4, 3)

    def test_depth_respected(self):
        model = PrismMLP(in_features=4, out_features=1, num_layers=4)
        # 4 × [Linear, act, Dropout] + 1 Linear output = 13 children
        assert len(list(model.net.children())) == 4 * 3 + 1

    def test_activation_gelu(self):
        model = PrismMLP(in_features=4, out_features=1, activation="gelu")
        acts = [m for m in model.net.children() if isinstance(m, nn.GELU)]
        assert len(acts) == 2  # default num_layers=2

    def test_dropout_deterministic_in_eval(self):
        model = PrismMLP(in_features=4, out_features=1, dropout=0.5)
        model.eval()
        x = th.randn(10, 4)
        assert th.allclose(model(x), model(x))

    def test_unknown_activation_falls_back_to_relu(self):
        model = PrismMLP(in_features=4, out_features=1, activation="unknown")
        acts = [m for m in model.net.children() if isinstance(m, nn.ReLU)]
        assert len(acts) == 2


# ---------------------------------------------------------------------------
# PrismLSTM
# ---------------------------------------------------------------------------


class TestPrismLSTM:
    def test_forward_shape(self):
        model = PrismLSTM(input_size=5, out_features=1)
        out = model(th.randn(4, 10, 5))
        assert out.shape == (4, 1)

    def test_bidirectional_head_size(self):
        model = PrismLSTM(input_size=5, out_features=2, hidden_size=32, bidirectional=True)
        assert model.head.in_features == 64

    def test_unidirectional_head_size(self):
        model = PrismLSTM(input_size=5, out_features=1, hidden_size=32, bidirectional=False)
        assert model.head.in_features == 32

    def test_uses_last_timestep(self):
        model = PrismLSTM(input_size=3, out_features=1, hidden_size=8)
        model.eval()
        out = model(th.randn(2, 5, 3))
        assert out.shape == (2, 1)

    def test_rnn_is_lstm(self):
        model = PrismLSTM(input_size=4, out_features=1)
        assert isinstance(model.rnn, nn.LSTM)


# ---------------------------------------------------------------------------
# PrismGRU
# ---------------------------------------------------------------------------


class TestPrismGRU:
    def test_forward_shape(self):
        model = PrismGRU(input_size=6, out_features=1)
        out = model(th.randn(3, 8, 6))
        assert out.shape == (3, 1)

    def test_bidirectional_head_size(self):
        model = PrismGRU(input_size=4, out_features=1, hidden_size=16, bidirectional=True)
        assert model.head.in_features == 32

    def test_rnn_is_gru(self):
        model = PrismGRU(input_size=4, out_features=1)
        assert isinstance(model.rnn, nn.GRU)

    def test_forward_matches_lstm_interface(self):
        model = PrismGRU(input_size=3, out_features=1, hidden_size=8)
        model.eval()
        out = model(th.randn(5, 4, 3))
        assert out.shape == (5, 1)


# ---------------------------------------------------------------------------
# TorchTrainer
# ---------------------------------------------------------------------------


class TestTorchTrainer:
    def _reg_data(self):
        th.manual_seed(0)
        X = th.randn(50, 4)
        y = X[:, 0] * 2 + X[:, 1] - X[:, 2] + th.randn(50) * 0.1
        return X, y

    def _cls_data(self):
        th.manual_seed(0)
        X = th.randn(60, 4)
        y = (X[:, 0] > 0).float()
        return X, y

    def test_fit_returns_float(self):
        X, y = self._reg_data()
        conf = TorchTrainer().fit(PrismMLP(4, 1), X, y, epochs=5)
        assert isinstance(conf, float)

    def test_regression_confidence_in_range(self):
        X, y = self._reg_data()
        conf = TorchTrainer().fit(PrismMLP(4, 1, hidden_units=64, num_layers=3), X, y, epochs=30)
        assert 0.0 <= conf <= 1.0

    def test_classification_confidence_in_range(self):
        X, y = self._cls_data()
        conf = TorchTrainer().fit(PrismMLP(4, 2), X, y, task_type="classification", epochs=20)
        assert 0.0 <= conf <= 1.0

    def test_sgd_optimizer(self):
        X, y = self._reg_data()
        conf = TorchTrainer().fit(PrismMLP(4, 1), X, y, epochs=5, optimizer_name="sgd")
        assert isinstance(conf, float)

    def test_batch_size_larger_than_dataset_clamped(self):
        X, y = th.randn(5, 3), th.randn(5)
        conf = TorchTrainer().fit(PrismMLP(3, 1), X, y, batch_size=100, epochs=3)
        assert isinstance(conf, float)

    def test_lstm_fit(self):
        X = th.randn(20, 5, 4)
        y = th.randn(20)
        conf = TorchTrainer().fit(
            PrismLSTM(input_size=4, out_features=1, hidden_size=16, num_layers=1),
            X, y, epochs=5,
        )
        assert 0.0 <= conf <= 1.0

    def test_gru_fit(self):
        X = th.randn(20, 5, 4)
        y = th.randn(20)
        conf = TorchTrainer().fit(
            PrismGRU(input_size=4, out_features=1, hidden_size=16, num_layers=1),
            X, y, epochs=5,
        )
        assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# _r2 helper
# ---------------------------------------------------------------------------


class TestR2:
    def test_perfect_prediction(self):
        y = th.tensor([1.0, 2.0, 3.0])
        assert _r2(y, y) == 1.0

    def test_constant_y_returns_zero(self):
        y = th.tensor([5.0, 5.0, 5.0])
        assert _r2(y, y) == 0.0

    def test_clamped_to_zero_on_negative(self):
        y = th.tensor([1.0, 2.0, 3.0])
        pred = th.tensor([3.0, 1.0, 2.0])
        assert _r2(y, pred) >= 0.0

    def test_result_is_float(self):
        y = th.tensor([1.0, 2.0, 3.0])
        assert isinstance(_r2(y, y), float)


# ---------------------------------------------------------------------------
# MLAssembler torch routing
# ---------------------------------------------------------------------------


class TestMLAssemblerTorchIntegration:
    def test_mlp_selected_for_high_dim_large_continuous(self):
        from prism_ml_assembler import MLAssembler

        asm = MLAssembler()
        rng = np.random.default_rng(42)
        X = rng.random((120, 30))  # n >= 100, n_features > 20
        y = X[:, 0] * 3 + rng.random(120) * 0.1  # continuous
        result = asm.run("test", X, y, translate=False)
        assert result.algorithm == "mlp"

    def test_lstm_selected_for_sequential_continuous(self):
        from prism_ml_assembler import MLAssembler

        asm = MLAssembler()
        rng = np.random.default_rng(0)
        X = rng.random((60, 5))
        y = X[:, 0] * 2 + rng.random(60) * 0.1
        result = asm.run("test", X, y, translate=False, sequential=True)
        assert result.algorithm == "lstm"

    def test_gru_selected_for_sequential_categorical(self):
        from prism_ml_assembler import MLAssembler

        asm = MLAssembler()
        rng = np.random.default_rng(1)
        X = rng.random((60, 5))
        y = np.array([0] * 30 + [1] * 30, dtype=float)
        result = asm.run("test", X, y, translate=False, sequential=True)
        assert result.algorithm == "gru"

    def test_mlp_run_returns_prediction_and_confidence(self):
        from prism_ml_assembler import MLAssembler

        asm = MLAssembler()
        rng = np.random.default_rng(7)
        X = rng.random((110, 25))
        y = X[:, 0] * 2 + rng.random(110) * 0.1
        result = asm.run("test", X, y, translate=False)
        assert result.algorithm == "mlp"
        assert result.prediction is not None
        assert 0.0 <= result.confidence <= 1.0

    def test_sequential_flag_false_does_not_select_lstm(self):
        from prism_ml_assembler import MLAssembler

        asm = MLAssembler()
        rng = np.random.default_rng(5)
        X = rng.random((60, 5))
        y = X[:, 0] * 2 + rng.random(60) * 0.1
        result = asm.run("test", X, y, translate=False, sequential=False)
        assert result.algorithm != "lstm"
        assert result.algorithm != "gru"
