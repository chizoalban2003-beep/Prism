"""
prism_torch_models.py
=====================
PyTorch model definitions for the Surgical ML Assembler.

All classes are imported lazily by prism_ml_assembler — this module is only
loaded when torch is available.  Install via:  pip install ".[torch]"

Models
------
  PrismMLP   — Dynamic feedforward array for high-dim tabular inference.
               Mutates num_layers, hidden_units, activation (ReLU/GELU), dropout.

  PrismLSTM  — Recurrent topology for sequential / time-series data.
               Configures hidden_size, num_layers, bidirectional flag.

  PrismGRU   — Lighter GRU variant.  Preferred for classification sequences.

TorchTrainer
------------
  Binds the right loss (HuberLoss / CrossEntropyLoss) to AdamW or SGD,
  runs a mini-batch training loop, and returns a confidence score
  (R² for regression, accuracy for classification).
"""
from __future__ import annotations

# PyTorch is an optional heavy dependency. This module is imported lazily and
# only from inside prism_ml_assembler's guarded try/except, which falls back to
# sklearn / a mean-predictor when the import fails — so a torch-less box (this
# one) degrades gracefully rather than crashing. We re-raise with a clear,
# actionable message instead of a bare ModuleNotFoundError for anyone who
# imports this directly.
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as _exc:  # pragma: no cover - depends on host having torch
    raise ImportError(
        "prism_torch_models needs PyTorch — install with "
        "`pip install \".[torch]\"`. Without it PRISM automatically falls "
        "back to sklearn / a mean-predictor (see prism_ml_assembler)."
    ) from _exc

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu":  nn.ReLU,
    "gelu":  nn.GELU,
    "tanh":  nn.Tanh,
    "silu":  nn.SiLU,
}


# ---------------------------------------------------------------------------
# PrismMLP
# ---------------------------------------------------------------------------

class PrismMLP(nn.Module):
    """
    Dynamically-sized feedforward network for tabular regression / classification.

    Parameters
    ----------
    in_features   : number of input features
    out_features  : output size (1 for regression; n_classes for classification)
    hidden_units  : width of each hidden layer
    num_layers    : depth (number of hidden layers)
    activation    : "relu" | "gelu" | "tanh" | "silu"
    dropout       : dropout rate applied after each hidden activation
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int   = 1,
        hidden_units: int   = 64,
        num_layers:   int   = 2,
        activation:   str   = "relu",
        dropout:      float = 0.2,
    ) -> None:
        super().__init__()
        act_cls = _ACTIVATIONS.get(activation.lower(), nn.ReLU)
        layers: list[nn.Module] = []
        prev = in_features
        for _ in range(num_layers):
            layers += [nn.Linear(prev, hidden_units), act_cls(), nn.Dropout(dropout)]
            prev = hidden_units
        layers.append(nn.Linear(prev, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# PrismLSTM
# ---------------------------------------------------------------------------

class PrismLSTM(nn.Module):
    """
    Recurrent LSTM topology.  Input expected as (batch, seq_len, input_size).

    Parameters
    ----------
    input_size    : number of features per time step
    out_features  : output size
    hidden_size   : LSTM hidden state width
    num_layers    : stacked LSTM depth
    bidirectional : if True, doubles effective hidden_size at the head
    dropout       : applied between LSTM layers (only when num_layers > 1)
    """

    def __init__(
        self,
        input_size:   int,
        out_features: int   = 1,
        hidden_size:  int   = 64,
        num_layers:   int   = 2,
        bidirectional: bool = False,
        dropout:      float = 0.1,
    ) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = bidirectional,
            dropout       = dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Linear(out_dim, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])  # last time-step


# ---------------------------------------------------------------------------
# PrismGRU
# ---------------------------------------------------------------------------

class PrismGRU(nn.Module):
    """
    Recurrent GRU topology.  Lighter than LSTM; preferred for classification.
    Same interface as PrismLSTM.
    """

    def __init__(
        self,
        input_size:   int,
        out_features: int   = 1,
        hidden_size:  int   = 64,
        num_layers:   int   = 2,
        bidirectional: bool = False,
        dropout:      float = 0.1,
    ) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = bidirectional,
            dropout       = dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Linear(out_dim, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


# ---------------------------------------------------------------------------
# TorchTrainer
# ---------------------------------------------------------------------------

class TorchTrainer:
    """
    Thin training harness shared by all three model types.

    Loss selection
    --------------
      regression     → nn.HuberLoss()      (robust to outliers)
      classification → nn.CrossEntropyLoss()

    Optimiser selection
    -------------------
      "adamw" (default) → torch.optim.AdamW
      "sgd"             → torch.optim.SGD  (momentum=0.9)
    """

    def fit(
        self,
        model:          nn.Module,
        X:              torch.Tensor,
        y:              torch.Tensor,
        task_type:      str   = "regression",
        lr:             float = 1e-3,
        weight_decay:   float = 1e-4,
        epochs:         int   = 50,
        batch_size:     int   = 32,
        optimizer_name: str   = "adamw",
    ) -> float:
        """
        Train *model* in-place and return confidence.

        Returns
        -------
        float
            R² score for regression; accuracy for classification. Clamped to [0, 1].
        """
        loss_fn: nn.Module = (
            nn.HuberLoss() if task_type == "regression" else nn.CrossEntropyLoss()
        )

        if optimizer_name.lower() == "sgd":
            opt: torch.optim.Optimizer = torch.optim.SGD(
                model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
            )
        else:
            opt = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )

        eff_batch = min(batch_size, len(X))
        loader = DataLoader(
            TensorDataset(X, y), batch_size=eff_batch, shuffle=True
        )

        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                opt.zero_grad()
                pred = model(xb)
                if task_type == "regression":
                    loss = loss_fn(pred.squeeze(-1), yb)
                else:
                    loss = loss_fn(pred, yb.long())
                loss.backward()
                opt.step()

        return self._confidence(model, X, y, task_type)

    @staticmethod
    def _confidence(
        model:     nn.Module,
        X:         torch.Tensor,
        y:         torch.Tensor,
        task_type: str,
    ) -> float:
        model.eval()
        with torch.no_grad():
            pred = model(X)
        if task_type == "regression":
            return _r2(y, pred.squeeze(-1))
        preds = pred.argmax(dim=-1)
        return float((preds == y.long()).float().mean())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r2(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot == 0.0:
        return 0.0
    return max(0.0, 1.0 - ss_res / ss_tot)
