from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from log import log


@dataclass
class LSTMConfig:
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 2048
    epochs: int = 5
    seed: int = 42


class LSTMRegressor(object):
    def __init__(self, cacheDir, input_size: int, config: Optional[LSTMConfig] = None):
        self.cacheDir = cacheDir
        self.config = config or LSTMConfig()
        self.input_size = int(input_size)
        self._torch = None
        self.model = None
        self.device = None
        self.weight_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights", "meow_lstm.pt"))

    def _lazy_init(self):
        if self.model is not None:
            return
        try:
            import torch
            import torch.nn as nn
        except Exception as e:
            raise ImportError(
                "PyTorch is required for LSTM. Install in Meow env: `pip install torch`."
            ) from e

        self._torch = torch
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        class _Net(nn.Module):
            def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                self.head = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                last = out[:, -1, :]
                return self.head(last).squeeze(-1)

        self.model = _Net(
            input_size=self.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(self, x_seq: np.ndarray, y: np.ndarray):
        self._lazy_init()
        torch = self._torch

        if x_seq.ndim != 3:
            raise ValueError(f"x_seq must be 3D (N,T,F), got shape={x_seq.shape}")

        x_seq = x_seq.astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)

        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(x_seq),
            torch.from_numpy(y),
        )
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )

        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        loss_fn = torch.nn.MSELoss()

        log.inf(
            "Fitting LSTM: epochs={}, batch_size={}, hidden={}, layers={} on {}".format(
                self.config.epochs,
                self.config.batch_size,
                self.config.hidden_size,
                self.config.num_layers,
                str(self.device),
            )
        )

        self.model.train()
        for epoch in range(1, self.config.epochs + 1):
            total = 0.0
            n = 0
            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                opt.zero_grad(set_to_none=True)
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

                bs = int(xb.shape[0])
                total += float(loss.detach().cpu().item()) * bs
                n += bs

            log.inf("Epoch {}/{}: train_mse={:.6g}".format(epoch, self.config.epochs, total / max(1, n)))

        log.inf("Done fitting")

        weights_dir = Path(os.path.dirname(self.weight_path))
        weights_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.weight_path)
        log.inf(f"Saved model weights to {self.weight_path}")

    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        self._lazy_init()
        torch = self._torch

        x_seq = x_seq.astype(np.float32, copy=False)
        ds = torch.utils.data.TensorDataset(torch.from_numpy(x_seq))
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        self.model.eval()
        outs = []
        with torch.no_grad():
            for (xb,) in dl:
                xb = xb.to(self.device)
                pred = self.model(xb).detach().cpu().numpy()
                outs.append(pred)

        if not outs:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(outs, axis=0)
