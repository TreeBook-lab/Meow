import os
from pathlib import Path
import sys
import argparse

import numpy as np
import pandas as pd


def _ensure_import_paths():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(this_dir, ".."))
    meow_dir = os.path.join(root_dir, "meow")
    for p in [this_dir, meow_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)


_ensure_import_paths()

from log import log
from dl import MeowDataLoader
from eval import MeowEvaluator
from tradingcalendar import Calendar

from feat_self import MeowSelfFeatureGenerator
from seq_ds import build_sequence_index
from patchtst import PatchTSTConfig, PatchTSTRegressor


def pearson_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)


def pearson_torch(pred, y, eps=1e-8):
    # differentiable correlation (batch)
    pred = pred - pred.mean()
    y = y - y.mean()
    num = (pred * y).sum()
    denom = (pred.pow(2).sum().sqrt() * y.pow(2).sum().sqrt()).clamp_min(eps)
    return num / denom


class MeowSelfTrainer(object):
    def __init__(
        self,
        h5dir: str,
        *,
        lookback: int,
        cross_day: bool,
        cfg_model: PatchTSTConfig,
        lr: float,
        weight_decay: float,
        epochs: int,
        batch_size: int,
        corr_weight: float,
        grad_clip: float,
        val_days: int,
        num_workers: int,
        seed: int = 1,
        max_train_samples: int = 0,
        max_val_samples: int = 0,
        max_test_samples: int = 0,
        select_metric: str = "pearson",
        warmup_days: int = 0,
        max_rows_per_date: int = 0,
    ):
        self.calendar = Calendar()
        self.h5dir = h5dir
        self.dloader = MeowDataLoader(h5dir=h5dir)
        self.feat = MeowSelfFeatureGenerator(cacheDir=None)
        self.feature_cols = self.feat.featureNames()
        self.ycol = self.feat.ycol
        self.lookback = int(lookback)
        self.cross_day = bool(cross_day)

        # Token block length (each token uses this many consecutive intervals).
        self.block_len = int(cfg_model.patch_len)
        if self.block_len < 1:
            raise ValueError(f"Invalid patch_len/block_len: {self.block_len}")
        if self.lookback % self.block_len != 0:
            raise ValueError(
                f"lookback ({self.lookback}) must be divisible by patch_len/block_len ({self.block_len}) "
                "for non-overlapping block tokens."
            )

        self.model = PatchTSTRegressor(input_size=len(self.feature_cols), config=cfg_model)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.corr_weight = float(corr_weight)
        self.grad_clip = float(grad_clip)
        self.val_days = int(val_days)
        self.num_workers = int(num_workers)

        self.seed = int(seed)
        self.max_train_samples = int(max_train_samples)
        self.max_val_samples = int(max_val_samples)
        self.max_test_samples = int(max_test_samples)
        self.select_metric = str(select_metric)
        self.warmup_days = int(warmup_days)
        self.max_rows_per_date = int(max_rows_per_date)
        self._rng = np.random.default_rng(self.seed)

        self._mu = None
        self._sigma = None

        self.evaluator = MeowEvaluator(cacheDir=None)

    def _filter_existing_dates(self, dates):
        """Keep only dates that have corresponding {date}.h5 under h5dir."""
        out = []
        for d in dates or []:
            p = os.path.join(self.h5dir, f"{int(d)}.h5")
            if os.path.exists(p):
                out.append(int(d))
        return out

    def _fit_scaler(self, xdf):
        x = xdf[self.feature_cols].to_numpy(dtype=np.float32, copy=False)
        mu = x.mean(axis=0)
        sigma = x.std(axis=0)
        sigma[sigma == 0] = 1.0
        self._mu, self._sigma = mu, sigma

    def _apply_scaler(self, xdf):
        x = xdf[self.feature_cols].to_numpy(dtype=np.float32, copy=False)
        x = (x - self._mu) / self._sigma
        out = xdf.copy()
        out.loc[:, self.feature_cols] = x
        return out

    def _make_loaders(
        self,
        xdf,
        ydf,
        *,
        max_samples: int = 0,
        shuffle: bool = True,
        allowed_dates: set = None,
    ):
        try:
            import torch
        except Exception as e:
            raise ImportError("PyTorch is required") from e

        class _WindowDataset(torch.utils.data.Dataset):
            def __init__(self, x_all_t, y_all_t, date_all_t, interval_all_t, end_pos, lookback: int, block_len: int):
                self.x_all_t = x_all_t
                self.y_all_t = y_all_t
                self.date_all_t = date_all_t
                self.interval_all_t = interval_all_t
                self.end_pos = end_pos
                self.lookback = int(lookback)
                self.block_len = int(block_len)

            def __len__(self):
                return int(self.end_pos.shape[0])

            def _apply_block_time_weights(self, x_win, date_win, interval_win):
                # x_win: (T, F)
                # Weight within each non-overlapping block of length `block_len`.
                # Use a monotonic timestamp so cross-day windows stay ordered.
                # Weight is linear in [0, 1] inside each block (larger timestamp => larger weight).
                import torch

                bl = self.block_len
                if bl <= 1:
                    return x_win
                t = x_win.shape[0]
                # date is YYYYMMDD, interval is like HHMMSS00. Combine to a monotonic timestamp.
                ts = date_win.to(torch.float32) * 1e9 + interval_win.to(torch.float32)
                w = torch.empty((t,), device=x_win.device, dtype=torch.float32)
                eps = 1e-6
                for s in range(0, t, bl):
                    e = min(s + bl, t)
                    ts_b = ts[s:e]
                    ts_min = ts_b.min()
                    denom = (ts_b.max() - ts_min).clamp_min(eps)
                    w[s:e] = (ts_b - ts_min) / denom
                return x_win * w.unsqueeze(-1)

            def __getitem__(self, i: int):
                e = int(self.end_pos[i])
                s = e - self.lookback + 1
                x_win = self.x_all_t[s : e + 1]
                date_win = self.date_all_t[s : e + 1]
                interval_win = self.interval_all_t[s : e + 1]
                x_win = self._apply_block_time_weights(x_win, date_win, interval_win)
                return x_win, self.y_all_t[e]

        idx = build_sequence_index(
            xdf,
            ydf,
            lookback=self.lookback,
            cross_day=self.cross_day,
            feature_cols=self.feature_cols,
            ycol=self.ycol,
        )

        end_pos = idx.end_pos
        if allowed_dates is not None:
            allowed_dates_set = set(int(d) for d in allowed_dates)
            if end_pos.shape[0] > 0:
                end_dates = idx.date_all[end_pos].astype(np.int64, copy=False)
                keep = np.isin(end_dates, np.asarray(sorted(allowed_dates_set), dtype=end_dates.dtype))
                end_pos = end_pos[keep]
        if max_samples and end_pos.shape[0] > max_samples:
            pick = self._rng.choice(end_pos.shape[0], size=max_samples, replace=False)
            end_pos = end_pos[pick]
            # Keep deterministic order for eval-like loaders
            if not shuffle:
                end_pos = np.sort(end_pos)

        if idx.end_pos.shape[0] == 0:
            raise ValueError(
                "No sequence samples were generated. "
                "Try reducing --lookback, or increasing the date range, "
                "or increasing --val-days for smoke tests, "
                "or set --cross-day 1 to allow per-symbol cross-day sequences."
            )

        x_all_t = torch.from_numpy(idx.x_all)
        y_all_t = torch.from_numpy(idx.y_all)
        date_all_t = torch.from_numpy(idx.date_all.astype(np.int64, copy=False))
        interval_all_t = torch.from_numpy(idx.interval_all.astype(np.int64, copy=False))
        ds = _WindowDataset(
            x_all_t,
            y_all_t,
            date_all_t,
            interval_all_t,
            end_pos,
            lookback=self.lookback,
            block_len=self.block_len,
        )
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=bool(shuffle),
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )
        return dl, idx, end_pos

    def fit(self, train_start: int, train_end: int):
        try:
            import torch
        except Exception as e:
            raise ImportError("PyTorch is required") from e

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        dates = self.calendar.range(train_start, train_end)
        if len(dates) <= self.val_days + 1:
            raise ValueError("Not enough train days for val split")
        val_dates = dates[-self.val_days :]
        tr_dates = dates[: -self.val_days]

        # Optional warmup: load extra history before the train/val windows.
        tr_warm = []
        va_warm = []
        if self.warmup_days and self.warmup_days > 0:
            tr_warm = self._filter_existing_dates(self.calendar.prevn(tr_dates[0], self.warmup_days) or [])
            va_warm = self._filter_existing_dates(self.calendar.prevn(val_dates[0], self.warmup_days) or [])

        log.inf(
            f"Loading train dates: {len(tr_dates)} (+warmup {len(tr_warm)}); "
            f"val dates: {len(val_dates)} (+warmup {len(va_warm)})"
        )
        tr_raw = self.dloader.loadDates(
            list(tr_warm) + list(tr_dates),
            max_rows_per_date=self.max_rows_per_date,
            seed=self.seed,
        )
        va_raw = self.dloader.loadDates(
            list(va_warm) + list(val_dates),
            max_rows_per_date=self.max_rows_per_date,
            seed=self.seed,
        )

        tr_xdf, tr_ydf = self.feat.genFeatures(tr_raw, cross_day=self.cross_day)
        va_xdf, va_ydf = self.feat.genFeatures(va_raw, cross_day=self.cross_day)

        self._fit_scaler(tr_xdf)
        tr_xdf = self._apply_scaler(tr_xdf)
        va_xdf = self._apply_scaler(va_xdf)

        tr_dl, _, _ = self._make_loaders(
            tr_xdf,
            tr_ydf,
            max_samples=self.max_train_samples,
            shuffle=True,
            allowed_dates=set(tr_dates),
        )
        va_dl, va_idx, va_end_pos = self._make_loaders(
            va_xdf,
            va_ydf,
            max_samples=self.max_val_samples,
            shuffle=False,
            allowed_dates=set(val_dates),
        )

        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        mse = torch.nn.MSELoss()

        try:
            from tqdm import tqdm
        except Exception as e:
            raise ImportError("tqdm is required for progress bar: pip install tqdm") from e

        best_val = -1e9
        best_state = None

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            pbar = tqdm(tr_dl, desc=f"epoch {epoch}/{self.epochs}")
            losses = []
            cors = []
            for xb, yb in pbar:
                xb = xb.to(device)
                yb = yb.to(device)

                opt.zero_grad(set_to_none=True)
                pred = self.model(xb)
                loss_mse = mse(pred, yb)
                # Train with pure MSE. Keep correlation only as a training-time metric.
                corr = pearson_torch(pred, yb)
                loss = loss_mse
                loss.backward()
                if self.grad_clip and self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                opt.step()

                losses.append(float(loss_mse.detach().cpu().item()))
                cors.append(float(corr.detach().cpu().item()))
                if len(losses) % 20 == 0:
                    pbar.set_postfix({"mse": np.mean(losses[-20:]), "corr": np.mean(cors[-20:])})

            # validation
            self.model.eval()
            vpred = []
            vy = []
            with torch.no_grad():
                for xb, yb in va_dl:
                    xb = xb.to(device)
                    pred = self.model(xb).detach().cpu().numpy()
                    vpred.append(pred)
                    vy.append(yb.numpy())

            vpred = np.concatenate(vpred)
            vy = np.concatenate(vy)

            # Reuse evaluator to compute global + IC-style metrics on validation set.
            va_end_pos_i = va_end_pos.astype(np.int64, copy=False)
            vdf = pd.DataFrame(
                {
                    "symbol": va_idx.symbol_all[va_end_pos_i],
                    "date": va_idx.date_all[va_end_pos_i].astype(int, copy=False),
                    "interval": va_idx.interval_all[va_end_pos_i].astype(int, copy=False),
                    "forecast": vpred.astype(np.float32, copy=False),
                    self.ycol: vy.astype(np.float32, copy=False),
                }
            ).set_index(["symbol", "date", "interval"])
            metrics = self.evaluator.eval(vdf)
            score = metrics.get(self.select_metric)
            if score is None:
                score = metrics.get("pearson")
            if score is None:
                score = float("-inf")

            log.inf(f"Epoch {epoch}: val select_metric={self.select_metric} score={score:.4f}")

            if score > best_val:
                best_val = score
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.model.state_dict().items()}

        if best_state is not None:
            self.model.model.load_state_dict(best_state)
            log.inf(f"Loaded best model by val {self.select_metric}={best_val:.4f}")

        weights_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights")))
        weights_dir.mkdir(parents=True, exist_ok=True)
        weight_path = weights_dir / "meow_self_patchtst.pt"
        torch.save(self.model.model.state_dict(), weight_path)
        log.inf(f"Saved model weights to {weight_path}")

    def eval(self, test_start: int, test_end: int):
        try:
            import torch
        except Exception as e:
            raise ImportError("PyTorch is required") from e

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        dates = self.calendar.range(test_start, test_end)
        warm = []
        if self.warmup_days and self.warmup_days > 0:
            warm = self._filter_existing_dates(self.calendar.prevn(dates[0], self.warmup_days) or [])
        raw = self.dloader.loadDates(
            list(warm) + list(dates),
            max_rows_per_date=self.max_rows_per_date,
            seed=self.seed,
        )
        xdf, ydf = self.feat.genFeatures(raw, cross_day=self.cross_day)
        xdf = self._apply_scaler(xdf)

        idx = build_sequence_index(
            xdf,
            ydf,
            lookback=self.lookback,
            cross_day=self.cross_day,
            feature_cols=self.feature_cols,
            ycol=self.ycol,
        )

        end_pos = idx.end_pos
        if end_pos.shape[0] > 0:
            end_dates = idx.date_all[end_pos].astype(np.int64, copy=False)
            keep = np.isin(end_dates, np.asarray(dates, dtype=end_dates.dtype))
            end_pos = end_pos[keep]
        if self.max_test_samples and end_pos.shape[0] > self.max_test_samples:
            pick = self._rng.choice(end_pos.shape[0], size=self.max_test_samples, replace=False)
            end_pos = np.sort(end_pos[pick])

        log.inf(f"Test sequences: {end_pos.shape[0]} samples")
        if end_pos.shape[0] == 0:
            raise ValueError(
                "No test sequence samples were generated. "
                "Try reducing --lookback, or expanding the test date range, "
                "or using --cross-day 1 with multiple test days."
            )

        class _WindowXDataset(torch.utils.data.Dataset):
            def __init__(self, x_all_t, end_pos, lookback: int):
                self.x_all_t = x_all_t
                self.end_pos = end_pos
                self.lookback = int(lookback)

            def __len__(self):
                return int(self.end_pos.shape[0])

            def __getitem__(self, i: int):
                e = int(self.end_pos[i])
                s = e - self.lookback + 1
                return self.x_all_t[s : e + 1]

        x_all_t = torch.from_numpy(idx.x_all).to(device)
        ds = _WindowXDataset(x_all_t, end_pos, lookback=self.lookback)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=False, drop_last=False)

        preds = []
        with torch.no_grad():
            for xb in dl:
                xb = xb.to(device)
                pred = self.model(xb).detach().cpu().numpy()
                preds.append(pred)
        preds = np.concatenate(preds)

        end_pos = end_pos.astype(np.int64, copy=False)
        sym_s = idx.symbol_all[end_pos]
        date_s = idx.date_all[end_pos]
        interval_s = idx.interval_all[end_pos]
        ydf_reset = ydf.reset_index()
        pred_df = pd.DataFrame(
            {
                "symbol": sym_s,
                "date": date_s.astype(int, copy=False),
                "interval": interval_s.astype(int, copy=False),
                "forecast": preds.astype(np.float32, copy=False),
            }
        )
        pred_df["symbol"] = pred_df["symbol"].astype(ydf_reset["symbol"].dtype, copy=False)

        aligned = ydf_reset.merge(pred_df, on=["symbol", "date", "interval"], how="right").set_index(
            ["symbol", "date", "interval"]
        )
        self.evaluator.eval(aligned)


if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    parser = argparse.ArgumentParser(description="MEOW Self: Feature engineering + PatchTST")
    parser.add_argument("--h5dir", default=os.path.join(root_dir, "archive"))

    parser.add_argument("--train-start", type=int, default=20230601)
    parser.add_argument("--train-end", type=int, default=20231130)
    parser.add_argument("--test-start", type=int, default=20231201)
    parser.add_argument("--test-end", type=int, default=20231229)

    parser.add_argument("--lookback", type=int, default=240)
    parser.add_argument(
        "--cross-day",
        type=int,
        default=0,
        help="1=build sequences per-symbol across dates; 0=within (symbol,date) only",
    )
    # Non-overlapping block tokenization: each token uses patch_len consecutive intervals.
    parser.add_argument("--patch-len", type=int, default=5)
    parser.add_argument("--patch-stride", type=int, default=5, help="Kept for compatibility; ignored in non-overlap mode")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument(
        "--attn-mode",
        type=str,
        default="reverse",
        help="Attention mask mode for patch tokens: reverse|causal|none. reverse=older attends newer; causal=newer attends older.",
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--corr-weight", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--val-days", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="0=use all samples; otherwise randomly subsample per epoch (helps CPU speed)",
    )
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)

    parser.add_argument(
        "--select-metric",
        type=str,
        default="pearson",
        help="Metric used to select best checkpoint. Options: pearson, spearman, mean_ic, mean_rank_ic, mean_sym_pearson, mean_sym_spearman",
    )

    parser.add_argument(
        "--warmup-days",
        type=int,
        default=0,
        help="Load extra trading days before train/val/test windows as input context. Samples ending in warmup dates are not used for training/eval.",
    )

    parser.add_argument(
        "--max-rows-per-date",
        type=int,
        default=0,
        help="Optional cap on raw rows loaded per trading day (random subsample) to reduce peak memory.",
    )

    args = parser.parse_args()

    cfg = PatchTSTConfig(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_feedforward=args.ff,
        dropout=args.dropout,
        attn_mode=args.attn_mode,
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
    )

    trainer = MeowSelfTrainer(
        h5dir=args.h5dir,
        lookback=args.lookback,
        cross_day=bool(args.cross_day),
        cfg_model=cfg,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        corr_weight=args.corr_weight,
        grad_clip=args.grad_clip,
        val_days=args.val_days,
        num_workers=args.num_workers,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        select_metric=args.select_metric,
        warmup_days=args.warmup_days,
        max_rows_per_date=args.max_rows_per_date,
    )

    trainer.fit(args.train_start, args.train_end)
    trainer.eval(args.test_start, args.test_end)
