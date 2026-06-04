import os
import sys
import argparse
import shutil

import numpy as np
import pandas as pd


def _ensure_import_paths():
    """Allow reusing baseline modules in ../meow without packaging changes."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(this_dir, ".."))
    meow_dir = os.path.join(root_dir, "meow")
    meow_self_dir = os.path.join(root_dir, "meow_self")

    for p in [this_dir, meow_dir, meow_self_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)


_ensure_import_paths()

from log import log
from dl import MeowDataLoader
from feat import MeowFeatureGenerator
try:
    from feat_self import MeowSelfFeatureGenerator
except Exception:
    MeowSelfFeatureGenerator = None
from eval import MeowEvaluator
from tradingcalendar import Calendar

from seq_ds import build_sequences
from lstm_mdl import LSTMConfig, LSTMRegressor


class MeowLSTMEngine(object):
    def __init__(
        self,
        h5dir,
        cacheDir,
        *,
        lookback: int,
        config: LSTMConfig,
        max_rows_per_date: int = 0,
        max_symbols_per_date: int = 0,
    ):
        self.calendar = Calendar()
        self.h5dir = h5dir
        if not os.path.exists(h5dir):
            raise ValueError("Data directory not exists: {}".format(self.h5dir))
        if not os.path.isdir(h5dir):
            raise ValueError("Invalid data directory: {}".format(self.h5dir))

        self.cacheDir = cacheDir
        self.lookback = int(lookback)
        self.max_rows_per_date = int(max_rows_per_date)
        self.max_symbols_per_date = int(max_symbols_per_date)
        self.dloader = MeowDataLoader(h5dir=h5dir)
        self.featGenerator = MeowFeatureGenerator(cacheDir=cacheDir)
        self.feature_cols = self.featGenerator.featureNames()
        self.ycol = self.featGenerator.ycol
        self.model = LSTMRegressor(cacheDir=cacheDir, input_size=len(self.feature_cols), config=config)
        self.evaluator = MeowEvaluator(cacheDir=cacheDir)

        self._mu = None
        self._sigma = None

    def _load_dates(self, dates):
        # For sequence models, random row subsampling can destroy continuity and
        # lead to 0 samples. Prefer capping by symbols instead.
        if self.max_rows_per_date and self.max_rows_per_date > 0:
            log.inf(
                "Warning: --max-rows-per-date may break sequences; prefer --max-symbols-per-date for LSTM."
            )

        dfs = []
        for d in dates:
            df = self.dloader.loadDate(int(d))
            if self.max_symbols_per_date and self.max_symbols_per_date > 0:
                syms = df["symbol"].dropna().unique().tolist()
                syms = sorted(syms)[: int(self.max_symbols_per_date)]
                df = df[df["symbol"].isin(syms)]
            dfs.append(df)
        if not dfs:
            raise ValueError("Dates empty")
        return pd.concat(dfs, axis=0)

    def _fit_scaler(self, xdf):
        x = xdf[self.feature_cols].to_numpy(dtype=np.float32, copy=False)
        mu = x.mean(axis=0)
        sigma = x.std(axis=0)
        sigma[sigma == 0] = 1.0
        self._mu = mu
        self._sigma = sigma

    def _apply_scaler(self, xdf):
        if self._mu is None or self._sigma is None:
            raise RuntimeError("Scaler not fitted")
        x = xdf[self.feature_cols].to_numpy(dtype=np.float32, copy=False)
        x = (x - self._mu) / self._sigma
        out = xdf.copy()
        out.loc[:, self.feature_cols] = x
        return out

    def fit(self, startDate, endDate):
        dates = self.calendar.range(startDate, endDate)
        rawData = self._load_dates(dates)
        log.inf("Running LSTM model fitting...")
        xdf, ydf = self.featGenerator.genFeatures(rawData)

        self._fit_scaler(xdf)
        xdf = self._apply_scaler(xdf)

        seq = build_sequences(
            xdf,
            ydf,
            lookback=self.lookback,
            feature_cols=self.feature_cols,
            ycol=self.ycol,
        )
        log.inf("Train sequences: {} samples".format(seq.x.shape[0]))
        self.model.fit(seq.x, seq.y)

    def eval(self, startDate, endDate):
        log.inf("Running model evaluation...")
        dates = self.calendar.range(startDate, endDate)
        rawData = self._load_dates(dates)
        xdf, ydf = self.featGenerator.genFeatures(rawData)
        xdf = self._apply_scaler(xdf)

        seq = build_sequences(
            xdf,
            ydf,
            lookback=self.lookback,
            feature_cols=self.feature_cols,
            ycol=self.ycol,
        )
        log.inf("Test sequences: {} samples".format(seq.x.shape[0]))
        pred = self.model.predict(seq.x)

        # Build ydf subset aligned to predictions
        idx = np.asarray(seq.indices, dtype=object)
        ydf_reset = ydf.reset_index()
        pred_df = pd.DataFrame(
            {
                "symbol": idx[:, 0],
                "date": idx[:, 1].astype(int),
                "interval": idx[:, 2].astype(int),
                "forecast": pred.astype(np.float32, copy=False),
            }
        )
        # Make sure merge keys have the same dtype
        pred_df["symbol"] = pred_df["symbol"].astype(ydf_reset["symbol"].dtype, copy=False)

        aligned = ydf_reset.merge(
            pred_df,
            on=["symbol", "date", "interval"],
            how="right",
        ).set_index(["symbol", "date", "interval"])

        self.evaluator.eval(aligned)


if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    parser = argparse.ArgumentParser(description="MEOW LSTM")
    parser.add_argument("--h5dir", default=os.path.join(root_dir, "archive"))
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--feature-set", choices=["baseline", "self"], default="baseline")
    parser.add_argument(
        "--max-rows-per-date",
        type=int,
        default=0,
        help="Optional cap on raw rows loaded per trading day (random subsample) to reduce peak memory.",
    )
    parser.add_argument(
        "--max-symbols-per-date",
        type=int,
        default=0,
        help="Optional cap on symbols per trading day (keeps full intraday sequences per symbol).",
    )

    parser.add_argument("--train-start", type=int, default=20230601)
    parser.add_argument("--train-end", type=int, default=20231031)
    parser.add_argument("--test-start", type=int, default=20231101)
    parser.add_argument("--test-end", type=int, default=20231229)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    cfg = LSTMConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    engine = MeowLSTMEngine(
        h5dir=args.h5dir,
        cacheDir=None,
        lookback=args.lookback,
        config=cfg,
        max_rows_per_date=args.max_rows_per_date,
        max_symbols_per_date=args.max_symbols_per_date,
    )
    if args.feature_set == "self":
        if MeowSelfFeatureGenerator is None:
            raise ImportError("meow_self feature generator not available")
        engine.featGenerator = MeowSelfFeatureGenerator(cacheDir=None)
        engine.feature_cols = engine.featGenerator.featureNames()
        engine.ycol = engine.featGenerator.ycol
        engine.model = LSTMRegressor(cacheDir=None, input_size=len(engine.feature_cols), config=cfg)

    engine.fit(args.train_start, args.train_end)
    engine.eval(args.test_start, args.test_end)

    # Preserve weights with a stable, feature-set-specific filename.
    weights_dir = os.path.abspath(os.path.join(root_dir, "weights"))
    src = os.path.join(weights_dir, "meow_lstm.pt")
    if os.path.exists(src):
        dst = os.path.join(weights_dir, f"meow_lstm_{args.feature_set}.pt")
        try:
            shutil.copy2(src, dst)
            log.inf(f"Copied model weights to {dst}")
        except Exception:
            pass
