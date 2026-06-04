import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
from log import log


@dataclass
class LGBRankConfig:
    n_estimators: int = 800
    learning_rate: float = 0.05
    max_depth: int = -1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    random_state: int = 42
    n_jobs: int = max(1, os.cpu_count() or 1)
    rank_bins: int = 5


class MeowLGBRankModel(object):
    def __init__(self, cacheDir=None, config: Optional[LGBRankConfig] = None):
        self.cacheDir = cacheDir
        self.config = config or LGBRankConfig()
        self.estimator = None

    def _build(self):
        try:
            from lightgbm import LGBMRanker
        except Exception as e:
            raise ImportError("lightgbm is required for LGBMRanker") from e

        cfg = self.config
        return LGBMRanker(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_alpha=cfg.reg_alpha,
            reg_lambda=cfg.reg_lambda,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
        )

    def fit(self, xdf, ydf):
        if self.estimator is None:
            self.estimator = self._build()

        x = xdf.to_numpy(dtype=np.float32, copy=False)
        y = ydf.to_numpy(copy=False)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        y = y.astype(np.float32, copy=False)

        log.inf(
            f"Fitting LGBMRanker: n_estimators={self.config.n_estimators}, lr={self.config.learning_rate}, max_depth={self.config.max_depth}..."
        )

        try:
            idx = ydf.index
            dates = idx.get_level_values("date")
        except Exception:
            dates = np.array([0] * x.shape[0])

        order = np.argsort(dates)
        x_ord = x[order]
        y_ord = y[order]
        dates_ord = np.asarray(dates)[order]

        _, counts = np.unique(dates_ord, return_counts=True)
        y_rel = np.empty_like(y_ord, dtype=np.int32)
        start = 0
        for cnt in counts:
            grp = y_ord[start : start + cnt]
            if cnt <= 1:
                y_rel[start : start + cnt] = 0
            else:
                # Quantile binning within each date keeps labels compact and valid for LightGBM ranking.
                bins = max(2, int(self.config.rank_bins))
                try:
                    edges = np.quantile(grp, q=np.linspace(0, 1, bins + 1))
                except Exception:
                    edges = np.linspace(np.min(grp), np.max(grp), bins + 1)
                labels = np.digitize(grp, edges[1:-1], right=True).astype(np.int32)
                y_rel[start : start + cnt] = labels
            start += cnt

        groups = counts.tolist()
        try:
            log.inf(f"LGB rank labels min/max/dtype: {y_rel.min()}/{y_rel.max()}/{y_rel.dtype}")
        except Exception:
            pass

        self.estimator.fit(x_ord, y_rel, group=groups)
        weights_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights")))
        weights_dir.mkdir(parents=True, exist_ok=True)
        weight_path = weights_dir / "meow_lgb_rank_model.txt"
        self.estimator.booster_.save_model(str(weight_path))
        log.inf(f"Saved model weights to {weight_path}")
        log.inf("Done fitting")

    def predict(self, xdf):
        if self.estimator is None:
            raise RuntimeError("Model not fitted")
        x = xdf.to_numpy(dtype=np.float32, copy=False)
        return self.estimator.predict(x)
