import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from log import log


@dataclass
class LGBConfig:
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


class MeowLGBModel(object):
    def __init__(self, cacheDir, config: Optional[LGBConfig] = None):
        self.cacheDir = cacheDir
        self.config = config or LGBConfig()
        self.estimator = None

    def _build_estimator(self):
        try:
            from lightgbm import LGBMRanker
        except Exception as e:
            raise ImportError("lightgbm is required for LGBMRanker. Install it in the env.") from e

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
            self.estimator = self._build_estimator()

        x = xdf.to_numpy(dtype=np.float32, copy=False)
        y = ydf.to_numpy(copy=False)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        y = y.astype(np.float32, copy=False)

        log.inf(f"Fitting LightGBM Ranker: n_estimators={self.config.n_estimators}, lr={self.config.learning_rate}...")

        # compute groups by date level in index
        try:
            idx = ydf.index
            dates = idx.get_level_values("date")
        except Exception:
            dates = np.array([0] * x.shape[0])

        order = np.argsort(dates)
        x_ord = x[order]
        y_ord = y[order]
        dates_ord = np.asarray(dates)[order]

        unique_dates, counts = np.unique(dates_ord, return_counts=True)
        # bin continuous labels into integer relevance per date
        bins = max(2, int(self.config.rank_bins))
        y_rel = np.empty_like(y_ord, dtype=np.int32)
        start = 0
        for cnt in counts:
            if cnt <= 1:
                y_rel[start : start + cnt] = 0
            else:
                grp = y_ord[start : start + cnt]
                try:
                    edges = np.quantile(grp, q=np.linspace(0, 1, bins + 1))
                except Exception:
                    edges = np.linspace(np.min(grp), np.max(grp), bins + 1)
                inds = np.digitize(grp, edges[1:-1], right=True)
                y_rel[start : start + cnt] = inds
            start += cnt

        groups = counts.tolist()
        try:
            log.inf(f"LGB rank labels min/max/dtype: {y_rel.min()}/{y_rel.max()}/{y_rel.dtype}")
        except Exception:
            pass

        # LightGBM expects group parameter as list of group sizes
        self.estimator.fit(x_ord, y_rel, group=groups)
        log.inf("Done fitting")

    def predict(self, xdf):
        if self.estimator is None:
            raise RuntimeError("Model not fitted")
        x = xdf.to_numpy(dtype=np.float32, copy=False)
        return self.estimator.predict(x)
