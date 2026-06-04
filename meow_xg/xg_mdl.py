import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from log import log


@dataclass
class XGBoostConfig:
    n_estimators: int = 800
    learning_rate: float = 0.05
    max_depth: int = 6
    min_child_weight: float = 1.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    gamma: float = 0.0
    random_state: int = 42
    n_jobs: int = max(1, os.cpu_count() or 1)
    tree_method: str = "hist"
    # objective: 'reg' for regression (default) or 'rank' for ranking (XGBRanker)
    objective: str = "reg"
    # number of relevance bins when using ranking objective
    rank_bins: int = 5


class MeowXGModel(object):
    def __init__(self, cacheDir, config: Optional[XGBoostConfig] = None):
        self.cacheDir = cacheDir
        self.config = config or XGBoostConfig()
        self.estimator = None

    def _build_estimator(self):
        try:
            from xgboost import XGBRegressor
            # XGBRanker may not be available in some older xgboost versions,
            # but try importing for ranking support.
            try:
                from xgboost import XGBRanker
            except Exception:
                XGBRanker = None
        except Exception as e:
            raise ImportError(
                "xgboost is required. Please install it in your conda env (e.g. conda install -c conda-forge xgboost)."
            ) from e

        cfg = self.config
        if cfg.objective == "rank":
            if XGBRanker is None:
                raise ImportError("XGBRanker not available in installed xgboost version")
            return XGBRanker(
                n_estimators=cfg.n_estimators,
                learning_rate=cfg.learning_rate,
                max_depth=cfg.max_depth,
                min_child_weight=cfg.min_child_weight,
                subsample=cfg.subsample,
                colsample_bytree=cfg.colsample_bytree,
                reg_alpha=cfg.reg_alpha,
                reg_lambda=cfg.reg_lambda,
                gamma=cfg.gamma,
                random_state=cfg.random_state,
                n_jobs=cfg.n_jobs,
                tree_method=cfg.tree_method,
            )

        return XGBRegressor(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
            min_child_weight=cfg.min_child_weight,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_alpha=cfg.reg_alpha,
            reg_lambda=cfg.reg_lambda,
            gamma=cfg.gamma,
            objective="reg:squarederror",
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
            tree_method=cfg.tree_method,
        )

    def fit(self, xdf, ydf):
        if self.estimator is None:
            self.estimator = self._build_estimator()

        x = xdf.to_numpy(dtype=np.float32, copy=False)
        y = ydf.to_numpy(copy=False)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        y = y.astype(np.float32, copy=False)

        log.inf(
            "Fitting XGBoost: n_estimators={}, lr={}, max_depth={}...".format(
                self.config.n_estimators, self.config.learning_rate, self.config.max_depth
            )
        )
        # If using a ranker, compute group sizes by date and ensure rows are
        # ordered by date so groups are contiguous.
        if self.config.objective == "rank":
            import numpy as _np

            # Attempt to extract date level from the index if available
            try:
                # xdf / ydf may be DataFrame with MultiIndex; try to get date values
                idx = ydf.index
                dates = idx.get_level_values("date")
            except Exception:
                # Fallback: no index info; treat whole set as single group
                dates = _np.array([0] * x.shape[0])

            order = _np.argsort(dates)
            x_ord = x[order]
            y_ord = y[order]
            dates_ord = _np.asarray(dates)[order]
            # bin continuous labels into integer relevance scores per date
            bins = max(2, int(self.config.rank_bins))
            y_rel = _np.empty_like(y_ord, dtype=_np.int32)
            start = 0
            unique_dates, counts = _np.unique(dates_ord, return_counts=True)
            for cnt, d in zip(counts, unique_dates):
                if cnt <= 1:
                    y_rel[start : start + cnt] = 0
                else:
                    grp = y_ord[start : start + cnt]
                    # compute quantile-based bins
                    try:
                        edges = _np.quantile(grp, q=_np.linspace(0, 1, bins + 1))
                    except Exception:
                        edges = _np.linspace(_np.min(grp), _np.max(grp), bins + 1)
                    # digitize into 0..bins-1
                    inds = _np.digitize(grp, edges[1:-1], right=True)
                    y_rel[start : start + cnt] = inds
                start += cnt
            groups = counts.tolist()
            try:
                from log import log as _log

                _log.inf(f"Rank labels min/max/dtype: {_np.min(y_rel)}/{_np.max(y_rel)}/{y_rel.dtype}")
            except Exception:
                pass
            self.estimator.fit(x_ord, y_rel, group=groups)
        else:
            self.estimator.fit(x, y)
        weights_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights")))
        weights_dir.mkdir(parents=True, exist_ok=True)
        weight_path = weights_dir / "meow_xg_model.json"
        self.estimator.save_model(str(weight_path))
        log.inf(f"Saved model weights to {weight_path}")
        log.inf("Done fitting")

    def predict(self, xdf):
        if self.estimator is None:
            raise RuntimeError("Model not fitted")
        x = xdf.to_numpy(dtype=np.float32, copy=False)
        return self.estimator.predict(x)
