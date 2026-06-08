import os
import sys
import argparse
from typing import Optional


def _ensure_import_paths():
    """Allow reusing baseline modules in ../meow without packaging changes."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(this_dir, ".."))
    meow_dir = os.path.join(root_dir, "meow")
    meow_self_dir = os.path.join(root_dir, "meow_self")

    # Prioritize baseline code and current directory.
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
from xg_mdl import MeowXGModel, XGBoostConfig


class MeowXGEngine(object):
    def __init__(
        self,
        h5dir,
        cache_dir,
        *,
        feature_set: str = "baseline",
        cross_day: int = 0,
        xgb_config: XGBoostConfig = None,
    ):
        self.calendar = Calendar()
        self.h5dir = h5dir
        if not os.path.exists(h5dir):
            raise ValueError("Data directory not exists: {}".format(self.h5dir))
        if not os.path.isdir(h5dir):
            raise ValueError("Invalid data directory: {}".format(self.h5dir))

        self.cache_dir = cache_dir  # not used in sample code
        self.dloader = MeowDataLoader(h5dir=h5dir)
        self.feature_set = str(feature_set)
        self.cross_day = int(cross_day)
        if self.feature_set == "self":
            if MeowSelfFeatureGenerator is None:
                raise ImportError("meow_self feature generator not available")
            self.feat_generator = MeowSelfFeatureGenerator(cache_dir=cache_dir)
        else:
            self.feat_generator = MeowFeatureGenerator(cache_dir=cache_dir)
        self.model = MeowXGModel(cache_dir=cache_dir, config=xgb_config)
        self.evaluator = MeowEvaluator(cache_dir=cache_dir)

    def _required_columns(self):
        if self.feature_set == "self":
            return None
        return [
            "symbol",
            "interval",
            "asize0",
            "bsize0",
            "asize0_4",
            "bsize0_4",
            "asize5_9",
            "bsize5_9",
            "tradeBuyQty",
            "tradeSellQty",
            "midpx",
            "fret12",
        ]

    def fit(self, start_date, end_date):
        dates = self.calendar.range(start_date, end_date)
        # Allow optional per-date row cap to reduce peak memory when generating
        # the (potentially large) `self` feature set. Set `self.max_rows_per_date`
        # externally if desired; default is 0 (no cap).
        raw_data = self.dloader.load_dates(
            dates, columns=self._required_columns(), max_rows_per_date=getattr(self, "max_rows_per_date", 0)
        )
        log.inf("Running XGBoost model fitting...")
        if self.feature_set == "self":
            xdf, ydf = self.feat_generator.gen_features(raw_data, cross_day=bool(self.cross_day))
        else:
            xdf, ydf = self.feat_generator.gen_features(raw_data)
        xdf, ydf = _maybe_subsample_xy(xdf, ydf, max_rows=getattr(self, "max_train_rows", None), seed=getattr(self, "seed", 1))
        self.model.fit(xdf, ydf)

    def predict(self, xdf):
        return self.model.predict(xdf)

    def eval(self, start_date, end_date):
        log.inf("Running model evaluation...")
        dates = self.calendar.range(start_date, end_date)
        raw_data = self.dloader.load_dates(
            dates, columns=self._required_columns(), max_rows_per_date=getattr(self, "max_rows_per_date", 0)
        )
        if self.feature_set == "self":
            xdf, ydf = self.feat_generator.gen_features(raw_data, cross_day=bool(self.cross_day))
        else:
            xdf, ydf = self.feat_generator.gen_features(raw_data)
        xdf, ydf = _maybe_subsample_xy(xdf, ydf, max_rows=getattr(self, "max_test_rows", None), seed=getattr(self, "seed", 1))
        ydf.loc[:, "forecast"] = self.predict(xdf)
        self.evaluator.eval(ydf)


def _maybe_subsample_xy(xdf, ydf, max_rows: Optional[int], seed: int = 1):
    if not max_rows or max_rows <= 0:
        return xdf, ydf
    n = int(xdf.shape[0])
    if n <= max_rows:
        return xdf, ydf

    import numpy as np

    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n, size=int(max_rows), replace=False)
    idx.sort()
    # Keep original index (often a MultiIndex with symbol/date/interval) so evaluation
    # can compute cross-sectional IC if desired.
    return xdf.iloc[idx], ydf.iloc[idx]


if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser(description="MEOW XGBoost baseline")
    parser.add_argument("--h5dir", default=os.path.join(root_dir, "archive"))
    parser.add_argument("--train-start", type=int, default=20230601)
    parser.add_argument("--train-end", type=int, default=20231130)
    parser.add_argument("--test-start", type=int, default=20231201)
    parser.add_argument("--test-end", type=int, default=20231229)
    parser.add_argument("--feature-set", choices=["baseline", "self"], default="baseline")
    parser.add_argument("--cross-day", type=int, default=0, help="Only used when --feature-set self")

    # XGBoost hyper-parameters
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--tree-method", type=str, default="hist", help="hist|gpu_hist (if supported)")
    parser.add_argument("--max-train-rows", type=int, default=0, help="Optional cap on training rows (random subsample)")
    parser.add_argument("--max-test-rows", type=int, default=0, help="Optional cap on test rows (random subsample)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--objective", choices=["reg", "rank"], default="reg", help="Model objective: regression or ranking")
    parser.add_argument("--use-lgbm-rank", action="store_true", help="Use LightGBM LambdaRank (LGBMRanker) instead of XGBoost ranker")
    args = parser.parse_args()

    xgb_cfg = XGBoostConfig(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        gamma=args.gamma,
        random_state=args.seed,
        tree_method=args.tree_method,
        objective=args.objective,
    )

    engine = MeowXGEngine(
        h5dir=args.h5dir,
        cache_dir=None,
        feature_set=args.feature_set,
        cross_day=args.cross_day,
        xgb_config=xgb_cfg,
    )
    # Optionally switch to LightGBM Ranker implementation
    if args.use_lgbm_rank:
        try:
            from lgb_mdl import MeowLGBModel, LGBConfig

            lgb_cfg = LGBConfig(
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                random_state=args.seed,
            )
            engine.model = MeowLGBModel(cache_dir=None, config=lgb_cfg)
        except Exception as e:
            raise
    # When using the heavier `self` feature set, cap rows per date to avoid OOM by default.
    if args.feature_set == "self":
        # Use a conservative per-date row cap to avoid OOM during heavy feature generation.
        engine.max_rows_per_date = 5000
    engine.max_train_rows = args.max_train_rows
    engine.max_test_rows = args.max_test_rows
    engine.seed = args.seed
    engine.fit(args.train_start, args.train_end)
    engine.eval(args.test_start, args.test_end)
