import argparse
import os
import sys
from pathlib import Path


def _ensure_import_paths():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(this_dir, ".."))
    meow_dir = os.path.join(root_dir, "meow")
    meow_self_dir = os.path.join(root_dir, "meow_self")

    for p in [this_dir, meow_dir, meow_self_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)


_ensure_import_paths()

from dl import MeowDataLoader
from eval import MeowEvaluator
from log import log
from tradingcalendar import Calendar

from feature import MeowLGBRankFeatureGenerator
from model import LGBRankConfig, MeowLGBRankModel


class MeowLGBRankEngine(object):
    def __init__(self, h5dir, cacheDir=None, lgb_config=None, max_rows_per_date: int = 5000):
        self.calendar = Calendar()
        self.h5dir = h5dir
        if not os.path.exists(h5dir):
            raise ValueError(f"Data directory not exists: {self.h5dir}")
        if not os.path.isdir(h5dir):
            raise ValueError(f"Invalid data directory: {self.h5dir}")

        self.cacheDir = cacheDir
        self.max_rows_per_date = int(max_rows_per_date)
        self.dloader = MeowDataLoader(h5dir=h5dir)
        self.featGenerator = MeowLGBRankFeatureGenerator(cacheDir=cacheDir)
        self.model = MeowLGBRankModel(cacheDir=cacheDir, config=lgb_config)
        self.evaluator = MeowEvaluator(cacheDir=cacheDir)

    def fit(self, startDate, endDate):
        dates = self.calendar.range(startDate, endDate)
        rawData = self.dloader.loadDates(dates, max_rows_per_date=self.max_rows_per_date)
        log.inf("Running LightGBM rank model fitting...")
        xdf, ydf = self.featGenerator.genFeatures(rawData)
        self.model.fit(xdf, ydf)

    def predict(self, xdf):
        return self.model.predict(xdf)

    def eval(self, startDate, endDate):
        log.inf("Running model evaluation...")
        dates = self.calendar.range(startDate, endDate)
        rawData = self.dloader.loadDates(dates, max_rows_per_date=self.max_rows_per_date)
        xdf, ydf = self.featGenerator.genFeatures(rawData)
        ydf.loc[:, "forecast"] = self.predict(xdf)
        # Convert ranker integer scores into percentile ranks per date so
        # regression-style metrics (Pearson, R2, etc.) are meaningful.
        try:
            ydf["forecast"] = ydf.groupby("date")["forecast"].rank(method="average", pct=True)
        except Exception:
            pass
        self.evaluator.eval(ydf)


if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser(description="MEOW LightGBM LambdaRank")
    parser.add_argument("--h5dir", default=os.path.join(root_dir, "archive"))
    parser.add_argument("--train-start", type=int, default=20230601)
    parser.add_argument("--train-end", type=int, default=20231031)
    parser.add_argument("--test-start", type=int, default=20231101)
    parser.add_argument("--test-end", type=int, default=20231229)
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-rows-per-date", type=int, default=5000)
    args = parser.parse_args()

    lgb_cfg = LGBRankConfig(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        random_state=args.seed,
    )

    engine = MeowLGBRankEngine(
        h5dir=args.h5dir,
        cacheDir=None,
        lgb_config=lgb_cfg,
        max_rows_per_date=args.max_rows_per_date,
    )
    engine.fit(args.train_start, args.train_end)
    engine.eval(args.test_start, args.test_end)
