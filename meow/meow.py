import os
import sys
import argparse
import shutil

# Allow importing sibling modules (../meow_self) when running as a script.
_this_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.abspath(os.path.join(_this_dir, ".."))
_meow_self_dir = os.path.join(_root_dir, "meow_self")
for _p in [_this_dir, _root_dir, _meow_self_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from log import log
from dl import MeowDataLoader
from feat import MeowFeatureGenerator
from mdl import MeowModel
from eval import MeowEvaluator
from tradingcalendar import Calendar

try:
    from feat_self import MeowSelfFeatureGenerator
except Exception:
    MeowSelfFeatureGenerator = None


class MeowEngine(object):
    def __init__(self, h5dir, cacheDir, *, max_rows_per_date: int = 0):
        self.calendar = Calendar()
        self.h5dir = h5dir
        if not os.path.exists(h5dir):
            raise ValueError("Data directory not exists: {}".format(self.h5dir))
        if not os.path.isdir(h5dir):
            raise ValueError("Invalid data directory: {}".format(self.h5dir))
        self.cacheDir = cacheDir # this is not used in sample code
        self.max_rows_per_date = int(max_rows_per_date)
        self.dloader = MeowDataLoader(h5dir=h5dir)
        self.featGenerator = MeowFeatureGenerator(cacheDir=cacheDir)
        self.model = MeowModel(cacheDir=cacheDir)
        self.evaluator = MeowEvaluator(cacheDir=cacheDir)

    def fit(self, startDate, endDate):
        dates = self.calendar.range(startDate, endDate)
        rawData = self.dloader.loadDates(dates, max_rows_per_date=self.max_rows_per_date)
        log.inf("Running model fitting...")
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
        self.evaluator.eval(ydf)


if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser(description="MEOW Ridge baseline")
    parser.add_argument("--h5dir", default=os.path.join(root_dir, "archive"))
    parser.add_argument("--train-start", type=int, default=20230601)
    parser.add_argument("--train-end", type=int, default=20231031)
    parser.add_argument("--test-start", type=int, default=20231101)
    parser.add_argument("--test-end", type=int, default=20231229)
    parser.add_argument("--feature-set", choices=["baseline", "self"], default="baseline")
    parser.add_argument(
        "--max-rows-per-date",
        type=int,
        default=0,
        help="Optional cap on raw rows loaded per trading day (random subsample) to reduce peak memory.",
    )
    args = parser.parse_args()

    engine = MeowEngine(h5dir=args.h5dir, cacheDir=None, max_rows_per_date=args.max_rows_per_date)
    if args.feature_set == "self":
        if MeowSelfFeatureGenerator is None:
            raise ImportError("meow_self feature generator not available")
        engine.featGenerator = MeowSelfFeatureGenerator(cacheDir=None)

    engine.fit(args.train_start, args.train_end)
    engine.eval(args.test_start, args.test_end)

    # Preserve weights with a stable, feature-set-specific filename.
    weights_dir = os.path.abspath(os.path.join(root_dir, "weights"))
    src = os.path.join(weights_dir, "meow_ridge.pkl")
    if os.path.exists(src):
        dst = os.path.join(weights_dir, f"meow_ridge_{args.feature_set}.pkl")
        try:
            shutil.copy2(src, dst)
            log.inf(f"Copied model weights to {dst}")
        except Exception:
            pass
