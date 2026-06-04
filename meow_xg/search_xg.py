#!/usr/bin/env python3
"""Hyperparameter search for meow_xg.

Default protocol:
- feature set: self
- search train: 20230601~20230929
- validation: 20231009~20231031
- final train: 20230601~20231031
- final test: 20231101~20231229
"""

import argparse
import csv
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def _ensure_import_paths():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(this_dir, ".."))
    meow_dir = os.path.join(root_dir, "meow")
    meow_self_dir = os.path.join(root_dir, "meow_self")
    for p in [this_dir, meow_dir, meow_self_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)


_ensure_import_paths()

from dl import MeowDataLoader  # noqa: E402
from eval import MeowEvaluator  # noqa: E402
from feat import MeowFeatureGenerator  # noqa: E402
try:
    from feat_self import MeowSelfFeatureGenerator  # noqa: E402
except Exception:
    MeowSelfFeatureGenerator = None
from log import log  # noqa: E402
from tradingcalendar import Calendar  # noqa: E402
from xg_mdl import MeowXGModel, XGBoostConfig  # noqa: E402


METRIC_KEYS = [
    "pearson",
    "spearman",
    "r2",
    "mse",
    "mean_ic",
    "median_ic",
    "mean_rank_ic",
    "median_rank_ic",
    "mean_sym_pearson",
    "median_sym_pearson",
    "mean_sym_spearman",
    "median_sym_spearman",
]


def required_columns(feature_set: str):
    if feature_set == "self":
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


def make_feature_generator(feature_set: str):
    if feature_set == "self":
        if MeowSelfFeatureGenerator is None:
            raise ImportError("meow_self feature generator not available")
        return MeowSelfFeatureGenerator(cacheDir=None)
    return MeowFeatureGenerator(cacheDir=None)


def load_xy(
    h5dir: str,
    start_date: int,
    end_date: int,
    feature_set: str,
    max_rows_per_date: int,
    seed: int,
    cross_day: int,
):
    calendar = Calendar()
    dates = calendar.range(start_date, end_date)
    loader = MeowDataLoader(h5dir=h5dir)
    raw = loader.loadDates(
        dates,
        columns=required_columns(feature_set),
        max_rows_per_date=max_rows_per_date,
        seed=seed,
    )
    feat_gen = make_feature_generator(feature_set)
    if feature_set == "self":
        return feat_gen.genFeatures(raw, cross_day=bool(cross_day))
    return feat_gen.genFeatures(raw)


def maybe_subsample_xy(xdf, ydf, max_rows: int, seed: int):
    if not max_rows or max_rows <= 0 or xdf.shape[0] <= max_rows:
        return xdf, ydf
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(int(xdf.shape[0]), size=int(max_rows), replace=False)
    idx.sort()
    return xdf.iloc[idx], ydf.iloc[idx]


def evaluate(model: MeowXGModel, xdf, ydf) -> Dict[str, float]:
    eval_ydf = ydf.copy()
    eval_ydf.loc[:, "forecast"] = model.predict(xdf)
    return MeowEvaluator(cacheDir=None).eval(eval_ydf)


def candidate_configs(seed: int, tree_method: str, n_jobs: int) -> List[XGBoostConfig]:
    raw = [
        dict(n_estimators=800, learning_rate=0.05, max_depth=6, min_child_weight=1.0, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0, gamma=0.0),
        dict(n_estimators=500, learning_rate=0.03, max_depth=4, min_child_weight=3.0, subsample=0.9, colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=2.0, gamma=0.0),
        dict(n_estimators=600, learning_rate=0.04, max_depth=5, min_child_weight=5.0, subsample=0.8, colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=3.0, gamma=0.0),
        dict(n_estimators=800, learning_rate=0.03, max_depth=3, min_child_weight=10.0, subsample=0.9, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=5.0, gamma=0.0),
        dict(n_estimators=1000, learning_rate=0.02, max_depth=4, min_child_weight=5.0, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=4.0, gamma=0.0),
        dict(n_estimators=400, learning_rate=0.06, max_depth=5, min_child_weight=2.0, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=2.0, gamma=0.05),
        dict(n_estimators=700, learning_rate=0.05, max_depth=3, min_child_weight=8.0, subsample=0.9, colsample_bytree=0.75, reg_alpha=0.05, reg_lambda=5.0, gamma=0.0),
        dict(n_estimators=500, learning_rate=0.08, max_depth=2, min_child_weight=10.0, subsample=0.95, colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=6.0, gamma=0.0),
        dict(n_estimators=800, learning_rate=0.04, max_depth=4, min_child_weight=6.0, subsample=0.75, colsample_bytree=0.75, reg_alpha=0.1, reg_lambda=5.0, gamma=0.05),
        dict(n_estimators=600, learning_rate=0.03, max_depth=6, min_child_weight=3.0, subsample=0.7, colsample_bytree=0.7, reg_alpha=0.0, reg_lambda=2.0, gamma=0.1),
        dict(n_estimators=1000, learning_rate=0.015, max_depth=5, min_child_weight=8.0, subsample=0.85, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=8.0, gamma=0.0),
        dict(n_estimators=300, learning_rate=0.10, max_depth=3, min_child_weight=6.0, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.05, reg_lambda=4.0, gamma=0.05),
    ]
    return [
        XGBoostConfig(
            **params,
            random_state=seed,
            n_jobs=n_jobs,
            tree_method=tree_method,
            objective="reg",
        )
        for params in raw
    ]


def score_value(metrics: Dict[str, float], score_metric: str) -> float:
    val = metrics.get(score_metric)
    if val is None or not np.isfinite(val):
        return float("-inf")
    return float(val)


def write_results_csv(path: Path, rows: Iterable[Dict]):
    rows = list(rows)
    if not rows:
        return
    fieldnames = ["trial", "score_metric", "score"] + list(asdict(rows[0]["config"]).keys()) + METRIC_KEYS
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            cfg = asdict(row["config"])
            out = {
                "trial": row["trial"],
                "score_metric": row["score_metric"],
                "score": row["score"],
                **cfg,
            }
            out.update({k: row["metrics"].get(k) for k in METRIC_KEYS})
            writer.writerow(out)


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser(description="meow_xg hyperparameter search")
    parser.add_argument("--h5dir", default=os.path.join(root_dir, "archive"))
    parser.add_argument("--feature-set", choices=["baseline", "self"], default="self")
    parser.add_argument("--cross-day", type=int, default=0)
    parser.add_argument("--search-train-start", type=int, default=20230601)
    parser.add_argument("--search-train-end", type=int, default=20230929)
    parser.add_argument("--val-start", type=int, default=20231009)
    parser.add_argument("--val-end", type=int, default=20231031)
    parser.add_argument("--final-train-start", type=int, default=20230601)
    parser.add_argument("--final-train-end", type=int, default=20231031)
    parser.add_argument("--test-start", type=int, default=20231101)
    parser.add_argument("--test-end", type=int, default=20231229)
    parser.add_argument("--max-rows-per-date", type=int, default=5000)
    parser.add_argument("--max-search-train-rows", type=int, default=0)
    parser.add_argument("--max-val-rows", type=int, default=0)
    parser.add_argument("--max-final-train-rows", type=int, default=0)
    parser.add_argument("--max-test-rows", type=int, default=0)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--score-metric", default="mean_rank_ic", choices=METRIC_KEYS)
    parser.add_argument("--tree-method", default="hist")
    parser.add_argument("--n-jobs", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-csv", default=os.path.join(root_dir, "weights", "meow_xg_hyperparam_search.csv"))
    parser.add_argument("--output-json", default=os.path.join(root_dir, "weights", "meow_xg_tuned_summary.json"))
    args = parser.parse_args()

    weights_dir = Path(root_dir) / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    log.inf("Loading search train and validation features...")
    x_train, y_train = load_xy(
        args.h5dir,
        args.search_train_start,
        args.search_train_end,
        args.feature_set,
        args.max_rows_per_date,
        args.seed,
        args.cross_day,
    )
    x_val, y_val = load_xy(
        args.h5dir,
        args.val_start,
        args.val_end,
        args.feature_set,
        args.max_rows_per_date,
        args.seed,
        args.cross_day,
    )
    x_train, y_train = maybe_subsample_xy(x_train, y_train, args.max_search_train_rows, args.seed)
    x_val, y_val = maybe_subsample_xy(x_val, y_val, args.max_val_rows, args.seed)
    log.inf(f"Search train shape: {x_train.shape}, validation shape: {x_val.shape}")

    configs = candidate_configs(seed=args.seed, tree_method=args.tree_method, n_jobs=args.n_jobs)[: max(1, args.trials)]
    results = []
    best = None

    for i, cfg in enumerate(configs, start=1):
        log.inf(f"Trial {i}/{len(configs)}: {asdict(cfg)}")
        model = MeowXGModel(cacheDir=None, config=cfg)
        model.fit(x_train, y_train)
        metrics = evaluate(model, x_val, y_val)
        score = score_value(metrics, args.score_metric)
        row = {"trial": i, "config": cfg, "metrics": metrics, "score_metric": args.score_metric, "score": score}
        results.append(row)
        write_results_csv(Path(args.output_csv), results)
        if best is None or score > best["score"]:
            best = row
            log.inf(f"New best trial {i}: {args.score_metric}={score:.6f}")

    if best is None:
        raise RuntimeError("No successful hyperparameter trial")

    best_cfg = best["config"]
    log.inf(f"Best validation config: {asdict(best_cfg)}")
    log.inf(f"Best validation {args.score_metric}: {best['score']:.6f}")

    log.inf("Loading final train and test features...")
    x_final, y_final = load_xy(
        args.h5dir,
        args.final_train_start,
        args.final_train_end,
        args.feature_set,
        args.max_rows_per_date,
        args.seed,
        args.cross_day,
    )
    x_test, y_test = load_xy(
        args.h5dir,
        args.test_start,
        args.test_end,
        args.feature_set,
        args.max_rows_per_date,
        args.seed,
        args.cross_day,
    )
    x_final, y_final = maybe_subsample_xy(x_final, y_final, args.max_final_train_rows, args.seed)
    x_test, y_test = maybe_subsample_xy(x_test, y_test, args.max_test_rows, args.seed)
    log.inf(f"Final train shape: {x_final.shape}, test shape: {x_test.shape}")

    final_model = MeowXGModel(cacheDir=None, config=best_cfg)
    final_model.fit(x_final, y_final)
    test_metrics = evaluate(final_model, x_test, y_test)

    default_model_path = weights_dir / "meow_xg_model.json"
    tuned_model_path = weights_dir / "meow_xg_tuned_self.json" if args.feature_set == "self" else weights_dir / "meow_xg_tuned_baseline.json"
    if default_model_path.exists():
        shutil.copy2(default_model_path, tuned_model_path)
        log.inf(f"Copied tuned model to {tuned_model_path}")

    summary = {
        "protocol": {
            "feature_set": args.feature_set,
            "cross_day": args.cross_day,
            "search_train": [args.search_train_start, args.search_train_end],
            "validation": [args.val_start, args.val_end],
            "final_train": [args.final_train_start, args.final_train_end],
            "test": [args.test_start, args.test_end],
            "max_rows_per_date": args.max_rows_per_date,
            "score_metric": args.score_metric,
        },
        "best_trial": best["trial"],
        "best_params": asdict(best_cfg),
        "validation_metrics": best["metrics"],
        "test_metrics": test_metrics,
        "search_results_csv": str(Path(args.output_csv).resolve()),
        "tuned_model_path": str(tuned_model_path.resolve()),
    }
    with Path(args.output_json).open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.inf(f"Wrote summary to {args.output_json}")
    log.inf("FINAL_TEST_METRICS " + json.dumps(test_metrics, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
