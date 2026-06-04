# MEOW XGBoost (`meow_xg/`)

This folder provides an XGBoost regressor version of the MEOW pipeline.
It **reuses the core data loader / evaluator** from `meow/`, and can train with either:

- `--feature-set baseline`: 6 baseline features from `meow/feat.py`
- `--feature-set self`: 84 engineered features from `meow_self/feat_self.py`

## Environment

- Recommended: create the conda env from the repo root:
  - `conda env create -f environment.meow.yml`
  - `conda activate Meow`

## Run

- Baseline features (fast):
  - `python meow_xg/meow_xg.py`

- Smoke test (small date range):
  - `python meow_xg/meow_xg.py --train-start 20230601 --train-end 20230605 --test-start 20230606 --test-end 20230607`

## Using self features (recommended)

- `python meow_xg/meow_xg.py \
  --feature-set self --cross-day 1 \
  --train-start 20230601 --train-end 20230731 \
  --test-start 20230801 --test-end 20230831 \
  --max-train-rows 400000 --max-test-rows 400000 --seed 1`

Notes:

- `--cross-day 1` controls how `meow_self` features are computed across days within each symbol.
- `--max-*-rows` caps rows by random subsample (keeps original MultiIndex so IC/RankIC stays computable).

## Metrics

Evaluation is done by `MeowEvaluator` (from `meow/eval.py`). It reports:

- Global correlations: `Pearson`, `Spearman`
- Cross-sectional metrics per `(date, interval)` averaged over time: `MeanIC` (Pearson), `MeanRankIC` (Spearman)
- Optional per-symbol time-series correlations: `MeanSymPearson`, `MeanSymSpearman`

In many quant/ranking settings, **MeanIC / MeanRankIC** is the primary signal; it can be much higher (or lower) than global Pearson.

## Tuning knobs

Common hyper-parameters exposed by CLI:

- `--n-estimators`, `--learning-rate`, `--max-depth`, `--min-child-weight`
- `--subsample`, `--colsample-bytree`
- `--reg-alpha`, `--reg-lambda`, `--gamma`
- `--tree-method hist|gpu_hist`

GPU acceleration:

- Try `--tree-method gpu_hist` if your XGBoost build supports CUDA.

## Reference results (Jun–Jul train / Aug test, 40w cap)

- Baseline (6 feats): Pearson≈0.0223, MeanIC≈0.0208, MeanRankIC≈0.0221
- Self feats (84 feats, `--cross-day 1`): Pearson≈0.0368, MeanIC≈0.0689, MeanRankIC≈0.0845

One “bigger” run did **not** improve on that reference (likely overfitting / mismatch):

- `n_estimators=2000, lr=0.03, max_depth=8`: MeanRankIC≈0.0760
