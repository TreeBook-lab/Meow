# MEOW Core (`meow/`)

This folder contains the **core baseline pipeline** shared by `meow_xg/`, `meow_lstm/`, and `meow_self/`:

- Data loading: `dl.py` (reads `archive/*.h5`)
- Baseline feature engineering: `feat.py`
- Evaluation / metrics: `eval.py` (`MeowEvaluator`)
- Simple baseline entrypoint: `meow.py`

## Data expectations

- Minute-level data lives in `archive/*.h5`.
- The baseline label is `fret12` (12-minute forward return).
- Most pipelines keep a MultiIndex like `(symbol, date, interval)`; evaluation also works if these are present as columns.

## How imports work

This repo is intentionally lightweight and **does not package** `meow/` as an installable Python module.
Scripts in other folders call a small helper (e.g. `_ensure_import_paths()`) that inserts `meow/` into `sys.path`.

Practical rule: run commands from the repo root (the folder that contains `archive/` and `meow/`).

## Metrics (what we report)

`MeowEvaluator.eval(ydf)` expects `ydf` to contain:

- `forecast`: model predictions
- `fret12`: ground-truth label
- optional `symbol`, `date`, `interval` (either as columns or in the index)

It reports:

- `pearson`: global Pearson correlation between `forecast` and `fret12`
- `spearman`: global Spearman correlation
- `mean_ic` / `median_ic`: **cross-sectional IC** averaged over each `(date, interval)` slice
- `mean_rank_ic` / `median_rank_ic`: **cross-sectional RankIC** (Spearman) averaged over each `(date, interval)` slice
- `mean_sym_pearson` / `mean_sym_spearman`: per-symbol time-series correlations (mean over symbols)

Note: global Pearson and cross-sectional IC can differ materially; for quant-style ranking tasks, IC/RankIC is often the primary metric.

## Quick run

- Baseline demo:
  - `python meow/meow.py`

For model training/evaluation, see the module READMEs:

- `meow_xg/README.md` (XGBoost baseline + self features)
- `meow_lstm/README.md` (LSTM baseline)
- `meow_self/README.md` (self feature engineering + PatchTST-like model)
