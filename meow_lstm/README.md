# MEOW LSTM (`meow_lstm/`)

This folder provides an LSTM (sequence) regressor version of the baseline MEOW pipeline.
It reuses the existing data loader / feature generator / evaluator in `meow/`.

## Environment (conda env: Meow)

Because the conda solver on this machine is old, installing deps via `conda install` may fail.
Use pip inside the existing `Meow` env:

- `source /home/treeboss/anaconda3/bin/activate Meow`
- `python -m pip install -U torch`

## Run

- Smoke test (small date range):
  - `python meow_lstm/meow_lstm.py --train-start 20230601 --train-end 20230605 --test-start 20230606 --test-end 20230607 --lookback 30 --epochs 2`

- Full run (defaults match baseline split):
  - `python meow_lstm/meow_lstm.py`

Notes

- Sequences are built within each (symbol, date) only (no cross-day sequences).
- Predictions are aligned to the last row of each lookback window, so the first (lookback-1) minutes per (symbol, date) are not evaluated.
- Metrics are reported by `MeowEvaluator` and include global correlations plus optional IC-style metrics (MeanIC/MeanRankIC) when `(date, interval)` are available.
