# MEOW Self (`meow_self/`)

Feature engineering + PatchTST-like Transformer.

## Goal

- Provide a **richer feature set** (84 features) and a tunable deep model for the MEOW minute-level dataset (`archive/*.h5`).
- Keep evaluation aligned with the same metrics used by other modules (`meow/eval.py`).

About the earlier “Pearson ≥ 0.2” expectation:

- On the current dataset split we tested, global Pearson / IC-style metrics are far below 0.2 for both baseline and stronger models.
- Practically, it’s more actionable to optimize **MeanIC / MeanRankIC** (cross-sectional IC) than to chase a fixed global Pearson threshold.

Pipeline design (logic-first)

1) Data understanding / logical integration
- The raw data includes:
  - Prices: `midpx`, `lastpx`, OHLC
  - Order book snapshots: best bid/ask and depth summaries (`bsize*`, `asize*`)
  - Trade statistics: buy/sell quantities, turnover, high/low, VWAD
  - Add/cancel order statistics
- Target is `fret12` (12-minute forward return).

2) Feature engineering (structured, not brute-force)
- Microstructure: spread, relative spread, depth imbalance, depth sums
- Deeper book & flow: 10-19 level depth imbalance, OFI-style net add/cancel flow
- Trade pressure: trade imbalance (qty & turnover), VWAD gaps, intraminute ranges
- Add/cancel pressure: add/cancel imbalances
- Time-of-day: `sin/cos` encoding from `interval`
- Time-series transforms within each `(symbol, date)`:
  - multi-lag returns (1/2/5/10/20)
  - rolling mean/std of key signals (5/10/20/60)
  - lag/ewm of key microstructure signals (trade imbalance, OFI, etc.)
- Cross-sectional de-meaning per `interval` for a few key signals (helps remove market-wide moves)

All these features are implemented in: `meow_self/feat_self.py`.

3) Sequence dataset
- Default: build sliding windows within each `(symbol, date)` only (no cross-day leakage).
- Optional: `--cross-day 1` builds sequences per `symbol` across all provided dates.
  - This is often important for long `--lookback` (240/480/960), because some symbols/days may have missing minutes.
  - In this dataset, a single day typically has ~226 intervals, so `lookback=240` cannot work with `--val-days 1`.
- Each sample uses a lookback window to predict the label at the last timestamp of that window.

Implemented in: `meow_self/seq_ds.py`.

4) Backbone model: PatchTST-like Transformer
- Instead of attending over every minute step directly, PatchTST-style patching reduces length:
  - Split the `(lookback, features)` sequence into overlapping patches.
  - Flatten each patch and project to `d_model`.
  - Run TransformerEncoder over patch tokens.
  - Predict using the last patch token.

Implemented in: `meow_self/patchtst.py`.

Attention mask mode

- `--attn-mode reverse`: older tokens attend to newer tokens ("far looks near")
- `--attn-mode causal`: newer tokens attend to older tokens (standard causal)
- `--attn-mode none`: no attention mask (bidirectional)

5) Training objective & evaluation

- Loss = `MSE(pred, y)`
- We still compute batch Pearson as a **training metric** (shown in tqdm), but it does not affect backprop.
- After each epoch, we run `MeowEvaluator.eval()` on the validation set and compute multiple metrics.
- Best checkpoint selection is controlled by `--select-metric` (defaults to `pearson`).

Implemented in: `meow_self/train_self.py`.

## Environment

- Activate conda env `Meow`:
  - `conda activate Meow`

If you want GPU training:

- Ensure NVIDIA driver works (`nvidia-smi`).
- Install a CUDA-enabled PyTorch build compatible with your driver.

Progress bar dependency:

- `python -m pip install -U tqdm`

## Run

- Smoke test (small range):
  - `python meow_self/train_self.py --train-start 20230601 --train-end 20230605 --test-start 20230606 --test-end 20230607 --lookback 60 --epochs 2 --batch-size 256 --val-days 1`

- Full split run (Jun–Jul train / Aug test, select best by RankIC):
  - `python meow_self/train_self.py \
    --train-start 20230601 --train-end 20230731 \
    --test-start 20230801 --test-end 20230831 \
    --lookback 240 --cross-day 1 \
    --epochs 5 --batch-size 1024 --val-days 10 \
    --d-model 256 --layers 4 --ff 1024 --nhead 8 \
    --patch-len 24 --patch-stride 12 --dropout 0.1 \
    --lr 0.0003 --corr-weight 0.2 --weight-decay 0.01 --grad-clip 1.0 \
    --max-train-samples 300000 --max-val-samples 80000 --max-test-samples 120000 \
    --select-metric mean_rank_ic`

## Speeding up large runs

- Large date ranges can create millions of sliding-window samples.
- Use these caps to iterate quickly:
  - `--max-train-samples 300000 --max-val-samples 80000 --max-test-samples 120000`

## Metrics and `--select-metric`

`--select-metric` chooses which validation metric defines the “best” checkpoint.
It must match a key returned by `MeowEvaluator.eval()`:

- `pearson`, `spearman`
- `mean_ic`, `mean_rank_ic`
- `mean_sym_pearson`, `mean_sym_spearman`

For cross-sectional ranking tasks, `mean_rank_ic` is usually the most stable selection criterion.

## Reference results (Jun–Jul train / Aug test)

On a representative run (5 epochs, capped samples as above):

- Best validation `mean_rank_ic` ≈ 0.0610 (epoch 3)
- Test: Pearson≈0.0306, MeanIC≈0.0458, MeanRankIC≈0.0464

For comparison on the same split, XGBoost with self features reached roughly:

- Test MeanRankIC ≈ 0.0845 (see `meow_xg/README.md`)

## Tuning tips (practical)

- Try selecting by `--select-metric mean_rank_ic` if the goal is IC/RankIC.
- Increase `epochs` (e.g. 20–80) and keep `lr` small (`1e-4`–`5e-4`).
- Increase context: `--lookback 480` or `--lookback 960` (PatchTST keeps it feasible).
- Increase capacity: `--d-model 512 --layers 8 --ff 2048`.
- Patch settings: larger `--patch-len` often helps (e.g. 24/32) with `--patch-stride` ≈ half.
- Correlation weight: if correlation metrics lag, try `--corr-weight 0.3`–`0.6`.

If you want ModernTCN / Mamba variants

- The current code uses a PatchTST-style Transformer because it’s robust and easy to implement without extra dependencies.
- I can add ModernTCN (pure conv) or a Mamba(-Transformer) variant if you confirm which library you prefer (e.g. `mamba-ssm`) and whether pip wheels are acceptable.
