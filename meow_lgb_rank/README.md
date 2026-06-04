# MEOW LightGBM LambdaRank

This folder contains a standalone LightGBM LambdaRank experiment for MEOW.

## What it does

- Uses the richer self feature set from `meow_self/feat_self.py`.
- Converts the target `fret12` into per-date dense integer relevance labels.
- Trains `LGBMRanker` on training dates.
- Evaluates on the requested test window with the existing MEOW evaluator.

## Files

- `model.py`: LightGBM ranker wrapper.
- `feature.py`: dedicated feature generator for ranking labels.
- `run.py`: training and evaluation entrypoint.

## Install

The environment needs LightGBM:

```bash
conda activate Meow
pip install lightgbm
```

## Run

```bash
conda activate Meow
python3 meow_lgb_rank/run.py \
  --train-start 20230601 \
  --train-end 20231031 \
  --test-start 20231101 \
  --test-end 20231229
```

If the full dataset is too large for your machine, lower the per-date cap:

```bash
python3 meow_lgb_rank/run.py \
  --max-rows-per-date 5000 \
  --train-start 20230601 \
  --train-end 20231031 \
  --test-start 20231101 \
  --test-end 20231229
```

## Notes

- The default feature generator uses all `self` features and does not cap rows per date.
- Labels are converted to relevance ranks per trading date, which is required by ranking models.

## Final Result

Using `--max-rows-per-date 5000` with the train/test split above, the final evaluation was:

- Pearson = 0.0170
- Spearman = -0.0115
- R2 = -495776.01807
- MSE = 10.98
- MeanIC = -0.0051, MedianIC = -0.0149
- MeanRankIC = -0.0067, MedianRankIC = -0.0078

