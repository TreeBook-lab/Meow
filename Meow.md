# MEOW金融时序分析

## 统一口径测试结果

统一口径说明：以下结果均来自同一测试窗口 `20231101 ~ 20231229`，训练窗口默认使用 `20230601 ~ 20231031`。指标来自 `MeowEvaluator`。

### 1. 原始数据 / baseline 特征

| 模型 | 特征 | Pearson | Spearman | R2 | MSE | MeanIC | MeanRankIC |
|---|---|---:|---:|---:|---:|---:|---:|
| `meow` | baseline 6 feats | 0.0215 | 0.0160 | 0.00034 | 0.00 | 0.0151 | 0.0102 |
| `meow_xg` | baseline 6 feats | 0.0313 | 0.0229 | 0.00002 | 0.00 | 0.0270 | 0.0207 |
| `meow_lstm` | baseline 6 feats | 0.0378 | 0.0437 | -0.00553 | 0.00 | 0.0494 | 0.0597 |

### 2. self 特征

| 模型 | 特征 | Pearson | Spearman | R2 | MSE | MeanIC | MeanRankIC |
|---|---|---:|---:|---:|---:|---:|---:|
| `meow` | self 84 feats | 0.0547 | 0.0566 | 0.00280 | 0.00 | 0.0624 | 0.0629 |
| `meow_xg` | self 84 feats | 0.0455 | 0.0536 | -0.00907 | 0.00 | 0.0598 | 0.0619 |
| `meow_xg_tuned` | self 84 feats + tuned XGBoost | 0.0560 | 0.0541 | 0.00160 | 0.00 | 0.0662 | 0.0620 |
| `meow_lstm` | self 84 feats | 0.0440 | 0.0618 | -0.09539 | 0.00 | 0.0634 | 0.0746 |
| `meow_lgb_rank` | self 84 feats + LambdaRank | 0.0170 | -0.0115 | -495776.01807 | 10.98 | -0.0051 | -0.0067 |
| `meow_self` | self 84 feats + PatchTST | -0.0027 | 0.0051 | -5.62269 | 0.00 | -0.0030 | -0.0039 |

#### `meow_xg_tuned` 超参数搜索

- 搜索方案：使用 self 84 特征，训练窗口 `20230601 ~ 20230929`，验证窗口 `20231009 ~ 20231031`，按 `mean_rank_ic` 选择最优参数；最终使用 `20230601 ~ 20231031` 重训，并在 `20231101 ~ 20231229` 测试。
- 搜索设置：`max_rows_per_date=5000`，共测试 12 组 XGBoost 回归参数。
- 最优验证结果：`mean_rank_ic=0.049807`，`mean_ic=0.046313`，`spearman=0.065804`。
- 最优参数：`n_estimators=1000`，`learning_rate=0.015`，`max_depth=5`，`min_child_weight=8.0`，`subsample=0.85`，`colsample_bytree=0.8`，`reg_alpha=0.1`，`reg_lambda=8.0`，`gamma=0.0`，`tree_method=hist`。
- 最终测试结果：Pearson `0.0560`，Spearman `0.0541`，R2 `0.00160`，MSE `0.00`，MeanIC `0.0662`，MeanRankIC `0.0620`。

### 3. 说明

- `meow_xg` 的 self 特征版本是当前这组测试里表现最好的模型之一，尤其在 `MeanIC / MeanRankIC` 上明显优于 baseline。
- `meow_lstm` 在 baseline 特征上的 `MeanRankIC` 也比较高，但其训练开销明显更大。
- `meow_lgb_rank` 采用排序目标后，没有在当前窗口上优于回归版本。
- `meow_self` 已补齐同口径窗口的 PatchTST 结果（为避免 OOM 使用了 `--max-rows-per-date` 和 sample cap）。
- `meow_lstm` 的 self 特征版本为保证序列连续性，采用了 `--max-symbols-per-date` 进行裁剪（否则随机行采样会导致 0 条序列）。

### 4. 当前可用的文件输出

- `weights/meow_ridge.pkl`
- `weights/meow_ridge_self.pkl`
- `weights/meow_xg_model.json`
- `weights/meow_xg_tuned_self.json`
- `weights/meow_xg_hyperparam_search.csv`
- `weights/meow_xg_tuned_summary.json`
- `weights/meow_lgb_rank_model.txt`
- `weights/meow_lstm.pt`
- `weights/meow_lstm_self.pt`
- `weights/meow_self_patchtst.pt`
















