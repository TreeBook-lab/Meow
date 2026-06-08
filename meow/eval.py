import os
import numpy as np
import pandas as pd
from log import log


class MeowEvaluator(object):
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.prediction_col = "forecast"
        self.ycol = "fret12"

    def eval(self, ydf):
        ydf = ydf.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Allow (date, interval) to be stored either as columns or in the index.
        ydf_for_group = ydf
        if not all(c in ydf_for_group.columns for c in ["date", "interval"]):
            try:
                idx_names = list(getattr(ydf_for_group.index, "names", []) or [])
            except Exception:
                idx_names = []
            if "date" in idx_names or "interval" in idx_names:
                ydf_for_group = ydf_for_group.reset_index()

        pcor = ydf[[self.prediction_col, self.ycol]].corr().to_numpy()[0, 1]
        spr = ydf[[self.prediction_col, self.ycol]].corr(method="spearman").to_numpy()[0, 1]
        r2 = 1 - ((ydf[self.prediction_col] - ydf[self.ycol]) ** 2).sum() / ydf[self.ycol].var() / ydf.shape[0]
        mse = ((ydf[self.prediction_col] - ydf[self.ycol]) ** 2).sum() / ydf.shape[0]
        msg = "Meow evaluation summary: Pearson={:.4f}, Spearman={:.4f}, R2={:.5f}, MSE={:.2f}".format(pcor, spr, r2, mse)

        metrics = {
            "pearson": float(pcor) if pcor == pcor else None,
            "spearman": float(spr) if spr == spr else None,
            "r2": float(r2) if r2 == r2 else None,
            "mse": float(mse) if mse == mse else None,
            "mean_ic": None,
            "median_ic": None,
            "mean_rank_ic": None,
            "median_rank_ic": None,
            "mean_sym_pearson": None,
            "median_sym_pearson": None,
            "mean_sym_spearman": None,
            "median_sym_spearman": None,
        }

        # Optional: cross-sectional IC by each (date, interval) slice (mean over time).
        # This is a common quant metric and can differ a lot from global Pearson.
        if all(c in ydf_for_group.columns for c in ["date", "interval", self.prediction_col, self.ycol]):
            def _slice_corr(g: pd.DataFrame, method: str = "pearson"):
                if g.shape[0] < 2:
                    return np.nan
                return g.corr(method=method).to_numpy()[0, 1]

            slice_df = ydf_for_group[["date", "interval", self.prediction_col, self.ycol]]
            gb = slice_df.groupby(["date", "interval"], sort=False)[[self.prediction_col, self.ycol]]
            ic_by_t = gb.apply(lambda g: _slice_corr(g, method="pearson"))
            ric_by_t = gb.apply(lambda g: _slice_corr(g, method="spearman"))

            ic_by_t = ic_by_t.replace([np.inf, -np.inf], np.nan).dropna()
            ric_by_t = ric_by_t.replace([np.inf, -np.inf], np.nan).dropna()

            if ic_by_t.shape[0] > 0:
                metrics["mean_ic"] = float(ic_by_t.mean())
                metrics["median_ic"] = float(ic_by_t.median())
                msg += ", MeanIC={:.4f}, MedianIC={:.4f}".format(metrics["mean_ic"], metrics["median_ic"])
            if ric_by_t.shape[0] > 0:
                metrics["mean_rank_ic"] = float(ric_by_t.mean())
                metrics["median_rank_ic"] = float(ric_by_t.median())
                msg += ", MeanRankIC={:.4f}, MedianRankIC={:.4f}".format(metrics["mean_rank_ic"], metrics["median_rank_ic"])

        # Optional: per-symbol time-series correlation (can be much larger/smaller than IC)
        if "symbol" in ydf_for_group.columns and all(
            c in ydf_for_group.columns for c in [self.prediction_col, self.ycol]
        ):
            sym_gb = ydf_for_group.groupby("symbol", sort=False)[[self.prediction_col, self.ycol]]
            sym_pc = sym_gb.apply(lambda g: _slice_corr(g, method="pearson"))
            sym_sp = sym_gb.apply(lambda g: _slice_corr(g, method="spearman"))
            sym_pc = sym_pc.replace([np.inf, -np.inf], np.nan).dropna()
            sym_sp = sym_sp.replace([np.inf, -np.inf], np.nan).dropna()
            if sym_pc.shape[0] > 0:
                metrics["mean_sym_pearson"] = float(sym_pc.mean())
                metrics["median_sym_pearson"] = float(sym_pc.median())
                msg += ", MeanSymPearson={:.4f}, MedianSymPearson={:.4f}".format(metrics["mean_sym_pearson"], metrics["median_sym_pearson"])
            if sym_sp.shape[0] > 0:
                metrics["mean_sym_spearman"] = float(sym_sp.mean())
                metrics["median_sym_spearman"] = float(sym_sp.median())
                msg += ", MeanSymSpearman={:.4f}, MedianSymSpearman={:.4f}".format(metrics["mean_sym_spearman"], metrics["median_sym_spearman"])

        log.inf(msg)
        return metrics
