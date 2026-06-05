import numpy as np
import pandas as pd
from log import log


class MeowEvaluator:
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir
        self.prediction_col = "forecast"
        self.ycol = "fret12"

    def eval(self, ydf):
        ydf = ydf.replace([np.inf, -np.inf], np.nan).fillna(0)
        pcor = ydf[[self.prediction_col, self.ycol]].corr().to_numpy()[0, 1]
        ss_res = ((ydf[self.prediction_col] - ydf[self.ycol]) ** 2).sum()
        ss_tot = ydf[self.ycol].var() * ydf.shape[0]
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        mse = ss_res / ydf.shape[0]
        log.inf("Meow evaluation summary: Pearson correlation={:.4f}, "
                "R2={:.5f}, MSE={:.2f}".format(pcor, r2, mse))
        return float(pcor), float(r2), float(mse)
