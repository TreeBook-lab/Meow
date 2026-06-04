import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd


def _ensure_import_paths():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(this_dir, ".."))
    meow_dir = os.path.join(root_dir, "meow")
    meow_self_dir = os.path.join(root_dir, "meow_self")

    for p in [this_dir, meow_dir, meow_self_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)


_ensure_import_paths()

from feat_self import MeowSelfFeatureGenerator


class MeowLGBRankFeatureGenerator(object):
    """Feature generator dedicated to LightGBM LambdaRank.

    It reuses the richer self-feature set and converts the continuous target
    into dense integer relevance labels within each trading date.
    """

    def __init__(self, cacheDir=None):
        self.cacheDir = cacheDir
        self.base = MeowSelfFeatureGenerator(cacheDir=cacheDir)

    def genFeatures(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Reuse the richer self features; the model will convert the continuous
        # target into compact relevance bins per date.
        # Use cross_day=True to avoid MultiIndex reset issues when input
        # DataFrame indexing varies between loads (safer and less memory
        # sensitive for initial experiments).
        return self.base.genFeatures(df, cross_day=True)
