from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


IndexTriple = Tuple[object, int, int]


@dataclass(frozen=True)
class SequenceData:
    x: np.ndarray  # (N, lookback, F)
    y: np.ndarray  # (N,)
    indices: List[IndexTriple]


def build_sequences(
    xdf,
    ydf,
    *,
    lookback: int,
    feature_cols: Sequence[str],
    ycol: str,
) -> SequenceData:
    """Build fixed-length sequences within each (symbol, date).

    xdf/ydf are expected to be indexed by MultiIndex: (symbol, date, interval).
    We do NOT create sequences across date boundaries.

    Returns samples aligned to the *last* row of each sequence.
    """
    if lookback < 2:
        raise ValueError(f"lookback must be >=2, got {lookback}")

    df = xdf[list(feature_cols)].join(ydf[[ycol]], how="inner")

    # index -> columns for grouping/sorting
    df = df.reset_index()
    required_cols = {"symbol", "date", "interval", ycol, *feature_cols}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in joined df: {sorted(missing)}")

    xs: List[np.ndarray] = []
    ys: List[float] = []
    indices: List[IndexTriple] = []

    for (symbol, date), g in df.groupby(["symbol", "date"], sort=False):
        g = g.sort_values("interval")
        x = g[list(feature_cols)].to_numpy(dtype=np.float32, copy=False)
        y = g[ycol].to_numpy(dtype=np.float32, copy=False)
        intervals = g["interval"].to_numpy(copy=False)

        if x.shape[0] < lookback:
            continue

        # sliding window
        for t in range(lookback - 1, x.shape[0]):
            seq = x[t - lookback + 1 : t + 1]
            xs.append(seq)
            ys.append(float(y[t]))
            indices.append((symbol, int(date), int(intervals[t])))

    if not xs:
        return SequenceData(
            x=np.zeros((0, lookback, len(feature_cols)), dtype=np.float32),
            y=np.zeros((0,), dtype=np.float32),
            indices=[],
        )

    x_arr = np.stack(xs, axis=0)
    y_arr = np.asarray(ys, dtype=np.float32)
    return SequenceData(x=x_arr, y=y_arr, indices=indices)
