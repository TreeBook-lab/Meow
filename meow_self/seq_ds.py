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


@dataclass(frozen=True)
class SequenceIndex:
    """Memory-efficient representation of a sequence dataset.

    Stores the full feature matrix once (2D) and a list of end positions for
    each sliding window sample. The sample's input window is a view into `x_all`.
    """

    x_all: np.ndarray  # (T, F)
    y_all: np.ndarray  # (T,)
    symbol_all: np.ndarray  # (T,)
    date_all: np.ndarray  # (T,)
    interval_all: np.ndarray  # (T,)
    end_pos: np.ndarray  # (N_samples,)


def build_sequences(
    xdf,
    ydf,
    *,
    lookback: int,
    feature_cols: Sequence[str],
    ycol: str,
    cross_day: bool = False,
) -> SequenceData:
    """Build fixed-length sequences.

    - cross_day=False: sequences are built within each (symbol, date).
    - cross_day=True: sequences are built within each symbol across all provided dates.
      This enables longer-context modeling at day boundaries (useful for long lookbacks).
    """
    if lookback < 2:
        raise ValueError(f"lookback must be >=2, got {lookback}")

    df = xdf[list(feature_cols)].join(ydf[[ycol]], how="inner")
    df = df.reset_index()

    xs: List[np.ndarray] = []
    ys: List[float] = []
    indices: List[IndexTriple] = []

    if cross_day:
        # within each symbol, sort by (date, interval)
        for symbol, g in df.groupby("symbol", sort=False):
            g = g.sort_values(["date", "interval"])
            x = g[list(feature_cols)].to_numpy(dtype=np.float32, copy=False)
            y = g[ycol].to_numpy(dtype=np.float32, copy=False)
            dates = g["date"].to_numpy(copy=False)
            intervals = g["interval"].to_numpy(copy=False)

            if x.shape[0] < lookback:
                continue

            for t in range(lookback - 1, x.shape[0]):
                xs.append(x[t - lookback + 1 : t + 1])
                ys.append(float(y[t]))
                indices.append((symbol, int(dates[t]), int(intervals[t])))
    else:
        # within each (symbol, date)
        for (symbol, date), g in df.groupby(["symbol", "date"], sort=False):
            g = g.sort_values("interval")
            x = g[list(feature_cols)].to_numpy(dtype=np.float32, copy=False)
            y = g[ycol].to_numpy(dtype=np.float32, copy=False)
            intervals = g["interval"].to_numpy(copy=False)
            if x.shape[0] < lookback:
                continue

            for t in range(lookback - 1, x.shape[0]):
                xs.append(x[t - lookback + 1 : t + 1])
                ys.append(float(y[t]))
                indices.append((symbol, int(date), int(intervals[t])))

    if not xs:
        return SequenceData(
            x=np.zeros((0, lookback, len(feature_cols)), dtype=np.float32),
            y=np.zeros((0,), dtype=np.float32),
            indices=[],
        )

    return SequenceData(
        x=np.stack(xs, axis=0),
        y=np.asarray(ys, dtype=np.float32),
        indices=indices,
    )


def build_sequence_index(
    xdf,
    ydf,
    *,
    lookback: int,
    feature_cols: Sequence[str],
    ycol: str,
    cross_day: bool = False,
) -> SequenceIndex:
    """Build a SequenceIndex without materializing all sliding windows.

    This is the preferred function for multi-week/month training.
    """

    if lookback < 2:
        raise ValueError(f"lookback must be >=2, got {lookback}")

    df = xdf[list(feature_cols)].join(ydf[[ycol]], how="inner").reset_index()

    # Sort once, then compute group boundaries by key changes.
    if cross_day:
        df = df.sort_values(["symbol", "date", "interval"])
        group_change = df["symbol"].ne(df["symbol"].shift(1)).to_numpy()
    else:
        df = df.sort_values(["symbol", "date", "interval"])
        sym = df["symbol"]
        dat = df["date"]
        group_change = (sym.ne(sym.shift(1)) | dat.ne(dat.shift(1))).to_numpy()

    x_all = df[list(feature_cols)].to_numpy(dtype=np.float32, copy=False)
    y_all = df[ycol].to_numpy(dtype=np.float32, copy=False)
    symbol_all = df["symbol"].to_numpy(copy=False)
    date_all = df["date"].to_numpy(copy=False)
    interval_all = df["interval"].to_numpy(copy=False)

    T = x_all.shape[0]
    if T == 0:
        return SequenceIndex(
            x_all=np.zeros((0, len(feature_cols)), dtype=np.float32),
            y_all=np.zeros((0,), dtype=np.float32),
            symbol_all=np.asarray([], dtype=object),
            date_all=np.zeros((0,), dtype=np.int32),
            interval_all=np.zeros((0,), dtype=np.int32),
            end_pos=np.zeros((0,), dtype=np.int32),
        )

    group_starts = np.flatnonzero(group_change)
    if group_starts.size == 0 or group_starts[0] != 0:
        group_starts = np.r_[0, group_starts]
    group_ends = np.r_[group_starts[1:], T]

    end_pos_chunks: List[np.ndarray] = []
    for s, e in zip(group_starts, group_ends):
        n = int(e - s)
        if n < lookback:
            continue
        ends = np.arange(s + lookback - 1, e, dtype=np.int32)
        end_pos_chunks.append(ends)

    if not end_pos_chunks:
        end_pos = np.zeros((0,), dtype=np.int32)
    else:
        end_pos = np.concatenate(end_pos_chunks, axis=0)

    return SequenceIndex(
        x_all=x_all,
        y_all=y_all,
        symbol_all=symbol_all,
        date_all=date_all,
        interval_all=interval_all,
        end_pos=end_pos,
    )
