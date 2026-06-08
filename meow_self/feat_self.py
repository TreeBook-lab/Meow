import numpy as np
import pandas as pd
from log import log


class MeowSelfFeatureGenerator(object):
    """Feature generator with more structured / logic-driven features.

    Output contract matches the baseline: returns (xdf, ydf) indexed by
    (symbol, date, interval).

    Notes
    - All rolling/lag features are computed within each (symbol, date).
    - Cross-sectional (interval) de-meaning is applied to selected features.
    """

    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir
        self.ycol = "fret12"
        self.mcols = ["symbol", "date", "interval"]

    @staticmethod
    def _safe_div(a, b):
        return a / (b.replace(0, np.nan))

    @staticmethod
    def _interval_to_minutes(interval_s: pd.Series) -> pd.Series:
        # interval is like 93000000 for 09:30:00; parse hh and mm.
        # Works for typical HHMMSSxx integer encodings used in this dataset.
        v = interval_s.astype(np.int64)
        hh = v // 10000000
        mm = (v // 100000) % 100
        return (hh * 60 + mm).astype(np.int32)

    @classmethod
    def feature_names(cls):
        base = [
            # prices
            "ret1",
            "ret2",
            "ret5",
            "ret10",
            "ret20",
            "logret1",
            "bret12",
            "lagret12",
            "hl_range",
            "oc_ret",
            # microstructure
            "spread0",
            "rspread0",
            "mid_last_gap",
            "ob_imb0",
            "ob_imb4",
            "ob_imb9",
            "ob_imb19",
            "depth_sum0",
            "depth_sum4",
            "depth_sum9",
            "depth_sum19",
            # trades
            "trade_qty",
            "trade_turnover",
            "trade_imb",
            "trade_imbema5",
            "trade_imb_qty",
            "trade_imb_turnover",
            "trade_qty_rate",
            "trade_turnover_rate",
            "buy_sell_vwad_gap",
            "trade_range_buy",
            "trade_range_sell",
            # add/cancel
            "add_imb_qty",
            "cxl_imb_qty",
            "add_imb_turnover",
            "cxl_imb_turnover",
            "net_ofi_qty",
            "net_ofi_turnover",
            # time-of-day
            "tod_sin",
            "tod_cos",
        ]

        # rolling stats (within symbol/date)
        roll = []
        for w in [5, 10, 20, 60]:
            roll += [
                f"ret1_mean{w}",
                f"ret1_std{w}",
                f"spread0_mean{w}",
                f"trade_imb_qty_mean{w}",
                f"ob_imb0_mean{w}",
                f"net_ofi_qty_mean{w}",
            ]

        # simple lags (within symbol/date)
        lags = []
        for k in [1, 2, 5, 10]:
            lags += [
                f"trade_imb_qty_lag{k}",
                f"ob_imb0_lag{k}",
                f"net_ofi_qty_lag{k}",
            ]

        # EWMs (within symbol/date)
        ewms = [
            "trade_imb_qty_ewm5",
            "net_ofi_qty_ewm5",
            "ret1_ewm10",
        ]

        # cross-sectional de-mean versions (interval mean across symbols)
        cs = [
            "ret1_cs",
            "spread0_cs",
            "trade_imb_qty_cs",
            "ob_imb0_cs",
            "net_ofi_qty_cs",
        ]

        return base + roll + lags + ewms + cs

    def gen_features(self, df: pd.DataFrame, *, cross_day: bool = False):
        log.inf("Generating {} self features from raw data...".format(len(self.feature_names())))

        df = df.copy()

        # ---------- price & return features ----------
        # returns within symbol (optionally across day)
        df.sort_values(["symbol", "date", "interval"], inplace=True)
        group_keys = ["symbol"] if cross_day else ["symbol", "date"]
        g = df.groupby(group_keys, sort=False)

        df.loc[:, "ret1"] = g["midpx"].pct_change(1)
        df.loc[:, "ret2"] = g["midpx"].pct_change(2)
        df.loc[:, "ret5"] = g["midpx"].pct_change(5)
        df.loc[:, "ret10"] = g["midpx"].pct_change(10)
        df.loc[:, "ret20"] = g["midpx"].pct_change(20)
        df.loc[:, "logret1"] = np.log(df["midpx"]) - np.log(g["midpx"].shift(1))

        # baseline-like backward return and interval de-mean (intraday seasonality removal)
        df.loc[:, "bret12"] = g["midpx"].pct_change(12)
        cx_bret = (
            df.groupby(["date", "interval"], sort=False)[["bret12"]]
            .mean()
            .reset_index()
            .rename(columns={"bret12": "cx_bret12"})
        )
        df = df.merge(cx_bret, on=["date", "interval"], how="left")
        df.loc[:, "lagret12"] = df["bret12"] - df["cx_bret12"]
        df.drop(columns=["cx_bret12"], inplace=True)

        # daily bar like range at minute-level (uses current minute OHLC columns)
        df.loc[:, "hl_range"] = self._safe_div(df["high"] - df["low"], df["midpx"])
        df.loc[:, "oc_ret"] = self._safe_div(df["midpx"] - df["open"], df["open"])

        # ---------- microstructure ----------
        df.loc[:, "spread0"] = df["ask0"] - df["bid0"]
        df.loc[:, "rspread0"] = self._safe_div(df["spread0"], df["midpx"])
        df.loc[:, "mid_last_gap"] = self._safe_div(df["lastpx"] - df["midpx"], df["midpx"])

        # align sign with baseline (ask - bid size imbalance)
        df.loc[:, "ob_imb0"] = self._safe_div(df["asize0"] - df["bsize0"], df["asize0"] + df["bsize0"])
        df.loc[:, "ob_imb4"] = self._safe_div(df["asize0_4"] - df["bsize0_4"], df["asize0_4"] + df["bsize0_4"])
        df.loc[:, "ob_imb9"] = self._safe_div(df["asize5_9"] - df["bsize5_9"], df["asize5_9"] + df["bsize5_9"])
        df.loc[:, "ob_imb19"] = self._safe_div(
            df["asize10_19"] - df["bsize10_19"], df["asize10_19"] + df["bsize10_19"]
        )

        df.loc[:, "depth_sum0"] = df["bsize0"] + df["asize0"]
        df.loc[:, "depth_sum4"] = df["bsize0_4"] + df["asize0_4"]
        df.loc[:, "depth_sum9"] = df["bsize5_9"] + df["asize5_9"]
        df.loc[:, "depth_sum19"] = df["bsize10_19"] + df["asize10_19"]

        # ---------- trades ----------
        df.loc[:, "trade_qty"] = df["tradeBuyQty"] + df["tradeSellQty"]
        df.loc[:, "trade_turnover"] = df["tradeBuyTurnover"] + df["tradeSellTurnover"]
        df.loc[:, "trade_imb"] = self._safe_div(
            df["tradeBuyQty"] - df["tradeSellQty"],
            df["tradeBuyQty"] + df["tradeSellQty"],
        )
        # pandas GroupBy objects can be sensitive to columns added after groupby creation;
        # rebuild grouping before using newly created columns.
        g = df.groupby(group_keys, sort=False)
        if cross_day:
            df.loc[:, "trade_imbema5"] = (
                g["trade_imb"].apply(lambda s: s.ewm(halflife=5).mean()).reset_index(level=[0], drop=True)
            )
        else:
            df.loc[:, "trade_imbema5"] = (
                g["trade_imb"].apply(lambda s: s.ewm(halflife=5).mean()).reset_index(level=[0, 1], drop=True)
            )
        df.loc[:, "trade_imb_qty"] = self._safe_div(df["tradeBuyQty"] - df["tradeSellQty"], df["tradeBuyQty"] + df["tradeSellQty"])
        df.loc[:, "trade_imb_turnover"] = self._safe_div(
            df["tradeBuyTurnover"] - df["tradeSellTurnover"],
            df["tradeBuyTurnover"] + df["tradeSellTurnover"],
        )
        df.loc[:, "buy_sell_vwad_gap"] = self._safe_div(df["buyVwad"] - df["sellVwad"], df["midpx"])
        df.loc[:, "trade_range_buy"] = self._safe_div(df["tradeBuyHigh"] - df["tradeBuyLow"], df["midpx"])
        df.loc[:, "trade_range_sell"] = self._safe_div(df["tradeSellHigh"] - df["tradeSellLow"], df["midpx"])

        # normalize trade activity by book depth (level-1)
        df.loc[:, "trade_qty_rate"] = self._safe_div(df["trade_qty"], df["depth_sum0"])
        df.loc[:, "trade_turnover_rate"] = self._safe_div(df["trade_turnover"], df["midpx"] * df["depth_sum0"])

        # ---------- add / cancel ----------
        df.loc[:, "add_imb_qty"] = self._safe_div(df["addBuyQty"] - df["addSellQty"], df["addBuyQty"] + df["addSellQty"])
        df.loc[:, "cxl_imb_qty"] = self._safe_div(df["cxlBuyQty"] - df["cxlSellQty"], df["cxlBuyQty"] + df["cxlSellQty"])
        df.loc[:, "add_imb_turnover"] = self._safe_div(
            df["addBuyTurnover"] - df["addSellTurnover"],
            df["addBuyTurnover"] + df["addSellTurnover"],
        )
        df.loc[:, "cxl_imb_turnover"] = self._safe_div(
            df["cxlBuyTurnover"] - df["cxlSellTurnover"],
            df["cxlBuyTurnover"] + df["cxlSellTurnover"],
        )

        # net order-flow imbalance (OFI-style)
        df.loc[:, "net_ofi_qty"] = (df["addBuyQty"] - df["cxlBuyQty"]) - (df["addSellQty"] - df["cxlSellQty"])
        df.loc[:, "net_ofi_turnover"] = (df["addBuyTurnover"] - df["cxlBuyTurnover"]) - (
            df["addSellTurnover"] - df["cxlSellTurnover"]
        )

        # ---------- time-of-day encoding ----------
        mins = self._interval_to_minutes(df["interval"]).astype(np.float32)
        # minute-of-day cyclical; robust even across lunch break
        ang = 2.0 * np.pi * (mins / 1440.0)
        df.loc[:, "tod_sin"] = np.sin(ang)
        df.loc[:, "tod_cos"] = np.cos(ang)

        # Rebuild groupby once more before all rolling/lag/ewm blocks that rely on
        # columns created above.
        g = df.groupby(group_keys, sort=False)

        # ---------- rolling features within group ----------
        for w in [5, 10, 20, 60]:
            if cross_day:
                df.loc[:, f"ret1_mean{w}"] = g["ret1"].rolling(window=w, min_periods=1).mean().reset_index(level=[0], drop=True)
                df.loc[:, f"ret1_std{w}"] = g["ret1"].rolling(window=w, min_periods=2).std().reset_index(level=[0], drop=True)
                df.loc[:, f"spread0_mean{w}"] = g["spread0"].rolling(window=w, min_periods=1).mean().reset_index(level=[0], drop=True)
                df.loc[:, f"trade_imb_qty_mean{w}"] = g["trade_imb_qty"].rolling(window=w, min_periods=1).mean().reset_index(level=[0], drop=True)
                df.loc[:, f"ob_imb0_mean{w}"] = g["ob_imb0"].rolling(window=w, min_periods=1).mean().reset_index(level=[0], drop=True)
                df.loc[:, f"net_ofi_qty_mean{w}"] = g["net_ofi_qty"].rolling(window=w, min_periods=1).mean().reset_index(level=[0], drop=True)
            else:
                df.loc[:, f"ret1_mean{w}"] = g["ret1"].rolling(window=w, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
                df.loc[:, f"ret1_std{w}"] = g["ret1"].rolling(window=w, min_periods=2).std().reset_index(level=[0, 1], drop=True)
                df.loc[:, f"spread0_mean{w}"] = g["spread0"].rolling(window=w, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
                df.loc[:, f"trade_imb_qty_mean{w}"] = g["trade_imb_qty"].rolling(window=w, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
                df.loc[:, f"ob_imb0_mean{w}"] = g["ob_imb0"].rolling(window=w, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
                df.loc[:, f"net_ofi_qty_mean{w}"] = g["net_ofi_qty"].rolling(window=w, min_periods=1).mean().reset_index(level=[0, 1], drop=True)

        # fill std NaNs (early windows)
        for w in [5, 10, 20, 60]:
            df.loc[:, f"ret1_std{w}"] = df[f"ret1_std{w}"].fillna(0)

        # ---------- lag/ewm features within group ----------
        for k in [1, 2, 5, 10]:
            df.loc[:, f"trade_imb_qty_lag{k}"] = g["trade_imb_qty"].shift(k)
            df.loc[:, f"ob_imb0_lag{k}"] = g["ob_imb0"].shift(k)
            df.loc[:, f"net_ofi_qty_lag{k}"] = g["net_ofi_qty"].shift(k)

        if cross_day:
            df.loc[:, "trade_imb_qty_ewm5"] = g["trade_imb_qty"].apply(lambda s: s.ewm(halflife=5).mean()).reset_index(level=[0], drop=True)
            df.loc[:, "net_ofi_qty_ewm5"] = g["net_ofi_qty"].apply(lambda s: s.ewm(halflife=5).mean()).reset_index(level=[0], drop=True)
            df.loc[:, "ret1_ewm10"] = g["ret1"].apply(lambda s: s.ewm(halflife=10).mean()).reset_index(level=[0], drop=True)
        else:
            df.loc[:, "trade_imb_qty_ewm5"] = g["trade_imb_qty"].apply(lambda s: s.ewm(halflife=5).mean()).reset_index(level=[0, 1], drop=True)
            df.loc[:, "net_ofi_qty_ewm5"] = g["net_ofi_qty"].apply(lambda s: s.ewm(halflife=5).mean()).reset_index(level=[0, 1], drop=True)
            df.loc[:, "ret1_ewm10"] = g["ret1"].apply(lambda s: s.ewm(halflife=10).mean()).reset_index(level=[0, 1], drop=True)

        # ---------- cross-sectional de-mean (per interval) ----------
        for base_col, out_col in [
            ("ret1", "ret1_cs"),
            ("spread0", "spread0_cs"),
            ("trade_imb_qty", "trade_imb_qty_cs"),
            ("ob_imb0", "ob_imb0_cs"),
            ("net_ofi_qty", "net_ofi_qty_cs"),
        ]:
            cx = (
                df.groupby(["date", "interval"])[[base_col]]
                .mean()
                .reset_index()
                .rename(columns={base_col: f"cx_{base_col}"})
            )
            df = df.merge(cx, on=["date", "interval"], how="left")
            df.loc[:, out_col] = df[base_col] - df[f"cx_{base_col}"]
            df.drop(columns=[f"cx_{base_col}"], inplace=True)

        # ---------- finalize ----------
        feats = self.feature_names()
        xdf = df[self.mcols + feats].set_index(self.mcols)
        ydf = df[self.mcols + [self.ycol]].set_index(self.mcols)

        xdf = xdf.replace([np.inf, -np.inf], np.nan).fillna(0)
        ydf = ydf.replace([np.inf, -np.inf], np.nan).fillna(0)
        return xdf, ydf
