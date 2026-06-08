import gc
import numpy as np
import pandas as pd
from parameters import PREPROCESSING_CONFIG
from log import log


class MeowFeatureGenerator:
    @classmethod
    def feature_names(cls):
        return [
            # Price / volatility (11)
            "ret1", "ret5", "ret10", "ret30",
            "vol5", "vol10", "vol20", "ret1_vol20",
            "vol_ratio_5_20", "ret1_sign",
            # Spread & depth (6)
            "spread", "spread4", "spread_ema5",
            "depth_imb", "depth_conc", "depth_total",
            # Order book (7)
            "ob_imb0", "ob_imb4", "ob_imb9",
            "ob_slope", "ob_curvature",
            "ob_imb0_ema10", "ob_imb0_ema30",
            # Trade (8)
            "trade_imb", "trade_imb_ema5", "trade_imb_ema30",
            "trade_count_imb", "log_trade_volume",
            "vwap_dev", "vwap_dev_ema5", "volume_intensity",
            # Cross-sectional (8)
            "cx_ret1", "cx_ret10",
            "cx_ob_imb0", "cx_trade_imb",
            "rank_ob_imb0", "rank_trade_imb",
            "cx_vol20", "rank_vol20",
            # Momentum / reversal (5)
            "lagret12", "mom_5_30",
            "ret5_ret1", "ret30_ret5", "rank_ret1",
            # Market quality (2)
            "spread_scaled", "price_impact",
            # Intraday cumulative (4)
            "cum_ret1", "cum_volume", "cum_imb", "cum_ob_imb0",
            # Time (2)
            "sin_time", "cos_time",
            # Non-linear: polynomial (4)
            "ret1_sq", "ob_imb0_sq", "trade_imb_sq", "spread_sq",
            # Non-linear: pairwise crosses (8)
            "ret1_x_spread", "ret1_x_depth_imb", "ret1_x_trade_imb",
            "ret1_x_ob_imb0", "ob_imb0_x_trade_imb",
            "spread_x_depth_imb", "vol20_x_spread", "vol20_x_depth_imb",
            # Non-linear: triple interactions (2)
            "ret1_x_spread_x_vol", "ret1_x_ob_x_trade",
            # Non-linear: transforms (3)
            "tanh_ret1_vol20", "sigmoid_spread", "log1p_abs_ret1",
            # Non-linear: rolling z-scores (4)
            "ret1_zscore", "spread_zscore", "ob_imb0_zscore", "trade_imb_zscore",
            # Non-linear: temporal ranks (2)
            "ret1_trank", "vol20_trank",
            # Non-linear: asymmetry + vol-of-features (4)
            "ret1_up", "ret1_down", "ret1_vol10", "spread_vol10",
            # ---- NEW: Return dynamics (3) ----
            "ret_ema_5", "ret_ema_20", "ret_accel",
            # ---- NEW: Volatility dynamics (2) ----
            "vol_of_vol", "vol_ratio_10_20",
            # ---- NEW: Order book dynamics (3) ----
            "ob_imb_change", "spread_change", "depth_skew",
            # ---- NEW: Trade / volume (2) ----
            "trade_imb_x_ob_imb", "volume_ratio_20",
            # ---- NEW: Cross-sectional (2) ----
            "cx_spread", "cx_depth_imb",
            # ---- NEW: Microstructure / distribution (3) ----
            "bid_ask_bounce", "ret_autocorr", "ret_skew_20",
        ]

    target_horizons = ["fret1", "fret6", "fret12", "fret24"]
    """Forward-return horizons predicted by the model."""

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.mcols = ["symbol", "date", "interval"]

    def gen_features(self, df):
        # log.inf("Generating {} features from raw data...".format(len(self.feature_names())))
        eps = PREPROCESSING_CONFIG.eps

        df = df.sort_values(["symbol", "interval"])

        needed_raw = [
            "bid0", "ask0", "bid4", "ask4",
            "bsize0", "asize0",
            "bsize0_4", "asize0_4", "bsize5_9", "asize5_9",
            "tradeBuyQty", "tradeSellQty", "tradeBuyTurnover", "tradeSellTurnover",
            "nTradeBuy", "nTradeSell",
        ]
        df = df[self.mcols + needed_raw]

        df["midpx"] = (df["bid0"] + df["ask0"]) / 2.0

        for horizon in [1, 6, 24]:
            shifted = df.groupby("symbol")["midpx"].shift(-horizon)
            df[f"fret{horizon}"] = shifted / df["midpx"] - 1.0
        if "fret12" not in df.columns:
            shifted = df.groupby("symbol")["midpx"].shift(-12)
            df["fret12"] = shifted / df["midpx"] - 1.0

        # Price returns
        g = df.groupby("symbol")
        for h in [1, 5, 10, 30]:
            df[f"ret{h}"] = g["midpx"].diff(h) / (g["midpx"].shift(h) + eps)

        # Volatility family
        g = df.groupby("symbol")
        df["vol5"] = g["ret1"].transform(
            lambda x: x.rolling(5, min_periods=3).std())
        df["vol10"] = g["ret1"].transform(
            lambda x: x.rolling(10, min_periods=5).std())
        df["vol20"] = g["ret1"].transform(
            lambda x: x.rolling(20, min_periods=5).std())
        df["ret1_vol20"] = df["ret1"] / (df["vol20"] + eps)
        df["vol_ratio_5_20"] = df["vol5"] / (df["vol20"] + eps)
        df["ret1_sign"] = np.sign(df["ret1"])

        # Spread & depth
        df["spread"] = (df["ask0"] - df["bid0"]) / (df["midpx"] + eps)
        df["spread4"] = (df["ask4"] - df["bid4"]) / (df["midpx"] + eps)
        df["depth_imb"] = np.log((df["asize0_4"] + eps) / (df["bsize0_4"] + eps))
        df["depth_conc"] = (df["bsize0"] + df["asize0"]) / (
            df["bsize0_4"] + df["asize0_4"] + eps)
        df["depth_total"] = np.log(df["bsize0"] + df["asize0"] + 1.0)

        # Order book imbalance
        df["ob_imb0"] = (df["asize0"] - df["bsize0"]) / (df["asize0"] + df["bsize0"] + eps)
        df["ob_imb4"] = (df["asize0_4"] - df["bsize0_4"]) / (df["asize0_4"] + df["bsize0_4"] + eps)
        df["ob_imb9"] = (df["asize5_9"] - df["bsize5_9"]) / (df["asize5_9"] + df["bsize5_9"] + eps)
        df["ob_slope"] = df["ob_imb0"] - df["ob_imb9"]
        df["ob_curvature"] = df["ob_imb0"] - 2 * df["ob_imb4"] + df["ob_imb9"]

        # Trade features
        df["trade_imb"] = (df["tradeBuyQty"] - df["tradeSellQty"]) / (
            df["tradeBuyQty"] + df["tradeSellQty"] + eps)
        df["trade_count_imb"] = (df["nTradeBuy"] - df["nTradeSell"]) / (
            df["nTradeBuy"] + df["nTradeSell"] + eps)
        df["log_trade_volume"] = np.log(df["tradeBuyQty"] + df["tradeSellQty"] + 1.0)
        buy_vwap = df["tradeBuyTurnover"] / (df["tradeBuyQty"] + eps)
        sell_vwap = df["tradeSellTurnover"] / (df["tradeSellQty"] + eps)
        df["vwap_dev"] = ((buy_vwap + sell_vwap) / 2.0 - df["midpx"]) / (df["midpx"] + eps)
        df["volume_intensity"] = df["log_trade_volume"] / (df["vol20"] + eps)

        # EMA families (multi-scale smoothing)
        g = df.groupby("symbol")
        for col, hl in [("ob_imb0", 10), ("ob_imb0", 30),
                        ("trade_imb", 5), ("trade_imb", 30),
                        ("spread", 5), ("vwap_dev", 5)]:
            df[f"{col}_ema{hl}"] = g[col].transform(
                lambda x, h=hl: x.ewm(halflife=h, min_periods=1).mean())

        # Cross-sectional (demeaned by interval)
        for col in ["ret1", "ret10", "ob_imb0", "trade_imb", "vol20"]:
            mu = df.groupby("interval")[col].transform("mean")
            df[f"cx_{col}"] = df[col] - mu

        # Percentile ranks (robust cross-sectional signal)
        for col in ["ob_imb0", "trade_imb", "ret1", "vol20"]:
            df[f"rank_{col}"] = df.groupby("interval")[col].transform(
                lambda x: x.rank(pct=True))

        # Momentum / reversal
        g = df.groupby("symbol")
        df["bret12"] = g["midpx"].diff(12) / (g["midpx"].shift(12) + eps)
        cx_bret12 = df.groupby("interval")["bret12"].transform("mean")
        df["lagret12"] = df["bret12"] - cx_bret12
        df["mom_5_30"] = df["ret5"] * df["ret30"]
        df["ret5_ret1"] = df["ret5"] / (np.abs(df["ret1"]) + eps)
        df["ret30_ret5"] = df["ret30"] / (np.abs(df["ret5"]) + eps)

        # Market quality
        df["price_impact"] = np.abs(df["ret1"]) / (df["log_trade_volume"] + eps)
        df["spread_scaled"] = df["spread"] / (df["vol20"] + eps)

        # Time-of-day
        minute_of_day = (df["interval"] / 60_000).astype(int) % 1440
        df["sin_time"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
        df["cos_time"] = np.cos(2 * np.pi * minute_of_day / 1440.0)

        # Intraday cumulative (per symbol, backward-looking)
        g = df.groupby("symbol")
        df["cum_ret1"] = g["ret1"].cumsum()
        df["cum_volume"] = (g["tradeBuyQty"].cumsum()
                            + g["tradeSellQty"].cumsum())
        df["cum_imb"] = g["trade_imb"].expanding().mean().reset_index(level=0, drop=True)
        df["cum_ob_imb0"] = g["ob_imb0"].expanding().mean().reset_index(level=0, drop=True)

        df = df.copy()

        # Non-linear & interaction features

        # Polynomial terms
        df["ret1_sq"] = df["ret1"] ** 2
        df["ob_imb0_sq"] = df["ob_imb0"] ** 2
        df["trade_imb_sq"] = df["trade_imb"] ** 2
        df["spread_sq"] = df["spread"] ** 2

        # Pairwise crosses
        df["ret1_x_spread"] = df["ret1"] * df["spread"]
        df["ret1_x_depth_imb"] = df["ret1"] * df["depth_imb"]
        df["ret1_x_trade_imb"] = df["ret1"] * df["trade_imb"]
        df["ret1_x_ob_imb0"] = df["ret1"] * df["ob_imb0"]
        df["ob_imb0_x_trade_imb"] = df["ob_imb0"] * df["trade_imb"]
        df["spread_x_depth_imb"] = df["spread"] * df["depth_imb"]
        df["vol20_x_spread"] = df["vol20"] * df["spread"]
        df["vol20_x_depth_imb"] = df["vol20"] * df["depth_imb"]

        # Triple interactions
        df["ret1_x_spread_x_vol"] = df["ret1"] * df["spread"] * df["vol20"]
        df["ret1_x_ob_x_trade"] = df["ret1"] * df["ob_imb0"] * df["trade_imb"]

        # Non-linear transforms
        df["tanh_ret1_vol20"] = np.tanh(df["ret1_vol20"])
        df["sigmoid_spread"] = 1.0 / (1.0 + np.exp(-df["spread"] * 10.0))
        df["log1p_abs_ret1"] = np.log1p(np.abs(df["ret1"]))

        # Rolling z-scores (per-symbol)
        g = df.groupby("symbol")
        for col in ["ret1", "spread", "ob_imb0", "trade_imb"]:
            rm = g[col].transform(lambda x: x.rolling(20, min_periods=10).mean())
            rs = g[col].transform(lambda x: x.rolling(20, min_periods=10).std())
            df[f"{col}_zscore"] = (df[col] - rm) / (rs + eps)

        # Temporal ranks
        for col in ["ret1", "vol20"]:
            df[f"{col}_trank"] = g[col].transform(
                lambda x: x.rolling(30, min_periods=15).apply(
                    lambda y: (y[-1] > y[:-1]).mean(), raw=True))

        # Return asymmetry
        df["ret1_up"] = np.clip(df["ret1"], 0, None)
        df["ret1_down"] = np.clip(df["ret1"], None, 0)

        # Vol-of-features
        df["ret1_vol10"] = g["ret1"].transform(
            lambda x: x.rolling(10, min_periods=5).std())
        df["spread_vol10"] = g["spread"].transform(
            lambda x: x.rolling(10, min_periods=5).std())

        df = df.copy()  # defragment before adding new columns

        # Return dynamics
        g = df.groupby("symbol")
        df["ret_ema_5"] = g["ret1"].transform(
            lambda x: x.ewm(halflife=5, min_periods=1).mean())
        df["ret_ema_20"] = g["ret1"].transform(
            lambda x: x.ewm(halflife=20, min_periods=1).mean())
        df["ret_accel"] = df["ret1"] - df.groupby("symbol")["ret1"].shift(1)

        # Volatility dynamics
        df["vol_of_vol"] = g["vol20"].transform(
            lambda x: x.rolling(50, min_periods=20).std())
        df["vol_ratio_10_20"] = df["vol10"] / (df["vol20"] + eps)

        # Order book dynamics
        df["ob_imb_change"] = df["ob_imb0"] - df.groupby("symbol")["ob_imb0"].shift(1)
        df["spread_change"] = df["spread"] - df.groupby("symbol")["spread"].shift(1)
        df["depth_skew"] = (df["bsize0"] - df["asize0"]) / (df["bsize0"] + df["asize0"] + eps)

        # Trade / volume
        df["trade_imb_x_ob_imb"] = df["trade_imb"] * df["ob_imb0"]
        df["volume_ratio_20"] = df["log_trade_volume"] / (
            g["log_trade_volume"].transform(
                lambda x: x.rolling(20, min_periods=10).mean()) + eps)

        # Cross-sectional
        df["cx_spread"] = df["spread"] - df.groupby("interval")["spread"].transform("mean")
        df["cx_depth_imb"] = df["depth_imb"] - df.groupby("interval")["depth_imb"].transform("mean")

        # Microstructure / distribution
        df["bid_ask_bounce"] = df["ret1"] * np.sign(
            df.groupby("symbol")["ret1"].shift(1))
        df["ret_autocorr"] = df["ret1"] * df.groupby("symbol")["ret1"].shift(1)
        df["ret_skew_20"] = g["ret1"].transform(
            lambda x: x.rolling(20, min_periods=10).skew())

        # Assemble output
        feature_names = self.feature_names()
        keep_cols = self.mcols + feature_names + self.target_horizons
        xdf = df[keep_cols].set_index(self.mcols)
        ydf = xdf[self.target_horizons].copy()
        xdf = xdf[feature_names]
        xdf = xdf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ydf = ydf.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        del df, buy_vwap, sell_vwap, cx_bret12, minute_of_day
        gc.collect()
        return xdf, ydf