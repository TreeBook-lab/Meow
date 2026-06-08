import numpy as np
import pandas as pd
from log import log


class MeowFeatureGenerator(object):
    @classmethod
    def feature_names(cls):
        return [
            # === Order Book Imbalance (13 features) ===
            "ob_imb0",
            "ob_imb4",
            "ob_imb9",
            "ob_imb19",
            "ob_tr_imb0",
            "ob_tr_imb4",
            "ob_tr_imb9",
            "ob_tr_imb19",
            "ob_spread",
            "ob_spread9",
            "ob_midpx_pos",
            "ob_depth_ratio",
            "ob_depth_ratio9",
            # === Trade Flow (7 features) ===
            "trade_imb",
            "trade_imbema5",
            "trade_imbema20",
            "trade_imbema60",
            "trade_turnover_imb",
            "trade_count_imb",
            "trade_intensity",
            # === Order Flow (Add/Cancel events) (8 features) ===
            "add_imb",
            "add_imbema5",
            "cxl_imb",
            "cxl_imbema5",
            "net_order_flow",
            "cancel_rate",
            "add_trade_ratio",
            "cxl_trade_ratio",
            # === Price Momentum (6 features) ===
            "bret1",
            "bret5",
            "bret12",
            "lagret1",
            "lagret5",
            "lagret12",
            # === Volatility & Range (4 features) ===
            "price_range",
            "realized_vol5",
            "realized_vol12",
            "spread_ema12",
            # === Volume (4 features) ===
            "total_trade_qty",
            "rel_trade_qty",
            "buy_qty_ratio",
            "total_qty_ema12",
            # === Cross-sectional (5 features) ===
            "cs_imb0",
            "cs_bret12",
            "cs_volume",
            "cs_add_imb",
            "cs_spread",
        ]

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.ycol = "fret12"
        self.mcols = ["symbol", "date", "interval"]

    def gen_features(self, df):
        n_features = len(self.feature_names())
        log.inf("Generating {} features from raw data...".format(n_features))

        # ---- Order Book Imbalance ----
        df.loc[:, "ob_imb0"] = (df["asize0"] - df["bsize0"]) / (df["asize0"] + df["bsize0"] + 1e-12)
        df.loc[:, "ob_imb4"] = (df["asize0_4"] - df["bsize0_4"]) / (df["asize0_4"] + df["bsize0_4"] + 1e-12)
        df.loc[:, "ob_imb9"] = (df["asize5_9"] - df["bsize5_9"]) / (df["asize5_9"] + df["bsize5_9"] + 1e-12)
        df.loc[:, "ob_imb19"] = (df["asize10_19"] - df["bsize10_19"]) / (df["asize10_19"] + df["bsize10_19"] + 1e-12)
        df.loc[:, "ob_tr_imb0"] = (df["ask0"] * df["asize0"] - df["bid0"] * df["bsize0"]) / (
            df["ask0"] * df["asize0"] + df["bid0"] * df["bsize0"] + 1e-12
        )
        df.loc[:, "ob_tr_imb4"] = (df["atr0_4"] - df["btr0_4"]) / (df["atr0_4"] + df["btr0_4"] + 1e-12)
        df.loc[:, "ob_tr_imb9"] = (df["atr5_9"] - df["btr5_9"]) / (df["atr5_9"] + df["btr5_9"] + 1e-12)
        df.loc[:, "ob_tr_imb19"] = (df["atr10_19"] - df["btr10_19"]) / (df["atr10_19"] + df["btr10_19"] + 1e-12)
        df.loc[:, "ob_spread"] = (df["ask0"] - df["bid0"]) / (df["midpx"] + 1e-12)
        df.loc[:, "ob_spread9"] = (df["ask9"] - df["bid9"]) / (df["midpx"] + 1e-12)
        df.loc[:, "ob_midpx_pos"] = (df["midpx"] - df["bid0"]) / (df["ask0"] - df["bid0"] + 1e-12)
        df.loc[:, "ob_depth_ratio"] = (df["asize0_4"] + df["bsize0_4"]) / (
            df["asize0_4"] + df["bsize0_4"] + df["asize5_9"] + df["bsize5_9"] + 1e-12
        )
        df.loc[:, "ob_depth_ratio9"] = (df["asize0_4"] + df["bsize0_4"]) / (
            df["asize0_4"] + df["bsize0_4"] + df["asize5_9"] + df["bsize5_9"]
            + df["asize10_19"] + df["bsize10_19"] + 1e-12
        )

        # ---- Trade Flow Imbalance ----
        df.loc[:, "trade_imb"] = (df["tradeBuyQty"] - df["tradeSellQty"]) / (
            df["tradeBuyQty"] + df["tradeSellQty"] + 1e-12
        )
        df.loc[:, "trade_imbema5"] = df["trade_imb"].ewm(halflife=5).mean()
        df.loc[:, "trade_imbema20"] = df["trade_imb"].ewm(halflife=20).mean()
        df.loc[:, "trade_imbema60"] = df["trade_imb"].ewm(halflife=60).mean()
        df.loc[:, "trade_turnover_imb"] = (df["tradeBuyTurnover"] - df["tradeSellTurnover"]) / (
            df["tradeBuyTurnover"] + df["tradeSellTurnover"] + 1e-12
        )
        df.loc[:, "trade_count_imb"] = (df["nTradeBuy"] - df["nTradeSell"]) / (
            df["nTradeBuy"] + df["nTradeSell"] + 1e-12
        )
        df.loc[:, "trade_intensity"] = df["nTradeBuy"] + df["nTradeSell"]

        # ---- Order Flow (Add/Cancel) ----
        df.loc[:, "add_imb"] = (df["addBuyQty"] - df["addSellQty"]) / (
            df["addBuyQty"] + df["addSellQty"] + 1e-12
        )
        df.loc[:, "add_imbema5"] = df["add_imb"].ewm(halflife=5).mean()
        df.loc[:, "cxl_imb"] = (df["cxlBuyQty"] - df["cxlSellQty"]) / (
            df["cxlBuyQty"] + df["cxlSellQty"] + 1e-12
        )
        df.loc[:, "cxl_imbema5"] = df["cxl_imb"].ewm(halflife=5).mean()
        df.loc[:, "net_order_flow"] = (
            (df["addBuyQty"] - df["cxlBuyQty"]) - (df["addSellQty"] - df["cxlSellQty"])
        ) / (df["addBuyQty"] + df["addSellQty"] + df["cxlBuyQty"] + df["cxlSellQty"] + 1e-12)
        df.loc[:, "cancel_rate"] = (df["cxlBuyQty"] + df["cxlSellQty"]) / (
            df["addBuyQty"] + df["addSellQty"] + 1e-12
        )
        df.loc[:, "add_trade_ratio"] = (df["addBuyQty"] + df["addSellQty"]) / (
            df["tradeBuyQty"] + df["tradeSellQty"] + 1e-12
        )
        df.loc[:, "cxl_trade_ratio"] = (df["cxlBuyQty"] + df["cxlSellQty"]) / (
            df["tradeBuyQty"] + df["tradeSellQty"] + 1e-12
        )

        # ---- Price Momentum (backward returns) ----
        df.loc[:, "bret1"] = (df["midpx"] - df["midpx"].shift(1)) / (df["midpx"].shift(1) + 1e-12)
        df.loc[:, "bret5"] = (df["midpx"] - df["midpx"].shift(5)) / (df["midpx"].shift(5) + 1e-12)
        df.loc[:, "bret12"] = (df["midpx"] - df["midpx"].shift(12)) / (df["midpx"].shift(12) + 1e-12)

        # ---- Cross-sectional normalization of returns ----
        for horizon, col in [(1, "bret1"), (5, "bret5"), (12, "bret12")]:
            cx_mean = df.groupby("interval")[[col]].mean().reset_index()
            cx_mean.columns = ["interval", "cx_{}".format(col)]
            df = df.merge(cx_mean, on="interval", how="left")
            df.loc[:, "lagret{}".format(horizon)] = df[col] - df["cx_{}".format(col)]
            df.drop(columns=["cx_{}".format(col)], inplace=True)

        # ---- Volatility & Range ----
        df.loc[:, "price_range"] = (df["high"] - df["low"]) / (df["midpx"] + 1e-12)
        df.loc[:, "realized_vol5"] = df["bret1"].rolling(5).std()
        df.loc[:, "realized_vol12"] = df["bret1"].rolling(12).std()
        df.loc[:, "spread_ema12"] = df["ob_spread"].ewm(halflife=12).mean()

        # ---- Volume ----
        df.loc[:, "total_trade_qty"] = df["tradeBuyQty"] + df["tradeSellQty"]
        df.loc[:, "rel_trade_qty"] = df["total_trade_qty"] / (df["asize0_4"] + df["bsize0_4"] + 1e-12)
        df.loc[:, "buy_qty_ratio"] = df["tradeBuyQty"] / (df["tradeBuyQty"] + df["tradeSellQty"] + 1e-12)
        df.loc[:, "total_qty_ema12"] = df["total_trade_qty"].ewm(halflife=12).mean()

        # ---- Cross-sectional Features ----
        cs_features = {
            "ob_imb0": "cs_imb0",
            "bret12": "cs_bret12",
            "total_trade_qty": "cs_volume",
            "add_imb": "cs_add_imb",
            "ob_spread": "cs_spread",
        }
        for src_col, dst_col in cs_features.items():
            cs_rank = df.groupby("interval")[[src_col]].rank(pct=True).rename(columns={src_col: dst_col})
            df = df.merge(cs_rank, left_index=True, right_index=True, how="left")

        # ---- Build feature matrix ----
        xdf = df[self.mcols + self.feature_names()].set_index(self.mcols)
        ydf = df[self.mcols + [self.ycol]].set_index(self.mcols)
        return xdf.fillna(0), ydf.fillna(0)
