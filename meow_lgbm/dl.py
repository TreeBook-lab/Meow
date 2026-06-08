import os
import pandas as pd
from tradingcalendar import Calendar
from log import log


class MeowDataLoader(object):
    def __init__(self, h5dir):
        self.h5dir = h5dir
        self.calendar = Calendar()

    def load_dates(self, dates):
        if len(dates) == 0:
            raise ValueError("Dates empty")
        log.inf("Loading data of {} dates from {} to {}...".format(len(dates), min(dates), max(dates)))
        return pd.concat(self.load_date(x) for x in dates)

    def load_date(self, date):
        if not self.calendar.is_trading_day(date):
            raise ValueError("Not a trading day: {}".format(date))
        h5_file = os.path.join(self.h5dir, "{}.h5".format(date))
        df = pd.read_hdf(h5_file)
        df.loc[:, "date"] = date
        keep_cols = [
            "symbol", "interval", "date",
            "fret12", "midpx", "high", "low",
            "bid0", "ask0", "bid9", "ask9",
            "bsize0", "asize0",
            "bsize0_4", "asize0_4", "bsize5_9", "asize5_9", "bsize10_19", "asize10_19",
            "btr0_4", "atr0_4", "btr5_9", "atr5_9", "btr10_19", "atr10_19",
            "nTradeBuy", "tradeBuyQty", "tradeBuyTurnover",
            "nTradeSell", "tradeSellQty", "tradeSellTurnover",
            "addBuyQty", "addSellQty",
            "cxlBuyQty", "cxlSellQty",
        ]
        df = df[[c for c in keep_cols if c in df.columns]]
        return df
