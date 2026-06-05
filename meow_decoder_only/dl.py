import os
import pandas as pd
from tradingcalendar import Calendar


class MeowDataLoader:
    def __init__(self, h5dir):
        self.h5dir = h5dir
        self.calendar = Calendar()

    def load_date(self, date):
        if not self.calendar.isTradingDay(date):
            raise ValueError("Not a trading day: {}".format(date))
        h5file = os.path.join(self.h5dir, "{}.h5".format(date))
        df = pd.read_hdf(h5file)
        df.loc[:, "date"] = date
        precols = ["symbol", "interval", "date"]
        df = df[precols + [x for x in df.columns if x not in precols]]
        return df
