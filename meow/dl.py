import os
import pandas as pd
from tradingcalendar import Calendar
from log import log


class MeowDataLoader(object):
    def __init__(self, h5dir):
        self.h5dir = h5dir
        self.calendar = Calendar()

    def loadDates(self, dates, *, columns=None, max_rows_per_date: int = 0, seed: int = 1):
        if len(dates) == 0:
            raise ValueError("Dates empty")
        log.inf("Loading data of {} dates from {} to {}...".format(len(dates), min(dates), max(dates)))
        return pd.concat(
            (self.loadDate(x, columns=columns, max_rows=max_rows_per_date, seed=seed) for x in dates),
            axis=0,
        )

    def loadDate(self, date, *, columns=None, max_rows: int = 0, seed: int = 1):
        if not self.calendar.isTradingDay(date):
            raise ValueError("Not a trading day: {}".format(date))
        h5File = os.path.join(self.h5dir, "{}.h5".format(date))
        df = pd.read_hdf(h5File)

        # Optional: limit columns early to reduce memory.
        if columns is not None:
            keep = list(columns)
            # Always keep core identifiers if present.
            for c in ["symbol", "interval"]:
                if c not in keep and c in df.columns:
                    keep.append(c)
            keep = [c for c in keep if c in df.columns]
            df = df[keep]

        # Optional: subsample rows per date to cap memory.
        if max_rows and int(max_rows) > 0 and df.shape[0] > int(max_rows):
            df = df.sample(n=int(max_rows), replace=False, random_state=int(seed)).sort_index()

        df.loc[:, "date"] = date
        precols = ["symbol", "interval", "date"]
        df = df[precols + [x for x in df.columns if x not in precols]] # re-arrange columns
        return df
    
if __name__ == "__main__":
    df = pd.read_hdf("/home/treeboss/WorkSpace/MEOW/archive/20230602.h5")
    print(df.head())

