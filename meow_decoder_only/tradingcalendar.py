import os
import bisect
from log import log


class Calendar(object):
    def __init__(self):
        calendar_file = os.path.join(os.path.dirname(__file__), "resources/calendar")
        with open(calendar_file) as f:
            tokens = f.read().splitlines()
            self.trading_days = sorted([int(x) for x in tokens])
            self.trading_day_set = set(self.trading_days)

    def is_trading_day(self, date):
        if not isinstance(date, int):
            date = int(date)
        return date in self.trading_day_set

    def to_trading_day(self, date):
        if not isinstance(date, int):
            date = int(date)
        index = bisect.bisect_left(self.trading_days, date)
        return self.trading_days[index]

    def next(self, date):
        if not isinstance(date, int):
            date = int(date)
        index = bisect.bisect_right(self.trading_days, date)
        if index >= len(self.trading_days):
            return None
        return self.trading_days[index]

    def prev(self, date):
        if not isinstance(date, int):
            date = int(date)
        index = bisect.bisect_left(self.trading_days, date)
        if index == 0:
            return None
        return self.trading_days[index - 1]

    def shift(self, date, n):
        if not isinstance(date, int):
            date = int(date)
        if not isinstance(n, int):
            log.red("Invalid shift n: {}".format(n))
            return None

        index = bisect.bisect_left(self.trading_days, date)
        if index == 0:
            log.red("Failed to shift for date {}, n={}".format(date, n))
            return None
        return self.trading_days[index + n]

    def prevn(self, date, n):
        if not isinstance(date, int):
            date = int(date)
        if not isinstance(n, int) or n < 1:
            log.red("Invalid prevn: date={},n={}".format(date, n))
            return None

        index = bisect.bisect_left(self.trading_days, date)
        if index == 0:
            log.red("Failed to find prev trading day for date {}".format(date))
            return None
        if index < n:
            log.yellow("Not enough days for prevn: date={},n={},index={}".format(date, n, index))

        return self.trading_days[max(index - n, 0) : index]

    def nextn(self, date, n):
        if not isinstance(date, int):
            date = int(date)
        if not isinstance(n, int) or n < 1:
            log.red("Invalid nextn: date={},n={}".format(date, n))
            return None

        index = bisect.bisect_right(self.trading_days, date)
        if index >= len(self.trading_days):
            log.red("Failed to find next trading day for date {}".format(date))
            return None
        if index + n > len(self.trading_days):
            log.yellow("Not enough days for next: date={},n={},index={}".format(date, n, index))

        return self.trading_days[index: min(index + n, len(self.trading_days))]

    def range(self, start_date, end_date):
        if not isinstance(start_date, int):
            start_date = int(start_date)
        if not isinstance(end_date, int):
            end_date = int(end_date)
        if start_date > end_date:
            log.red("Invalid range - start_date is larger than end_date: start_date={},end_date={}".format(start_date, end_date))
            return None

        start_index = bisect.bisect_left(self.trading_days, start_date)
        if (start_index == len(self.trading_days)):
            log.red("No valid trading days found within the range [{}, {})".format(start_date, end_date))
            return None

        end_index = bisect.bisect_right(self.trading_days, end_date)
        return self.trading_days[start_index : end_index]
