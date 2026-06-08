import gc
import os
import random
import numpy as np
import pandas as pd
from log import log
from dl import MeowDataLoader
from feat import MeowFeatureGenerator
from mdl import MeowModel
from eval import MeowEvaluator
from tradingcalendar import Calendar
from parameters import TRAINING_CONFIG, PREPROCESSING_CONFIG


class MeowEngine:
    def __init__(self, h5dir, cache_dir):
        self.calendar = Calendar()
        self.h5dir = h5dir
        if not os.path.exists(h5dir):
            raise ValueError("Data directory not exists: {}".format(self.h5dir))
        if not os.path.isdir(h5dir):
            raise ValueError("Invalid data directory: {}".format(self.h5dir))
        self.dloader = MeowDataLoader(h5dir=h5dir)
        self.feat_gen = MeowFeatureGenerator(cache_dir=cache_dir)
        self.model = MeowModel()
        self.evaluator = MeowEvaluator(cache_dir=cache_dir)

    def fit(self, start_date, end_date):
        train_dates = self.calendar.range(start_date, end_date)
        n_epochs = TRAINING_CONFIG.n_epochs

        n_fit = min(PREPROCESSING_CONFIG.preprocessing_fit_days, len(train_dates))
        fit_dates = train_dates[:n_fit]

        log.inf("Fitting preprocessing on first {} dates...".format(len(fit_dates)))
        xdfs, ydfs = [], []
        for date in fit_dates:
            raw = self.dloader.load_date(date)
            xdf, ydf = self.feat_gen.gen_features(raw)
            xdfs.append(xdf)
            ydfs.append(ydf)
        self.model.fit_preprocessing(pd.concat(xdfs), pd.concat(ydfs))
        del xdfs, ydfs, xdf, ydf, raw
        gc.collect()
        self.model.set_scheduler(steps_per_epoch=len(train_dates), n_epochs=n_epochs)

        log.inf("Training on {} dates for {} epochs...".format(
            len(train_dates), n_epochs))
        for epoch in range(n_epochs):
            shuffled = list(train_dates)
            random.shuffle(shuffled)
            for date in shuffled:
                raw = self.dloader.load_date(date)
                xdf, ydf = self.feat_gen.gen_features(raw)
                self.model.partial_fit(xdf, ydf)
                del raw, xdf, ydf
                gc.collect()
        log.inf("Done fitting")

    def predict(self, xdf, denormalize=True):
        return self.model.predict(xdf, denormalize=denormalize)

    def eval(self, start_date, end_date):
        log.inf("Running model evaluation...")
        dates = self.calendar.range(start_date, end_date)

        all_dfs = []
        for date in dates:
            raw = self.dloader.load_date(date)
            xdf, ydf = self.feat_gen.gen_features(raw)

            p = self.predict(xdf, denormalize=True)

            eval_df = ydf.copy()
            eval_df[self.evaluator.prediction_col] = p
            all_dfs.append(eval_df)

        self.evaluator.eval(pd.concat(all_dfs))


if __name__ == "__main__":
    engine = MeowEngine(h5dir="archive/", cache_dir=None)
    engine.fit(20230601, 20231130)
    engine.eval(20231201, 20231229)