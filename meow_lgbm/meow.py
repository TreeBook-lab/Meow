import os
from log import log
from dl import MeowDataLoader
from feat import MeowFeatureGenerator
from mdl import MeowModel
from eval import MeowEvaluator
from tradingcalendar import Calendar


class MeowEngine(object):
    def __init__(self, h5dir, cache_dir=None, model_type="mlp"):
        self.calendar = Calendar()
        self.h5dir = h5dir
        if not os.path.exists(h5dir):
            raise ValueError("Data directory not exists: {}".format(self.h5dir))
        if not os.path.isdir(h5dir):
            raise ValueError("Invalid data directory: {}".format(self.h5dir))
        self.cache_dir = cache_dir
        self.dloader = MeowDataLoader(h5dir=h5dir)
        self.feat_generator = MeowFeatureGenerator(cache_dir=cache_dir)
        self.model = MeowModel(cache_dir=cache_dir, model_type=model_type)
        self.evaluator = MeowEvaluator(cache_dir=cache_dir)

    def fit(self, start_date, end_date):
        dates = self.calendar.range(start_date, end_date)
        raw_data = self.dloader.load_dates(dates)
        log.inf("Running model fitting...")
        xdf, ydf = self.feat_generator.gen_features(raw_data)
        self.model.fit(xdf, ydf)

    def predict(self, xdf):
        return self.model.predict(xdf)

    def eval(self, start_date, end_date):
        log.inf("Running model evaluation...")
        dates = self.calendar.range(start_date, end_date)
        raw_data = self.dloader.load_dates(dates)
        xdf, ydf = self.feat_generator.gen_features(raw_data)
        ydf.loc[:, "forecast"] = self.predict(xdf)
        self.evaluator.eval(ydf)


if __name__ == "__main__":
    import os
    data_dir = "archive"
    engine = MeowEngine(h5dir=data_dir, cache_dir=None, model_type="lgb")
    engine.fit(20230601, 20231130)
    engine.eval(20231201, 20231229)
