import os
import pickle
from sklearn.linear_model import Ridge
from log import log


class MeowModel(object):
    def __init__(self, cache_dir):
        self.estimator = Ridge(
            alpha=0.5,
            random_state=None,
            fit_intercept=False,
            tol=1e-8
        )

    def fit(self, xdf, ydf):
        self.estimator.fit(
            X=xdf.to_numpy(),
            y=ydf.to_numpy(),
        )
        weights_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights"))
        os.makedirs(weights_dir, exist_ok=True)
        weight_path = os.path.join(weights_dir, "meow_ridge.pkl")
        with open(weight_path, "wb") as f:
            pickle.dump(self.estimator, f)
        log.inf(f"Saved model weights to {weight_path}")
        log.inf("Done fitting")

    def predict(self, xdf):
        return self.estimator.predict(xdf.to_numpy())
