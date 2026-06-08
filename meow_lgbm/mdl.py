import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from log import log


class MeowModel(object):
    def __init__(self, cache_dir, model_type="mlp"):
        self.cache_dir = cache_dir
        self.model_type = model_type
        self.scaler = StandardScaler()
        if model_type == "ridge":
            self.estimator = Ridge(
                alpha=0.5,
                random_state=None,
                fit_intercept=False,
                tol=1e-8
            )
        elif model_type == "mlp":
            self.estimator = MLPRegressor(
                hidden_layer_sizes=(256, 128, 64, 32),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=4096,
                learning_rate="adaptive",
                learning_rate_init=1e-3,
                max_iter=200,
                shuffle=True,
                random_state=42,
                tol=1e-6,
                verbose=False,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
            )
        elif model_type == "lgb":
            try:
                import lightgbm as lgb
                self.estimator = lgb.LGBMRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=63,
                    max_depth=10,
                    min_child_samples=200,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    max_bin=127,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                )
            except ImportError:
                log.yellow("LightGBM not installed, falling back to MLP")
                self.model_type = "mlp"
                self.estimator = MLPRegressor(
                    hidden_layer_sizes=(256, 128, 64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=4096,
                    learning_rate="adaptive",
                    learning_rate_init=1e-3,
                    max_iter=200,
                    shuffle=True,
                    random_state=42,
                    tol=1e-6,
                    verbose=False,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=10,
                )
        else:
            raise ValueError("Unknown model type: {}".format(model_type))
        log.inf("MeowModel initialized with type={}".format(self.model_type))

    def fit(self, xdf, ydf):
        import gc
        n_total = xdf.shape[0]
        if n_total > 2000000:
            log.inf("Sampling 2M rows from {} for memory efficiency...".format(n_total))
            idx = np.random.RandomState(42).choice(n_total, size=2000000, replace=False)
            xdf = xdf.iloc[idx]
            ydf = ydf.iloc[idx]
        X = xdf.to_numpy().astype(np.float32)
        y = ydf.to_numpy().astype(np.float32).ravel()
        del xdf, ydf
        gc.collect()
        X = self.scaler.fit_transform(X).astype(np.float32)
        gc.collect()
        log.inf("Training {} model on {} samples with {} features...".format(
            self.model_type, X.shape[0], X.shape[1]
        ))
        self.estimator.fit(X, y)
        log.inf("Done fitting")

    def predict(self, xdf):
        X = xdf.to_numpy().astype(np.float64)
        X = self.scaler.transform(X)
        return self.estimator.predict(X)
