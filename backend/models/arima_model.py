"""ARIMA 模型：pmdarima.auto_arima 自动定阶 + 单步滚动预测。"""

import os
import numpy as np
import joblib
import pmdarima as pm

import config


class ArimaModel:
    name = "ARIMA"

    def __init__(self):
        self.model = None
        self.test_pred = None
        self.future_pred = None
        self.future_conf = None  # (lower, upper)

    def fit(self, train, test):
        train = np.asarray(train, dtype=float).reshape(-1)
        test = np.asarray(test, dtype=float).reshape(-1)

        self.model = pm.auto_arima(
            train,
            start_p=1, start_q=1, max_p=10, max_q=10,
            seasonal=False,
            information_criterion="aic",
            trace=False,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
            scoring="mse",
        )

        # 测试期：单步预测 + 滚动更新（沿用 notebook 做法）
        preds, confs = [], []
        for x in test:
            p, conf = self.model.predict(n_periods=1, return_conf_int=True)
            preds.append(p.tolist()[0])
            confs.append(conf.tolist()[0])
            self.model.update(x)
        self.test_pred = np.array(preds)

        # 未来 horizon 天：直接用更新后的模型多步预测
        future, conf = self.model.predict(n_periods=config.FORECAST_HORIZON_MAX, return_conf_int=True)
        self.future_pred = np.asarray(future, dtype=float)
        self.future_conf = (np.asarray(conf)[:, 0], np.asarray(conf)[:, 1])
        return self

    def forecast(self, horizon):
        return self.future_pred[:horizon]

    def forecast_conf(self, horizon):
        return self.future_conf[0][:horizon], self.future_conf[1][:horizon]

    # ---------- 持久化 ----------
    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.model, os.path.join(directory, "model.joblib"))
        np.save(os.path.join(directory, "test_pred.npy"), self.test_pred)
        np.save(os.path.join(directory, "future_pred.npy"), self.future_pred)
        np.save(os.path.join(directory, "future_conf_low.npy"), self.future_conf[0])
        np.save(os.path.join(directory, "future_conf_high.npy"), self.future_conf[1])

    @classmethod
    def load(cls, directory):
        obj = cls()
        obj.model = joblib.load(os.path.join(directory, "model.joblib"))
        obj.test_pred = np.load(os.path.join(directory, "test_pred.npy"))
        obj.future_pred = np.load(os.path.join(directory, "future_pred.npy"))
        obj.future_conf = (
            np.load(os.path.join(directory, "future_conf_low.npy")),
            np.load(os.path.join(directory, "future_conf_high.npy")),
        )
        return obj
