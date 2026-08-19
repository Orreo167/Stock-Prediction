"""SVR 模型：RBF 核 + 前一日开盘价预测次日开盘价，递归预测未来。"""

import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import config


class SvrModel:
    name = "SVR"

    def __init__(self):
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None
        self.test_pred = None
        self.future_pred = None
        self.train_last = None

    def _build_samples(self, series):
        """X=昨日开盘价, y=今日开盘价（与 notebook 一致）。"""
        x = series[:-1].reshape(-1, 1)
        y = series[1:].reshape(-1, 1)
        return x, y

    def fit(self, train, test):
        train = np.asarray(train, dtype=float).reshape(-1)
        test = np.asarray(test, dtype=float).reshape(-1)
        self.train_last = train[-1]

        X_train, y_train = self._build_samples(train)
        X_train_s = self.scaler_x.fit_transform(X_train)
        y_train_s = self.scaler_y.fit_transform(y_train)

        self.model = SVR(kernel="rbf", C=config.SVR_C,
                         epsilon=config.SVR_EPSILON, gamma=config.SVR_GAMMA)
        self.model.fit(X_train_s, y_train_s.ravel())

        # 测试期：用真实前一日开盘价做单步预测
        prev = np.concatenate([[self.train_last], test[:-1]]).reshape(-1, 1)
        prev_s = self.scaler_x.transform(prev)
        pred_s = self.model.predict(prev_s)
        self.test_pred = self.scaler_y.inverse_transform(pred_s.reshape(-1, 1)).reshape(-1)
        return self

    def forecast(self, horizon):
        """递归：用上一步预测值作为下一步输入。"""
        cur = self.test_pred[-1] if len(self.test_pred) else self.train_last
        out = []
        for _ in range(horizon):
            x_s = self.scaler_x.transform(np.array([[cur]]))
            y_s = self.model.predict(x_s)
            cur = float(self.scaler_y.inverse_transform(y_s.reshape(-1, 1)).reshape(-1)[0])
            out.append(cur)
        self.future_pred = np.array(out)
        return self.future_pred

    # ---------- 持久化 ----------
    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.scaler_x, os.path.join(directory, "scaler_x.joblib"))
        joblib.dump(self.scaler_y, os.path.join(directory, "scaler_y.joblib"))
        joblib.dump(self.model, os.path.join(directory, "model.joblib"))
        np.save(os.path.join(directory, "test_pred.npy"), self.test_pred)
        with open(os.path.join(directory, "train_last.txt"), "w") as f:
            f.write(str(self.train_last))

    @classmethod
    def load(cls, directory):
        obj = cls()
        obj.scaler_x = joblib.load(os.path.join(directory, "scaler_x.joblib"))
        obj.scaler_y = joblib.load(os.path.join(directory, "scaler_y.joblib"))
        obj.model = joblib.load(os.path.join(directory, "model.joblib"))
        obj.test_pred = np.load(os.path.join(directory, "test_pred.npy"))
        with open(os.path.join(directory, "train_last.txt")) as f:
            obj.train_last = float(f.read().strip())
        return obj
