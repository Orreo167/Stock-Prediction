"""ARIMA-LSTM 混合模型：ARIMA 主预测 + LSTM 学习 ARIMA 残差。

训练/预测流程：
1. ARIMA 在训练集上拟合，得到训练集拟合值与测试期单步滚动预测；
2. 训练集残差 = 真实值 - ARIMA拟合值，用 LSTM 学习该残差序列（滑窗）；
3. 测试期混合预测 = ARIMA测试预测 + LSTM残差预测（递归）；
4. 未来预测 = ARIMA未来预测 + LSTM残差递归预测。
"""

import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import pmdarima as pm
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

import config
from utils import create_dataset


class HybridModel:
    name = "ARIMA-LSTM"

    def __init__(self, params=None):
        params = params or {}
        self.max_p = int(params.get("max_p", config.ARIMA_MAX_P))
        self.max_q = int(params.get("max_q", config.ARIMA_MAX_Q))
        self.time_step = int(params.get("time_step", config.LSTM_TIME_STEP))
        self.units = int(params.get("units", config.LSTM_UNITS))
        self.dropout = float(params.get("dropout", config.LSTM_DROPOUT))
        self.epochs = int(params.get("epochs", config.LSTM_EPOCHS))
        self.batch = int(params.get("batch", config.LSTM_BATCH))
        self.arima = None
        self.resid_scaler = StandardScaler()
        self.resid_model = None
        self.test_pred = None
        self.future_pred = None
        self.future_conf = None
        self.train_resid = None
        self.test_arima = None
        self.future_arima = None

    def _fit_arima(self, train, test):
        train = np.asarray(train, dtype=float).reshape(-1)
        test = np.asarray(test, dtype=float).reshape(-1)

        self.arima = pm.auto_arima(
            train, start_p=1, start_q=1,
            max_p=self.max_p, max_q=self.max_q,
            seasonal=False, information_criterion="aic", trace=False,
            error_action="ignore", suppress_warnings=True, stepwise=True,
            scoring="mse",
        )

        # 训练集拟合值（用于残差序列）
        fitted = np.asarray(self.arima.predict_in_sample(), dtype=float)
        if len(fitted) < len(train):
            fitted = np.concatenate([np.full(len(train) - len(fitted), np.nan), fitted])
        self.train_resid = train - fitted
        self.train_resid = self.train_resid[~np.isnan(self.train_resid)]

        # 测试期单步滚动预测
        preds = []
        for x in test:
            p, conf = self.arima.predict(n_periods=1, return_conf_int=True)
            preds.append(p.tolist()[0])
            self.arima.update(x)
        self.test_arima = np.array(preds)

        # 未来多步预测 + 置信区间
        future, conf = self.arima.predict(
            n_periods=config.FORECAST_HORIZON_MAX, return_conf_int=True)
        self.future_arima = np.asarray(future, dtype=float)
        self.future_conf = (np.asarray(conf)[:, 0], np.asarray(conf)[:, 1])
        return train, test

    def _fit_resid_lstm(self):
        resid_scaled = self.resid_scaler.fit_transform(
            self.train_resid.reshape(-1, 1)).reshape(-1)
        X, y = create_dataset(resid_scaled, self.time_step)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        tf.keras.backend.clear_session()
        model = Sequential([
            LSTM(units=self.units, return_sequences=False,
                 input_shape=(self.time_step, 1)),
            Dropout(self.dropout),
            Dense(units=1),
        ])
        model.compile(optimizer="rmsprop", loss="mean_squared_error")
        model.fit(X, y, epochs=self.epochs,
                  batch_size=self.batch, verbose=0)
        self.resid_model = model

    def _forecast_resid(self, start_window, steps):
        window = list(start_window)
        out = []
        for _ in range(steps):
            x = np.array(window[-self.time_step:]).reshape(1, self.time_step, 1)
            y = float(self.resid_model.predict(x, verbose=0).reshape(-1)[0])
            out.append(y)
            window.append(y)
        return np.array(out)

    def fit(self, train, test):
        train, test = self._fit_arima(train, test)
        self._fit_resid_lstm()

        # 测试期残差预测：从训练残差末尾递归预测 len(test) 步
        resid_scaled = self.resid_scaler.transform(
            self.train_resid.reshape(-1, 1)).reshape(-1)
        last_win = resid_scaled[-self.time_step:]
        resid_pred_scaled = self._forecast_resid(last_win, len(test))
        resid_pred = self.resid_scaler.inverse_transform(
            resid_pred_scaled.reshape(-1, 1)).reshape(-1)
        self.test_pred = self.test_arima + resid_pred
        return self

    def forecast(self, horizon):
        resid_scaled = self.resid_scaler.transform(
            self.train_resid.reshape(-1, 1)).reshape(-1)
        last_win = resid_scaled[-self.time_step:]
        resid_pred_scaled = self._forecast_resid(last_win, horizon)
        resid_pred = self.resid_scaler.inverse_transform(
            resid_pred_scaled.reshape(-1, 1)).reshape(-1)
        self.future_pred = self.future_arima[:horizon] + resid_pred
        return self.future_pred

    def forecast_conf(self, horizon):
        return self.future_conf[0][:horizon], self.future_conf[1][:horizon]

    # ---------- 持久化 ----------
    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.arima, os.path.join(directory, "arima.joblib"))
        self.resid_model.save(os.path.join(directory, "resid_model.keras"))
        joblib.dump(self.resid_scaler, os.path.join(directory, "resid_scaler.joblib"))
        np.save(os.path.join(directory, "train_resid.npy"), self.train_resid)
        np.save(os.path.join(directory, "test_pred.npy"), self.test_pred)
        with open(os.path.join(directory, "time_step.txt"), "w") as f:
            f.write(str(self.time_step))

    @classmethod
    def load(cls, directory):
        obj = cls()
        obj.arima = joblib.load(os.path.join(directory, "arima.joblib"))
        obj.resid_model = load_model(os.path.join(directory, "resid_model.keras"))
        obj.resid_scaler = joblib.load(os.path.join(directory, "resid_scaler.joblib"))
        obj.train_resid = np.load(os.path.join(directory, "train_resid.npy"))
        obj.test_pred = np.load(os.path.join(directory, "test_pred.npy"))
        with open(os.path.join(directory, "time_step.txt")) as f:
            obj.time_step = int(f.read().strip())
        # 重新推导未来预测与置信区间（ARIMA 已含测试期滚动更新状态）
        future, conf = obj.arima.predict(
            n_periods=config.FORECAST_HORIZON_MAX, return_conf_int=True)
        obj.future_arima = np.asarray(future, dtype=float)
        obj.future_conf = (np.asarray(conf)[:, 0], np.asarray(conf)[:, 1])
        return obj
