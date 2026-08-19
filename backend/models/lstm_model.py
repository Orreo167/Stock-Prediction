"""LSTM 模型：滑窗 + StandardScaler + 单层 LSTM，递归多步预测。"""

import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

import config
from utils import create_dataset, sliding_predict_1d, recursive_forecast_1d


class LstmModel:
    name = "LSTM"

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.time_step = config.LSTM_TIME_STEP
        self.test_pred = None
        self.future_pred = None
        self.full_series = None

    def fit(self, train, test):
        train = np.asarray(train, dtype=float).reshape(-1)
        test = np.asarray(test, dtype=float).reshape(-1)

        # 标准化：只用训练集拟合 scaler
        train_scaled = self.scaler.fit_transform(train.reshape(-1, 1)).reshape(-1)
        self.full_series = np.concatenate([train, test])
        full_scaled = self.scaler.transform(self.full_series.reshape(-1, 1)).reshape(-1)

        X_train, y_train = create_dataset(train_scaled, self.time_step)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        tf.keras.backend.clear_session()
        model = Sequential([
            LSTM(units=config.LSTM_UNITS, return_sequences=False,
                 input_shape=(self.time_step, 1)),
            Dropout(config.LSTM_DROPOUT),
            Dense(units=1),
        ])
        model.compile(optimizer="rmsprop", loss="mean_squared_error")
        model.fit(X_train, y_train, epochs=config.LSTM_EPOCHS,
                  batch_size=config.LSTM_BATCH, verbose=0)
        self.model = model

        # 测试期预测：在完整序列上滑窗（与测试集日期对齐）
        test_pred_scaled = sliding_predict_1d(model, full_scaled, self.time_step)
        self.test_pred = self.scaler.inverse_transform(
            test_pred_scaled.reshape(-1, 1)).reshape(-1)[len(train):]
        return self

    def forecast(self, horizon):
        full_scaled = self.scaler.transform(self.full_series.reshape(-1, 1)).reshape(-1)
        last_window = full_scaled[-(self.time_step):]
        future_scaled = recursive_forecast_1d(self.model, last_window, horizon, channel_dim=True)
        self.future_pred = self.scaler.inverse_transform(
            future_scaled.reshape(-1, 1)).reshape(-1)
        return self.future_pred

    # ---------- 持久化 ----------
    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        self.model.save(os.path.join(directory, "model.keras"))
        joblib.dump(self.scaler, os.path.join(directory, "scaler.joblib"))
        np.save(os.path.join(directory, "full_series.npy"), self.full_series)
        np.save(os.path.join(directory, "test_pred.npy"), self.test_pred)
        with open(os.path.join(directory, "time_step.txt"), "w") as f:
            f.write(str(self.time_step))

    @classmethod
    def load(cls, directory):
        obj = cls()
        obj.model = load_model(os.path.join(directory, "model.keras"))
        obj.scaler = joblib.load(os.path.join(directory, "scaler.joblib"))
        obj.full_series = np.load(os.path.join(directory, "full_series.npy"))
        obj.test_pred = np.load(os.path.join(directory, "test_pred.npy"))
        with open(os.path.join(directory, "time_step.txt")) as f:
            obj.time_step = int(f.read().strip())
        return obj
