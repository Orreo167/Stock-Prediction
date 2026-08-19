"""BP 神经网络：滑窗 + MinMaxScaler + 全连接网络，递归多步预测。"""

import os
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense

import config
from utils import create_dataset, sliding_predict_1d, recursive_forecast_1d


class BpModel:
    name = "BP"

    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.time_step = config.BP_TIME_STEP
        self.test_pred = None
        self.future_pred = None
        self.full_series = None

    def fit(self, train, test):
        train = np.asarray(train, dtype=float).reshape(-1)
        test = np.asarray(test, dtype=float).reshape(-1)

        train_scaled = self.scaler.fit_transform(train.reshape(-1, 1)).reshape(-1)
        self.full_series = np.concatenate([train, test])
        full_scaled = self.scaler.transform(self.full_series.reshape(-1, 1)).reshape(-1)

        X_train, y_train = create_dataset(train_scaled, self.time_step)

        tf.keras.backend.clear_session()
        model = Sequential([
            Dense(units=config.BP_UNITS_1, input_dim=self.time_step, activation="relu"),
            Dense(units=config.BP_UNITS_2, activation="relu"),
            Dense(units=1),
        ])
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(X_train, y_train, epochs=config.BP_EPOCHS,
                  batch_size=config.BP_BATCH, verbose=0)
        self.model = model

        test_pred_scaled = sliding_predict_1d(model, full_scaled, self.time_step)
        self.test_pred = self.scaler.inverse_transform(
            test_pred_scaled.reshape(-1, 1)).reshape(-1)[len(train):]
        return self

    def forecast(self, horizon):
        full_scaled = self.scaler.transform(self.full_series.reshape(-1, 1)).reshape(-1)
        last_window = full_scaled[-(self.time_step):]
        future_scaled = recursive_forecast_1d(self.model, last_window, horizon, channel_dim=False)
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
