"""公共工具：指标计算、滑窗构造、递归预测。"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(actual, pred):
    """计算 MAE/MSE/RMSE/R²。pred 中 NaN 位置跳过。"""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(p))
    a, p = a[mask], p[mask]
    if len(a) == 0:
        return {"MAE": float("nan"), "MSE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}
    mse = mean_squared_error(a, p)
    return {
        "MAE": float(mean_absolute_error(a, p)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(a, p)),
    }


def create_dataset(data, time_step):
    """滑窗：data[i:i+time_step] -> data[i+time_step]。返回 (X, y)。"""
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i + time_step])
        y.append(data[i + time_step])
    return np.array(X), np.array(y)


def sliding_predict_1d(model, series, time_step):
    """对整条序列逐点预测，返回与 series 等长的数组（前 time_step 位为 NaN）。

    每个位置 i 使用 series[i-time_step:i] 作为输入，输出 series[i] 的预测。
    """
    n = len(series)
    out = np.full(n, np.nan)
    if n <= time_step:
        return out
    X = np.array([series[i - time_step:i] for i in range(time_step, n)])
    pred = model.predict(X, verbose=0)
    out[time_step:] = pred.reshape(-1)
    return out


def recursive_forecast_1d(model, last_window, horizon, channel_dim=False):
    """用最后一段窗口递归预测未来 horizon 个点。

    channel_dim=True 时输入为 [1, steps, 1]（LSTM 等）；否则为 [1, steps]（Dense/BP）。
    """
    steps = len(last_window)
    window = list(last_window)
    out = []
    for _ in range(horizon):
        arr = np.array(window[-steps:])
        x = arr.reshape(1, steps, 1) if channel_dim else arr.reshape(1, steps)
        y = float(model.predict(x, verbose=0).reshape(-1)[0])
        out.append(y)
        window.append(y)
    return np.array(out)
