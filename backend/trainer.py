"""训练编排：串行训练 5 个模型，进度回调，最优模型评选，结果缓存。"""

import os
import json
import joblib
import numpy as np
import pandas as pd

import config
import data_service
from garch import generate_paths
from utils import compute_metrics
from models import MODELS


class Trainer:
    def __init__(self, progress_cb=None):
        self.progress_cb = progress_cb or (lambda *a, **k: None)
        self.code = None
        self.df = None
        self.source = None
        self.warn = None
        self.train = None
        self.test = None
        self.results = {}      # model_name -> fitted model
        self.metrics = {}      # model_name -> dict
        self.test_actual = None

    # ---------- 进度 ----------
    def _progress(self, stage, pct, detail=""):
        self.progress_cb(stage=stage, pct=pct, detail=detail)

    # ---------- 数据 ----------
    def _load_data(self, code):
        self.code = str(code).strip()
        self._progress("数据", 0.05, f"拉取股票 {self.code} 日K线…")
        self.df, self.source, self.warn = data_service.get_stock_data(self.code)
        n = len(self.df)
        if n < config.MIN_ROWS_WARN:
            raise ValueError(
                f"股票 {self.code} 数据仅 {n} 个交易日，少于最低要求 "
                f"{config.MIN_ROWS_WARN} 天，拒绝训练。")
        split = int(n * config.TRAIN_SIZE)
        self.train = self.df["开盘"].values[:split]
        self.test = self.df["开盘"].values[split:]
        self.test_actual = self.test
        self._progress("数据", 0.1, f"共 {n} 个交易日（训练 {len(self.train)} / 测试 {len(self.test)}）")

    # ---------- 训练 ----------
    def _train_all(self):
        order = config.MODEL_ORDER
        total = len(order)
        for i, name in enumerate(order):
            try:
                self._progress(name, (i + 0.3) / total, f"正在训练 {name}…")
                model = MODELS[name]()
                model.fit(self.train, self.test)
                self.results[name] = model
                self.metrics[name] = compute_metrics(self.test, model.test_pred)
                self._progress(name, (i + 1) / total, f"{name} 完成")
            except Exception as exc:
                print(f"[trainer] {name} 训练失败: {exc}")
                self.metrics[name] = None

    # ---------- 预测 ----------
    def _forecast_all(self, horizon):
        forecasts = {}
        for name, model in self.results.items():
            try:
                forecasts[name] = model.forecast(horizon)
            except Exception as exc:
                print(f"[trainer] {name} 预测失败: {exc}")
                forecasts[name] = None
        return forecasts

    # ---------- 评选 ----------
    def _best_model(self):
        valid = {k: v for k, v in self.metrics.items() if v}
        if not valid:
            raise ValueError("所有模型均训练失败，请稍后重试。")
        # 用户指定：预测固定使用 ARIMA-LSTM 混合模型（ARIMA 主预测 + LSTM 残差，预测带变化）
        if "ARIMA-LSTM" in self.results:
            return "ARIMA-LSTM"
        # 回退：R² 最高优先，并列取 RMSE 更低
        return max(valid, key=lambda k: (valid[k]["R2"], -valid[k]["RMSE"]))

    # ---------- 结果组装 ----------
    def _build_result(self, horizon, forecasts):
        dates = self.df["日期"].dt.strftime("%Y-%m-%d").tolist()
        opens = self.df["开盘"].round(3).tolist()
        n_train = len(self.train)
        test_dates = dates[n_train:]
        test_actual = np.round(self.test_actual, 3).tolist()

        test_models = {}
        for name, model in self.results.items():
            test_models[name] = np.round(model.test_pred, 3).tolist()

        best = self._best_model()
        model = self.results[best]
        fc = np.asarray(forecasts[best], dtype=float)

        # 方案3：GARCH 波动率蒙特卡洛路径（中心 = 模型点预测）
        try:
            gen = generate_paths(self.df["开盘"].values, fc, horizon)
            median = np.round(gen["median"], 3).tolist()
            lower = np.round(gen["q5"], 3).tolist()
            upper = np.round(gen["q95"], 3).tolist()
            sample_paths = np.round(
                gen["paths"][: config.N_PATHS_SHOW], 3).tolist()
            vol = np.round(gen["sigma"], 6).tolist()
            path_method = gen["method"]
        except Exception as exc:
            print(f"[trainer] GARCH 路径生成失败，回退模型置信区间: {exc}")
            fc = np.round(fc, 3).tolist()
            if hasattr(model, "forecast_conf"):
                lo, hi = model.forecast_conf(horizon)
                lower = np.round(lo, 3).tolist()
                upper = np.round(hi, 3).tolist()
            else:
                resid = self.test - model.test_pred
                std = float(np.nanstd(resid))
                lower = np.round(np.asarray(fc) - 1.96 * std, 3).tolist()
                upper = np.round(np.asarray(fc) + 1.96 * std, 3).tolist()
            median = fc
            sample_paths = []
            vol = []
            path_method = "model_conf"

        fc_dates = self._next_trading_days(dates[-1], horizon)

        source_map = {"eastmoney": "在线拉取（东方财富）", "tencent": "在线拉取（腾讯）",
                      "sina": "在线拉取（新浪）", "netease": "在线拉取（网易）",
                      "efinance": "在线拉取（efinance）", "local": "本地缓存",
                      "sample": "内置示例数据"}
        stock = {
            "code": self.code,
            "name": self._stock_name(),
            "start": dates[0],
            "end": dates[-1],
            "samples": len(self.df),
            "cache": source_map.get(self.source, self.source),
        }
        if self.warn:
            stock["warn"] = self.warn

        return {
            "stock": stock,
            "history": {"dates": dates, "open": opens},
            "test": {"dates": test_dates, "actual": test_actual, "models": test_models},
            "metrics": {k: (v or None) for k, v in self.metrics.items()},
            "forecast": {
                "model": best,
                "horizon": horizon,
                "dates": fc_dates,
                "values": median,      # 中位数路径（主预测线）
                "lower": lower,        # 5% 分位
                "upper": upper,        # 95% 分位
                "paths": sample_paths, # 抽样灰色路径
                "vol": vol,            # 每日波动率（小数）
                "path_method": path_method,
            },
            "note": (f"预测模型：{best}（ARIMA 主预测 + LSTM 学习残差）；"
                     if best == "ARIMA-LSTM" else
                     f"预测模型：{best}（R² 最高）；")
                    + f"历史为真实开盘价；数据来源：{stock['cache']}",
        }

    def _stock_name(self):
        """从本地数据缓存读取股票名称（efinance/示例数据均含该列）。"""
        try:
            local_file = os.path.join(config.DATA_DIR, f"{self.code}.csv")
            if os.path.exists(local_file):
                df = pd.read_csv(local_file, encoding="utf-8", nrows=1)
                if "股票名称" in df.columns:
                    return str(df["股票名称"].iloc[0])
        except Exception:
            pass
        return ""

    def _next_trading_days(self, last_date, n):
        """从最后交易日往后生成 n 个工作日（交易日近似）。"""
        d = pd.Timestamp(last_date)
        out = []
        while len(out) < n:
            d = d + pd.Timedelta(days=1)
            if d.weekday() < 5:
                out.append(d.strftime("%Y-%m-%d"))
        return out

    # ---------- 缓存 ----------
    def _cache_dir(self, code):
        return os.path.join(config.CACHE_DIR, str(code).strip())

    def _cache_fingerprint(self):
        """数据指纹：最后交易日 + 样本量，数据变化时缓存自动失效。"""
        last = self.df["日期"].iloc[-1].strftime("%Y-%m-%d")
        return f"{last}|{len(self.df)}"

    def _load_cached(self, code):
        """尝试加载已训练模型；成功返回 True。数据指纹不一致时视为失效。"""
        d = self._cache_dir(code)
        meta_file = os.path.join(d, "meta.json")
        if not os.path.exists(meta_file):
            return False
        try:
            with open(meta_file, encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("fingerprint") != self._cache_fingerprint():
                print("[trainer] 数据已更新，缓存失效，重新训练")
                return False
            for name, cls in MODELS.items():
                mdir = os.path.join(d, name)
                if os.path.isdir(mdir):
                    self.results[name] = cls.load(mdir)
            self.metrics = {k: v for k, v in meta.get("metrics", {}).items()}
            self.source = meta.get("source", "local")
            self.warn = meta.get("warn")
            return True
        except Exception as exc:
            print(f"[trainer] 缓存加载失败，重新训练: {exc}")
            return False

    def _save_cache(self):
        d = self._cache_dir(self.code)
        os.makedirs(d, exist_ok=True)
        try:
            for name, model in self.results.items():
                model.save(os.path.join(d, name))
            with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "fingerprint": self._cache_fingerprint(),
                    "metrics": {k: v for k, v in self.metrics.items() if v},
                    "source": self.source,
                    "warn": self.warn,
                }, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"[trainer] 缓存保存失败（不影响本次结果）: {exc}")

    # ---------- 主流程 ----------
    def run(self, code, horizon, force_retrain=False):
        horizon = int(horizon)
        horizon = max(config.FORECAST_HORIZON_MIN,
                      min(config.FORECAST_HORIZON_MAX, horizon))
        self._load_data(code)
        cached = (not force_retrain) and self._load_cached(code)
        if not cached:
            self._train_all()
            self._save_cache()
        self._progress("预测", 0.95, "生成未来预测…")
        forecasts = self._forecast_all(horizon)
        result = self._build_result(horizon, forecasts)
        self._progress("完成", 1.0, "全部完成")
        return result


def run_training(code, horizon, progress_cb=None, force_retrain=False):
    """后台线程入口。"""
    trainer = Trainer(progress_cb=progress_cb)
    return trainer.run(code, horizon, force_retrain=force_retrain)
