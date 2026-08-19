"""GARCH 波动率蒙特卡洛路径生成（方案 3）。

思路：
1. 中心趋势 = 训练好的模型的递归点预测（绝对价格）
2. 波动率 = GARCH(1,1) 拟合历史对数收益率，预测未来 N 天条件波动率
   （能捕捉"波动聚集"：大跌后波动更大，波动尺度与近期市场一致）
3. 未来每天收益率 = 中心漂移 + sigma_t * z_t（z 为标准化学生 t 扰动），
   从最后实际价递推生成 N_PATHS 条模拟路径
4. 输出中位数路径（主预测线）、5%/95% 分位带、抽样灰色路径

arch 库不可用时自动降级：用最近 GARCH_WINDOW_FALLBACK 天收益率标准差
作为常数波动率（波动尺度仍与近期市场一致）。
"""

import numpy as np

import config


def _returns_from_prices(prices):
    """对数收益率（小数）。"""
    prices = np.asarray(prices, dtype=float)
    r = np.diff(np.log(prices))
    return r[~np.isnan(r)]


def _garch_sigma(returns, horizon):
    """用 GARCH(1,1) 拟合收益率，返回 (未来 horizon 天波动率, t 分布自由度)。

    返回的 sigma 为小数尺度（如 0.02 表示单日 2% 波动）。
    """
    from arch import arch_model

    am = arch_model(returns * 100.0, vol="GARCH",
                    p=config.GARCH_P, q=config.GARCH_Q,
                    mean="Zero", dist=config.GARCH_DIST, rescale=True)
    res = am.fit(disp="off", show_warning=False)
    fcast = res.forecast(horizon=horizon)
    var = fcast.variance.iloc[-1].values.astype(float)  # 百分比方差
    sigma = np.sqrt(np.maximum(var, 1e-12)) / 100.0     # 还原为小数波动率
    nu = float(res.params.get("nu", 8.0))
    return sigma, nu


def _fallback_sigma(returns, horizon):
    """降级：最近 window 天收益率标准差作为常数波动率。"""
    window = config.GARCH_WINDOW_FALLBACK
    recent = returns[-window:] if len(returns) > window else returns
    sigma0 = float(np.std(recent))
    return np.full(horizon, max(sigma0, 1e-6)), 8.0


def generate_paths(prices, center, horizon, n_paths=None, seed=None):
    """生成未来 horizon 天模拟路径。

    prices : 历史开盘价序列（拟合 GARCH 波动率）
    center : 模型点预测（绝对价格，长度必须等于 horizon）
    返回 dict: paths/median/q5/q95/sigma/method
    """
    prices = np.asarray(prices, dtype=float)
    center = np.asarray(center, dtype=float).reshape(-1)
    horizon = int(horizon)
    if len(center) != horizon:
        raise ValueError("center 长度必须等于 horizon")
    n_paths = config.N_PATHS if n_paths is None else int(n_paths)
    seed = config.PATH_SEED if seed is None else int(seed)

    p0 = float(prices[-1])
    returns = _returns_from_prices(prices)
    if len(returns) < 50:
        raise ValueError("历史数据不足，无法拟合波动率")

    try:
        sigma, nu = _garch_sigma(returns, horizon)
        method = "garch"
    except Exception as exc:
        print(f"[garch] arch 不可用，使用降级波动率: {exc}")
        sigma, nu = _fallback_sigma(returns, horizon)
        method = "fallback"

    # 中心漂移收益率：mu_t = center_t / center_{t-1} - 1（起点为最后实际价）
    mu = center / np.concatenate([[p0], center[:-1]]) - 1.0

    rng = np.random.default_rng(seed)
    if nu > 2.0:
        z = rng.standard_t(nu, size=(n_paths, horizon))
        z = z / np.sqrt(nu / (nu - 2.0))  # 单位方差化，sigma 即标准差
    else:
        z = rng.normal(size=(n_paths, horizon))

    ret = mu[None, :] + sigma[None, :] * z
    log_paths = np.log(p0) + np.cumsum(np.log1p(ret), axis=1)
    paths = np.exp(log_paths)

    return {
        "paths": paths,
        "median": np.median(paths, axis=0),
        "q5": np.percentile(paths, 5, axis=0),
        "q95": np.percentile(paths, 95, axis=0),
        "sigma": sigma,
        "method": method,
    }
