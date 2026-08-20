"""数据服务：拉取股票日K线、本地缓存与样本回退。

数据源优先级：
1. 东方财富 K 线接口直连（按 A 股规则构造 secid，重试 + 多通道）
2. 网易历史行情接口（备选，返回 2020 年至今全量日K）
3. efinance 库（备选）
4. 本地缓存 / 内置示例数据（离线回退）
"""

import io
import os
import threading
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

import config

_lock = threading.Lock()


def _load_legacy_sample():
    """从原项目文件夹加载示例数据（本机开发用）。"""
    if os.path.exists(config.LEGACY_SAMPLE_FILE):
        return pd.read_csv(config.LEGACY_SAMPLE_FILE, encoding="utf-8")
    return None


_EM_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Referer": "https://quote.eastmoney.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "close",
}
_EM_UT = "fa5fd1943c7b386f172d6893dbfba10b"  # 东财接口常用 token
_EM_BEG = "20200101"  # 起点：约 6.5 年，保证可用交易日 >= 1300

_COMMON_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

_EM_COLUMNS = ["日期", "开盘", "收盘", "最高", "最低",
               "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]


def _em_secid(code):
    """A 股规则：6/9 开头沪市 -> 1.xxx；其余深市 -> 0.xxx。"""
    if str(code).startswith(("6", "9")):
        return "1." + str(code)
    return "0." + str(code)


def _fetch_eastmoney(code):
    """直连东方财富日 K 接口（重试 + 多通道），返回 DataFrame（含股票名称列）。

    通道组合依次尝试：https/直连、https/走系统代理、http/直连、http/走代理；
    整体重试 2 轮并带退避。失败（网络/风控/解析）时抛异常，由调用方回退。
    """
    code = str(code).strip()
    last_exc = None
    channels = [(True, False), (True, True), (False, False), (False, True)]
    for attempt in range(2):
        ts = int(time.time() * 1000)  # 破除接口缓存
        for https, use_proxy in channels:
            scheme = "https" if https else "http"
            url = (
                f"{scheme}://push2his.eastmoney.com/api/qt/stock/kline/get"
                "?fields1=f1,f2,f3,f4,f5,f6"
                "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
                f"&secid={_em_secid(code)}&klt=101&fqt=1"
                f"&beg={_EM_BEG}&end=20500101&rtntype=6"
                f"&ut={_EM_UT}&_={ts}"
            )
            try:
                session = requests.Session()
                session.trust_env = use_proxy
                resp = session.get(url, headers=_EM_HEADERS, timeout=20)
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}")
                payload = resp.json()
                data = payload.get("data")
                if not data or not data.get("klines"):
                    raise RuntimeError(f"接口返回空 data（疑似限流/风控）：{resp.text[:120]}")
                name = data.get("name", "")
                rows = []
                for line in data["klines"]:
                    parts = line.split(",")
                    if len(parts) >= len(_EM_COLUMNS):
                        rows.append([parts[0]] + parts[1:len(_EM_COLUMNS)])
                if not rows:
                    raise RuntimeError("K 线解析结果为空")
                df = pd.DataFrame(rows, columns=_EM_COLUMNS)
                df.insert(0, "股票名称", name)
                df.insert(1, "股票代码", code)
                return df
            except Exception as exc:
                last_exc = exc
                continue
        time.sleep(1.0 * (attempt + 1))
    raise last_exc or RuntimeError("eastmoney 接口未知错误")


def _normalize(df):
    """统一列名并只保留需要的字段，按日期升序去重。"""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    date_col = next((c for c in df.columns if "日期" in c or c.lower() == "date"), None)
    price_col = "收盘" if "收盘" in df.columns else None
    if date_col is None or price_col is None:
        raise ValueError("数据缺少日期或收盘列")
    df = df[[date_col, price_col]].rename(columns={date_col: "日期", price_col: "收盘"})
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    df["收盘"] = pd.to_numeric(df["收盘"], errors="coerce")
    df = df.dropna().drop_duplicates(subset=["日期"]).sort_values("日期").reset_index(drop=True)
    return df


def _finalize(df, source, extra_warn=""):
    """按数据量与新鲜度给出告警，返回 (df, source, warn)。"""
    n = len(df)
    warns = []
    if extra_warn:
        warns.append(extra_warn)
    if n < config.MIN_ROWS_OK:
        warns.append(f"数据仅 {n} 个交易日，少于推荐的 {config.MIN_ROWS_OK} 天，结果仅供参考")
    return df, source, "；".join(warns)


def _is_fresh(df):
    """本地缓存是否足够新（最后交易日距今不超过 CACHE_MAX_AGE_DAYS 天）。"""
    if df is None or len(df) == 0:
        return False
    last = pd.Timestamp(df["日期"].iloc[-1])
    return (pd.Timestamp.now() - last).days <= config.CACHE_MAX_AGE_DAYS


def _fetch_tencent(code):
    """腾讯证券日K接口（备选数据源），按日期分段拉取，保证可用交易日 >= 1300。

    返回前复权(qfq)日K，键名 day/open/close/high/low/volume 位于行内数组。
    """
    code = str(code).strip()
    symbol = ("sh" if code.startswith(("6", "9")) else "sz") + code
    headers = dict(_COMMON_HEADERS, Referer="https://gu.qq.com/")
    start = "2019-09-01"
    step = 800  # 腾讯单次返回上限（约 800 条）
    frames = []
    cur_end = datetime.now().strftime("%Y-%m-%d")
    guard = 0
    while guard < 10:
        guard += 1
        url = ("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
               f"?param={symbol},day,{start},{cur_end},{step},qfq")
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        data = (payload.get("data") or {}).get(symbol) or {}
        rows = data.get("qfqday") or data.get("day") or []
        if not rows:
            break
        frames.append(rows)
        first_day = rows[0][0]
        if len(rows) < step or first_day <= start:
            break
        cur_end = (pd.Timestamp(first_day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if not frames:
        raise ValueError("腾讯接口未返回数据")
    # 股票名称：qt 字段形如 [序号, 名称, 代码, ...]
    name = ""
    try:
        qt = data.get("qt") or {}
        qt_inner = qt.get(symbol) if isinstance(qt, dict) else None
        if isinstance(qt_inner, list) and len(qt_inner) > 1:
            name = str(qt_inner[1])
        elif isinstance(qt_inner, dict):
            name = str(qt_inner.get("name", ""))
    except Exception:
        pass
    merged = {}
    for rows in frames:
        for r in rows:
            merged[r[0]] = r
    items = sorted(merged.items())
    if len(items) < 100:
        raise ValueError(f"腾讯接口数据不足：{len(items)} 行")
    df = pd.DataFrame([r for _, r in items]).iloc[:, :6]
    df.columns = ["日期", "开盘", "收盘", "最高", "最低", "成交量"]
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    df["收盘"] = pd.to_numeric(df["收盘"], errors="coerce")
    df = df.dropna(subset=["日期", "收盘"]).reset_index(drop=True)
    df = df[["日期", "收盘"]].copy()
    df.insert(0, "股票名称", name)
    df.insert(1, "股票代码", code)
    return df


def _fetch_sina(code):
    """新浪日K接口（备选数据源），返回正序日K。"""
    code = str(code).strip()
    symbol = ("sh" if code.startswith(("6", "9")) else "sz") + code
    headers = dict(_COMMON_HEADERS, Referer="https://finance.sina.com.cn/")
    url = ("https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData"
           f"?symbol={symbol}&scale=240&ma=no&datalen=1600")
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        raise ValueError("新浪接口未返回数据")
    name = ""
    try:
        hq = requests.get(f"https://hq.sinajs.cn/list={symbol}",
                          headers=dict(_COMMON_HEADERS,
                                       Referer="https://finance.sina.com.cn/"),
                          timeout=10)
        hq.encoding = "gbk"
        m = __import__("re").search(r'="([^,"]+),', hq.text)
        if m:
            name = m.group(1)
    except Exception:
        pass
    df = pd.DataFrame(rows)
    df["日期"] = pd.to_datetime(df["day"], errors="coerce")
    df["收盘"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["日期", "收盘"]).reset_index(drop=True)
    df = df[["日期", "收盘"]].copy()
    df.insert(0, "股票名称", name)
    df.insert(1, "股票代码", code)
    return df


def _fetch_netease(code):
    """网易历史行情接口（备选数据源），返回含名称/代码列的日K DataFrame。

    按日期区间返回 2020 年至今全量日K（含停牌日空行，已剔除），稳定且无频控。
    """
    code = str(code).strip()
    prefix = "0" if code.startswith(("6", "9")) else "1"  # 沪 0xxxxx / 深 1xxxxx
    end = datetime.now().strftime("%Y%m%d")
    url = (
        "http://quotes.money.163.com/service/chddata.html"
        f"?code={prefix}{code}&start={_EM_BEG}&end={end}"
        "&fields=TOPEN;TCLOSE;HIGH;LOW;VOTURNOVER;VATURNOVER"
    )
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    resp.encoding = "gb18030"
    df = pd.read_csv(io.StringIO(resp.text))
    if df is None or len(df) == 0:
        raise ValueError("网易接口未返回数据")
    if "日期" not in df.columns or "收盘价" not in df.columns:
        raise ValueError(f"网易接口返回格式异常：{list(df.columns)[:8]}")
    name = str(df["名称"].dropna().iloc[0]) if "名称" in df.columns else ""
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    df["收盘"] = pd.to_numeric(df["收盘价"], errors="coerce")
    df = df.dropna(subset=["日期", "收盘"])
    df = df.sort_values("日期").reset_index(drop=True)  # 网易返回倒序，转正序
    df = df[["日期", "收盘"]].copy()
    df.insert(0, "股票名称", name)
    df.insert(1, "股票代码", code)
    return df


def _fetch_efinance(code):
    """efinance 库拉取（备选数据源）。"""
    import efinance as ef
    raw = ef.stock.get_quote_history(code)
    if raw is None or len(raw) == 0:
        raise ValueError("efinance 未返回数据")
    return raw


def get_stock_data(code, max_rows=None):
    """获取指定股票的日K线（收盘价），返回 (df, source, warn)。

    策略：本地缓存新鲜则直接用；过期则尝试在线刷新（东财直连/efinance）；
    在线失败才回退旧缓存/示例数据，并提示数据可能滞后。
    df 列：日期/收盘；source：'eastmoney' | 'efinance' | 'local' | 'sample'；
    warn：数据滞后或不足时的提示字符串（可为空）。
    """
    code = str(code).strip()
    max_rows = config.MAX_ROWS if max_rows is None else int(max_rows)
    _errors = []
    with _lock:
        local_file = os.path.join(config.DATA_DIR, f"{code}.csv")
        cached = None
        if os.path.exists(local_file):
            try:
                cached = _normalize(pd.read_csv(local_file))
            except Exception:
                cached = None

        # 1) 缓存新鲜：直接用
        if cached is not None and _is_fresh(cached) and len(cached) >= config.MIN_ROWS_WARN:
            return _finalize(cached, "local")

        # 2) 在线拉取：东财 -> 腾讯 -> 新浪 -> 网易 -> efinance（缓存过期/缺失时刷新到最新）
        for fetcher, label in ((_fetch_eastmoney, "eastmoney"),
                               (_fetch_tencent, "tencent"),
                               (_fetch_sina, "sina"),
                               (_fetch_netease, "netease"),
                               (_fetch_efinance, "efinance")):
            try:
                raw = fetcher(code)
                if raw is not None and len(raw) > 0:
                    # 保存完整原始数据（含股票名称/股票代码列）到本地缓存
                    save_df = raw.copy()
                    if len(save_df) > max_rows:
                        save_df = save_df.iloc[-max_rows:].reset_index(drop=True)
                    os.makedirs(config.DATA_DIR, exist_ok=True)
                    save_df.to_csv(local_file, index=False, encoding="utf-8")
                    df = _normalize(save_df)
                    return _finalize(df, label)
            except Exception as exc:
                _errors.append(f"{label}: {exc}")
                print(f"[data_service] {label} 拉取失败: {exc}")

        # 3) 回退：旧缓存 / 示例数据（提示数据滞后）
        if cached is not None and len(cached) >= config.MIN_ROWS_WARN:
            warn = f"在线刷新失败，使用本地缓存（数据截至 {cached['日期'].iloc[-1].date()}）"
            return _finalize(cached, "local", warn)

        df = _load_legacy_sample()
        if df is not None and str(df.iloc[0]["股票代码"]) == code:
            os.makedirs(config.DATA_DIR, exist_ok=True)
            df.to_csv(os.path.join(config.DATA_DIR, f"{code}.csv"), index=False, encoding="utf-8")
            return _finalize(_normalize(df), "sample",
                             "在线刷新失败，使用内置示例数据")

        detail = "；".join(_errors) if _errors else "全部数据源均失败"
        raise ValueError(
            f"无法获取股票 {code} 的数据：{detail}。"
            "请检查网络后重试，或改用示例代码 003015。"
        )
