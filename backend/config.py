"""全局配置：路径与关键常量。"""

import os

# 项目根目录（webapp/）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

# 示例数据（离线演示/兜底用），随仓库提交
SAMPLE_DATA_FILE = os.path.join(DATA_DIR, "003015.csv")
# 源数据（原项目文件夹，仅本机存在；未提交到仓库）
LEGACY_SAMPLE_FILE = os.path.join(os.path.dirname(BASE_DIR), "代码", "数据003015.csv")

# 数据要求
MIN_ROWS_OK = 1300          # 理想下限
MIN_ROWS_WARN = 500         # 低于此值拒绝训练
MAX_ROWS = 1500             # 训练最多使用的交易日数（超出截取最近部分，控制训练时间）
MAX_ROWS_MIN = 1000         # 高级选项中数据天数上限的可调下限
MAX_ROWS_MAX = 3000         # 高级选项中数据天数上限的可调上限
CACHE_MAX_AGE_DAYS = 7      # 本地缓存超过 N 天未更新则强制在线刷新
FORECAST_HORIZON_DEFAULT = 30
FORECAST_HORIZON_MIN = 5
FORECAST_HORIZON_MAX = 60

# 训练配置（沿用 notebook 验证过的最优/常用参数）
ARIMA_MAX_P = 10            # auto_arima 最大 p 阶
ARIMA_MAX_Q = 10            # auto_arima 最大 q 阶

LSTM_TIME_STEP = 5
LSTM_UNITS = 48
LSTM_DROPOUT = 0.1
LSTM_EPOCHS = 50
LSTM_BATCH = 32

BP_TIME_STEP = 30
BP_UNITS_1 = 64
BP_UNITS_2 = 32
BP_EPOCHS = 100
BP_BATCH = 32

SVR_C = 100
SVR_EPSILON = 0.1
SVR_GAMMA = "scale"

TRAIN_SIZE = 0.8
TRAIN_SIZE_MIN = 0.5        # 高级选项中训练/测试划分比例的可调下限
TRAIN_SIZE_MAX = 0.9        # 高级选项中训练/测试划分比例的可调上限

# 高级选项中的模型参数默认值（前端“恢复默认”与此保持一致）
MODEL_PARAM_DEFAULTS = {
    "ARIMA": {"max_p": ARIMA_MAX_P, "max_q": ARIMA_MAX_Q},
    "LSTM": {
        "time_step": LSTM_TIME_STEP,
        "units": LSTM_UNITS,
        "dropout": LSTM_DROPOUT,
        "epochs": LSTM_EPOCHS,
        "batch": LSTM_BATCH,
    },
    "SVR": {"C": SVR_C, "epsilon": SVR_EPSILON, "gamma": SVR_GAMMA},
    "BP": {
        "time_step": BP_TIME_STEP,
        "units_1": BP_UNITS_1,
        "units_2": BP_UNITS_2,
        "epochs": BP_EPOCHS,
        "batch": BP_BATCH,
    },
    "ARIMA-LSTM": {
        "max_p": ARIMA_MAX_P,
        "max_q": ARIMA_MAX_Q,
        "time_step": LSTM_TIME_STEP,
        "units": LSTM_UNITS,
        "dropout": LSTM_DROPOUT,
        "epochs": LSTM_EPOCHS,
        "batch": LSTM_BATCH,
    },
}

# 高级选项中模型参数的可调范围（None 表示不限制，如 SVR gamma 字符串）
MODEL_PARAM_LIMITS = {
    "ARIMA": {"max_p": (1, 30), "max_q": (1, 30)},
    "LSTM": {
        "time_step": (1, 30), "units": (8, 256), "dropout": (0, 0.5),
        "epochs": (10, 200), "batch": (8, 128),
    },
    "SVR": {"C": (0.1, 1000), "epsilon": (0.01, 1), "gamma": None},
    "BP": {
        "time_step": (1, 60), "units_1": (8, 256), "units_2": (4, 128),
        "epochs": (10, 200), "batch": (8, 128),
    },
    "ARIMA-LSTM": {
        "max_p": (1, 30), "max_q": (1, 30),
        "time_step": (1, 30), "units": (8, 256), "dropout": (0, 0.5),
        "epochs": (10, 200), "batch": (8, 128),
    },
}

# 模型训练顺序（串行）
MODEL_ORDER = ["ARIMA", "LSTM", "SVR", "BP", "ARIMA-LSTM"]

# GARCH 蒙特卡洛路径（方案 3：递归点预测中心 + GARCH 波动率 + 随机扰动）
N_PATHS = 500            # 模拟路径条数
N_PATHS_SHOW = 20        # 前端展示的灰色样本路径条数
PATH_SEED = 42           # 随机种子（结果可复现）
GARCH_P = 1              # GARCH(p, q) 阶数
GARCH_Q = 1
GARCH_DIST = "t"         # 扰动分布：学生 t（肥尾，更接近真实收益）
GARCH_WINDOW_FALLBACK = 60  # arch 不可用时的降级波动窗口（最近 N 天收益率 std）
