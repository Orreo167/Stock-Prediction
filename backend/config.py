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
CACHE_MAX_AGE_DAYS = 7      # 本地缓存超过 N 天未更新则强制在线刷新
FORECAST_HORIZON_DEFAULT = 30
FORECAST_HORIZON_MIN = 5
FORECAST_HORIZON_MAX = 60

# 训练配置（沿用 notebook 验证过的最优/常用参数）
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

# 模型训练顺序（串行）
MODEL_ORDER = ["ARIMA", "LSTM", "SVR", "BP", "ARIMA-LSTM"]
