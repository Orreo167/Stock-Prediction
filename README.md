# 股票开盘价预测系统

基于 **ARIMA-LSTM 混合模型** 的 A 股开盘价预测 Web 应用。输入股票代码即可自动拉取日 K 线、训练 5 个模型（ARIMA / LSTM / SVR / BP / ARIMA-LSTM），对比测试集表现，并给出未来 5–60 天（默认 30 天）的开盘价预测。

## 功能

- 输入 6 位 A 股代码（如 `003015`），自动在线拉取日 K 线（≥1300 个交易日；东方财富直连优先，腾讯/新浪/网易/efinance 备选）
- 训练并对比 5 个模型：**ARIMA、LSTM、SVR、BP、ARIMA-LSTM 混合模型**（ARIMA 主预测 + LSTM 学习 ARIMA 残差）
- 测试集 80/20 划分（时间顺序），展示 MAE / MSE / RMSE / R² 指标对比
- 两张核心图表：
  1. **测试期预测对比图**：真实开盘价 vs 各模型
  2. **未来预测图**：GARCH 波动率蒙特卡洛——模型点预测为中心趋势，叠加 GARCH(1,1) 时变波动率采样生成多条模拟路径，展示中位数预测线 + 5%/95% 分位带 + 灰色样本路径（波动幅度接近真实股票）
- 预测天数滑杆 5–60 天，默认 30 天，超过 30 天提示"预测步长越大，误差越大"
- 按股票代码缓存训练产物，二次查询秒出；改预测天数只重新预测、不重训
- 历史记录：输入框聚焦自动下拉最近 20 条查询（代码 + 名称），点击只填充代码，支持单条删除与清空历史（localStorage 保存）
- 高级选项：可调数据天数上限（1000–3000）、训练/测试划分比例（50%–90%）、模型启用开关、GARCH 随机种子，以及 ARIMA/LSTM/SVR/BP/ARIMA-LSTM 各模型参数；按钮旁灰色提示"修改参数可能导致预测效果变差并触发重新训练，推荐使用默认值"，修改后自动触发重训
- 训练进度：阶段级进度条 + 动态加载动画
- 内置示例数据 `data/003015.csv`，离线或拉取失败时可演示

## 项目结构

```
webapp/
├─ backend/
│  ├─ main.py            # FastAPI 入口：静态托管 + 异步任务 + 轮询
│  ├─ trainer.py         # 串行训练编排、最优模型评选、结果缓存
│  ├─ data_service.py    # 在线拉取（东财直连/腾讯/新浪/网易/efinance）+ 本地缓存 + 示例回退
│  ├─ config.py          # 常量与路径
│  ├─ utils.py           # 指标计算、滑窗、递归预测
│  └─ models/            # 五个模型实现（含持久化）
├─ frontend/
│  ├─ index.html         # 单页前端
│  ├─ css/style.css
│  └─ js/                # app.js + 本地 echarts.min.js
├─ data/                 # 数据缓存（003015.csv 为随仓库示例）
├─ cache/                # 训练产物缓存（.gitignore）
├─ requirements.txt      # 依赖下限（可复现基线）
├─ requirements-lock.txt # 精确锁定版本
├─ run.bat / run.sh      # 一键启动
└─ README.md
```

## 环境要求

- **Python 3.8 ~ 3.10**（tensorflow 2.9 不支持 3.11+）
- 已验证环境：Python 3.10.9 + tensorflow 2.9.0（见 `requirements-lock.txt`）

## 安装与启动

```bash
# 1. 创建虚拟环境（可选但推荐）
python -m venv venv
# Windows: venv\Scripts\activate   |   macOS/Linux: source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt
# 或精确复现：pip install -r requirements-lock.txt

# 3. 启动服务
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
# 或直接运行：Windows 双击 run.bat；Linux/macOS ./run.sh
```

浏览器打开 <http://127.0.0.1:8000>。

> 提示：首次训练一个股票约需 1–3 分钟（CPU），期间页面显示进度条；同一股票二次查询直接命中缓存。

## 使用说明

1. 输入 6 位股票代码（如 `003015`），或点击快捷示例
2. 拖动滑杆选择预测天数（5–60，默认 30）
3. 点击"开始预测"，等待进度条完成
4. 查看指标表、测试期对比图、未来预测图

## 模型说明

- **ARIMA**：`pmdarima.auto_arima` 自动定阶；测试期单步滚动预测 + 模型更新
- **LSTM**：滑窗 5 天，LSTM(48) + Dropout(0.1) + Dense(1)，StandardScaler 标准化，50 epochs
- **SVR**：RBF 核，C=100 / epsilon=0.1 / gamma=scale，前一日开盘价预测次日
- **BP**：滑窗 30 天，Dense 64→32→1，MinMaxScaler，100 epochs
- **ARIMA-LSTM 混合**：ARIMA 预测主序列，LSTM 学习并预测 ARIMA 残差，二者相加
- 预测模型：自动选择测试期指标最优模型（R² 最高，并列取 RMSE 更低者）
- 波动率：**GARCH(1,1)**（`arch` 库）拟合历史收益率预测未来时变波动率；`arch` 未安装时自动降级为最近 60 天历史标准差

## 数据来源

- 在线：直连东方财富 K 线接口（按 A 股规则构造 secid，含重试/多通道，失败时依次回退腾讯/新浪/网易接口、最后回退 `efinance`），拉取到**最新交易日**
- 本地缓存：`data/<code>.csv`，缓存超过 7 天未更新时自动在线刷新
- 在线失败：回退旧缓存/示例数据并提示"数据截至 XX 日期"，避免静默使用过期数据
- 示例：`data/003015.csv`（日久光电）——离线演示回退
- 训练默认使用最近 1500 个交易日（可在高级选项调整为 1000–3000）；数据不足 1300 天时提示"结果仅供参考"；不足 500 天拒绝训练
- 注意：后端必须在你本机正常启动（非沙箱/代理限制环境），三个在线数据源才能联网拉取
